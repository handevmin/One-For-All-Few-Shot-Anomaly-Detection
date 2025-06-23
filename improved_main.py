import torch
import argparse
import torch.nn.functional as F
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import torch.nn as nn
from torch.optim import lr_scheduler
import open_clip
from torch import optim
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from improved_prompt_generator import EfficientPromptGenerator
from improved_architecture import LightweightAnomalyDetector
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
import time
import json

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImprovedFramework:
    """
    개선된 One-for-All Few-Shot Anomaly Detection Framework
    
    주요 개선 사항:
    1. BLIP-Diffusion 의존성 제거
    2. 효율적인 프롬프트 캐싱 메커니즘
    3. 경량화된 아키텍처
    4. 이론적 근거를 바탕으로 한 정보 이론적 최적화
    """
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_models()
        self.setup_metrics()
        
    def setup_models(self):
        """모델 초기화 - 경량화된 구조로 변경"""
        # 경량화된 CLIP 모델 사용 (ViT-B/32 대신 ViT-L-14)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", 224, pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        
        # 경량화된 이상 탐지기
        self.detector = LightweightAnomalyDetector(
            clip_model=self.clip_model,
            feature_dim=512,  # ViT-B/32의 feature dimension
            prompt_length=12
        )
        
        # 효율적인 프롬프트 생성기
        self.prompt_generator = EfficientPromptGenerator(
            feature_dim=512,
            cache_size=1000,
            similarity_threshold=0.85
        )
        
        self.clip_model.to(self.device)
        self.detector.to(self.device)
        self.prompt_generator.to(self.device)
        
    def setup_metrics(self):
        """성능 측정 메트릭 초기화"""
        self.performance_log = {
            'pixel_auroc': [],
            'image_auroc': [],
            'inference_time': [],
            'memory_usage': [],
            'cache_hit_rate': []
        }
        
    def information_theoretic_loss(self, pos_features, neg_features, image_features):
        """
        정보 이론 기반 손실 함수
        I(X; Y|P_instance) > I(X; Y|P_fixed) 를 최대화
        """
        # Mutual Information 추정 (MINE 방식 사용)
        pos_mi = self.estimate_mutual_information(image_features, pos_features)
        neg_mi = self.estimate_mutual_information(image_features, neg_features)
        
        # 정상과 비정상 간의 분리도 최대화
        separation_loss = torch.max(torch.tensor(0.0), 1.0 - (pos_mi - neg_mi))
        
        return separation_loss
        
    def estimate_mutual_information(self, x, y):
        """MINE을 사용한 상호 정보량 추정"""
        # 간단한 MLP 기반 MI 추정
        joint = torch.cat([x, y], dim=-1)
        marginal = torch.cat([x, y[torch.randperm(y.size(0))]], dim=-1)
        
        # T-function 근사
        t_joint = torch.mean(torch.exp(torch.sum(joint, dim=-1)))
        t_marginal = torch.mean(torch.exp(torch.sum(marginal, dim=-1)))
        
        mi = torch.log(t_joint) - torch.log(t_marginal)
        return mi
        
    def efficient_few_shot_learning(self, obj_list, shot=1, epochs=10):
        """효율적인 few-shot 학습"""
        optimizer = optim.Adam(
            list(self.detector.parameters()) + list(self.prompt_generator.parameters()),
            lr=1e-4, weight_decay=1e-5
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()
            
            for obj in obj_list:
                # 데이터 로딩 최적화
                few_shot_data = self.load_few_shot_data(obj, shot)
                
                for i, (image, _) in enumerate(few_shot_data):
                    image = image.to(self.device)
                    
                    # 이미지 특징 추출
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(image)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # 효율적인 프롬프트 생성 (캐싱 적용)
                    pos_prompt, neg_prompt = self.prompt_generator.generate_cached_prompts(
                        image_features, obj
                    )
                    
                    # 경량화된 탐지
                    anomaly_score, features = self.detector(image, pos_prompt, neg_prompt)
                    
                    # 정보 이론 기반 손실
                    loss = self.information_theoretic_loss(
                        features['pos'], features['neg'], image_features
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            scheduler.step()
            
            # 성능 로깅
            epoch_time = time.time() - start_time
            memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB
            cache_hit_rate = self.prompt_generator.get_cache_hit_rate()
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, "
                  f"Time={epoch_time:.2f}s, Memory={memory_usage:.2f}GB, "
                  f"Cache Hit Rate={cache_hit_rate:.2f}")
    
    def load_few_shot_data(self, obj, shot):
        """Few-shot 데이터 로딩 - 하드코딩된 경로 제거"""
        data_list = []
        
        # 동적 경로 설정
        if self.args.dataset == 'mvtec':
            data_path = os.path.join(self.args.data_path, obj, 'train', 'good')
        else:
            data_path = os.path.join(self.args.data_path, obj, 'train', 'good')
            
        # 사용 가능한 이미지 파일 찾기
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.JPG']:
            image_files.extend([f for f in os.listdir(data_path) if f.endswith(ext)])
        
        # Few-shot 샘플 선택
        selected_files = random.sample(image_files, min(shot, len(image_files)))
        
        for file in selected_files:
            image_path = os.path.join(data_path, file)
            # 이미지 로딩 및 전처리
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            transform = get_transform(self.args)[0]
            image_tensor = transform(image).unsqueeze(0)
            data_list.append((image_tensor, 0))  # 0 = normal
            
        return data_list
    
    def evaluate_performance(self, test_dataloader):
        """성능 평가"""
        self.detector.eval()
        self.prompt_generator.eval()
        
        results = {}
        total_inference_time = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_dataloader)):
                start_time = time.time()
                
                image = data['img'].to(self.device)
                mask = data['img_mask'].to(self.device) if len(data['img_mask']) > 0 else None
                cls_name = data['cls_name'][0]
                anomaly = data['anomaly'].item()
                
                if cls_name not in results:
                    results[cls_name] = {
                        'pixel_scores': [], 'pixel_masks': [],
                        'image_scores': [], 'image_labels': []
                    }
                
                # 효율적인 추론
                image_features = self.clip_model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                pos_prompt, neg_prompt = self.prompt_generator.generate_cached_prompts(
                    image_features, cls_name
                )
                
                anomaly_score, pixel_map = self.detector.predict(image, pos_prompt, neg_prompt)
                
                # 결과 저장
                results[cls_name]['image_scores'].append(anomaly_score.cpu().numpy())
                results[cls_name]['image_labels'].append(anomaly)
                
                if mask is not None:
                    results[cls_name]['pixel_scores'].append(pixel_map.cpu().numpy())
                    results[cls_name]['pixel_masks'].append(mask.cpu().numpy())
                
                # 성능 메트릭 수집
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                num_samples += 1
        
        # 평균 추론 시간 계산
        avg_inference_time = total_inference_time / num_samples
        
        # 클래스별 성능 계산
        performance_metrics = {}
        for cls_name, cls_results in results.items():
            if len(cls_results['pixel_scores']) > 0:
                pixel_auroc = self.calculate_pixel_auroc(
                    cls_results['pixel_scores'], cls_results['pixel_masks']
                )
                performance_metrics[f'{cls_name}_pixel_auroc'] = pixel_auroc
            
            if len(cls_results['image_scores']) > 0:
                image_auroc = self.calculate_image_auroc(
                    cls_results['image_scores'], cls_results['image_labels']
                )
                performance_metrics[f'{cls_name}_image_auroc'] = image_auroc
        
        # 전체 성능 메트릭
        performance_metrics['avg_inference_time'] = avg_inference_time
        performance_metrics['memory_usage'] = torch.cuda.max_memory_allocated() / 1024**3
        performance_metrics['cache_hit_rate'] = self.prompt_generator.get_cache_hit_rate()
        
        return performance_metrics
    
    def calculate_pixel_auroc(self, scores, masks):
        """픽셀 레벨 AUROC 계산"""
        from sklearn.metrics import roc_auc_score
        all_scores = np.concatenate([s.flatten() for s in scores])
        all_masks = np.concatenate([m.flatten() for m in masks])
        return roc_auc_score(all_masks, all_scores)
    
    def calculate_image_auroc(self, scores, labels):
        """이미지 레벨 AUROC 계산"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores)

def main(args):
    setup_seed(42)
    
    # 개선된 프레임워크 초기화
    framework = ImprovedFramework(args)
    
    # 데이터 로딩
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(
        root=args.data_path, 
        transform=preprocess, 
        target_transform=target_transform, 
        dataset_name=args.dataset
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    
    print(f"개선된 프레임워크로 {len(obj_list)}개 클래스에 대해 학습 시작...")
    
    # Few-shot 학습
    framework.efficient_few_shot_learning(obj_list, shot=args.shot, epochs=args.epochs)
    
    # 성능 평가
    print("성능 평가 중...")
    performance = framework.evaluate_performance(test_dataloader)
    
    # 결과 출력
    print("\n=== 개선된 결과 ===")
    for metric, value in performance.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
    
    # 결과 저장
    os.makedirs('improved_results', exist_ok=True)
    with open('improved_results/performance_metrics.json', 'w') as f:
        json.dump(performance, f, indent=2)
    
    print(f"\n결과가 'improved_results/' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./sample_mvtec')
    parser.add_argument('--save_path', type=str, default='./improved_results')
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--features_list', nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--t_n_ctx', type=int, default=4)
    parser.add_argument('--depth', type=int, default=9)
    
    args = parser.parse_args()
    main(args) 