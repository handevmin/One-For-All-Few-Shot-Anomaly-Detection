import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math
import time

class LightweightAnomalyDetector(nn.Module):
    """
    경량화된 이상 탐지기
    
    주요 개선 사항:
    1. BLIP-Diffusion 의존성 제거
    2. 효율적인 특징 융합
    3. 적응적 임계값 설정
    4. 실시간 처리를 위한 최적화
    """
    
    def __init__(self, clip_model, feature_dim=512, prompt_length=12):
        super(LightweightAnomalyDetector, self).__init__()
        
        self.clip_model = clip_model
        self.feature_dim = feature_dim
        self.prompt_length = prompt_length
        
        # 특징 융합 네트워크
        self.feature_fusion = FeatureFusionNetwork(feature_dim)
        
        # 이상 점수 계산기
        self.anomaly_scorer = AnomalyScoreCalculator(feature_dim)
        
        # 적응적 임계값 모듈
        self.adaptive_threshold = AdaptiveThresholdModule(feature_dim)
        
        # 공간적 어텐션 (픽셀 레벨 탐지용)
        self.spatial_attention = SpatialAttentionModule(feature_dim)
        
        self.freeze_clip_backbone()
    
    def freeze_clip_backbone(self):
        """CLIP 백본 고정하여 효율성 향상"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, image, pos_prompt, neg_prompt):
        """
        Args:
            image: [batch_size, 3, H, W]
            pos_prompt: [batch_size, prompt_length, feature_dim]
            neg_prompt: [batch_size, prompt_length, feature_dim]
        
        Returns:
            anomaly_score: [batch_size] - 이미지 레벨 이상 점수
            features: Dict - 중간 특징들
        """
        batch_size = image.size(0)
        
        # CLIP 이미지 인코딩 (그래디언트 차단)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 프롬프트 인코딩
        pos_text_features = self.encode_prompt_features(pos_prompt)
        neg_text_features = self.encode_prompt_features(neg_prompt)
        
        # 특징 융합
        fused_features = self.feature_fusion(
            image_features, pos_text_features, neg_text_features
        )
        
        # 이상 점수 계산
        anomaly_score = self.anomaly_scorer(fused_features)
        
        # 적응적 임계값 적용
        adjusted_score = self.adaptive_threshold(anomaly_score, image_features)
        
        features = {
            'pos': pos_text_features,
            'neg': neg_text_features,
            'fused': fused_features,
            'image': image_features
        }
        
        return adjusted_score, features
    
    def predict(self, image, pos_prompt, neg_prompt):
        """
        추론 전용 메서드 (픽셀 맵 포함)
        
        Returns:
            anomaly_score: 이미지 레벨 점수
            pixel_map: 픽셀 레벨 이상 맵
        """
        self.eval()
        
        with torch.no_grad():
            # 기본 예측
            anomaly_score, features = self.forward(image, pos_prompt, neg_prompt)
            
            # 픽셀 레벨 맵 생성
            pixel_map = self.generate_pixel_map(image, features)
            
        return anomaly_score, pixel_map
    
    def generate_pixel_map(self, image, features):
        """픽셀 레벨 이상 맵 생성"""
        # 공간적 어텐션을 통한 픽셀 맵 생성
        pixel_map = self.spatial_attention(
            features['image'], features['fused']
        )
        
        # 이미지 크기로 업샘플링
        target_size = (image.size(2), image.size(3))
        pixel_map = F.interpolate(
            pixel_map.unsqueeze(1), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        return pixel_map
    
    def encode_prompt_features(self, prompt):
        """프롬프트를 텍스트 특징으로 인코딩"""
        # 프롬프트를 평균내어 단일 특징 벡터로 변환
        prompt_features = torch.mean(prompt, dim=1)  # [batch_size, feature_dim]
        prompt_features = F.normalize(prompt_features, dim=-1)
        return prompt_features


class FeatureFusionNetwork(nn.Module):
    """효율적인 특징 융합 네트워크"""
    
    def __init__(self, feature_dim=512):
        super(FeatureFusionNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 크로스 어텐션 (경량화)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 특징 변환
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 게이팅 메커니즘
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, image_features, pos_features, neg_features):
        """
        Args:
            image_features: [batch_size, feature_dim]
            pos_features: [batch_size, feature_dim]  
            neg_features: [batch_size, feature_dim]
        
        Returns:
            fused_features: [batch_size, feature_dim]
        """
        batch_size = image_features.size(0)
        
        # 모든 특징을 시퀀스로 변환
        features_seq = torch.stack([
            image_features, pos_features, neg_features
        ], dim=1)  # [batch_size, 3, feature_dim]
        
        # 크로스 어텐션 적용
        attended_features, _ = self.cross_attention(
            features_seq, features_seq, features_seq
        )
        
        # 특징 연결 및 변환
        concatenated = attended_features.view(batch_size, -1)  # [batch_size, 3*feature_dim]
        transformed = self.feature_transform(concatenated)
        
        # 게이팅을 통한 적응적 융합
        gate_weights = self.gate(image_features)
        fused_features = gate_weights * transformed + (1 - gate_weights) * image_features
        
        return fused_features


class AnomalyScoreCalculator(nn.Module):
    """이상 점수 계산기"""
    
    def __init__(self, feature_dim=512):
        super(AnomalyScoreCalculator, self).__init__()
        
        # 다층 퍼셉트론 기반 점수 계산
        self.score_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 앙상블을 위한 추가 헤드
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, feature_dim]
        
        Returns:
            anomaly_score: [batch_size]
        """
        # 메인 점수
        main_score = self.score_network(features).squeeze(-1)
        
        # 앙상블 점수
        ensemble_scores = []
        for head in self.ensemble_heads:
            score = head(features).squeeze(-1)
            ensemble_scores.append(score)
        
        # 가중 평균
        ensemble_score = torch.mean(torch.stack(ensemble_scores), dim=0)
        final_score = 0.7 * main_score + 0.3 * ensemble_score
        
        return final_score


class AdaptiveThresholdModule(nn.Module):
    """적응적 임계값 모듈"""
    
    def __init__(self, feature_dim=512):
        super(AdaptiveThresholdModule, self).__init__()
        
        # 임계값 예측 네트워크
        self.threshold_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 동적 조정 파라미터
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.momentum = 0.1
        
    def forward(self, anomaly_score, image_features):
        """
        Args:
            anomaly_score: [batch_size]
            image_features: [batch_size, feature_dim]
        
        Returns:
            adjusted_score: [batch_size]
        """
        # 이미지 특징 기반 임계값 예측
        predicted_threshold = self.threshold_predictor(image_features).squeeze(-1)
        
        # 통계 기반 정규화
        if self.training:
            batch_mean = torch.mean(anomaly_score)
            batch_var = torch.var(anomaly_score)
            
            # 이동 평균 업데이트
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        # 점수 정규화
        normalized_score = (anomaly_score - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        
        # 적응적 조정
        adjusted_score = torch.sigmoid(normalized_score + predicted_threshold)
        
        return adjusted_score


class SpatialAttentionModule(nn.Module):
    """공간적 어텐션 모듈 (픽셀 레벨 탐지용)"""
    
    def __init__(self, feature_dim=512):
        super(SpatialAttentionModule, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 공간적 특징 추출기
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 글로벌 컨텍스트 인코더
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image_features, fused_features):
        """
        Args:
            image_features: [batch_size, feature_dim] - 글로벌 이미지 특징
            fused_features: [batch_size, feature_dim] - 융합된 특징
        
        Returns:
            spatial_map: [batch_size, H', W'] - 공간적 이상 맵
        """
        batch_size = image_features.size(0)
        
        # 간단한 공간 맵 생성 (실제 구현에서는 더 복잡한 로직 필요)
        # 여기서는 특징 차이를 기반으로 한 간단한 맵 생성
        feature_diff = torch.abs(image_features - fused_features)  # [batch_size, feature_dim]
        
        # 공간 맵으로 변환 (7x7 크기로 가정)
        spatial_size = 7
        spatial_map = feature_diff.view(batch_size, 1, 1, -1)
        spatial_map = F.adaptive_avg_pool2d(spatial_map, (spatial_size, spatial_size))
        spatial_map = spatial_map.view(batch_size, spatial_size, spatial_size)
        
        return spatial_map


class EfficientMemoryBank:
    """효율적인 메모리 뱅크 (클래스별 특징 저장)"""
    
    def __init__(self, feature_dim=512, max_samples_per_class=100):
        self.feature_dim = feature_dim
        self.max_samples_per_class = max_samples_per_class
        self.memory_banks = {}  # {class_name: torch.Tensor}
        
    def update(self, class_name, features):
        """클래스별 메모리 뱅크 업데이트"""
        if class_name not in self.memory_banks:
            self.memory_banks[class_name] = []
        
        # 특징 추가
        self.memory_banks[class_name].append(features.detach().cpu())
        
        # 크기 제한
        if len(self.memory_banks[class_name]) > self.max_samples_per_class:
            # 오래된 특징 제거 (FIFO)
            self.memory_banks[class_name].pop(0)
    
    def get_class_prototype(self, class_name):
        """클래스 프로토타입 계산"""
        if class_name not in self.memory_banks or len(self.memory_banks[class_name]) == 0:
            return None
        
        # 평균 특징 계산
        features_stack = torch.stack(self.memory_banks[class_name])
        prototype = torch.mean(features_stack, dim=0)
        
        return prototype
    
    def compute_anomaly_score(self, class_name, query_features):
        """메모리 뱅크 기반 이상 점수 계산"""
        prototype = self.get_class_prototype(class_name)
        
        if prototype is None:
            return torch.tensor(0.5)  # 기본값
        
        # 코사인 유사도 기반 점수
        prototype = prototype.to(query_features.device)
        similarity = F.cosine_similarity(
            query_features.squeeze(), prototype, dim=0
        )
        
        # 유사도를 이상 점수로 변환 (낮은 유사도 = 높은 이상 점수)
        anomaly_score = 1.0 - similarity
        
        return torch.clamp(anomaly_score, 0.0, 1.0)


class ModelEfficiencyAnalyzer:
    """모델 효율성 분석기"""
    
    @staticmethod
    def analyze_model_complexity(model):
        """모델 복잡도 분석"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 메모리 사용량 추정 (MB)
        memory_mb = total_params * 4 / (1024 * 1024)
        
        # FLOPs 추정 (간단한 방식)
        flops_estimate = trainable_params * 2  # 대략적인 추정
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_usage_mb': memory_mb,
            'flops_estimate': flops_estimate,
            'parameter_efficiency': trainable_params / total_params
        }
    
    @staticmethod
    def benchmark_inference_speed(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
        """추론 속도 벤치마킹"""
        model.eval()
        
        # 더미 입력 생성
        dummy_image = torch.randn(input_size).to(device)
        dummy_pos_prompt = torch.randn(1, 12, 512).to(device)
        dummy_neg_prompt = torch.randn(1, 12, 512).to(device)
        
        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_image, dummy_pos_prompt, dummy_neg_prompt)
        
        # 실제 측정
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_image, dummy_pos_prompt, dummy_neg_prompt)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        fps = 1000 / avg_time_ms
        
        return {
            'avg_inference_time_ms': avg_time_ms,
            'fps': fps,
            'total_time_s': end_time - start_time
        } 