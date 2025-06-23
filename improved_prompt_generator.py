import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
import time

class EfficientPromptGenerator(nn.Module):
    """
    효율적인 프롬프트 생성기
    
    주요 개선 사항:
    1. 프롬프트 캐싱 메커니즘
    2. KNN 기반 유사 프롬프트 재사용
    3. 경량화된 네트워크 구조
    4. 메모리 효율적인 특징 저장
    """
    
    def __init__(self, feature_dim=512, prompt_length=12, cache_size=1000, similarity_threshold=0.85):
        super(EfficientPromptGenerator, self).__init__()
        
        self.feature_dim = feature_dim
        self.prompt_length = prompt_length
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        # 경량화된 프롬프트 생성 네트워크
        self.prompt_network = PromptGenerationNetwork(feature_dim, prompt_length)
        
        # 프롬프트 캐시
        self.prompt_cache = OrderedDict()  # {feature_hash: (pos_prompt, neg_prompt)}
        self.feature_cache = OrderedDict()  # {feature_hash: feature_vector}
        
        # 성능 메트릭
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 초기 프롬프트 템플릿
        self.register_buffer('pos_template', torch.randn(1, prompt_length, feature_dim))
        self.register_buffer('neg_template', torch.randn(1, prompt_length, feature_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.pos_template)
        nn.init.xavier_uniform_(self.neg_template)
    
    def forward(self, image_features, class_name=None):
        """순전파 - 프롬프트 생성"""
        pos_prompt, neg_prompt = self.prompt_network(image_features)
        return pos_prompt, neg_prompt
    
    def generate_cached_prompts(self, image_features, class_name):
        """캐싱 메커니즘을 활용한 프롬프트 생성"""
        # 특징 해시 생성
        feature_hash = self._compute_feature_hash(image_features)
        
        # 캐시에서 찾기
        cached_prompt = self._search_cache(feature_hash, image_features)
        
        if cached_prompt is not None:
            self.cache_hits += 1
            return cached_prompt
        
        # 캐시 미스 - 새로운 프롬프트 생성
        self.cache_misses += 1
        pos_prompt, neg_prompt = self.forward(image_features)
        
        # 캐시에 저장
        self._update_cache(feature_hash, image_features, (pos_prompt, neg_prompt))
        
        return pos_prompt, neg_prompt
    
    def _compute_feature_hash(self, features):
        """특징 벡터의 해시 계산"""
        # 특징을 정규화하고 양자화하여 해시 생성
        normalized_features = F.normalize(features, dim=-1)
        quantized = torch.round(normalized_features * 1000).long()
        feature_hash = hash(quantized.cpu().numpy().tobytes())
        return feature_hash
    
    def _search_cache(self, feature_hash, current_features):
        """캐시에서 유사한 프롬프트 검색"""
        # 정확한 해시 매치 먼저 확인
        if feature_hash in self.prompt_cache:
            return self.prompt_cache[feature_hash]
        
        # KNN 기반 유사도 검색
        if len(self.feature_cache) > 0:
            similar_prompt = self._find_similar_cached_prompt(current_features)
            if similar_prompt is not None:
                return similar_prompt
        
        return None
    
    def _find_similar_cached_prompt(self, current_features):
        """KNN을 사용하여 유사한 프롬프트 찾기"""
        max_similarity = 0
        best_prompt = None
        
        current_features_norm = F.normalize(current_features, dim=-1)
        
        # 최대 50개의 최근 캐시만 검색 (효율성을 위해)
        search_limit = min(50, len(self.feature_cache))
        cache_items = list(self.feature_cache.items())[-search_limit:]
        
        for feature_hash, cached_features in cache_items:
            # 코사인 유사도 계산
            cached_features_norm = F.normalize(cached_features, dim=-1)
            similarity = torch.cosine_similarity(
                current_features_norm, cached_features_norm, dim=-1
            ).item()
            
            if similarity > max_similarity and similarity > self.similarity_threshold:
                max_similarity = similarity
                if feature_hash in self.prompt_cache:
                    best_prompt = self.prompt_cache[feature_hash]
        
        return best_prompt
    
    def _update_cache(self, feature_hash, features, prompts):
        """캐시 업데이트"""
        # 캐시 크기 제한
        if len(self.prompt_cache) >= self.cache_size:
            # 오래된 항목 제거 (LRU)
            oldest_hash = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_hash]
            if oldest_hash in self.feature_cache:
                del self.feature_cache[oldest_hash]
        
        # 새 항목 추가
        self.prompt_cache[feature_hash] = prompts
        self.feature_cache[feature_hash] = features.detach().clone()
    
    def get_cache_hit_rate(self):
        """캐시 히트율 계산"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    def clear_cache(self):
        """캐시 초기화"""
        self.prompt_cache.clear()
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class PromptGenerationNetwork(nn.Module):
    """
    경량화된 프롬프트 생성 네트워크
    원본의 복잡한 구조를 단순화하면서도 성능 유지
    """
    
    def __init__(self, feature_dim=512, prompt_length=12):
        super(PromptGenerationNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.prompt_length = prompt_length
        
        # 공유 특징 인코더
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU()
        )
        
        # 정상 프롬프트 생성기
        self.positive_generator = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, prompt_length * feature_dim)
        )
        
        # 비정상 프롬프트 생성기
        self.negative_generator = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, prompt_length * feature_dim)
        )
        
        # 어텐션 메커니즘 (경량화)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=4,  # 원본의 1개에서 4개로 증가하여 표현력 향상
            dropout=0.1,
            batch_first=True
        )
        
        # 잔차 연결을 위한 프로젝션
        self.residual_proj = nn.Linear(feature_dim, feature_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_features):
        """
        Args:
            image_features: [batch_size, feature_dim]
        
        Returns:
            pos_prompt: [batch_size, prompt_length, feature_dim]
            neg_prompt: [batch_size, prompt_length, feature_dim]
        """
        batch_size = image_features.size(0)
        
        # 공유 특징 인코딩
        encoded_features = self.feature_encoder(image_features)
        
        # 정상/비정상 프롬프트 생성
        pos_flat = self.positive_generator(encoded_features)
        neg_flat = self.negative_generator(encoded_features)
        
        # reshape to [batch_size, prompt_length, feature_dim]
        pos_prompt = pos_flat.view(batch_size, self.prompt_length, self.feature_dim)
        neg_prompt = neg_flat.view(batch_size, self.prompt_length, self.feature_dim)
        
        # Self-attention for refinement
        pos_prompt_refined, _ = self.attention(pos_prompt, pos_prompt, pos_prompt)
        neg_prompt_refined, _ = self.attention(neg_prompt, neg_prompt, neg_prompt)
        
        # 잔차 연결
        pos_prompt = pos_prompt + self.residual_proj(pos_prompt_refined)
        neg_prompt = neg_prompt + self.residual_proj(neg_prompt_refined)
        
        # 정규화
        pos_prompt = F.normalize(pos_prompt, dim=-1)
        neg_prompt = F.normalize(neg_prompt, dim=-1)
        
        return pos_prompt, neg_prompt


class InformationTheoreticPromptOptimizer:
    """
    정보 이론 기반 프롬프트 최적화기
    상호 정보량을 최대화하여 더 효과적인 프롬프트 생성
    """
    
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
        
        # MINE (Mutual Information Neural Estimation) 네트워크
        self.mine_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def estimate_mutual_information(self, x, y):
        """
        MINE을 사용한 상호 정보량 추정
        
        Args:
            x: [batch_size, feature_dim] - 이미지 특징
            y: [batch_size, feature_dim] - 프롬프트 특징
        
        Returns:
            mi_estimate: 상호 정보량 추정값
        """
        batch_size = x.size(0)
        
        # Joint distribution
        joint = torch.cat([x, y], dim=-1)
        joint_scores = self.mine_network(joint)
        
        # Marginal distribution (shuffle y)
        y_shuffle = y[torch.randperm(batch_size)]
        marginal = torch.cat([x, y_shuffle], dim=-1)
        marginal_scores = self.mine_network(marginal)
        
        # MINE objective
        mi_estimate = torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores)))
        
        return mi_estimate
    
    def optimize_prompts(self, image_features, pos_prompts, neg_prompts):
        """
        정보 이론 기반 프롬프트 최적화
        
        Args:
            image_features: [batch_size, feature_dim]
            pos_prompts: [batch_size, prompt_length, feature_dim]
            neg_prompts: [batch_size, prompt_length, feature_dim]
        
        Returns:
            optimization_loss: 최적화 손실
        """
        # 프롬프트를 평균내어 단일 특징 벡터로 변환
        pos_features = torch.mean(pos_prompts, dim=1)  # [batch_size, feature_dim]
        neg_features = torch.mean(neg_prompts, dim=1)  # [batch_size, feature_dim]
        
        # 정상 프롬프트와 이미지 간의 상호 정보량 최대화
        pos_mi = self.estimate_mutual_information(image_features, pos_features)
        
        # 비정상 프롬프트와 이미지 간의 상호 정보량 최소화
        neg_mi = self.estimate_mutual_information(image_features, neg_features)
        
        # 최적화 목표: 정상 MI 최대화, 비정상 MI 최소화
        optimization_loss = -pos_mi + 0.5 * neg_mi
        
        return optimization_loss


# 성능 벤치마킹을 위한 유틸리티
class PromptGeneratorBenchmark:
    """프롬프트 생성기 성능 벤치마킹"""
    
    @staticmethod
    def benchmark_generation_speed(generator, num_samples=1000, feature_dim=512):
        """프롬프트 생성 속도 벤치마킹"""
        device = next(generator.parameters()).device
        
        # 랜덤 특징 생성
        random_features = torch.randn(num_samples, feature_dim).to(device)
        
        # 속도 측정
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(num_samples):
                features = random_features[i:i+1]
                pos_prompt, neg_prompt = generator.generate_cached_prompts(features, f"class_{i%10}")
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time_per_sample = (end_time - start_time) / num_samples * 1000  # ms
        cache_hit_rate = generator.get_cache_hit_rate()
        
        return {
            'avg_generation_time_ms': avg_time_per_sample,
            'cache_hit_rate': cache_hit_rate,
            'total_time_s': end_time - start_time
        }
    
    @staticmethod
    def compare_memory_usage(original_generator, improved_generator):
        """메모리 사용량 비교"""
        # 파라미터 수 계산
        original_params = sum(p.numel() for p in original_generator.parameters())
        improved_params = sum(p.numel() for p in improved_generator.parameters())
        
        # 메모리 사용량 추정 (4 bytes per float32 parameter)
        original_memory_mb = original_params * 4 / (1024 * 1024)
        improved_memory_mb = improved_params * 4 / (1024 * 1024)
        
        reduction_ratio = (original_memory_mb - improved_memory_mb) / original_memory_mb
        
        return {
            'original_params': original_params,
            'improved_params': improved_params,
            'original_memory_mb': original_memory_mb,
            'improved_memory_mb': improved_memory_mb,
            'memory_reduction_ratio': reduction_ratio
        } 