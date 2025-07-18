"""
Adaptive matching using CLIP-based global descriptors
"""
import numpy as np
import cv2
import torch
from .imports_utils import CLIP_AVAILABLE

if CLIP_AVAILABLE:
    import clip
    from PIL import Image as PILImage


class AdaptiveMatcher:
    """Adaptive Matching: CLIP 기반 global descriptor, cosine similarity, 상위 N개 쌍 선정"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.top_k = config.get('adaptive_top_k', 20)
        self.global_descriptors = {}
        
        # CLIP 모델 로드 (선택적)
        self.clip_available = CLIP_AVAILABLE
        if self.clip_available:
            try:
                # 전역에서 이미 로드된 모델 사용
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                print("✓ CLIP model loaded for adaptive matching")
            except Exception as e:
                print(f"⚠️  CLIP model loading failed: {e}, using fallback descriptors")
                self.clip_available = False
        else:
            print("⚠️  CLIP not available, using fallback global descriptors")
    
    def compute_global_descriptors(self, image_paths):
        """전역 이미지 descriptor 계산"""
        print("  Computing global descriptors...")
        
        if self.clip_available:
            return self._compute_clip_descriptors(image_paths)
        else:
            return self._compute_fallback_descriptors(image_paths)
    
    def _compute_clip_descriptors(self, image_paths):
        """CLIP을 사용한 전역 descriptor 계산"""
        global_descs = {}
        
        if not self.clip_available:
            print("    CLIP not available, falling back to image statistics")
            return self._compute_fallback_descriptors(image_paths)
        
        for i, path in enumerate(image_paths):
            try:
                img = PILImage.open(path).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    desc = self.model.encode_image(img_tensor).cpu().numpy().flatten()
                    desc = desc / (np.linalg.norm(desc) + 1e-10)
                
                global_descs[i] = desc
                
            except Exception as e:
                print(f"    Failed to compute CLIP descriptor for {path}: {e}")
                # Fallback descriptor
                global_descs[i] = np.random.randn(512).astype(np.float32)
                global_descs[i] = global_descs[i] / np.linalg.norm(global_descs[i])
        
        return global_descs
    
    def _compute_fallback_descriptors(self, image_paths):
        """Fallback 전역 descriptor 계산 (SuperPoint 특징점 기반)"""
        global_descs = {}
        
        for i, path in enumerate(image_paths):
            try:
                # 간단한 이미지 통계 기반 descriptor
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # 이미지 통계 계산
                mean_intensity = np.mean(img)
                std_intensity = np.std(img)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
                hist = hist / (np.sum(hist) + 1e-10)
                
                # 간단한 descriptor 생성
                desc = np.concatenate([
                    [mean_intensity / 255.0, std_intensity / 255.0],
                    hist[:128],  # 히스토그램의 절반만 사용
                    hist[128:]
                ]).astype(np.float32)
                
                # 정규화
                desc = desc / (np.linalg.norm(desc) + 1e-10)
                global_descs[i] = desc
                
            except Exception as e:
                print(f"    Failed to compute fallback descriptor for {path}: {e}")
                # 랜덤 descriptor
                global_descs[i] = np.random.randn(258).astype(np.float32)
                global_descs[i] = global_descs[i] / np.linalg.norm(global_descs[i])
        
        return global_descs
    
    def select_topk_pairs(self, global_descs):
        """상위 K개 이미지 쌍 선택"""
        print(f"  Selecting top-{self.top_k} pairs...")
        
        n_images = len(global_descs)
        if n_images < 2:
            return []
        
        # 유사도 행렬 계산
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                if i in global_descs and j in global_descs:
                    sim = np.dot(global_descs[i], global_descs[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # 상위 K개 쌍 선택
        pairs = set()
        
        for i in range(n_images):
            # 각 이미지에 대해 가장 유사한 K개 이미지 선택
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            for j in top_indices:
                if i != j and similarities[j] > 0.1:  # 최소 유사도 임계값
                    pairs.add(tuple(sorted([i, j])))
        
        selected_pairs = list(pairs)
        print(f"    Selected {len(selected_pairs)} candidate pairs")
        
        return selected_pairs
    
    def compute_pair_similarity(self, cam_i, cam_j, global_descs):
        """두 이미지 간의 유사도 계산"""
        if cam_i in global_descs and cam_j in global_descs:
            return np.dot(global_descs[cam_i], global_descs[cam_j])
        return 0.0
    
    def filter_pairs_by_similarity(self, pairs, global_descs, min_similarity=0.1):
        """유사도 기반 쌍 필터링"""
        filtered_pairs = []
        
        for cam_i, cam_j in pairs:
            similarity = self.compute_pair_similarity(cam_i, cam_j, global_descs)
            if similarity >= min_similarity:
                filtered_pairs.append((cam_i, cam_j))
        
        return filtered_pairs 