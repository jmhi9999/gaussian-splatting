# superglue_matcher.py
# 3DGS 리포지토리에 통합할 SuperGlue 인터페이스

import glob
import os
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# SuperGlue 모델 import
from models.matching import Matching
from models.utils import frame2tensor

class SuperGlueMatcher:
    """3DGS용 SuperGlue 매칭 인터페이스"""
    
    def __init__(self, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 기본 설정 - 고해상도 이미지에 최적화
        if config is None:
            config = {
                'superpoint': {
                    'nms_radius': 3,  # 증가
                    'keypoint_threshold': 0.001,
                    'max_keypoints': 4096  # 더 많은 특징점
                },
                'superglue': {
                    'weights': 'outdoor',  # indoor 가중치 사용
                    'sinkhorn_iterations': 100,  # 증가
                    'match_threshold': 0.1,  # 증가
                }
            }
        
        # 모델 로드
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        print(f'SuperGlue loaded on {self.device}')
    
    def match_image_pair(self, image_path0, image_path1, resize=None):
        """두 이미지 간 매칭 수행 - 적응형 resize 적용"""
        
        # 적응형 resize 계산
        if resize is None:
            resize = self._calculate_adaptive_resize(image_path0, image_path1)
        
        # 이미지 로드 및 전처리
        image0 = self._load_image(image_path0, resize)
        image1 = self._load_image(image_path1, resize)
        
        if image0 is None or image1 is None:
            return None
        
        # 텐서 변환
        inp0 = frame2tensor(image0, self.device)
        inp1 = frame2tensor(image1, self.device)
        
        # SuperPoint 특징점 추출
        pred0 = self.matching.superpoint({'image': inp0})
        pred1 = self.matching.superpoint({'image': inp1})
        
        # 데이터 준비
        data0 = {k+'0': pred0[k] for k in self.keys}
        data1 = {k+'1': pred1[k] for k in self.keys}
        data0['image0'] = inp0
        data1['image1'] = inp1
        
        # SuperGlue 매칭
        pred = self.matching({**data0, **data1})
        
        # 결과 추출
        kpts0 = pred0['keypoints'][0].cpu().numpy()
        kpts1 = pred1['keypoints'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        
        # 유효한 매칭만 필터링
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = confidence[valid]
        
        # 원본 해상도로 좌표 변환
        if resize is not None and len(resize) == 2:
            scale_x = resize[0] / image0.shape[1]
            scale_y = resize[1] / image0.shape[0]
            mkpts0[:, 0] *= scale_x
            mkpts0[:, 1] *= scale_y
            mkpts1[:, 0] *= scale_x
            mkpts1[:, 1] *= scale_y
        
        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'matches': matches,
            'confidence': confidence,
            'matched_kpts0': mkpts0,
            'matched_kpts1': mkpts1,
            'match_confidence': mconf,
            'resize_used': resize
        }
    
    def match_with_quality_check(self, image_path0, image_path1, min_matches=10, max_retries=3):
        """품질 체크가 포함된 매칭 - 매칭 개수가 부족하면 설정을 조정하여 재시도"""
        
        for attempt in range(max_retries):
            # 시도별 설정 조정
            if attempt == 0:
                # 첫 번째 시도: 기본 설정
                config = self._get_config_for_attempt(0)
            elif attempt == 1:
                # 두 번째 시도: 더 관대한 설정
                config = self._get_config_for_attempt(1)
            else:
                # 세 번째 시도: 가장 관대한 설정
                config = self._get_config_for_attempt(2)
            
            # 임시 매처 생성
            temp_matcher = SuperGlueMatcher(config, device=str(self.device))
            
            # 매칭 시도
            result = temp_matcher.match_image_pair(image_path0, image_path1)
            
            if result is not None and len(result['matched_kpts0']) >= min_matches:
                print(f"  Successful matching with {len(result['matched_kpts0'])} matches (attempt {attempt + 1})")
                return result
        
        print(f"  Failed to get sufficient matches after {max_retries} attempts")
        return None
    
    def _get_config_for_attempt(self, attempt):
        """시도별 설정 반환"""
        if attempt == 0:
            # 기본 설정
            return {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 4096
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 50,
                    'match_threshold': 0.15,
                }
            }
        elif attempt == 1:
            # 더 관대한 설정
            return {
                'superpoint': {
                    'nms_radius': 3,  # 감소
                    'keypoint_threshold': 0.003,  # 감소
                    'max_keypoints': 6144  # 증가
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 30,  # 감소
                    'match_threshold': 0.1,  # 감소
                }
            }
        else:
            # 가장 관대한 설정
            return {
                'superpoint': {
                    'nms_radius': 2,  # 더 감소
                    'keypoint_threshold': 0.001,  # 더 감소
                    'max_keypoints': 8192  # 더 증가
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 20,  # 더 감소
                    'match_threshold': 0.05,  # 더 감소
                }
            }
    
    def match_multiple_images_with_quality(self, image_paths, min_matches=10):
        """품질 체크가 포함된 다중 이미지 매칭"""
        results = {}
        n_images = len(image_paths)
        
        print(f"Matching {n_images} images with quality check...")
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                pair_key = f"{i}_{j}"
                print(f"Matching pair {i}-{j}")
                
                result = self.match_with_quality_check(
                    image_paths[i], 
                    image_paths[j], 
                    min_matches
                )
                
                if result is not None:
                    results[pair_key] = result
        
        return results
    
    def _calculate_adaptive_resize(self, image_path0, image_path1):
        """이미지 해상도에 따른 적응형 resize 계산 - 더 높은 해상도 지원"""
        # 이미지 크기 확인
        img0 = cv2.imread(str(image_path0))
        img1 = cv2.imread(str(image_path1))
        
        if img0 is None or img1 is None:
            return [1024, 768]  # 기본값
        
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        
        # 최대 해상도 계산
        max_dim = max(h0, w0, h1, w1)
        
        # 적응형 resize 규칙 - 더 높은 해상도 지원
        if max_dim <= 1024:
            # 작은 이미지는 원본 크기 유지
            return None
        elif max_dim <= 2048:
            # 중간 크기는 1024로 resize
            scale = 1024 / max_dim
            return [int(w0 * scale), int(h0 * scale)]
        elif max_dim <= 4096:
            # 큰 이미지는 2048로 resize (증가)
            scale = 2048 / max_dim
            return [int(w0 * scale), int(h0 * scale)]
        elif max_dim <= 8192:
            # 매우 큰 이미지는 3072로 resize (증가)
            scale = 3072 / max_dim
            return [int(w0 * scale), int(h0 * scale)]
        else:
            # 극도로 큰 이미지는 4096으로 제한
            scale = 4096 / max_dim
            return [int(w0 * scale), int(h0 * scale)]
    
    def match_multiple_images(self, image_paths, resize=None):
        """여러 이미지 간 전체 매칭"""
        results = {}
        n_images = len(image_paths)
        
        print(f"Matching {n_images} images...")
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                pair_key = f"{i}_{j}"
                print(f"Matching pair {i}-{j}")
                
                result = self.match_image_pair(
                    image_paths[i], 
                    image_paths[j], 
                    resize
                )
                
                if result is not None:
                    results[pair_key] = result
        
        return results
    
    def _load_image(self, image_path, resize):
        """이미지 로드 및 전처리"""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # 리사이즈
        if resize is None:
            pass  # 원본 크기 유지
        elif len(resize) == 2:
            image = cv2.resize(image, tuple(resize))
        elif len(resize) == 1 and resize[0] > 0:
            h, w = image.shape
            scale = resize[0] / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        return image.astype(np.float32)
    
    def extract_features_for_3dgs(self, image_paths):
        """3DGS 학습에 필요한 형태로 특징점 추출"""
        # COLMAP 형식과 유사한 출력 생성
        cameras = {}
        images = {}
        points3d = {}
        
        # TODO: SuperGlue 매칭 결과를 COLMAP 형식으로 변환
        # - 카메라 파라미터 추정
        # - 3D 포인트 triangulation
        # - Bundle adjustment
        
        return cameras, images, points3d

# 3DGS scene/dataset_readers.py에 추가할 함수
def readSuperGlueSceneInfo(path, images, eval=False):
    """SuperGlue 기반 scene 정보 읽기"""
    
    # SuperGlue 매처 초기화
    matcher = SuperGlueMatcher()
    
    # 이미지 경로 수집
    images_folder = os.path.join(path, images)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
    
    image_paths.sort()
    
    # SuperGlue 매칭 수행 (적응형 resize 사용)
    print(f"Processing {len(image_paths)} images with SuperGlue...")
    matching_results = matcher.match_multiple_images(image_paths, resize=None)  # 적응형 resize
    
    # 3DGS 형식으로 변환
    cameras, images_info, points3d = matcher.extract_features_for_3dgs(image_paths)
    
    # CameraInfo 리스트 생성
    cam_infos = []
    for i, image_path in enumerate(image_paths):
        # 임시 카메라 정보 (실제로는 pose estimation 필요)
        cam_info = CameraInfo(
            uid=i,
            R=np.eye(3),  # 임시
            T=np.zeros(3),  # 임시
            FovY=np.pi/3,  # 임시
            FovX=np.pi/3,  # 임시
            image_path=image_path,
            image_name=Path(image_path).name,
            width=640,  # 임시
            height=480,  # 임시
            depth_params=None,
            depth_path="",
            is_test=False
        )
        cam_infos.append(cam_info)
    
    # SceneInfo 생성
    scene_info = SceneInfo(
        point_cloud=None,  # SuperGlue로부터 생성된 포인트 클라우드
        train_cameras=cam_infos,
        test_cameras=[],
        nerf_normalization={"translate": np.zeros(3), "radius": 1.0},
        ply_path="",
        is_nerf_synthetic=False
    )
    
    return scene_info

# 사용 예시
if __name__ == "__main__":
    # SuperGlue 매칭 테스트
    matcher = SuperGlueMatcher()
    
    # 두 이미지 매칭
    result = matcher.match_image_pair(
        'path/to/image1.jpg',
        'path/to/image2.jpg'
    )
    
    if result:
        print(f"Found {len(result['matched_kpts0'])} matches")
        print(f"Average confidence: {result['match_confidence'].mean():.3f}")