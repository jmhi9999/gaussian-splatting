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
        
        # 기본 설정
        if config is None:
            config = {
                'superpoint': {
                    'nms_radius': 3,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 2048
                },
                'superglue': {
                    'weights': 'indoor',  # 'indoor' 또는 'outdoor'
                    'sinkhorn_iterations': 30,
                    'match_threshold': 0.1,
                }
            }
        
        # 모델 로드
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        print(f'SuperGlue loaded on {self.device}')
    
    def match_image_pair(self, image_path0, image_path1, resize=[-1]):
        """두 이미지 간 매칭 수행"""
        
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
        
        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'matches': matches,
            'confidence': confidence,
            'matched_kpts0': mkpts0,
            'matched_kpts1': mkpts1,
            'match_confidence': mconf
        }
    
    def match_multiple_images(self, image_paths, resize=[-1]):
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
        if len(resize) == 2:
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
    
    # SuperGlue 매칭 수행
    print(f"Processing {len(image_paths)} images with SuperGlue...")
    matching_results = matcher.match_multiple_images(image_paths)
    
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