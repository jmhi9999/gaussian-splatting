# superglue_colmap_pipeline.py
# SuperGlue + COLMAP 하이브리드 SfM 파이프라인

import os
import sys
import sqlite3
import numpy as np
import cv2
import torch
from pathlib import Path
import subprocess
import logging
from typing import List, Dict, Tuple, Optional
import json

# SuperGlue 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from superglue_matcher import SuperGlueMatcher
from models.matching import Matching
from models.utils import frame2tensor

class SuperGlueColmapPipeline:
    """SuperGlue + COLMAP 하이브리드 SfM 파이프라인"""
    
    def __init__(self, colmap_path="colmap", device='cuda'):
        self.colmap_path = colmap_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # SuperGlue 초기화
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 2048
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching = Matching(config).eval().to(self.device)
        
        # COLMAP database 경로
        self.database_path = None
        self.image_path = None
        self.sparse_path = None
        
        print(f"SuperGlue-Colmap Pipeline initialized on {self.device}")
    
    def run_pipeline(self, image_dir: str, output_dir: str, max_images: int = 100):
        """완전한 하이브리드 SfM 파이프라인 실행"""
        
        print(f"=== SuperGlue + COLMAP Pipeline ===")
        print(f"Input: {image_dir}")
        print(f"Output: {output_dir}")
        
        # 디렉토리 설정
        self._setup_directories(output_dir)
        
        # 1. 이미지 경로 수집
        image_paths = self._collect_images(image_dir, max_images)
        print(f"Found {len(image_paths)} images")
        
        # 2. SuperGlue로 특징점 추출 및 매칭
        print("1. Extracting features with SuperGlue...")
        self._extract_features_superglue(image_paths)
        
        # 3. COLMAP database 생성
        print("2. Creating COLMAP database...")
        self._create_colmap_database(image_paths)
        
        # 4. COLMAP bundle adjustment
        print("3. Running COLMAP bundle adjustment...")
        self._run_colmap_mapper()
        
        # 5. 결과 정리
        print("4. Finalizing results...")
        self._finalize_results()
        
        print("Pipeline completed successfully!")
        return True
    
    def _setup_directories(self, output_dir: str):
        """필요한 디렉토리 생성"""
        self.database_path = os.path.join(output_dir, "database.db")
        self.image_path = os.path.join(output_dir, "images")
        self.sparse_path = os.path.join(output_dir, "sparse")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.sparse_path, exist_ok=True)
    
    def _collect_images(self, image_dir: str, max_images: int) -> List[str]:
        """이미지 파일 경로 수집"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        
        image_paths = sorted([str(p) for p in image_paths])[:max_images]
        return image_paths
    
    def _extract_features_superglue(self, image_paths: List[str]):
        """SuperGlue로 특징점 추출"""
        features = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"  Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # SuperPoint 특징점 추출
            inp = frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            
            # 결과 저장
            kpts = pred['keypoints'][0].cpu().numpy()
            desc = pred['descriptors'][0].cpu().numpy()
            scores = pred['scores'][0].cpu().numpy()
            
            features[i] = {
                'keypoints': kpts,
                'descriptors': desc,
                'scores': scores,
                'image_path': image_path,
                'image': image
            }
        
        self.features = features
    
    def _create_colmap_database(self, image_paths: List[str]):
        """COLMAP database 생성"""
        # COLMAP database 초기화
        self._init_colmap_database()
        
        # 카메라 정보 추가
        self._add_cameras_to_database(image_paths)
        
        # 이미지 정보 추가
        self._add_images_to_database(image_paths)
        
        # SuperGlue 특징점을 COLMAP 형식으로 추가
        self._add_features_to_database()
        
        # SuperGlue 매칭 결과를 COLMAP 형식으로 추가
        self._add_matches_to_database()
    
    def _init_colmap_database(self):
        """COLMAP database 초기화"""
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
        
        # COLMAP database 생성
        cmd = f"{self.colmap_path} database_creator --database_path {self.database_path}"
        subprocess.run(cmd, shell=True, check=True)
    
    def _add_cameras_to_database(self, image_paths: List[str]):
        """카메라 정보를 database에 추가"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 첫 번째 이미지로 카메라 파라미터 추정
        first_image = cv2.imread(image_paths[0])
        height, width = first_image.shape[:2]
        
        # 대략적인 focal length 추정
        focal = max(width, height) * 0.8
        
        # 카메라 정보 삽입
        cursor.execute("""
            INSERT INTO cameras (camera_id, model, width, height, params)
            VALUES (?, ?, ?, ?, ?)
        """, (1, 1, width, height, 
              np.array([focal, focal, width/2, height/2], dtype=np.float64).tobytes()))
        
        conn.commit()
        conn.close()
    
    def _add_images_to_database(self, image_paths: List[str]):
        """이미지 정보를 database에 추가"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        for i, image_path in enumerate(image_paths):
            image_name = Path(image_path).name
            
            # 이미지 정보 삽입
            cursor.execute("""
                INSERT INTO images (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz,
                                 prior_tx, prior_ty, prior_tz)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (i+1, image_name, 1, 1, 0, 0, 0, 0, 0, 0))
        
        conn.commit()
        conn.close()
    
    def _add_features_to_database(self):
        """SuperGlue 특징점을 COLMAP database에 추가"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        for image_id, feature_data in self.features.items():
            keypoints = feature_data['keypoints']
            descriptors = feature_data['descriptors']
            
            # 특징점 정보 삽입
            for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
                cursor.execute("""
                    INSERT INTO keypoints (image_id, rows, cols, data)
                    VALUES (?, ?, ?, ?)
                """, (image_id + 1, 1, 2, kp.astype(np.float32).tobytes()))
                
                cursor.execute("""
                    INSERT INTO descriptors (image_id, rows, cols, data)
                    VALUES (?, ?, ?, ?)
                """, (image_id + 1, 1, len(desc), desc.astype(np.float32).tobytes()))
        
        conn.commit()
        conn.close()
    
    def _add_matches_to_database(self):
        """SuperGlue 매칭 결과를 COLMAP database에 추가"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 모든 이미지 쌍에 대해 매칭 수행
        image_ids = list(self.features.keys())
        
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):
                matches = self._match_pair(image_ids[i], image_ids[j])
                
                if len(matches) > 10:  # 최소 매칭 수
                    # 매칭 정보 삽입
                    matches_data = np.array(matches, dtype=np.uint32)
                    cursor.execute("""
                        INSERT INTO two_view_geometries (pair_id, rows, cols, data, config, F, E, H)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (i*len(image_ids) + j, len(matches), 2, matches_data.tobytes(), 
                          1, None, None, None))
        
        conn.commit()
        conn.close()
    
    def _match_pair(self, img1_id: int, img2_id: int) -> List[Tuple[int, int]]:
        """두 이미지 간 SuperGlue 매칭"""
        feat1 = self.features[img1_id]
        feat2 = self.features[img2_id]
        
        # SuperGlue 매칭
        data = {
            'keypoints0': torch.from_numpy(feat1['keypoints']).unsqueeze(0).to(self.device),
            'keypoints1': torch.from_numpy(feat2['keypoints']).unsqueeze(0).to(self.device),
            'descriptors0': torch.from_numpy(feat1['descriptors']).unsqueeze(0).to(self.device),
            'descriptors1': torch.from_numpy(feat2['descriptors']).unsqueeze(0).to(self.device),
            'scores0': torch.from_numpy(feat1['scores']).unsqueeze(0).to(self.device),
            'scores1': torch.from_numpy(feat2['scores']).unsqueeze(0).to(self.device),
            'image0': torch.zeros(1, 1, 480, 640).to(self.device),
            'image1': torch.zeros(1, 1, 480, 640).to(self.device),
        }
        
        with torch.no_grad():
            pred = self.matching.superglue(data)
        
        # 유효한 매칭 추출
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        
        valid_matches = []
        for idx in np.where(matches > -1)[0]:
            match_idx = matches[idx]
            conf = confidence[idx]
            if conf > 0.3:  # 신뢰도 임계값
                valid_matches.append((idx, match_idx))
        
        return valid_matches
    
    def _run_colmap_mapper(self):
        """COLMAP mapper 실행"""
        cmd = f"{self.colmap_path} mapper " \
              f"--database_path {self.database_path} " \
              f"--image_path {self.image_path} " \
              f"--output_path {self.sparse_path} " \
              f"--Mapper.ba_global_function_tolerance=0.000001"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print("  COLMAP mapper completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  COLMAP mapper failed: {e}")
            raise
    
    def _finalize_results(self):
        """결과 정리"""
        # sparse/0 디렉토리로 결과 이동
        sparse_0 = os.path.join(self.sparse_path, "0")
        if os.path.exists(sparse_0):
            # 이미 존재하는 경우 삭제
            import shutil
            shutil.rmtree(sparse_0)
        
        # sparse 디렉토리의 내용을 sparse/0으로 이동
        files = os.listdir(self.sparse_path)
        os.makedirs(sparse_0, exist_ok=True)
        
        for file in files:
            if file != "0":
                source = os.path.join(self.sparse_path, file)
                dest = os.path.join(sparse_0, file)
                if os.path.isfile(source):
                    import shutil
                    shutil.move(source, dest)
        
        print(f"  Results saved to {sparse_0}")
    
    def load_results(self):
        """결과 로딩 (SIBR 형식)"""
        from SIBR_viewers.src.core.assets.InputCamera import InputCamera
        
        sparse_0 = os.path.join(self.sparse_path, "0")
        if not os.path.exists(sparse_0):
            raise FileNotFoundError(f"Results not found at {sparse_0}")
        
        # COLMAP 결과 로딩
        cameras = InputCamera.loadColmap(sparse_0, 0.01, 1000, 1)
        
        return cameras

def run_superglue_colmap_pipeline(image_dir: str, output_dir: str, max_images: int = 100):
    """SuperGlue + COLMAP 파이프라인 실행 함수"""
    
    pipeline = SuperGlueColmapPipeline()
    
    try:
        success = pipeline.run_pipeline(image_dir, output_dir, max_images)
        if success:
            cameras = pipeline.load_results()
            print(f"Successfully loaded {len(cameras)} cameras")
            return cameras
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperGlue + COLMAP Pipeline")
    parser.add_argument("--image_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_images", type=int, default=100, help="Maximum number of images")
    parser.add_argument("--colmap_path", default="colmap", help="COLMAP executable path")
    
    args = parser.parse_args()
    
    # COLMAP 경로 설정
    pipeline = SuperGlueColmapPipeline(colmap_path=args.colmap_path)
    pipeline.run_pipeline(args.image_dir, args.output_dir, args.max_images) 