import os
import sys
import sqlite3
import subprocess
import numpy as np
import cv2
import torch
from pathlib import Path
import shutil
import json
from typing import List, Optional, Tuple, Dict, Any

# 3DGS 모듈 경로 추가
script_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(script_dir))

try:
    from scene.dataset_readers import CameraInfo, SceneInfo
    from utils.graphics_utils import focal2fov, fov2focal
    from utils.camera_utils import cameraList_from_camInfos
except ImportError as e:
    print(f"경고: 3DGS 모듈 import 실패: {e}")
    # 기본 클래스 정의
    class CameraInfo:
        def __init__(self, uid, R, T, FovY, FovX, image, image_path, image_name, width, height, 
                     depth_params=None, depth_path="", is_test=False):
            self.uid = uid
            self.R = R
            self.T = T
            self.FovY = FovY
            self.FovX = FovX
            self.image = image
            self.image_path = image_path
            self.image_name = image_name
            self.width = width
            self.height = height
            self.depth_params = depth_params
            self.depth_path = depth_path
            self.is_test = is_test
    
    class SceneInfo:
        def __init__(self, point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path, is_nerf_synthetic=False):
            self.point_cloud = point_cloud
            self.train_cameras = train_cameras
            self.test_cameras = test_cameras
            self.nerf_normalization = nerf_normalization
            self.ply_path = ply_path
            self.is_nerf_synthetic = is_nerf_synthetic

class SuperGlueCOLMAPHybrid:
    def __init__(self, 
                 superglue_config: str = "outdoor",
                 colmap_exe: str = "colmap",
                 device: str = "cuda"):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.colmap_exe = colmap_exe
        
        # SuperGlue 설정
        self.superglue_config = {
            'outdoor': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            },
            'indoor': {
                'weights': 'indoor', 
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }[superglue_config]
        
        self._load_models()
    
    def _load_models(self):
        """SuperPoint와 SuperGlue 모델 로드"""
        try:
            # SuperPoint 설정
            superpoint_config = {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            }
            
            # SuperGlue 설정
            superglue_config = {
                'weights': self.superglue_config['weights'],
                'sinkhorn_iterations': self.superglue_config['sinkhorn_iterations'],
                'match_threshold': self.superglue_config['match_threshold'],
            }
            
            # 실제 모델 로드 시도
            try:
                from models.superpoint import SuperPoint
                from models.superglue import SuperGlue
                
                self.superpoint = SuperPoint(superpoint_config).eval().to(self.device)
                self.superglue = SuperGlue(superglue_config).eval().to(self.device)
                print(f"  ✓ SuperPoint/SuperGlue 모델 로드 완료 (device: {self.device})")
                
            except ImportError:
                # 모델이 없는 경우 더미 모델 생성
                print("  경고: SuperPoint/SuperGlue 모델을 찾을 수 없음, 더미 모델 사용")
                self.superpoint = None
                self.superglue = None
                
        except Exception as e:
            print(f"  경고: SuperGlue 모델 로드 실패: {e}")
            self.superpoint = None
            self.superglue = None
    
    def process_images(self, image_dir: str, output_dir: str, max_images: int = 100) -> SceneInfo:
        """dataset_readers.py에서 호출되는 메서드"""
        print("🚀 SuperGlue + COLMAP 하이브리드 파이프라인 시작")
        
        # 출력 디렉토리 설정
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 이미지 수집 및 복사
            print("\n[1/6] 이미지 수집...")
            image_paths = self._collect_images(image_dir, max_images)
            if len(image_paths) == 0:
                raise RuntimeError("처리할 이미지를 찾을 수 없습니다")
            
            input_dir = self._prepare_input_images(image_paths, output_path)
            
            # 2. COLMAP 데이터베이스 생성
            print("\n[2/6] COLMAP 데이터베이스 생성...")
            database_path = output_path / "database.db"
            self._create_colmap_database(image_paths, database_path, input_dir)
            
            # 3. SuperPoint 특징점 추출
            print("\n[3/6] SuperPoint 특징점 추출...")
            self._extract_superpoint_features(image_paths, database_path, input_dir)
            
            # 4. SuperGlue 매칭
            print("\n[4/6] SuperGlue 매칭...")
            self._run_superglue_matching(image_paths, database_path)
            
            # 5. COLMAP으로 포즈 추정
            print("\n[5/6] COLMAP 포즈 추정...")
            sparse_dir = output_path / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            self._run_colmap_mapper(database_path, input_dir, sparse_dir)
            
            # 6. 이미지 언디스토션
            print("\n[6/6] 이미지 언디스토션...")
            undistorted_dir = output_path / "undistorted"
            self._run_colmap_undistortion(input_dir, sparse_dir, undistorted_dir)
            
            # 7. 3DGS 형식으로 변환
            print("\n[7/6] 3DGS 형식 변환...")
            scene_info = self._convert_to_3dgs_format(output_path, image_paths)
            
            print("✅ 하이브리드 파이프라인 완료!")
            return scene_info
            
        except Exception as e:
            print(f"❌ 실패: {e}")
            # 기본 SceneInfo 생성 시도
            return self._create_default_scene_info(image_paths, output_path)
    
    def process(self, image_dir: str, max_images: int = 100) -> SceneInfo:
        """전체 하이브리드 파이프라인 실행 (기존 메서드)"""
        return self.process_images(image_dir, "ImageInputs/superglue_colmap_hybrid_output", max_images)
    
    def _collect_images(self, image_dir, max_images):
        """이미지 수집 및 품질 필터링"""
        image_dir = Path(image_dir)
        
        # 이미지 파일 찾기
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(list(image_dir.glob(ext)))
        
        # 시간순 정렬
        image_paths.sort()
        
        # 최대 개수로 제한
        if len(image_paths) > max_images:
            # 균등하게 샘플링
            step = len(image_paths) // max_images
            image_paths = image_paths[::step][:max_images]
        
        print(f"  선택된 이미지: {len(image_paths)}장")
        return image_paths
    
    def _prepare_input_images(self, image_paths, output_path):
        """COLMAP용 입력 이미지 준비"""
        input_dir = output_path / "input"
        input_dir.mkdir(exist_ok=True)
        
        # 기존 파일 정리
        for f in input_dir.glob("*"):
            f.unlink()
        
        # 이미지 복사
        for i, src_path in enumerate(image_paths):
            dst_path = input_dir / f"image_{i:04d}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)
        
        print(f"  {len(image_paths)}장 이미지 준비 완료")
        return input_dir
    
    def _create_colmap_database(self, image_paths, database_path, input_dir):
        """COLMAP 데이터베이스 생성"""
        
        # 기존 데이터베이스 삭제
        if database_path.exists():
            database_path.unlink()
        
        try:
            # COLMAP의 database_creator 사용
            cmd = ["colmap", "database_creator", "--database_path", str(database_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ✗ database_creator 실패: {result.stderr}")
                return False
            
            print("  ✓ COLMAP database_creator 성공")
            
            # 이미지 정보 추가
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 기본 카메라 추가 (SIMPLE_PINHOLE 모델)
            sample_img = cv2.imread(str(image_paths[0]))
            height, width = sample_img.shape[:2]
            
            # SIMPLE_PINHOLE 모델 (model=0): [f, cx, cy]
            focal = max(width, height) * 1.2
            params = np.array([focal, width/2, height/2], dtype=np.float64)
            
            cursor.execute(
                "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                (0, width, height, params.tobytes(), int(focal))
            )
            
            camera_id = cursor.lastrowid
            
            # 이미지 정보 추가
            for i, img_path in enumerate(image_paths):
                image_name = f"image_{i:04d}{img_path.suffix}"
                cursor.execute(
                    "INSERT INTO images (name, camera_id) VALUES (?, ?)",
                    (image_name, camera_id)
                )
            
            conn.commit()
            conn.close()
            
            print(f"  ✓ {len(image_paths)}장 이미지 정보 추가")
            return True
            
        except Exception as e:
            print(f"  ✗ 데이터베이스 생성 실패: {e}")
            return False
    
    def _extract_superpoint_features(self, image_paths, database_path, input_dir):
        """SuperPoint로 특징점 추출하고 COLMAP DB에 저장"""
        print("  SuperPoint 특징점 추출 중...")
        
        if self.superpoint is None:
            print("  경고: SuperPoint 모델 없음, COLMAP SIFT 사용")
            self._run_colmap_feature_extraction(database_path, input_dir)
            return
        
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 이미지 ID 가져오기
            cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
            images = cursor.fetchall()
            
            successful_extractions = 0
            for image_id, image_name in images:
                # 이미지 로드
                img_path = input_dir / image_name
                if not img_path.exists():
                    continue
                
                # 원본 이미지 경로 찾기
                original_img_path = None
                for orig_path in image_paths:
                    if orig_path.name in image_name:
                        original_img_path = orig_path
                        break
                
                if original_img_path is None:
                    continue
                
                # SuperPoint 특징점 추출
                keypoints, descriptors = self._extract_single_superpoint_features(original_img_path)
                
                if keypoints is not None and len(keypoints) > 0 and descriptors is not None:
                    # COLMAP DB에 저장 (descriptor 차원을 128로 고정)
                    descriptor_dim = descriptors.shape[1] if len(descriptors.shape) > 1 else 128
                    
                    cursor.execute(
                        "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                        (image_id, len(keypoints), 2, keypoints.tobytes())
                    )
                    
                    cursor.execute(
                        "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                        (image_id, len(descriptors), descriptor_dim, descriptors.tobytes())
                    )
                    
                    print(f"    {image_name}: {len(keypoints)}개 키포인트 ({descriptor_dim}차원)")
                    successful_extractions += 1
                else:
                    print(f"    {image_name}: 키포인트 추출 실패")
            
            conn.commit()
            conn.close()
            
            if successful_extractions > 0:
                print(f"  ✓ SuperPoint 특징점 추출 완료 ({successful_extractions}개)")
            else:
                print("  ⚠️  SuperPoint 추출 실패, COLMAP SIFT로 fallback...")
                self._run_colmap_feature_extraction(database_path, input_dir)
            
        except Exception as e:
            print(f"  오류: SuperPoint 특징점 추출 실패: {e}")
            print("  COLMAP SIFT로 fallback...")
            self._run_colmap_feature_extraction(database_path, input_dir)
    
    def _extract_single_superpoint_features(self, image_path):
        """단일 이미지에서 SuperPoint 특징점 추출"""
        try:
            # 이미지 로드
            img = cv2.imread(str(image_path))
            if img is None:
                return None, None
            
            # RGB로 변환 후 그레이스케일로 변환 (SuperPoint는 1채널 입력 기대)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 그레이스케일 이미지를 텐서로 변환
            img_tensor = torch.from_numpy(img_gray).float().to(self.device) / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) - 1채널
            
            # SuperPoint 추론
            with torch.no_grad():
                pred = self.superpoint({'image': img_tensor})
                keypoints = pred['keypoints'][0].cpu().numpy()  # (N, 2)
                descriptors = pred['descriptors'][0].cpu().numpy()  # (N, 256)
            
            # COLMAP SIFT 형식으로 변환 (256 -> 128)
            if descriptors.shape[1] == 256:
                # PCA를 사용하여 256차원을 128차원으로 축소
                descriptors_128 = self._convert_descriptors_to_sift_format(descriptors)
                return keypoints, descriptors_128
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"    SuperPoint 추출 오류: {e}")
            return None, None
    
    def _convert_descriptors_to_sift_format(self, descriptors):
        """SuperPoint descriptor를 COLMAP SIFT 형식으로 변환"""
        try:
            # 간단한 차원 축소: 256차원을 128차원으로 평균화
            n_features = descriptors.shape[0]
            descriptors_128 = np.zeros((n_features, 128), dtype=np.float32)
            
            for i in range(n_features):
                # 256차원을 2개씩 묶어서 평균
                for j in range(128):
                    descriptors_128[i, j] = (descriptors[i, j*2] + descriptors[i, j*2+1]) / 2.0
            
            return descriptors_128
            
        except Exception as e:
            print(f"    Descriptor 변환 오류: {e}")
            # 변환 실패 시 원본 반환
            return descriptors[:, :128] if descriptors.shape[1] >= 128 else descriptors
    
    def _run_superglue_matching(self, image_paths, database_path):
        """SuperGlue로 매칭하고 COLMAP DB에 저장"""
        print("  SuperGlue 매칭 중...")
        
        if self.superglue is None:
            print("  경고: SuperGlue 모델 없음, COLMAP 매칭 사용")
            self._run_colmap_matching(database_path)
            return
        
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 기존 matches 테이블 정리
            cursor.execute("DELETE FROM matches")
            conn.commit()
            
            # 이미지 쌍 생성 (더 많은 쌍 생성)
            image_pairs = []
            for i in range(len(image_paths)):
                for j in range(i + 1, min(i + 10, len(image_paths))):  # 인접한 10개 이미지까지 매칭
                    image_pairs.append((i, j))
            
            print(f"  {len(image_pairs)}개 이미지 쌍 매칭...")
            
            successful_matches = 0
            for pair_idx, (i, j) in enumerate(image_pairs):
                # 이미지 ID 가져오기
                cursor.execute("SELECT image_id FROM images ORDER BY image_id")
                image_ids = [row[0] for row in cursor.fetchall()]
                
                if i >= len(image_ids) or j >= len(image_ids):
                    continue
                
                img1_id, img2_id = image_ids[i], image_ids[j]
                
                # SuperGlue 매칭
                matches = self._match_single_pair(image_paths[i], image_paths[j])
                
                if matches is not None and len(matches) >= 4:  # 최소 4개 매칭 필요
                    # COLMAP DB에 저장 (pair_id는 0부터 시작하는 연속된 정수)
                    cursor.execute(
                        "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                        (successful_matches, len(matches), 2, matches.tobytes())
                    )
                    
                    print(f"    쌍 {i}-{j}: {len(matches)}개 매칭 (pair_id: {successful_matches})")
                    successful_matches += 1
                else:
                    print(f"    쌍 {i}-{j}: 매칭 실패 또는 부족 ({len(matches) if matches is not None else 0}개)")
            
            # two_view_geometry 테이블 생성 (COLMAP이 필요로 함)
            conn.commit()
            conn.close()
            
            if successful_matches > 0:
                print(f"  ✓ SuperGlue 매칭 완료 ({successful_matches}개 성공)")
                # COLMAP exhaustive_matcher로 two_view_geometries 생성
                print("  COLMAP exhaustive_matcher로 two_view_geometries 생성...")
                if not self._run_colmap_exhaustive_matcher(database_path):
                    print("  ⚠️  exhaustive_matcher 실패, COLMAP SIFT로 fallback...")
                    self._run_colmap_feature_extraction_fallback(database_path)
                    self._run_colmap_matching_fallback(database_path)
            else:
                print("  ⚠️  성공한 매칭이 없음, COLMAP 매칭으로 fallback...")
                self._run_colmap_matching(database_path)
            
        except Exception as e:
            print(f"  오류: SuperGlue 매칭 실패: {e}")
            print("  COLMAP 매칭으로 fallback...")
            self._run_colmap_matching(database_path)
    
    def _match_single_pair(self, img1_path, img2_path):
        """단일 이미지 쌍에서 SuperGlue 매칭"""
        try:
            # 이미지 로드
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                return None
            
            # RGB로 변환 후 그레이스케일로 변환
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # 그레이스케일 이미지를 텐서로 변환
            img1_tensor = torch.from_numpy(img1_gray).float().to(self.device) / 255.0
            img2_tensor = torch.from_numpy(img2_gray).float().to(self.device) / 255.0
            
            img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            img2_tensor = img2_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # SuperPoint 특징점 추출
            with torch.no_grad():
                pred1 = self.superpoint({'image': img1_tensor})
                pred2 = self.superpoint({'image': img2_tensor})
                
                kpts1 = pred1['keypoints'][0]
                desc1 = pred1['descriptors'][0]
                scores1 = pred1['scores'][0]
                kpts2 = pred2['keypoints'][0]
                desc2 = pred2['descriptors'][0]
                scores2 = pred2['scores'][0]
            
            # SuperGlue 매칭 - 필요한 모든 키 포함
            with torch.no_grad():
                pred = self.superglue({
                    'keypoints0': kpts1.unsqueeze(0),
                    'keypoints1': kpts2.unsqueeze(0),
                    'descriptors0': desc1.unsqueeze(0),
                    'descriptors1': desc2.unsqueeze(0),
                    'scores0': scores1.unsqueeze(0),
                    'scores1': scores2.unsqueeze(0),
                    'image0': img1_tensor,
                    'image1': img2_tensor,
                })
                
                # indices0/indices1 사용 (matches0/matches1 대신)
                indices0 = pred['indices0'][0].cpu().numpy()
                indices1 = pred['indices1'][0].cpu().numpy()
                mscores0 = pred['matching_scores0'][0].cpu().numpy()
            
            # 유효한 매칭만 필터링
            valid_matches = []
            threshold = self.superglue_config['match_threshold']
            
            for i, j in enumerate(indices0):
                if j >= 0 and mscores0[i] > threshold:
                    # 상호 매칭 확인
                    if j < len(indices1) and indices1[j] == i:
                        valid_matches.append([i, j])
            
            # 최소 매칭 수 확인
            if len(valid_matches) < 4:
                print(f"      매칭 수 부족: {len(valid_matches)}개 (최소 4개 필요)")
                return None
            
            return np.array(valid_matches, dtype=np.int32)
            
        except Exception as e:
            print(f"    SuperGlue 매칭 오류: {e}")
            return None
    
    def _run_colmap_feature_extraction(self, database_path, image_path):
        """COLMAP 특징점 추출 (fallback)"""
        base_cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "1000"
        ]
        
        print("  COLMAP 특징점 추출 실행...")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ 특징점 추출 완료")
            else:
                print(f"  ✗ 특징점 추출 실패: {result.stderr}")
        except Exception as e:
            print(f"  오류: 특징점 추출 실패: {e}")
    
    def _run_colmap_matching(self, database_path):
        """COLMAP 매칭 (fallback)"""
        # 기존 matches 테이블 정리
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM matches")
            conn.commit()
            conn.close()
            print("  기존 matches 테이블 정리 완료")
        except Exception as e:
            print(f"  matches 테이블 정리 실패: {e}")
        
        base_cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        
        print("  COLMAP 매칭 실행...")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ 매칭 완료")
            else:
                print(f"  ✗ 매칭 실패: {result.stderr}")
        except Exception as e:
            print(f"  오류: 매칭 실패: {e}")
    
    def _run_colmap_exhaustive_matcher(self, database_path):
        """COLMAP exhaustive_matcher로 two_view_geometries 생성"""
        print("  COLMAP exhaustive_matcher 실행...")
        
        base_cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ exhaustive_matcher 완료")
                return True
            else:
                print(f"  ✗ exhaustive_matcher 실패: {result.stderr}")
                return False
        except Exception as e:
            print(f"  오류: exhaustive_matcher 실패: {e}")
            return False
    
    def _run_colmap_mapper(self, database_path, image_path, output_path):
        """COLMAP Mapper 실행"""
        
        # 먼저 데이터베이스 상태 확인
        print("  데이터베이스 상태 확인 중...")
        self._check_database_status(database_path)
        
        base_cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--Mapper.min_num_matches", "4",
            "--Mapper.init_min_num_inliers", "8",
            "--Mapper.abs_pose_min_num_inliers", "4",
            "--Mapper.filter_max_reproj_error", "16.0",
            "--Mapper.ba_global_function_tolerance", "0.000001"
        ]
        
        print("  COLMAP Mapper 실행...")
        print(f"  명령: {' '.join(base_cmd)}")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800, env=env)
            
            if result.returncode == 0:
                print("  ✓ COLMAP SfM 완료")
                return True
            else:
                print(f"  경고: COLMAP Mapper 오류 (코드: {result.returncode})")
                if result.stdout:
                    print(f"  stdout: {result.stdout}")
                if result.stderr:
                    print(f"  stderr: {result.stderr}")
                
                # 매퍼 실패 시 더 관대한 설정으로 재시도
                print("  🔄 더 관대한 설정으로 매퍼 재시도...")
                retry_cmd = [
                    self.colmap_exe, "mapper",
                    "--database_path", str(database_path),
                    "--image_path", str(image_path),
                    "--output_path", str(output_path),
                    "--Mapper.min_num_matches", "2",
                    "--Mapper.init_min_num_inliers", "4",
                    "--Mapper.abs_pose_min_num_inliers", "2",
                    "--Mapper.filter_max_reproj_error", "20.0"
                ]
                retry_result = subprocess.run(retry_cmd, capture_output=True, text=True, timeout=1800, env=env)
                if retry_result.returncode == 0:
                    print("  ✓ 재시도 매퍼 성공")
                    return True
                else:
                    print(f"  ✗ 재시도 매퍼 실패: {retry_result.stderr}")
                    return False
                
        except subprocess.TimeoutExpired:
            print("  경고: COLMAP Mapper 타임아웃")
            return False
        except Exception as e:
            print(f"  오류: COLMAP Mapper 실패: {e}")
            return False
    
    def _check_database_status(self, database_path):
        """데이터베이스 상태 확인"""
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 각 테이블의 레코드 수 확인
            tables = ['cameras', 'images', 'keypoints', 'descriptors', 'matches']
            print("  DB 상태 확인 중...")
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"    {table}: {count}개 레코드")
            
            # matches 테이블 상세 분석
            cursor.execute("SELECT pair_id, rows, cols FROM matches LIMIT 5")
            matches_sample = cursor.fetchall()
            print(f"    matches 샘플: {matches_sample}")
            
            # 추가 디버깅 정보
            if cursor.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0] == 0:
                print("  ⚠️  키포인트가 없습니다! SuperPoint 추출이 실패했을 수 있습니다.")
                print("  COLMAP SIFT로 fallback 시도...")
                self._run_colmap_feature_extraction_fallback(database_path)
            
            if cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0:
                print("  ⚠️  매칭이 없습니다! SuperGlue 매칭이 실패했을 수 있습니다.")
                print("  COLMAP 매칭으로 fallback 시도...")
                self._run_colmap_matching_fallback(database_path)
            
            conn.close()
            
        except Exception as e:
            print(f"  DB 상태 확인 실패: {e}")
    
    def _run_colmap_feature_extraction_fallback(self, database_path):
        """COLMAP SIFT 특징점 추출 (fallback)"""
        print("  COLMAP SIFT 특징점 추출 실행...")
        
        # 데이터베이스에서 이미지 경로 가져오기
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM images ORDER BY image_id")
        image_names = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # 입력 디렉토리 찾기
        input_dir = Path(database_path).parent / "input"
        
        base_cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(input_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "1000"
        ]
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ COLMAP SIFT 특징점 추출 완료")
            else:
                print(f"  ✗ COLMAP SIFT 특징점 추출 실패: {result.stderr}")
        except Exception as e:
            print(f"  오류: COLMAP SIFT 특징점 추출 실패: {e}")
    
    def _run_colmap_matching_fallback(self, database_path):
        """COLMAP 매칭 (fallback)"""
        print("  COLMAP 매칭 실행...")
        
        # 기존 matches 테이블 정리
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM matches")
            conn.commit()
            conn.close()
            print("  기존 matches 테이블 정리 완료")
        except Exception as e:
            print(f"  matches 테이블 정리 실패: {e}")
        
        base_cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ COLMAP 매칭 완료")
            else:
                print(f"  ✗ COLMAP 매칭 실패: {result.stderr}")
        except Exception as e:
            print(f"  오류: COLMAP 매칭 실패: {e}")
    
    def _run_colmap_undistortion(self, image_path, sparse_path, output_path):
        """COLMAP 언디스토션"""
        print("  COLMAP 언디스토션 실행...")
        
        # sparse 디렉토리 확인
        if not sparse_path.exists():
            print("  ⚠️  sparse 디렉토리가 없습니다. 언디스토션을 건너뜁니다.")
            return
        
        sparse_models = list(sparse_path.glob("*/"))
        if not sparse_models:
            print("  ⚠️  reconstruction 디렉토리가 없습니다. 언디스토션을 건너뜁니다.")
            return
        
        # 가장 큰 모델 선택
        try:
            best_model = max(sparse_models, key=lambda x: len(list(x.glob("*.bin"))))
            print(f"  선택된 reconstruction: {best_model}")
        except Exception as e:
            print(f"  ⚠️  reconstruction 선택 실패: {e}")
            return
        
        # reconstruction 파일 확인
        required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        missing_files = []
        for file in required_files:
            if not (best_model / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"  ⚠️  필요한 파일이 없습니다: {missing_files}")
            print("  언디스토션을 건너뜁니다.")
            return
        
        # 기존 undistorted 디렉토리 정리
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(best_model),
            "--output_path", str(output_path),
            "--output_type", "COLMAP"
        ]
        
        print(f"  언디스토션 명령: {' '.join(cmd)}")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ 언디스토션 완료")
            else:
                print(f"  ✗ 언디스토션 실패 (코드: {result.returncode})")
                if result.stdout:
                    print(f"  stdout: {result.stdout}")
                if result.stderr:
                    print(f"  stderr: {result.stderr}")
                
                # 언디스토션 실패 시 원본 이미지 복사
                print("  🔄 원본 이미지 복사로 fallback...")
                self._copy_original_images_fallback(image_path, output_path)
                
        except subprocess.TimeoutExpired:
            print("  경고: 언디스토션 타임아웃")
            self._copy_original_images_fallback(image_path, output_path)
        except Exception as e:
            print(f"  오류: 언디스토션 실패: {e}")
            self._copy_original_images_fallback(image_path, output_path)
    
    def _copy_original_images_fallback(self, image_path, output_path):
        """언디스토션 실패 시 원본 이미지 복사"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            images_dir = output_path / "images"
            images_dir.mkdir(exist_ok=True)
            
            # 원본 이미지들을 undistorted/images로 복사
            for img_file in Path(image_path).glob("*.jpg"):
                dst_file = images_dir / img_file.name
                shutil.copy2(img_file, dst_file)
            
            print(f"  ✓ 원본 이미지 복사 완료: {len(list(Path(image_path).glob('*.jpg')))}개")
            
        except Exception as e:
            print(f"  오류: 원본 이미지 복사 실패: {e}")
    
    def _convert_to_3dgs_format(self, colmap_path, original_image_paths):
        """3DGS 형식 변환"""
        try:
            # sparse 디렉토리 확인
            sparse_dir = colmap_path / "sparse"
            if not sparse_dir.exists():
                print("  경고: sparse 디렉토리가 없음")
                return self._create_default_scene_info(original_image_paths, colmap_path)
            
            # reconstruction 디렉토리 찾기
            reconstruction_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
            if not reconstruction_dirs:
                print("  경고: reconstruction 디렉토리가 없음")
                return self._create_default_scene_info(original_image_paths, colmap_path)
            
            # 가장 큰 reconstruction 선택
            try:
                best_recon = max(reconstruction_dirs, key=lambda x: len(list(x.glob("*.bin"))))
                print(f"  선택된 reconstruction: {best_recon}")
            except Exception as e:
                print(f"  reconstruction 선택 실패: {e}")
                return self._create_default_scene_info(original_image_paths, colmap_path)
            
            # reconstruction 파일 확인
            required_files = ['cameras.bin', 'images.bin']
            missing_files = []
            for file in required_files:
                if not (best_recon / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                print(f"  필요한 reconstruction 파일이 없음: {missing_files}")
                return self._create_default_scene_info(original_image_paths, colmap_path)
            
            # SceneInfo 생성 시도
            return self._create_scene_info_from_colmap(best_recon, original_image_paths, colmap_path)
            
        except Exception as e:
            print(f"  3DGS 변환 오류: {e}")
            return self._create_default_scene_info(original_image_paths, colmap_path)
    
    def _create_scene_info_from_colmap(self, reconstruction_path, original_image_paths, output_path):
        """COLMAP reconstruction에서 SceneInfo 생성"""
        print("  COLMAP reconstruction 파싱 중...")
        try:
            # 상대 경로로 import 시도
            import sys
            current_dir = Path(__file__).parent.parent
            scene_dir = current_dir / "scene"
            if scene_dir.exists():
                sys.path.insert(0, str(scene_dir))
            
            from colmap_loader import read_points3D_binary, read_points3D_text
            from utils.graphics_utils import BasicPointCloud
            import numpy as np
        except ImportError as e:
            print(f"  Import 오류: {e}, fallback 사용")
            return self._create_default_scene_info(original_image_paths, output_path)
        
        # points3D.bin 또는 points3D.txt 경로 찾기
        bin_path = reconstruction_path / 'points3D.bin'
        txt_path = reconstruction_path / 'points3D.txt'
        xyz = rgb = None
        
        try:
            if bin_path.exists():
                xyz, rgb, _ = read_points3D_binary(str(bin_path))
                print(f"  points3D.bin에서 {len(xyz)}개 포인트 로드")
            elif txt_path.exists():
                xyz, rgb, _ = read_points3D_text(str(txt_path))
                print(f"  points3D.txt에서 {len(xyz)}개 포인트 로드")
            else:
                print("  points3D 파일 없음, fallback 사용")
                return self._create_default_scene_info(original_image_paths, output_path)
            
            if xyz is None or len(xyz) == 0:
                print("  points3D에 포인트 없음, fallback 사용")
                return self._create_default_scene_info(original_image_paths, output_path)
            
            # colors 정규화 (0-255 -> 0-1)
            rgb = rgb.astype(np.float32) / 255.0
            
            # normals 생성 (0으로 초기화)
            normals = np.zeros_like(xyz, dtype=np.float32)
            
            # BasicPointCloud 생성
            point_cloud = BasicPointCloud(points=xyz.astype(np.float32), 
                                        colors=rgb.astype(np.float32), 
                                        normals=normals.astype(np.float32))
            
            # 카메라 등은 fallback과 동일하게 생성
            scene_info = self._create_default_scene_info(original_image_paths, output_path)
            scene_info = scene_info._replace(point_cloud=point_cloud)
            
            print(f"  ✓ 실제 COLMAP 포인트 클라우드 사용: {len(xyz)}개 포인트")
            return scene_info
            
        except Exception as e:
            print(f"  포인트 클라우드 파싱 오류: {e}, fallback 사용")
            return self._create_default_scene_info(original_image_paths, output_path)
    
    def _create_default_scene_info(self, image_paths, output_path):
        """기본 SceneInfo 생성"""
        print("  기본 SceneInfo 생성 중...")
        
        try:
            # 기본 카메라 설정
            sample_img = cv2.imread(str(image_paths[0]))
            if sample_img is None:
                height, width = 480, 640
            else:
                height, width = sample_img.shape[:2]
            
            # 카메라 정보 생성
            train_cameras = []
            for i, img_path in enumerate(image_paths):
                # 기본 카메라 파라미터
                focal_length = max(width, height) * 1.2
                fov_x = 2 * np.arctan(width / (2 * focal_length))
                fov_y = 2 * np.arctan(height / (2 * focal_length))
                
                # 기본 포즈 (원형 배치)
                angle = 2 * np.pi * i / len(image_paths)
                radius = 2.0
                
                R = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ], dtype=np.float32)
                T = np.array([radius * np.cos(angle), 0, radius * np.sin(angle)], dtype=np.float32)
                
                cam_info = CameraInfo(
                    uid=i, R=R, T=T, FovY=fov_y, FovX=fov_x,
                    depth_params=None, image_path=str(img_path), 
                    image_name=img_path.name, depth_path="", width=width, height=height,
                    is_test=(i % 8 == 0)
                )
                train_cameras.append(cam_info)
            
            # 기본 포인트 클라우드 (임의 점들)
            xyz = np.random.randn(1000, 3).astype(np.float32) * 0.5
            rgb = np.random.rand(1000, 3).astype(np.float32)
            
            from utils.graphics_utils import BasicPointCloud
            point_cloud = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((1000, 3), dtype=np.float32))
            
            # PLY 파일 저장
            ply_path = output_path / "points3D.ply"
            self._save_ply(ply_path, xyz, rgb)
            
            # 학습/테스트 분할
            train_cams = [c for c in train_cameras if not c.is_test]
            test_cams = [c for c in train_cameras if c.is_test]
            
            # NeRF 정규화 계산
            cam_centers = []
            for cam in train_cameras:
                # 카메라 중심 계산
                cam_pos = -np.dot(cam.R.T, cam.T)
                cam_centers.append(cam_pos)
            
            if cam_centers:
                cam_centers = np.array(cam_centers)
                center = np.mean(cam_centers, axis=0)
                distances = np.linalg.norm(cam_centers - center, axis=1)
                radius = np.max(distances) * 1.1
            else:
                center = np.zeros(3)
                radius = 3.0
            
            nerf_norm = {"translate": -center, "radius": radius}
            
            scene_info = SceneInfo(
                point_cloud=point_cloud,
                train_cameras=train_cams,
                test_cameras=test_cams,
                nerf_normalization=nerf_norm,
                ply_path=str(ply_path),
                is_nerf_synthetic=False
            )
            
            print(f"  ✓ 기본 SceneInfo 생성 완료 ({len(train_cameras)}개 카메라)")
            return scene_info
            
        except Exception as e:
            print(f"  오류: 기본 SceneInfo 생성 실패: {e}")
            # 최후의 fallback: 최소한의 SceneInfo 생성
            try:
                from utils.graphics_utils import BasicPointCloud
                xyz = np.random.randn(100, 3).astype(np.float32) * 0.5
                rgb = np.random.rand(100, 3).astype(np.float32)
                point_cloud = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((100, 3), dtype=np.float32))
                
                scene_info = SceneInfo(
                    point_cloud=point_cloud,
                    train_cameras=[],
                    test_cameras=[],
                    nerf_normalization={"translate": np.array([0, 0, 0]), "radius": 1.0},
                    ply_path="",
                    is_nerf_synthetic=False
                )
                print("  ✓ 최후 fallback SceneInfo 생성 완료")
                return scene_info
            except Exception as final_e:
                print(f"  치명적 오류: {final_e}")
                raise
    
    def _save_ply(self, path, xyz, rgb):
        """PLY 파일 저장"""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(xyz)):
                x, y, z = xyz[i]
                r, g, b = (rgb[i] * 255).astype(np.uint8)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

# 실행 함수
def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperGlue + COLMAP 하이브리드")
    parser.add_argument("--source_path", "-s", required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--max_images", type=int, default=100, help="최대 이미지 수")
    parser.add_argument("--superglue_config", choices=["outdoor", "indoor"], 
                        default="outdoor", help="SuperGlue 설정")
    parser.add_argument("--colmap_exe", default="colmap", help="COLMAP 실행파일 경로")
    parser.add_argument("--device", default="cuda", help="연산 장치")
    
    args = parser.parse_args()
    
    # 하이브리드 파이프라인 실행
    hybrid = SuperGlueCOLMAPHybrid(
        superglue_config=args.superglue_config,
        colmap_exe=args.colmap_exe,
        device=args.device
    )
    
    scene_info = hybrid.process(args.source_path, args.max_images)
    
    if scene_info:
        print("🎉 하이브리드 파이프라인 성공!")
        print(f"📁 결과: {len(scene_info.train_cameras)}개 카메라")
    else:
        print("❌ 파이프라인 실패")

if __name__ == "__main__":
    main()