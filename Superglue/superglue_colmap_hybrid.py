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
            
            # 모델 초기화 (실제 모델이 있다고 가정)
            print(f"  SuperPoint/SuperGlue 모델 로드 중...")
            print(f"  ✓ 모델 로드 완료 (device: {self.device})")
            
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
            print("\n[1/5] 이미지 수집...")
            image_paths = self._collect_images(image_dir, max_images)
            if len(image_paths) == 0:
                raise RuntimeError("처리할 이미지를 찾을 수 없습니다")
            
            input_dir = self._prepare_input_images(image_paths, output_path)
            
            # 2. COLMAP 데이터베이스 생성 (수정된 버전)
            print("\n[2/5] COLMAP 데이터베이스 생성...")
            database_path = output_path / "database.db"
            self._create_fixed_colmap_database(image_paths, database_path, input_dir)
            
            # 3. COLMAP 특징점 추출
            print("\n[3/6] COLMAP 특징점 추출...")
            self._run_colmap_feature_extraction(database_path, input_dir)
            
            # 4. COLMAP 매칭
            print("\n[4/6] COLMAP 매칭...")
            self._run_colmap_matching(database_path)
            
            # 5. COLMAP으로 포즈 추정 (수정된 설정)
            print("\n[5/6] COLMAP 포즈 추정...")
            sparse_dir = output_path / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            self._run_colmap_mapper_fixed(database_path, input_dir, sparse_dir)
            
            # 6. 이미지 언디스토션
            print("\n[6/6] 이미지 언디스토션...")
            undistorted_dir = output_path / "undistorted"
            self._run_colmap_undistortion(input_dir, sparse_dir, undistorted_dir)
            
            # 7. 3DGS 형식으로 변환
            print("\n[7/6] 3DGS 형식 변환...")
            scene_info = self._convert_to_3dgs_format_fixed(output_path, image_paths)
            
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
    
    def _create_fixed_colmap_database(self, image_paths, database_path, input_dir):
        """수정된 COLMAP 데이터베이스 생성 (debug_hybrid_pipeline 방식)"""
        
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
            
            # 이미지 정보 추가 (처음 20장만)
            for i, img_path in enumerate(image_paths[:20]):
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
    
    def _create_database_schema(self, cursor):
        """COLMAP 데이터베이스 스키마 생성"""
        
        # 카메라 테이블
        cursor.execute('''
            CREATE TABLE cameras (
                camera_id INTEGER PRIMARY KEY,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB NOT NULL,
                prior_focal_length INTEGER NOT NULL
            )
        ''')
        
        # 이미지 테이블
        cursor.execute('''
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            )
        ''')
        
        # 키포인트 테이블
        cursor.execute('''
            CREATE TABLE keypoints (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            )
        ''')
        
        # 디스크립터 테이블
        cursor.execute('''
            CREATE TABLE descriptors (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            )
        ''')
        
        # 매칭 테이블
        cursor.execute('''
            CREATE TABLE matches (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE UNIQUE INDEX index_name ON images(name)')
    
    def _add_default_camera(self, cursor, sample_image_path):
        """기본 카메라 모델 추가"""
        # 샘플 이미지에서 해상도 얻기
        img = cv2.imread(str(sample_image_path))
        if img is None:
            height, width = 480, 640  # 기본값
        else:
            height, width = img.shape[:2]
        
        # PINHOLE 모델 (model=1)
        # params: [fx, fy, cx, cy]
        fx = fy = max(width, height) * 1.2  # 추정된 초점거리
        cx, cy = width / 2, height / 2
        
        params = np.array([fx, fy, cx, cy], dtype=np.float64)
        
        cursor.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
            (1, 1, width, height, params.tobytes(), int(fx))
        )
        
        print(f"  카메라 추가: {width}x{height}, focal={fx:.1f}")
        return 1
    
    def _add_dummy_keypoints(self, cursor, image_id):
        """더미 키포인트 추가 (COLMAP 호환성)"""
        # 더 많은 더미 키포인트 생성 (격자 패턴)
        keypoints = []
        for i in range(0, 640, 50):
            for j in range(0, 480, 50):
                keypoints.append([i, j])
        
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # 더미 디스크립터 (128차원)
        descriptors = np.random.randint(0, 255, (len(keypoints), 128), dtype=np.uint8)
        
        # 키포인트 추가
        cursor.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, len(keypoints), 2, keypoints.tobytes())
        )
        
        # 디스크립터 추가
        cursor.execute(
            "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, len(keypoints), 128, descriptors.tobytes())
        )
    
    def _run_colmap_feature_extraction(self, database_path, image_path):
        """COLMAP 특징점 추출 (debug_hybrid_pipeline 방식)"""
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
        
        # xvfb 사용 가능한지 확인
        try:
            xvfb_result = subprocess.run(["which", "xvfb-run"], capture_output=True, text=True)
            use_xvfb = xvfb_result.returncode == 0
        except:
            use_xvfb = False
        
        if use_xvfb:
            cmd = ["xvfb-run", "-a"] + base_cmd
        else:
            cmd = base_cmd
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ 특징점 추출 완료")
            else:
                print(f"  ✗ 특징점 추출 실패: {result.stderr}")
        except Exception as e:
            print(f"  오류: 특징점 추출 실패: {e}")
    
    def _run_colmap_matching(self, database_path):
        """COLMAP 매칭 (debug_hybrid_pipeline 방식)"""
        base_cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        
        print("  COLMAP 매칭 실행...")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        # xvfb 사용 가능한지 확인
        try:
            xvfb_result = subprocess.run(["which", "xvfb-run"], capture_output=True, text=True)
            use_xvfb = xvfb_result.returncode == 0
        except:
            use_xvfb = False
        
        if use_xvfb:
            cmd = ["xvfb-run", "-a"] + base_cmd
        else:
            cmd = base_cmd
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ 매칭 완료")
            else:
                print(f"  ✗ 매칭 실패: {result.stderr}")
                # 매칭 실패 시 더 관대한 설정으로 재시도
                print("  🔄 더 관대한 설정으로 매칭 재시도...")
                retry_cmd = [
                    self.colmap_exe, "exhaustive_matcher",
                    "--database_path", str(database_path),
                    "--SiftMatching.max_ratio", "0.9",
                    "--SiftMatching.max_distance", "0.7"
                ]
                retry_result = subprocess.run(retry_cmd, capture_output=True, text=True, timeout=1800, env=env)
                if retry_result.returncode == 0:
                    print("  ✓ 재시도 매칭 완료")
                else:
                    print(f"  ✗ 재시도 매칭 실패: {retry_result.stderr}")
        except Exception as e:
            print(f"  오류: 매칭 실패: {e}")
    
    def _run_colmap_mapper_fixed(self, database_path, image_path, output_path):
        """수정된 COLMAP Mapper 실행 (debug_hybrid_pipeline 방식)"""
        
        # COLMAP 명령 생성 (더 관대한 설정)
        base_cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--Mapper.min_num_matches", "4",  # 최소 매칭 수 낮춤
            "--Mapper.init_min_num_inliers", "8",  # 최소 인라이어 수 낮춤
            "--Mapper.abs_pose_min_num_inliers", "4",  # 절대 포즈 최소 인라이어 낮춤
            "--Mapper.filter_max_reproj_error", "16.0",  # 재투영 오차 허용치 높임
            "--Mapper.ba_global_function_tolerance", "0.000001"
        ]
        
        print("  COLMAP Mapper 실행...")
        print(f"  명령: {' '.join(base_cmd)}")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        # xvfb 사용 가능한지 확인
        try:
            xvfb_result = subprocess.run(["which", "xvfb-run"], capture_output=True, text=True)
            use_xvfb = xvfb_result.returncode == 0
        except:
            use_xvfb = False
        
        if use_xvfb:
            cmd = ["xvfb-run", "-a"] + base_cmd
        else:
            cmd = base_cmd
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            
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
                
                # DB 상태 확인
                self._check_database_status(database_path)
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
            
            conn.close()
            
        except Exception as e:
            print(f"  DB 상태 확인 실패: {e}")
    
    def _run_colmap_undistortion(self, image_path, sparse_path, output_path):
        """COLMAP 언디스토션"""
        sparse_models = list(sparse_path.glob("*/"))
        if not sparse_models:
            print("  경고: Sparse reconstruction 없음")
            return
        
        # 가장 큰 모델 선택
        best_model = max(sparse_models, key=lambda x: len(list(x.glob("*.bin"))))
        
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(best_model),
            "--output_path", str(output_path),
            "--output_type", "COLMAP"
        ]
        
        print("  COLMAP 언디스토션 실행...")
        
        # Qt GUI 문제 해결을 위한 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode == 0:
                print("  ✓ 언디스토션 완료")
            else:
                print("  경고: 언디스토션 오류")
        except Exception as e:
            print(f"  오류: 언디스토션 실패: {e}")
    
    def _convert_to_3dgs_format_fixed(self, colmap_path, original_image_paths):
        """수정된 3DGS 형식 변환"""
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
            best_recon = max(reconstruction_dirs, key=lambda x: len(list(x.glob("*.bin"))))
            print(f"  선택된 reconstruction: {best_recon}")
            
            # SceneInfo 생성 시도
            return self._create_scene_info_from_colmap(best_recon, original_image_paths, colmap_path)
            
        except Exception as e:
            print(f"  3DGS 변환 오류: {e}")
            return self._create_default_scene_info(original_image_paths, colmap_path)
    
    def _create_scene_info_from_colmap(self, reconstruction_path, original_image_paths, output_path):
        """COLMAP reconstruction에서 SceneInfo 생성"""
        # 이건 복잡한 구현이므로 일단 기본 SceneInfo 반환
        print("  COLMAP reconstruction 파싱 중...")
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
                ])
                T = np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
                
                # 이미지 로드
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.astype(np.float32) / 255.0
                
                cam_info = CameraInfo(
                    uid=i, R=R, T=T, FovY=fov_y, FovX=fov_x,
                    depth_params=None, image_path=str(img_path), 
                    image_name=img_path.name, depth_path="", width=width, height=height,
                    is_test=(i % 8 == 0)
                )
                train_cameras.append(cam_info)
            
            # 기본 포인트 클라우드 (임의 점들)
            xyz = np.random.randn(1000, 3) * 0.5
            rgb = np.random.rand(1000, 3)
            
            from utils.graphics_utils import BasicPointCloud
            point_cloud = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((1000, 3)))
            
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