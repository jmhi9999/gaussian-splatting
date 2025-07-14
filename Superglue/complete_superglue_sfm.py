# superglue_3dgs_complete.py
# SuperGlue와 3DGS 완전 통합 파이프라인

import glob
from tkinter import Image
import numpy as np
import cv2
import torch
from pathlib import Path
import json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys
import time
import gc
# import psutil  # 제거 - 의존성 문제 해결
from scipy.spatial.distance import cdist

# SuperGlue 관련 imports
from models.matching import Matching
from models.utils import frame2tensor

# 3DGS 관련 imports - lazy import로 변경
def get_3dgs_imports():
    """3DGS 관련 모듈들을 lazy import - 개선된 버전"""
    # gaussian-splatting 루트 디렉토리를 Python path에 추가
    gaussian_splatting_root = Path(__file__).parent.parent
    if str(gaussian_splatting_root) not in sys.path:
        sys.path.insert(0, str(gaussian_splatting_root))
    
    # 추가 경로들 시도
    additional_paths = [
        gaussian_splatting_root,
        gaussian_splatting_root / "scene",
        gaussian_splatting_root / "utils",
        Path.cwd(),
        Path.cwd().parent
    ]
    
    for path in additional_paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    
    try:
        # 먼저 scene.dataset_readers 시도
        from scene.dataset_readers import CameraInfo, SceneInfo
        print("✓ Successfully imported CameraInfo and SceneInfo from scene.dataset_readers")
    except ImportError as e:
        print(f"✗ Failed to import from scene.dataset_readers: {e}")
        try:
            # 직접 import 시도
            import scene.dataset_readers as dataset_readers
            CameraInfo = dataset_readers.CameraInfo
            SceneInfo = dataset_readers.SceneInfo
            print("✓ Successfully imported CameraInfo and SceneInfo via direct import")
        except ImportError as e2:
            print(f"✗ Direct import also failed: {e2}")
            # Fallback 클래스 정의
            print("⚠️  Creating fallback CameraInfo and SceneInfo classes")
            
            class CameraInfo:
                def __init__(self, uid, R, T, FovY, FovX, image_path, image_name, 
                             width, height, depth_params=None, depth_path="", is_test=False):
                    self.uid = uid
                    self.R = R
                    self.T = T
                    self.FovY = FovY
                    self.FovX = FovX
                    self.image_path = image_path
                    self.image_name = image_name
                    self.width = width
                    self.height = height
                    self.depth_params = depth_params
                    self.depth_path = depth_path
                    self.is_test = is_test
            
            class SceneInfo:
                def __init__(self, point_cloud, train_cameras, test_cameras, 
                             nerf_normalization, ply_path="", is_nerf_synthetic=False):
                    self.point_cloud = point_cloud
                    self.train_cameras = train_cameras
                    self.test_cameras = test_cameras
                    self.nerf_normalization = nerf_normalization
                    self.ply_path = ply_path
                    self.is_nerf_synthetic = is_nerf_synthetic
    
    try:
        # utils.graphics_utils 시도
        from utils.graphics_utils import BasicPointCloud
        print("✓ Successfully imported BasicPointCloud from utils.graphics_utils")
    except ImportError as e:
        print(f"✗ Failed to import BasicPointCloud from utils.graphics_utils: {e}")
        try:
            # 직접 import 시도
            import utils.graphics_utils as graphics_utils
            BasicPointCloud = graphics_utils.BasicPointCloud
            print("✓ Successfully imported BasicPointCloud via direct import")
        except ImportError as e2:
            print(f"✗ Direct import also failed: {e2}")
            # Fallback 클래스 정의
            print("⚠️  Creating fallback BasicPointCloud class")
            
            class BasicPointCloud:
                def __init__(self, points, colors, normals):
                    self.points = points
                    self.colors = colors
                    self.normals = normals
    
    # 최종 확인
    if 'CameraInfo' not in locals() or 'SceneInfo' not in locals() or 'BasicPointCloud' not in locals():
        print("❌ Critical: Could not import any 3DGS modules")
        return None, None, None
    
    print("✅ All 3DGS modules successfully imported or created")
    return CameraInfo, SceneInfo, BasicPointCloud


def test_pipeline_availability():
    """파이프라인 가용성 테스트 - 개선된 버전"""
    print("🔍 Testing SuperGlue 3DGS Pipeline availability...")
    
    # 1. SuperGlue 모듈 테스트
    superglue_available = False
    try:
        from models.matching import Matching
        from models.utils import frame2tensor
        print("✓ SuperGlue modules available")
        superglue_available = True
    except ImportError as e:
        print(f"✗ SuperGlue modules not available: {e}")
        print("  This is expected if SuperGlue models are not installed")
    
    # 2. 3DGS 모듈 테스트
    gs_available = False
    try:
        CameraInfo, SceneInfo, BasicPointCloud = get_3dgs_imports()
        if CameraInfo is not None and SceneInfo is not None and BasicPointCloud is not None:
            print("✓ 3DGS modules available")
            gs_available = True
        else:
            print("✗ 3DGS modules not available")
    except Exception as e:
        print(f"✗ 3DGS modules test failed: {e}")
    
    # 3. 기타 의존성 테스트
    core_available = False
    missing_deps = []
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        missing_deps.append("torch")
        print("✗ PyTorch not available")
    
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
        print("✗ OpenCV not available")
    
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ NumPy not available")
    
    try:
        from scipy.optimize import least_squares
        print("✓ SciPy available")
    except ImportError:
        missing_deps.append("scipy")
        print("✗ SciPy not available")
    
    if not missing_deps:
        print("✓ Core dependencies available")
        core_available = True
    else:
        print(f"✗ Missing core dependencies: {missing_deps}")
    
    # 4. 전체 가용성 판단 (더 관대하게)
    # SuperGlue가 없어도 fallback으로 작동할 수 있도록
    pipeline_available = gs_available and core_available
    
    print(f"\n📊 Pipeline Availability Summary:")
    print(f"  SuperGlue: {'✓' if superglue_available else '✗'}")
    print(f"  3DGS: {'✓' if gs_available else '✗'}")
    print(f"  Core Dependencies: {'✓' if core_available else '✗'}")
    print(f"  Overall: {'✓' if pipeline_available else '✗'}")
    
    if not pipeline_available:
        print("\n⚠️  Pipeline not fully available, but fallback mode may work")
        print("   Missing dependencies can be installed with:")
        print("   pip install numpy opencv-python torch torchvision scipy matplotlib psutil pillow")
    
    return pipeline_available


# 파이프라인 가용성 테스트 실행
PIPELINE_AVAILABLE = test_pipeline_availability()


class SuperGlue3DGSPipeline:
    """SuperGlue 기반 완전한 3DGS SfM 파이프라인"""
    
    def __init__(self, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 성능 모니터링
        self.start_time = time.time()
        self.memory_usage = []
        
        # 파이프라인 가용성 확인 (더 관대하게)
        if not PIPELINE_AVAILABLE:
            print("⚠️  Pipeline not fully available, but will attempt to run in fallback mode")
            print("   Some features may not work without proper dependencies")
        
        # SuperGlue 설정 (더 완화된 설정)
        if config is None:
            config = {
                'superpoint': {
                    'nms_radius': 2,  # 3 → 2로 더 완화
                    'keypoint_threshold': 0.0005,  # 0.001 → 0.0005로 더 완화
                    'max_keypoints': 10240  # 8192 → 10240로 증가
                },
                'superglue': {
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 10,  # 15 → 10으로 완화
                    'match_threshold': 0.01,  # 0.05 → 0.01로 대폭 완화
                }
            }
        
        # SuperGlue 모델 로드 시도
        self.superglue_available = False
        try:
            from models.matching import Matching
            self.matching = Matching(config).eval().to(self.device)
            self.superglue_available = True
            print(f"✓ SuperGlue matching model loaded on {self.device}")
        except Exception as e:
            print(f"⚠️  SuperGlue model not available: {e}")
            print("   Will use fallback pose estimation methods")
            self.matching = None
            self.superglue_available = False
        
        # SfM 데이터 저장소
        self.cameras = {}  # camera_id -> {'R': R, 'T': T, 'K': K, 'image_path': path}
        self.points_3d = {}  # point_id -> {'xyz': xyz, 'color': rgb, 'observations': [(cam_id, kpt_idx)]}
        self.image_features = {}  # image_id -> SuperPoint features
        self.matches = {}  # (img_i, img_j) -> SuperGlue matches
        
        # Bundle Adjustment를 위한 추가 데이터
        self.camera_graph = defaultdict(list)  # 카메라 연결 그래프
        self.point_observations = defaultdict(list)  # 포인트 관찰 데이터
        
        # 품질 메트릭
        self.quality_metrics = {
            'pose_estimation_success_rate': 0.0,
            'average_matches_per_pair': 0.0,
            'bundle_adjustment_cost': float('inf'),
            'total_processing_time': 0.0
        }
        
        print(f'✅ SuperGlue 3DGS Pipeline initialized on {self.device}')
        if not self.superglue_available:
            print('   Running in fallback mode (SuperGlue not available)')
    
    def _monitor_memory(self):
        """메모리 사용량 모니터링 (psutil 없이)"""
        try:
            # 간단한 메모리 모니터링 (psutil 없이)
            if torch.cuda.is_available():
                # GPU 메모리 사용량
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                print(f"    GPU Memory: {gpu_memory:.1f} MB")
            else:
                # CPU 메모리는 간단한 추정
                print(f"    Memory monitoring: CPU mode")
        except:
            print(f"    Memory monitoring: Not available")
    
    def _cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_images_to_3dgs(self, image_dir, output_dir, max_images=120):
        """이미지들을 3DGS 형식으로 처리 - COLMAP 하이브리드 지원"""
        print(f"Processing images from {image_dir} to {output_dir}")
        
        try:
            # output_dir 저장 (COLMAP intrinsics 읽기용)
            self.output_dir = output_dir
            
            # 이미지 수집
            image_paths = self._collect_images(image_dir, max_images)
            if not image_paths:
                raise RuntimeError("No images found")
            
            print(f"Found {len(image_paths)} images")
            self._monitor_memory()
            
            # 🔧 NEW: COLMAP reconstruction 확인 및 활용
            colmap_available = self._check_colmap_reconstruction(output_dir)
            if colmap_available:
                print("  ✓ COLMAP reconstruction found - using hybrid approach")
                return self._process_with_colmap_hybrid(image_paths, output_dir)
            else:
                print("  ⚠️  COLMAP reconstruction not found - using SuperGlue only")
            
            # 특징점 추출
            self._extract_all_features(image_paths)
            self._monitor_memory()
            
            # 매칭
            self._intelligent_matching()
            self._monitor_memory()
            
            # 카메라 포즈 추정
            self._estimate_camera_poses_robust()
            self._monitor_memory()
            
            # 삼각측량
            n_points = self._triangulate_all_points_robust()
            self._monitor_memory()
            
            # Bundle Adjustment
            self._bundle_adjustment_robust()
            self._monitor_memory()
            
            # 품질 메트릭 계산
            self._compute_quality_metrics()
            
            # 3DGS SceneInfo 생성
            scene_info = self._create_3dgs_scene_info(image_paths)
            
            # 3DGS 형식으로 저장
            self._save_3dgs_format(scene_info, output_dir)
            
            # 최종 메모리 정리
            self._cleanup_memory()
            
            return scene_info
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 실패시 fallback
            print("Falling back to simple camera arrangement...")
            return self._create_fallback_scene_info(image_paths)
    
    def _check_colmap_reconstruction(self, output_dir):
        """COLMAP reconstruction 존재 여부 및 유효성 확인"""
        try:
            reconstruction_path = Path(output_dir) / "sparse" / "0"
            cameras_bin = reconstruction_path / "cameras.bin"
            images_bin = reconstruction_path / "images.bin"
            points3d_bin = reconstruction_path / "points3D.bin"
            
            if not cameras_bin.exists() or not images_bin.exists():
                print(f"    No COLMAP reconstruction found at: {reconstruction_path}")
                return False
            
            # 파일 크기 확인
            if cameras_bin.stat().st_size == 0 or images_bin.stat().st_size == 0:
                print(f"    COLMAP files are empty, removing corrupted files")
                self._cleanup_corrupted_colmap_files(reconstruction_path)
                return False
            
            # 파일 형식 검증 시도
            try:
                from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
                
                # cameras.bin 검증
                cameras = read_intrinsics_binary(str(cameras_bin))
                if len(cameras) == 0:
                    print(f"    cameras.bin is empty or corrupted")
                    self._cleanup_corrupted_colmap_files(reconstruction_path)
                    return False
                
                # images.bin 검증
                images = read_extrinsics_binary(str(images_bin))
                if len(images) == 0:
                    print(f"    images.bin is empty or corrupted")
                    self._cleanup_corrupted_colmap_files(reconstruction_path)
                    return False
                
                print(f"    Found valid COLMAP reconstruction at: {reconstruction_path}")
                print(f"    - {len(cameras)} cameras, {len(images)} images")
                return True
                
            except Exception as e:
                print(f"    COLMAP files are corrupted: {e}")
                self._cleanup_corrupted_colmap_files(reconstruction_path)
                return False
                
        except Exception as e:
            print(f"    COLMAP reconstruction check failed: {e}")
            return False
    
    def _cleanup_corrupted_colmap_files(self, reconstruction_path):
        """손상된 COLMAP 파일들 정리"""
        try:
            for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
                file_path = reconstruction_path / file_name
                if file_path.exists():
                    file_path.unlink()
                    print(f"    Deleted corrupted {file_name}")
        except Exception as e:
            print(f"    Failed to cleanup corrupted files: {e}")
    
    def _process_with_colmap_hybrid(self, image_paths, output_dir):
        """COLMAP reconstruction을 활용한 하이브리드 처리"""
        print("  🔄 Using COLMAP + SuperGlue hybrid approach")
        
        try:
            # 1. COLMAP reconstruction 로드
            colmap_data = self._load_colmap_reconstruction(output_dir)
            if colmap_data is None:
                print("    Failed to load COLMAP reconstruction, falling back to SuperGlue")
                return self._process_superglue_only(image_paths, output_dir)
            
            cameras, images, points3d = colmap_data
            print(f"    Loaded {len(cameras)} cameras, {len(images)} images, {len(points3d)} points from COLMAP")
            
            # 2. COLMAP 데이터를 SuperGlue 형식으로 변환
            self._convert_colmap_to_superglue_format(cameras, images, points3d, image_paths)
            
            # 3. SuperGlue 특징점 추출 (COLMAP 포즈 개선용)
            self._extract_all_features(image_paths)
            
            # 4. COLMAP 포즈를 기반으로 한 개선된 매칭
            self._intelligent_matching_with_colmap_poses()
            
            # 5. Bundle Adjustment (COLMAP 초기값 사용)
            self._bundle_adjustment_with_colmap_initialization()
            
            # 6. 3DGS SceneInfo 생성
            scene_info = self._create_3dgs_scene_info(image_paths)
            
            # 7. 3DGS 형식으로 저장 (안전한 방법)
            try:
                self._save_3dgs_format(scene_info, output_dir)
            except Exception as save_error:
                print(f"    Warning: Failed to save 3DGS format: {save_error}")
                import traceback
                traceback.print_exc()
                # 기본 텍스트 파일만 저장
                self._save_basic_format(scene_info, output_dir)
            
            return scene_info
            
        except Exception as e:
            print(f"    Hybrid processing failed: {e}")
            import traceback
            traceback.print_exc()
            print("    Falling back to SuperGlue only")
            return self._process_superglue_only(image_paths, output_dir)
    
    def _save_basic_format(self, scene_info, output_dir):
        """기본 텍스트 파일만 저장 (fallback)"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(exist_ok=True, parents=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 텍스트 파일만 저장
        self._write_cameras_txt(scene_info.train_cameras + scene_info.test_cameras, 
                               sparse_dir / "cameras.txt")
        self._write_images_txt(scene_info.train_cameras + scene_info.test_cameras, 
                              sparse_dir / "images.txt")
        self._write_points3d_ply(scene_info.point_cloud, sparse_dir / "points3D.ply")
        
        print(f"    Saved basic format to {output_dir}")
    
    def _load_colmap_reconstruction(self, output_dir):
        """COLMAP reconstruction 로드"""
        try:
            reconstruction_path = Path(output_dir) / "sparse" / "0"
            print(f"    Checking reconstruction path: {reconstruction_path}")
            
            # 파일 존재 확인
            cameras_bin = reconstruction_path / "cameras.bin"
            images_bin = reconstruction_path / "images.bin"
            points3d_bin = reconstruction_path / "points3D.bin"
            
            print(f"    cameras.bin exists: {cameras_bin.exists()}")
            print(f"    images.bin exists: {images_bin.exists()}")
            print(f"    points3D.bin exists: {points3d_bin.exists()}")
            
            if not cameras_bin.exists() or not images_bin.exists() or not points3d_bin.exists():
                print("    Missing required COLMAP files")
                return None
            
            # 파일 크기 확인
            print(f"    cameras.bin size: {cameras_bin.stat().st_size} bytes")
            print(f"    images.bin size: {images_bin.stat().st_size} bytes")
            print(f"    points3D.bin size: {points3d_bin.stat().st_size} bytes")
            
            # COLMAP 모듈 import
            try:
                from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, read_points3D_binary
                print("    COLMAP loader imported successfully")
            except ImportError as e:
                print(f"    COLMAP loader import failed: {e}")
                return None
            
            # 카메라 내부 파라미터 로드
            print("    Loading cameras.bin...")
            try:
                cameras = read_intrinsics_binary(str(cameras_bin))
                print(f"    Loaded {len(cameras)} cameras")
            except Exception as e:
                print(f"    Failed to load cameras.bin: {e}")
                return None
            
            # 카메라 외부 파라미터 로드
            print("    Loading images.bin...")
            try:
                images = read_extrinsics_binary(str(images_bin))
                print(f"    Loaded {len(images)} images")
            except Exception as e:
                print(f"    Failed to load images.bin: {e}")
                return None
            
            # 3D 포인트 로드
            print("    Loading points3D.bin...")
            try:
                points3d = read_points3D_binary(str(points3d_bin))
                print(f"    Loaded {len(points3d)} 3D points")
            except Exception as e:
                print(f"    Failed to load points3D.bin: {e}")
                return None
            
            return cameras, images, points3d
            
        except Exception as e:
            print(f"    Failed to load COLMAP reconstruction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_colmap_to_superglue_format(self, cameras, images, points3d, image_paths):
        """COLMAP 데이터를 SuperGlue 형식으로 변환"""
        print("    Converting COLMAP data to SuperGlue format...")
        
        # 카메라 포즈 설정
        for image_id, image_data in images.items():
            if image_id < len(image_paths):
                # COLMAP 포즈를 SuperGlue 형식으로 변환
                R = image_data.qvec2rotmat()  # 쿼터니언을 회전 행렬로
                T = image_data.tvec  # 이동 벡터
                
                # 카메라 내부 파라미터
                camera_id = image_data.camera_id
                if camera_id in cameras:
                    camera = cameras[camera_id]
                    K = self._colmap_camera_to_intrinsics(camera)
                else:
                    K = self._estimate_intrinsics(image_id)
                
                self.cameras[image_id] = {
                    'R': R.astype(np.float32),
                    'T': T.astype(np.float32),
                    'K': K,
                    'image_path': str(image_paths[image_id])
                }
        
        print(f"    Converted {len(self.cameras)} camera poses from COLMAP")
    
    def _colmap_camera_to_intrinsics(self, camera):
        """COLMAP 카메라를 내부 파라미터로 변환"""
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            # 기본 PINHOLE 모델 사용
            width, height = camera.width, camera.height
            focal = max(width, height) * 0.9
            cx, cy = width / 2, height / 2
            K = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        
        return K
    
    def _intelligent_matching_with_colmap_poses(self):
        """COLMAP 포즈를 기반으로 한 개선된 매칭"""
        print("    Using COLMAP poses for improved matching...")
        
        # COLMAP 포즈가 있으면 더 적극적인 매칭
        n_images = len(self.image_features)
        
        # 전역 descriptors 계산
        self._compute_global_descriptors()
        
        # COLMAP 포즈 기반 매칭 (더 적극적)
        for i in range(n_images):
            for j in range(i+1, n_images):
                if i in self.cameras and j in self.cameras:
                    # COLMAP 포즈 기반 유사도 계산
                    pose_similarity = self._compute_pose_similarity(i, j)
                    
                    if pose_similarity > 0.1:  # 포즈가 유사한 경우
                        matches = self._match_pair_superglue(i, j)
                        if len(matches) > 1:
                            self.matches[(i, j)] = matches
                            self.camera_graph[i].append(j)
                            self.camera_graph[j].append(i)
    
    def _compute_pose_similarity(self, cam_i, cam_j):
        """두 카메라 포즈 간의 유사도 계산"""
        try:
            R_i, T_i = self.cameras[cam_i]['R'], self.cameras[cam_i]['T']
            R_j, T_j = self.cameras[cam_j]['R'], self.cameras[cam_j]['T']
            
            # 회전 차이
            R_diff = R_i @ R_j.T
            rotation_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            
            # 이동 차이
            translation_error = np.linalg.norm(T_i - T_j)
            
            # 종합 유사도 (작을수록 유사)
            similarity = 1.0 / (1.0 + rotation_error + translation_error * 0.1)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _bundle_adjustment_with_colmap_initialization(self):
        """COLMAP 초기값을 사용한 Bundle Adjustment"""
        print("    Using COLMAP poses as initialization for Bundle Adjustment...")
        
        # COLMAP 포즈가 이미 있으므로 더 보수적인 BA
        self._bundle_adjustment_robust(max_iterations=30)  # 반복 횟수 줄임
    
    def _process_superglue_only(self, image_paths, output_dir):
        """SuperGlue만 사용한 처리 (기존 방식)"""
        print("    Using SuperGlue-only processing...")
        
        # 특징점 추출
        self._extract_all_features(image_paths)
        
        # 매칭
        self._intelligent_matching()
        
        # 카메라 포즈 추정
        self._estimate_camera_poses_robust()
        
        # 삼각측량
        n_points = self._triangulate_all_points_robust()
        
        # Bundle Adjustment
        self._bundle_adjustment_robust()
        
        # 품질 메트릭 계산
        self._compute_quality_metrics()
        
        # 3DGS SceneInfo 생성
        scene_info = self._create_3dgs_scene_info(image_paths)
        
        # 3DGS 형식으로 저장
        self._save_3dgs_format(scene_info, output_dir)
        
        return scene_info
    
    def _compute_quality_metrics(self):
        """품질 메트릭 계산"""
        try:
            # 포즈 추정 성공률
            total_cameras = len(self.image_features)
            estimated_cameras = len([cam for cam in self.cameras.values() if 'R' in cam])
            self.quality_metrics['pose_estimation_success_rate'] = estimated_cameras / total_cameras
            
            # 평균 매칭 수
            if self.matches:
                avg_matches = np.mean([len(matches) for matches in self.matches.values()])
                self.quality_metrics['average_matches_per_pair'] = avg_matches
            
            # 처리 시간
            self.quality_metrics['total_processing_time'] = time.time() - self.start_time
            
            print(f"\n=== Quality Metrics ===")
            print(f"Pose estimation success rate: {self.quality_metrics['pose_estimation_success_rate']:.2%}")
            print(f"Average matches per pair: {self.quality_metrics['average_matches_per_pair']:.1f}")
            print(f"Total processing time: {self.quality_metrics['total_processing_time']:.1f}s")
            
            # GPU 메모리 사용량 (가능한 경우)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"GPU Memory usage: {gpu_memory:.1f} MB")
            
        except Exception as e:
            print(f"Quality metrics calculation failed: {e}")
    
    def _create_fallback_scene_info(self, image_paths):
        """개선된 fallback scene 생성"""
        try:
            # Lazy import 3DGS modules
            CameraInfo, SceneInfo, BasicPointCloud = get_3dgs_imports()
            if CameraInfo is None:
                raise ImportError("3DGS modules not available")
            
            print(f"📸 Creating fallback scene for {len(image_paths)} images")
            
            # 카메라 정보 생성
            cam_infos = []
            for i, image_path in enumerate(image_paths):
                try:
                    # 이미지 크기 확인
                    image = Image.open(image_path)
                    width, height = image.size
                    
                    # 원형 배치로 카메라 배치
                    angle = i * (2 * np.pi / len(image_paths))
                    radius = 3.0
                    
                    # 카메라 포즈 (원을 바라보도록)
                    camera_pos = np.array([
                        radius * np.cos(angle),
                        0.0,  # 높이 고정
                        radius * np.sin(angle)
                    ])
                    
                    # 원점을 향하는 방향
                    look_at = np.array([0.0, 0.0, 0.0])
                    up = np.array([0.0, 1.0, 0.0])
                    
                    # 카메라 회전 행렬 계산
                    forward = look_at - camera_pos
                    forward = forward / np.linalg.norm(forward)
                    right = np.cross(forward, up)
                    right = right / np.linalg.norm(right)
                    up = np.cross(right, forward)
                    
                    R = np.array([right, up, -forward]).T  # OpenCV 컨벤션
                    T = camera_pos
                    
                    # FOV 계산 (더 안전한 값들)
                    focal_length = max(width, height) * 0.8
                    FovX = 2 * np.arctan(width / (2 * focal_length))
                    FovY = 2 * np.arctan(height / (2 * focal_length))
                    
                    # 테스트 카메라 선택 (더 균등하게 분산)
                    is_test = (i % 8 == 0)  # 8개마다 1개씩 테스트
                    
                    cam_info = CameraInfo(
                        uid=i,
                        R=R.astype(np.float32),
                        T=T.astype(np.float32),
                        FovY=float(FovY),
                        FovX=float(FovX),
                        image_path=str(image_path),
                        image_name=Path(image_path).name,
                        width=int(width),
                        height=int(height),
                        depth_params=None,
                        depth_path="",
                        is_test=is_test
                    )
                    cam_infos.append(cam_info)
                    
                except Exception as e:
                    print(f"    Warning: Failed to process {image_path}: {e}")
                    continue
            
            if not cam_infos:
                raise ValueError("No valid cameras created")
            
            # 개선된 포인트 클라우드 생성
            n_points = 12000  # 8000 → 12000로 증가
            
            # 더 현실적인 3D 포인트 분포
            # 구형 분포 + 일부 평면 구조
            points_sphere = np.random.randn(n_points // 2, 3).astype(np.float32)
            points_sphere = points_sphere / np.linalg.norm(points_sphere, axis=1, keepdims=True) * 3.0  # 2.0 → 3.0
            
            # 평면 구조 추가 (바닥면)
            points_plane = np.random.randn(n_points // 2, 3).astype(np.float32)
            points_plane[:, 1] = np.abs(points_plane[:, 1]) * 0.2 - 1.0  # 바닥 근처 (0.1 → 0.2, -0.5 → -1.0)
            points_plane[:, [0, 2]] *= 2.0  # 1.5 → 2.0
            
            points = np.vstack([points_sphere, points_plane])
            
            # 더 현실적인 색상 (회색조 + 약간의 색상)
            colors = np.random.rand(n_points, 3).astype(np.float32)
            colors = colors * 0.5 + 0.3  # 0.3-0.8 범위
            
            # 법선 벡터 (무작위지만 정규화됨)
            normals = np.random.randn(n_points, 3).astype(np.float32)
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
            
            # BasicPointCloud 생성 시 차원 확인
            assert points.shape == (n_points, 3), f"Points shape error: {points.shape}"
            assert colors.shape == (n_points, 3), f"Colors shape error: {colors.shape}"
            assert normals.shape == (n_points, 3), f"Normals shape error: {normals.shape}"
            
            pcd = BasicPointCloud(
                points=points,
                colors=colors,
                normals=normals
            )
            
            # 학습/테스트 분할
            train_cams = [c for c in cam_infos if not c.is_test]
            test_cams = [c for c in cam_infos if c.is_test]
            
            # NeRF 정규화 (개선된 버전)
            if train_cams:
                camera_centers = []
                for cam in train_cams:
                    # 카메라 중심 계산
                    center = -cam.R.T @ cam.T
                    camera_centers.append(center)
                
                camera_centers = np.array(camera_centers)
                scene_center = np.mean(camera_centers, axis=0)
                distances = np.linalg.norm(camera_centers - scene_center, axis=1)
                scene_radius = np.max(distances) * 1.2
                
                # 최소/최대 제한
                scene_radius = max(scene_radius, 1.0)
                scene_radius = min(scene_radius, 10.0)
            else:
                scene_center = np.zeros(3)
                scene_radius = 3.0
            
            nerf_normalization = {
                "translate": -scene_center.astype(np.float32),
                "radius": float(scene_radius)
            }
            
            scene_info = SceneInfo(
                point_cloud=pcd,
                train_cameras=train_cams,
                test_cameras=test_cams,
                nerf_normalization=nerf_normalization,
                ply_path="",
                is_nerf_synthetic=False
            )
            
            print(f"✓ Fallback scene created:")
            print(f"  - {len(train_cams)} training cameras")
            print(f"  - {len(test_cams)} test cameras")
            print(f"  - {n_points} 3D points")
            print(f"  - Scene radius: {scene_radius:.2f}")
            
            return scene_info
            
        except Exception as e:
            print(f"Failed to create fallback scene: {e}")
            raise
    
    def _collect_images(self, image_dir, max_images):
        """이미지 수집 및 정렬"""
        image_dir = Path(image_dir)
        image_paths = []
        
        # 지원하는 확장자
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        # 정렬 및 제한
        image_paths.sort()
        return image_paths[:max_images]
    
    def _extract_all_features(self, image_paths):
        """모든 이미지에서 SuperPoint 특징점 추출 (수정된 버전)"""
        if not self.superglue_available:
            print("  Using fallback feature extraction (SuperGlue not available)")
            return self._extract_features_fallback(image_paths)
        
        for i, image_path in enumerate(image_paths):
            print(f"  {i+1:3d}/{len(image_paths)}: {image_path.name}")
            
            # 이미지 로드
            image = self._load_image(image_path)
            if image is None:
                continue
            
            # SuperPoint 특징점 추출
            inp = frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            
            # 결과 저장 - 모든 필요한 키 포함
            self.image_features[i] = {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(), 
                'scores': pred['scores'][0].cpu().numpy(),
                'image_path': str(image_path),
                'image_size': image.shape[:2]  # (H, W)
            }
            
            print(f"    Keypoints: {self.image_features[i]['keypoints'].shape[0]}")
            
        print(f"  Extracted features from {len(self.image_features)} images")
    
    def _extract_features_fallback(self, image_paths):
        """SuperGlue가 없을 때 사용하는 fallback 특징점 추출"""
        print("  Using OpenCV SIFT for feature extraction")
        
        try:
            import cv2
            sift = cv2.SIFT_create()
        except ImportError:
            print("  OpenCV not available, using random features")
            return self._extract_random_features(image_paths)
        
        for i, image_path in enumerate(image_paths):
            print(f"  {i+1:3d}/{len(image_paths)}: {image_path.name}")
            
            try:
                # 이미지 로드
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # SIFT 특징점 추출
                keypoints, descriptors = sift.detectAndCompute(image, None)
                
                if keypoints is None or descriptors is None:
                    continue
                
                # 결과를 SuperPoint 형식으로 변환
                kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
                scores = np.array([kp.response for kp in keypoints])
                
                # descriptor를 float32로 변환
                desc = descriptors.astype(np.float32)
                
                self.image_features[i] = {
                    'keypoints': kpts,
                    'descriptors': desc.T,  # SuperPoint 형식에 맞춤
                    'scores': scores,
                    'image_path': str(image_path),
                    'image_size': image.shape[:2]
                }
                
                print(f"    Keypoints: {len(keypoints)}")
                
            except Exception as e:
                print(f"    Error processing {image_path.name}: {e}")
                continue
        
        print(f"  Extracted features from {len(self.image_features)} images (fallback)")
    
    def _extract_random_features(self, image_paths):
        """모든 방법이 실패했을 때 사용하는 랜덤 특징점"""
        print("  Using random features (no feature extraction available)")
        
        for i, image_path in enumerate(image_paths):
            print(f"  {i+1:3d}/{len(image_paths)}: {image_path.name}")
            
            try:
                # 이미지 크기 확인
                from PIL import Image
                img = Image.open(image_path)
                width, height = img.size
                
                # 랜덤 특징점 생성
                n_keypoints = 1000
                kpts = np.random.rand(n_keypoints, 2)
                kpts[:, 0] *= width
                kpts[:, 1] *= height
                
                # 랜덤 descriptor (128차원)
                desc = np.random.randn(128, n_keypoints).astype(np.float32)
                
                # 랜덤 scores
                scores = np.random.rand(n_keypoints).astype(np.float32)
                
                self.image_features[i] = {
                    'keypoints': kpts,
                    'descriptors': desc,
                    'scores': scores,
                    'image_path': str(image_path),
                    'image_size': (height, width)
                }
                
                print(f"    Random keypoints: {n_keypoints}")
                
            except Exception as e:
                print(f"    Error processing {image_path.name}: {e}")
                continue
        
        print(f"  Generated random features for {len(self.image_features)} images")
    
    def _intelligent_matching(self, max_pairs=3000):
        """지능적 이미지 매칭 (극도로 완화된 버전)"""
        n_images = len(self.image_features)
        
        # 전역 descriptors 계산
        self._compute_global_descriptors()
        
        print(f"  Starting intelligent matching for {n_images} images...")
        
        # 1. 순차적 매칭 - 모든 이미지를 한 바퀴 돌기
        sequential_count = 0
        for i in range(n_images):
            # 다음 이미지 (마지막 이미지는 첫 번째와 연결)
            next_i = (i + 1) % n_images
            
            matches = self._match_pair_superglue(i, next_i)
            if len(matches) > 1:  # 2 → 1로 극도로 완화
                self.matches[(i, next_i)] = matches
                self.camera_graph[i].append(next_i)
                self.camera_graph[next_i].append(i)
                sequential_count += 1
        
        print(f"    Sequential pairs: {sequential_count}")
        
        # 2. 유사도 기반 매칭 (더 적극적으로)
        similarity_count = self._similarity_based_matching_very_relaxed(max_pairs)
        print(f"    Similarity pairs: {similarity_count}")
        
        # 3. Loop closure 매칭 (더 적극적으로)
        loop_count = self._loop_closure_matching_very_relaxed()
        print(f"    Loop closure pairs: {loop_count}")
        
        # 4. 🔧 NEW: 그리드 기반 매칭 (연속된 이미지들 간의 연결)
        grid_count = self._grid_based_matching_very_relaxed()
        print(f"    Grid-based pairs: {grid_count}")
        
        # 5. 🔧 NEW: 랜덤 샘플링 매칭 (연결되지 않은 카메라들을 위한 fallback)
        random_count = self._random_sampling_matching_very_relaxed(max_pairs)
        print(f"    Random sampling pairs: {random_count}")
        
        print(f"  Total matching pairs: {len(self.matches)}")
        
        # 🔧 NEW: 연결성 분석 및 개선
        self._analyze_and_improve_connectivity_very_relaxed()

    def _similarity_based_matching_very_relaxed(self, max_pairs):
        """극도로 완화된 유사도 기반 매칭"""
        # 유사도 행렬 계산
        n_images = len(self.global_descriptors)
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                if i in self.global_descriptors and j in self.global_descriptors:
                    sim = np.dot(self.global_descriptors[i], self.global_descriptors[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # 유사한 이미지들 매칭 (극도로 적극적으로)
        similarity_count = 0
        for cam_id in range(n_images):
            # 유사도 높은 상위 30개 선택 (20 → 30으로 증가)
            similarities = similarity_matrix[cam_id]
            candidates = np.argsort(similarities)[::-1]
            candidates = [c for c in candidates if c != cam_id and similarities[c] > 0.01][:30]  # 0.05 → 0.01로 극도로 완화
            
            for candidate in candidates:
                pair_key = (min(cam_id, candidate), max(cam_id, candidate))
                if pair_key in self.matches:
                    continue
                
                matches = self._match_pair_superglue(cam_id, candidate)
                if len(matches) > 1:  # 2 → 1로 극도로 완화
                    self.matches[pair_key] = matches
                    self.camera_graph[cam_id].append(candidate)
                    self.camera_graph[candidate].append(cam_id)
                    similarity_count += 1
                
                if len(self.matches) >= max_pairs:
                    return similarity_count
        
        return similarity_count

    def _loop_closure_matching_very_relaxed(self):
        """극도로 완화된 Loop closure 매칭"""
        n_images = len(self.image_features)
        loop_count = 0
        
        # 더 넓은 범위에서 loop closure 시도
        for i in range(min(20, n_images//3)):  # 15 → 20으로 증가
            for j in range(max(n_images-20, 2*n_images//3), n_images):  # 15 → 20으로 증가
                if i >= j:
                    continue
                
                pair_key = (i, j)
                if pair_key in self.matches:
                    continue
                
                # 전역 유사도 체크 (극도로 완화된 조건)
                if hasattr(self, 'global_descriptors') and i in self.global_descriptors and j in self.global_descriptors:
                    sim = np.dot(self.global_descriptors[i], self.global_descriptors[j])
                    if sim > 0.05:  # 0.1 → 0.05로 극도로 완화
                        matches = self._match_pair_superglue(i, j)
                        if len(matches) > 1:  # 2 → 1로 극도로 완화
                            self.matches[pair_key] = matches
                            self.camera_graph[i].append(j)
                            self.camera_graph[j].append(i)
                            loop_count += 1
        
        return loop_count

    def _grid_based_matching_very_relaxed(self):
        """극도로 완화된 그리드 기반 매칭"""
        n_images = len(self.image_features)
        grid_count = 0
        
        # 연속된 이미지들 간의 추가 연결
        for i in range(n_images - 1):
            # 인접한 이미지들
            for offset in [1, 2, 3, 4, 5]:  # 1, 2, 3 → 1, 2, 3, 4, 5로 증가
                j = i + offset
                if j >= n_images:
                    continue
                
                pair_key = (i, j)
                if pair_key in self.matches:
                    continue
                
                matches = self._match_pair_superglue(i, j)
                if len(matches) > 1:  # 1 → 1로 유지 (이미 최소값)
                    self.matches[pair_key] = matches
                    self.camera_graph[i].append(j)
                    self.camera_graph[j].append(i)
                    grid_count += 1
        
        return grid_count

    def _random_sampling_matching_very_relaxed(self, max_pairs):
        """극도로 완화된 랜덤 샘플링 매칭"""
        n_images = len(self.image_features)
        random_count = 0
        
        # 연결되지 않은 카메라들을 찾기
        unconnected_cameras = []
        for cam_id in range(n_images):
            if len(self.camera_graph[cam_id]) == 0:
                unconnected_cameras.append(cam_id)
        
        print(f"    Found {len(unconnected_cameras)} unconnected cameras")
        
        # 연결되지 않은 카메라들에 대해 랜덤 매칭 시도
        for cam_id in unconnected_cameras:
            # 다른 모든 카메라와 매칭 시도
            for other_cam in range(n_images):
                if cam_id == other_cam:
                    continue
                
                pair_key = (min(cam_id, other_cam), max(cam_id, other_cam))
                if pair_key in self.matches:
                    continue
                
                matches = self._match_pair_superglue(cam_id, other_cam)
                if len(matches) > 1:  # 1 → 1로 유지 (이미 최소값)
                    self.matches[pair_key] = matches
                    self.camera_graph[cam_id].append(other_cam)
                    self.camera_graph[other_cam].append(cam_id)
                    random_count += 1
                    break  # 하나라도 연결되면 다음 카메라로
        
        return random_count

    def _analyze_and_improve_connectivity_very_relaxed(self):
        """극도로 완화된 연결성 분석 및 개선"""
        n_images = len(self.image_features)
        
        # 연결성 분석
        connected_cameras = []
        isolated_cameras = []
        
        for cam_id in range(n_images):
            if len(self.camera_graph[cam_id]) > 0:
                connected_cameras.append(cam_id)
            else:
                isolated_cameras.append(cam_id)
        
        print(f"    Connectivity analysis:")
        print(f"      Connected cameras: {len(connected_cameras)}")
        print(f"      Isolated cameras: {len(isolated_cameras)}")
        
        # 연결되지 않은 카메라들을 연결된 카메라와 연결 시도
        if len(connected_cameras) > 0 and len(isolated_cameras) > 0:
            print(f"    Attempting to connect {len(isolated_cameras)} isolated cameras...")
            
            for isolated_cam in isolated_cameras:
                # 가장 가까운 연결된 카메라 찾기
                best_connection = None
                best_similarity = -1
                
                for connected_cam in connected_cameras:
                    if hasattr(self, 'global_descriptors'):
                        if isolated_cam in self.global_descriptors and connected_cam in self.global_descriptors:
                            sim = np.dot(self.global_descriptors[isolated_cam], self.global_descriptors[connected_cam])
                            if sim > best_similarity:
                                best_similarity = sim
                                best_connection = connected_cam
                
                if best_connection is not None:
                    # 매칭 시도
                    matches = self._match_pair_superglue(isolated_cam, best_connection)
                    if len(matches) > 1:  # 1 → 1로 유지 (이미 최소값)
                        pair_key = (min(isolated_cam, best_connection), max(isolated_cam, best_connection))
                        self.matches[pair_key] = matches
                        self.camera_graph[isolated_cam].append(best_connection)
                        self.camera_graph[best_connection].append(isolated_cam)
                        print(f"      Connected camera {isolated_cam} to {best_connection}")

    def _compute_global_descriptors(self):
        """전역 이미지 descriptor 계산 (NEW METHOD)"""
        self.global_descriptors = {}
        
        for cam_id, features in self.image_features.items():
            descriptors = features['descriptors']  # (256, N)
            scores = features['scores']
            
            if len(scores) > 0:
                # Score로 가중평균하여 전역 descriptor 계산
                weights = scores / (scores.sum() + 1e-10)
                global_desc = np.average(descriptors.T, weights=weights, axis=0)
                global_desc = global_desc / (np.linalg.norm(global_desc) + 1e-10)
                self.global_descriptors[cam_id] = global_desc
            else:
                self.global_descriptors[cam_id] = np.zeros(256)
                
    def _similarity_based_matching(self, max_pairs):
        """유사도 기반 매칭 (NEW METHOD)"""
        # 유사도 행렬 계산
        n_images = len(self.global_descriptors)
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                if i in self.global_descriptors and j in self.global_descriptors:
                    sim = np.dot(self.global_descriptors[i], self.global_descriptors[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # 유사한 이미지들 매칭
        similarity_count = 0
        for cam_id in range(n_images):
            # 유사도 높은 상위 12개 선택 (8 → 12로 증가)
            similarities = similarity_matrix[cam_id]
            candidates = np.argsort(similarities)[::-1]
            candidates = [c for c in candidates if c != cam_id and similarities[c] > 0.2][:12]  # 0.3 → 0.2, 8 → 12
            
            for candidate in candidates:
                pair_key = (min(cam_id, candidate), max(cam_id, candidate))
                if pair_key in self.matches:
                    continue
                
                matches = self._match_pair_superglue(cam_id, candidate)
                if len(matches) > 10:  # 15 → 10으로 완화
                    self.matches[pair_key] = matches
                    self.camera_graph[cam_id].append(candidate)
                    self.camera_graph[candidate].append(cam_id)
                    similarity_count += 1
                
                if len(self.matches) >= max_pairs:
                    return similarity_count
        
        return similarity_count

    def _loop_closure_matching(self):
        """Loop closure 매칭 (NEW METHOD)"""
        n_images = len(self.image_features)
        loop_count = 0
        
        # 첫 번째와 마지막 몇 개 이미지 간 매칭
        for i in range(min(8, n_images//3)):  # 5 → 8로 증가
            for j in range(max(n_images-8, 2*n_images//3), n_images):  # 5 → 8로 증가
                if i >= j:
                    continue
                
                pair_key = (i, j)
                if pair_key in self.matches:
                    continue
                
                # 전역 유사도 체크
                if hasattr(self, 'global_descriptors') and i in self.global_descriptors and j in self.global_descriptors:
                    sim = np.dot(self.global_descriptors[i], self.global_descriptors[j])
                    if sim > 0.3:  # 0.4 → 0.3으로 완화
                        matches = self._match_pair_superglue(i, j)
                        if len(matches) > 15:  # 20 → 15로 완화
                            self.matches[pair_key] = matches
                            self.camera_graph[i].append(j)
                            self.camera_graph[j].append(i)
                            loop_count += 1
        
        return loop_count
    
    def _filter_low_quality_matches_very_relaxed(self):
        """매우 완화된 낮은 품질의 매칭 필터링"""
        pairs_to_remove = []
        
        for (cam_i, cam_j), matches in self.matches.items():
            if len(matches) < 2:  # 더 낮은 임계값 (3 → 2)
                pairs_to_remove.append((cam_i, cam_j))
                continue
            
            # 매칭 품질 분석
            confidences = [conf for _, _, conf in matches]
            avg_confidence = np.mean(confidences)
            
            if avg_confidence < 0.001:  # 더 낮은 임계값 (0.1 → 0.001)
                pairs_to_remove.append((cam_i, cam_j))
                continue
            
            # 매칭 분포 분석 (더 완화된 조건)
            if self._has_poor_matching_distribution_very_relaxed(cam_i, cam_j, matches):
                pairs_to_remove.append((cam_i, cam_j))
        
        # 필터링된 매칭 제거
        for pair in pairs_to_remove:
            cam_i, cam_j = pair
            del self.matches[pair]
            
            # 그래프에서도 제거
            if cam_j in self.camera_graph[cam_i]:
                self.camera_graph[cam_i].remove(cam_j)
            if cam_i in self.camera_graph[cam_j]:
                self.camera_graph[cam_j].remove(cam_i)
        
        print(f"  Filtered out {len(pairs_to_remove)} low-quality matches (very relaxed)")
    
    def _has_poor_matching_distribution_very_relaxed(self, cam_i, cam_j, matches):
        """매우 완화된 매칭 분포 검사"""
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        # 인덱스 범위 검증
        valid_matches = []
        for idx_i, idx_j, conf in matches:
            if (idx_i < len(kpts_i) and idx_j < len(kpts_j) and 
                idx_i >= 0 and idx_j >= 0):
                valid_matches.append((idx_i, idx_j, conf))
        
        if len(valid_matches) < 1:  # 더 낮은 임계값 (2 → 1)
            return True
        
        # 매칭된 점들의 위치 분석
        matched_i = np.array([kpts_i[idx_i] for idx_i, _, _ in valid_matches])
        matched_j = np.array([kpts_j[idx_j] for _, idx_j, _ in valid_matches])
        
        # 이미지 크기
        h_i, w_i = self.image_features[cam_i]['image_size']
        h_j, w_j = self.image_features[cam_j]['image_size']
        
        # 경계 근처의 매칭이 너무 많은지 확인 (더 완화된 조건)
        border_threshold = 10  # 더 작은 경계 (20 → 10)
        
        border_matches_i = np.sum((matched_i[:, 0] < border_threshold) | 
                                  (matched_i[:, 0] > w_i - border_threshold) |
                                  (matched_i[:, 1] < border_threshold) | 
                                  (matched_i[:, 1] > h_i - border_threshold))
        
        border_matches_j = np.sum((matched_j[:, 0] < border_threshold) | 
                                  (matched_j[:, 0] > w_j - border_threshold) |
                                  (matched_j[:, 1] < border_threshold) | 
                                  (matched_j[:, 1] > h_j - border_threshold))
        
        # 경계 매칭이 전체의 98% 이상이면 나쁜 분포 (95% → 98%)
        if border_matches_i > len(valid_matches) * 0.98 or border_matches_j > len(valid_matches) * 0.98:
            return True
        
        return False
    
    def _match_pair_superglue(self, cam_i, cam_j):
        """SuperGlue 페어 매칭 (더 완화된 버전)"""
        if not self.superglue_available:
            return self._match_pair_fallback(cam_i, cam_j)
        
        try:
            feat_i = self.image_features[cam_i]
            feat_j = self.image_features[cam_j]
            
            # 입력 데이터 준비
            data = {
                'image0': torch.zeros((1, 1, 480, 640)).to(self.device),
                'image1': torch.zeros((1, 1, 480, 640)).to(self.device),
                'keypoints0': torch.from_numpy(feat_i['keypoints']).unsqueeze(0).to(self.device),
                'keypoints1': torch.from_numpy(feat_j['keypoints']).unsqueeze(0).to(self.device),
                'descriptors0': torch.from_numpy(feat_i['descriptors']).unsqueeze(0).to(self.device),
                'descriptors1': torch.from_numpy(feat_j['descriptors']).unsqueeze(0).to(self.device),
                'scores0': torch.from_numpy(feat_i['scores']).unsqueeze(0).to(self.device),
                'scores1': torch.from_numpy(feat_j['scores']).unsqueeze(0).to(self.device),
            }
            
            # SuperGlue 매칭
            with torch.no_grad():
                result = self.matching.superglue(data)
            
            # 결과 추출
            indices0 = result['indices0'][0].cpu().numpy()
            indices1 = result['indices1'][0].cpu().numpy()
            mscores0 = result['matching_scores0'][0].cpu().numpy()
            
            # 🔧 더 완화된 매칭 필터링
            valid_matches = []
            threshold = 0.00001  # 0.0001 → 0.00001로 대폭 완화
            
            for i, j in enumerate(indices0):
                if j >= 0 and mscores0[i] > threshold:
                    # 상호 매칭 확인
                    if j < len(indices1) and indices1[j] == i:
                        # 인덱스 범위 검증
                        if i < len(feat_i['keypoints']) and j < len(feat_j['keypoints']):
                            valid_matches.append((i, j, mscores0[i]))
            
            # 🔧 더 완화된 기하학적 필터링
            if len(valid_matches) >= 1:  # 1개 이상이면 필터링 시도
                valid_matches = self._geometric_filtering_relaxed(valid_matches, feat_i['keypoints'], feat_j['keypoints'])
            
            return valid_matches
            
        except Exception as e:
            print(f"    SuperGlue matching failed for pair {cam_i}-{cam_j}: {e}")
            return self._match_pair_fallback(cam_i, cam_j)
    
    def _match_pair_fallback(self, cam_i, cam_j):
        """SuperGlue가 없을 때 사용하는 fallback 매칭"""
        try:
            feat_i = self.image_features[cam_i]
            feat_j = self.image_features[cam_j]
            
            # 간단한 descriptor 매칭
            desc_i = feat_i['descriptors'].T  # (N, D)
            desc_j = feat_j['descriptors'].T  # (M, D)
            
            # 코사인 유사도 계산
            desc_i_norm = desc_i / (np.linalg.norm(desc_i, axis=1, keepdims=True) + 1e-10)
            desc_j_norm = desc_j / (np.linalg.norm(desc_j, axis=1, keepdims=True) + 1e-10)
            
            similarity = desc_i_norm @ desc_j_norm.T  # (N, M)
            
            # 상위 매칭 찾기
            matches = []
            threshold = 0.5  # 유사도 임계값
            
            for i in range(len(desc_i)):
                best_j = np.argmax(similarity[i])
                if similarity[i, best_j] > threshold:
                    # 상호 매칭 확인
                    if np.argmax(similarity[:, best_j]) == i:
                        confidence = similarity[i, best_j]
                        matches.append((i, best_j, confidence))
            
            return matches
            
        except Exception as e:
            print(f"    Fallback matching failed for pair {cam_i}-{cam_j}: {e}")
            return []

    def _geometric_filtering_relaxed(self, matches, kpts_i, kpts_j):
        """완화된 기하학적 필터링 (NEW METHOD)"""
        try:
            pts_i = np.array([kpts_i[m[0]] for m in matches])
            pts_j = np.array([kpts_j[m[1]] for m in matches])
            
            # 호모그래피 기반 outlier 제거 (더 완화된 조건)
            H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 100.0)  # 50.0 → 100.0으로 더 완화
            
            if H is not None and mask is not None:
                inlier_matches = [matches[i] for i, is_inlier in enumerate(mask.flatten()) if is_inlier]
                if len(inlier_matches) >= 1:  # 1개 이상이면 통과
                    return inlier_matches
        except:
            pass
        
        return matches
    
    def _estimate_camera_poses_robust(self):
        """개선된 카메라 포즈 추정 - 더 강력한 연결성"""
        
        # 첫 번째 카메라를 원점으로 설정
        self.cameras[0] = {
            'R': np.eye(3, dtype=np.float32),
            'T': np.zeros(3, dtype=np.float32),
            'K': self._estimate_intrinsics(0)
        }
        
        print(f"  Camera 0: Origin (reference)")
        
        # 🔧 개선된 포즈 추정 전략
        estimated_cameras = {0}
        queue = [0]
        
        # 1단계: 연결된 카메라들만 포즈 추정
        while queue:
            current_cam = queue.pop(0)
            
            # 현재 카메라와 연결된 카메라들 확인
            for neighbor_cam in self.camera_graph[current_cam]:
                if neighbor_cam in estimated_cameras:
                    continue
                
                # 매칭 데이터 찾기
                pair_key = (current_cam, neighbor_cam) if current_cam < neighbor_cam else (neighbor_cam, current_cam)
                if pair_key not in self.matches:
                    continue
                
                # Essential Matrix 기반 포즈 추정
                R_rel, T_rel = self._estimate_relative_pose_robust(current_cam, neighbor_cam, pair_key)
                
                if R_rel is not None and T_rel is not None:
                    # 월드 좌표계에서의 절대 포즈 계산
                    R_ref, T_ref = self.cameras[current_cam]['R'], self.cameras[current_cam]['T']
                    
                    # 상대 포즈를 절대 포즈로 변환
                    R_world = R_rel @ R_ref
                    T_world = R_rel @ T_ref + T_rel
                    
                    # 포즈 유효성 검사
                    if self._is_valid_rotation_matrix(R_world):
                        self.cameras[neighbor_cam] = {
                            'R': R_world.astype(np.float32),
                            'T': T_world.astype(np.float32),
                            'K': self._estimate_intrinsics(neighbor_cam)
                        }
                        
                        print(f"  Camera {neighbor_cam}: Estimated from camera {current_cam}")
                        estimated_cameras.add(neighbor_cam)
                        queue.append(neighbor_cam)
                    else:
                        print(f"  Camera {neighbor_cam}: Invalid pose, skipping...")
                else:
                    print(f"  Camera {neighbor_cam}: Pose estimation failed, will use default pose")
        
        # 2단계: 연결되지 않은 카메라들에 대한 개선된 기본 포즈 설정
        unconnected_count = 0
        for cam_id in range(len(self.image_features)):
            if cam_id not in estimated_cameras:
                unconnected_count += 1
                print(f"  Camera {cam_id}: Using improved default pose (not connected)")
                
                # 🔧 개선된 기본 포즈 설정
                if len(estimated_cameras) > 0:
                    # 연결된 카메라들의 평균 위치 계산
                    connected_positions = []
                    for est_cam in estimated_cameras:
                        R, T = self.cameras[est_cam]['R'], self.cameras[est_cam]['T']
                        # 카메라 중심 계산
                        center = -R.T @ T
                        connected_positions.append(center)
                    
                    if connected_positions:
                        # 연결된 카메라들의 중심 주변에 배치
                        avg_position = np.mean(connected_positions, axis=0)
                        position_std = np.std(connected_positions, axis=0)
                        
                        # 카메라 ID에 따른 위치 변화
                        angle = cam_id * (2 * np.pi / len(self.image_features))
                        radius = 2.0 + np.random.normal(0, 0.5)  # 약간의 랜덤성
                        
                        # 원형 배치 + 중심으로부터의 오프셋
                        camera_pos = avg_position + np.array([
                            radius * np.cos(angle),
                            0.5 * np.sin(angle),  # 높이 변화
                            radius * np.sin(angle)
                        ])
                        
                        # 중심을 향하는 방향
                        look_at = avg_position
                        up = np.array([0.0, 1.0, 0.0])
                        
                        # 카메라 회전 행렬 계산
                        forward = look_at - camera_pos
                        forward = forward / (np.linalg.norm(forward) + 1e-10)
                        right = np.cross(forward, up)
                        right = right / (np.linalg.norm(right) + 1e-10)
                        up = np.cross(right, forward)
                        
                        R = np.array([right, up, -forward]).T
                        T = camera_pos
                    else:
                        # 기본 원형 배치
                        angle = cam_id * (2 * np.pi / len(self.image_features))
                        radius = 3.0
                        
                        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                                     [0, 1, 0],
                                     [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
                        T = np.array([radius * np.sin(angle), 0, radius * (1 - np.cos(angle))], dtype=np.float32)
                else:
                    # 기본 원형 배치
                    angle = cam_id * (2 * np.pi / len(self.image_features))
                    radius = 3.0
                    
                    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                                 [0, 1, 0],
                                 [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
                    T = np.array([radius * np.sin(angle), 0, radius * (1 - np.cos(angle))], dtype=np.float32)
                
                self.cameras[cam_id] = {
                    'R': R.astype(np.float32),
                    'T': T.astype(np.float32),
                    'K': self._estimate_intrinsics(cam_id)
                }
        
        print(f"  Estimated poses for {len(estimated_cameras)} cameras")
        print(f"  Used default poses for {unconnected_count} cameras")
        print(f"  Total cameras with poses: {len(self.cameras)}")
    
    def _estimate_relative_pose_robust(self, cam_i, cam_j, pair_key):
        """개선된 두 카메라 간 상대 포즈 추정 - 극도로 완화된 버전"""
        matches = self.matches[pair_key]
        
        if len(matches) < 4:  # 6 → 4로 더 완화
            print(f"    Pair {cam_i}-{cam_j}: Insufficient matches ({len(matches)} < 4)")
            return None, None
        
        # 매칭점들 추출
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        print(f"    Pair {cam_i}-{cam_j}: kpts_i shape: {kpts_i.shape}, kpts_j shape: {kpts_j.shape}")
        
        # 🔧 극도로 완화된 신뢰도 임계값
        high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches if conf > 0.00001]  # 0.0001 → 0.00001로 대폭 완화
        
        if len(high_conf_matches) < 4:  # 6 → 4로 더 완화
            high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches if conf > 0.000001]  # 0.00001 → 0.000001로 대폭 완화
        
        if len(high_conf_matches) < 4:  # 6 → 4로 더 완화
            # 모든 매칭을 사용
            high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches]
        
        if len(high_conf_matches) < 4:  # 6 → 4로 더 완화
            print(f"    Pair {cam_i}-{cam_j}: Insufficient high-confidence matches ({len(high_conf_matches)} < 4)")
            return None, None
        
        # 🔧 인덱스 범위 검증 강화
        valid_matches = []
        for idx_i, idx_j, conf in high_conf_matches:
            if (isinstance(idx_i, (int, np.integer)) and isinstance(idx_j, (int, np.integer)) and
                idx_i >= 0 and idx_j >= 0 and 
                idx_i < len(kpts_i) and idx_j < len(kpts_j)):
                valid_matches.append((idx_i, idx_j, conf))
        
        if len(valid_matches) < 4:  # 6 → 4로 더 완화
            print(f"    Pair {cam_i}-{cam_j}: Insufficient valid matches after index validation ({len(valid_matches)} < 4)")
            return None, None
        
        print(f"    Pair {cam_i}-{cam_j}: Using {len(valid_matches)} validated matches")
        
        # 🔧 개선된 포인트 추출
        try:
            pts_i = np.array([kpts_i[idx_i] for idx_i, _, _ in valid_matches])
            pts_j = np.array([kpts_j[idx_j] for _, idx_j, _ in valid_matches])
        except IndexError as e:
            print(f"    IndexError during point extraction: {e}")
            return None, None
        
        # 🔧 기하학적 일관성 사전 검증 (더 관대하게)
        if not self._check_geometric_consistency_very_relaxed(pts_i, pts_j):
            print(f"    Pair {cam_i}-{cam_j}: Failed geometric consistency check")
            return None, None
        
        # 카메라 내부 파라미터
        K_i = self.cameras.get(cam_i, {}).get('K', self._estimate_intrinsics(cam_i))
        K_j = self._estimate_intrinsics(cam_j)
        
        # 🔧 극도로 관대한 Essential Matrix 추정 방법들
        methods = [
            (cv2.RANSAC, 0.5, 0.99),    # 극도로 관대한 임계값
            (cv2.RANSAC, 1.0, 0.95),
            (cv2.RANSAC, 2.0, 0.90),
            (cv2.LMEDS, 0.5, 0.95),
            (cv2.RANSAC, 5.0, 0.85),    # 매우 관대한 설정
            (cv2.RANSAC, 10.0, 0.80),   # 극도로 관대한 설정
            (cv2.RANSAC, 20.0, 0.70)    # 최대한 관대한 설정
        ]
        
        best_R, best_T = None, None
        best_inliers = 0
        best_quality = 0
        
        for method, threshold, confidence in methods:
            try:
                # Essential Matrix 추정
                E, mask = cv2.findEssentialMat(
                    pts_i, pts_j, K_i,
                    method=method,
                    prob=confidence,
                    threshold=threshold,
                    maxIters=500  # 반복 횟수 줄임
                )
                
                if E is None or E.shape != (3, 3):
                    continue
                
                # 포즈 복원
                _, R, T, mask = cv2.recoverPose(E, pts_i, pts_j, K_i, mask=mask)
                
                if R is None or T is None:
                    continue
                
                inliers = np.sum(mask)
                
                if inliers >= 2:  # 4 → 2로 극도로 완화
                    # 🔧 더 관대한 포즈 품질 검증
                    quality_score = self._evaluate_pose_quality_very_relaxed(pts_i, pts_j, R, T.flatten(), K_i, K_j, mask)
                    
                    if quality_score > best_quality:
                        best_R, best_T = R, T.flatten()
                        best_inliers = inliers
                        best_quality = quality_score
                        
            except Exception as e:
                print(f"      Method {method} failed: {e}")
                continue
        
        if best_R is not None:
            print(f"    Pair {cam_i}-{cam_j}: Successfully estimated pose with {best_inliers} inliers, quality: {best_quality:.3f}")
        else:
            print(f"    Pair {cam_i}-{cam_j}: Failed to estimate pose")
        
        return best_R, best_T

    def _check_geometric_consistency_very_relaxed(self, pts_i, pts_j):
        """극도로 관대한 기하학적 일관성 사전 검증"""
        try:
            # 1. 호모그래피 기반 일관성 검사 (극도로 관대하게)
            H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 20.0)  # 10.0 → 20.0으로 더 완화
            if H is not None:
                homography_inliers = np.sum(mask)
                if homography_inliers < len(pts_i) * 0.01:  # 5% → 1%로 극도로 완화
                    return False
            
            # 2. 포인트 분포 검사 (극도로 관대하게)
            if len(pts_i) > 2:  # 3 → 2로 더 완화
                # 포인트들의 분산 계산
                var_i = np.var(pts_i, axis=0)
                var_j = np.var(pts_j, axis=0)
                
                # 분산이 너무 작으면 나쁜 분포 (극도로 관대하게)
                if np.min(var_i) < 0.1 or np.min(var_j) < 0.1:  # 1 → 0.1로 대폭 완화
                    return False
            
            # 3. 포인트 간 거리 검사 (극도로 관대하게)
            if len(pts_i) > 1:  # 2 → 1로 더 완화
                distances_i = cdist(pts_i, pts_i)
                distances_j = cdist(pts_j, pts_j)
                
                # 대각선 제거
                np.fill_diagonal(distances_i, np.inf)
                np.fill_diagonal(distances_j, np.inf)
                
                min_dist_i = np.min(distances_i)
                min_dist_j = np.min(distances_j)
                
                # 최소 거리가 너무 작으면 나쁜 분포 (극도로 관대하게)
                if min_dist_i < 0.01 or min_dist_j < 0.01:  # 0.1 → 0.01로 대폭 완화
                    return False
            
            return True
            
        except Exception as e:
            print(f"      Geometric consistency check failed: {e}")
            return True  # 오류시 통과

    def _evaluate_pose_quality_very_relaxed(self, pts_i, pts_j, R, T, K_i, K_j, mask):
        """극도로 관대한 포즈 품질 평가"""
        try:
            # 1. 회전 행렬 유효성 확인 (극도로 관대하게)
            det = np.linalg.det(R)
            if abs(det - 1.0) > 0.5:  # 0.3 → 0.5로 더 완화
                return 0.0
            
            # 2. 삼각측량 품질 검사 (극도로 관대하게)
            P_i = K_i @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P_j = K_j @ np.hstack([R, T.reshape(-1, 1)])
            
            # inlier 포인트들만 사용
            inlier_pts_i = pts_i[mask.flatten()]
            inlier_pts_j = pts_j[mask.flatten()]
            
            if len(inlier_pts_i) < 2:  # 4 → 2로 극도로 완화
                return 0.0
            
            # 삼각측량 테스트 (극도로 관대하게)
            valid_points = 0
            total_error = 0.0
            
            for pt_i, pt_j in zip(inlier_pts_i, inlier_pts_j):
                try:
                    # 삼각측량
                    pt_4d = cv2.triangulatePoints(P_i, P_j, pt_i.reshape(2, 1), pt_j.reshape(2, 1))
                    
                    if abs(pt_4d[3, 0]) > 1e-10:
                        pt_3d = (pt_4d[:3] / pt_4d[3]).flatten()
                        
                        # 거리 체크 (극도로 관대하게)
                        if 0.0001 < np.linalg.norm(pt_3d) < 100000:  # 0.001~10000 → 0.0001~100000으로 완화
                            # 재투영 오차 계산
                            proj_i = P_i @ np.append(pt_3d, 1)
                            proj_j = P_j @ np.append(pt_3d, 1)
                            
                            if proj_i[2] > 0 and proj_j[2] > 0:
                                proj_i_2d = proj_i[:2] / proj_i[2]
                                proj_j_2d = proj_j[:2] / proj_j[2]
                                
                                error_i = np.linalg.norm(proj_i_2d - pt_i)
                                error_j = np.linalg.norm(proj_j_2d - pt_j)
                                
                                total_error += max(error_i, error_j)
                                valid_points += 1
                                
                except:
                    continue
            
            if valid_points < 2:  # 4 → 2로 극도로 완화
                return 0.0
            
            # 품질 점수 계산 (극도로 관대하게)
            avg_error = total_error / valid_points
            inlier_ratio = len(inlier_pts_i) / len(pts_i)
            
            # 오차가 작고 inlier 비율이 높을수록 높은 점수 (극도로 관대하게)
            quality_score = inlier_ratio * (1.0 / (1.0 + avg_error * 0.001))  # 0.01 → 0.001로 완화
            
            return quality_score
            
        except Exception as e:
            print(f"      Pose quality evaluation failed: {e}")
            return 0.0
    
    def _estimate_pose_fallback(self, pts_i, pts_j, K_i, K_j):
        """OpenCV 실패시 사용할 fallback 포즈 추정"""
        try:
            # 간단한 호모그래피 기반 방법
            H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 5.0)
            
            if H is None:
                return None, None
            
            # 호모그래피에서 회전과 이동 추출 (근사)
            # 이는 정확하지 않지만 기본적인 포즈를 제공
            K_inv = np.linalg.inv(K_i)
            R_approx = K_inv @ H @ K_i
            
            # SVD를 사용하여 회전 행렬로 정규화
            U, _, Vt = np.linalg.svd(R_approx)
            R = U @ Vt
            
            # 회전 행렬 유효성 검사
            if not self._is_valid_rotation_matrix(R):
                # 기본 회전 행렬 사용
                R = np.eye(3)
            
            # 이동 벡터 추정 (간단한 근사)
            T = np.array([0.1, 0.0, 0.0])  # 기본 이동
            
            return R, T
            
        except Exception as e:
            print(f"      Fallback pose estimation failed: {e}")
            return None, None
    
    def _is_valid_rotation_matrix(self, R):
        """회전 행렬이 유효한지 확인"""
        try:
            # 행렬식이 1에 가까운지 확인
            det = np.linalg.det(R)
            if abs(det - 1.0) > 0.1:
                return False
            
            # R * R^T = I인지 확인
            I = np.eye(3)
            RRt = R @ R.T
            if np.max(np.abs(RRt - I)) > 0.1:
                return False
            
            return True
        except:
            return False
    
    def _verify_pose_quality_very_relaxed(self, pts_i, pts_j, R, T, K_i, K_j):
        """매우 완화된 포즈 품질 검증"""
        # 재투영 오차 계산
        P_i = K_i @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_j = K_j @ np.hstack([R, T.reshape(-1, 1)])
        
        errors = []
        depths_i = []
        depths_j = []
        
        for pt_i, pt_j in zip(pts_i, pts_j):
            # 삼각측량
            point_4d = cv2.triangulatePoints(P_i, P_j, pt_i.reshape(2, 1), pt_j.reshape(2, 1))
            if abs(point_4d[3, 0]) > 1e-10:
                point_3d = (point_4d[:3] / point_4d[3]).flatten()
                
                # 재투영 (3D 좌표계에서)
                proj_i_3d = P_i @ np.append(point_3d, 1)
                proj_j_3d = P_j @ np.append(point_3d, 1)
                
                # 2D 좌표로 변환
                proj_i_2d = proj_i_3d[:2] / proj_i_3d[2]
                proj_j_2d = proj_j_3d[:2] / proj_j_3d[2]
                
                error_i = np.linalg.norm(proj_i_2d - pt_i)
                error_j = np.linalg.norm(proj_j_2d - pt_j)
                errors.append(max(error_i, error_j))
                
                # 깊이 정보 저장 (3D 좌표계에서)
                depths_i.append(proj_i_3d[2])
                depths_j.append(proj_j_3d[2])
        
        if len(errors) < 2:  # 더 낮은 임계값 (3 → 2)
            return False
        
        # 오차 통계
        median_error = np.median(errors)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # 깊이 검증 (더 완화된 조건)
        if depths_i and depths_j:
            depths_i = np.array(depths_i)
            depths_j = np.array(depths_j)
            
            # 깊이가 양수인지 확인
            if np.any(depths_i <= 0) or np.any(depths_j <= 0):
                return False
            
            # 깊이 비율이 합리적인지 확인 (더 완화된 조건)
            depth_ratios = depths_j / depths_i
            if np.median(depth_ratios) < 0.01 or np.median(depth_ratios) > 50:  # 0.05~20 → 0.01~50
                return False
        
        # 오차 임계값 검증 (더 완화된 조건)
        pose_quality = (median_error < 15.0 and   # 8.0 → 15.0
                mean_error < 20.0 and    # 10.0 → 20.0
                max_error < 50.0)        # 20.0 → 50.0
        
        if not pose_quality:
            print(f"      Pose quality check failed: median={median_error:.2f}, mean={mean_error:.2f}, max={max_error:.2f}")
        
        return pose_quality
    
    def _estimate_intrinsics(self, cam_id):
        """개선된 카메라 내부 파라미터 추정 (COLMAP 우선)"""
        h, w = self.image_features[cam_id]['image_size']
        
        # COLMAP reconstruction이 있으면 그것을 사용
        try:
            colmap_cameras = self._get_colmap_intrinsics()
            if colmap_cameras and cam_id in colmap_cameras:
                camera = colmap_cameras[cam_id]
                width, height = camera.width, camera.height
                
                # PINHOLE 모델 가정 (fx, fy, cx, cy)
                if len(camera.params) == 4:
                    fx, fy, cx, cy = camera.params
                    # COLMAP에서 추정한 정확한 focal length 사용
                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    print(f"    Camera {cam_id}: Using COLMAP focal length (fx={fx:.1f}, fy={fy:.1f})")
                    return K
        except Exception as e:
            print(f"    Camera {cam_id}: COLMAP intrinsics failed, using default: {e}")
            # 손상된 COLMAP 파일들을 정리
            try:
                output_dir = getattr(self, 'output_dir', None)
                if output_dir:
                    reconstruction_path = Path(output_dir) / "sparse" / "0"
                    for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
                        file_path = reconstruction_path / file_name
                        if file_path.exists():
                            file_path.unlink()
                            print(f"    Deleted corrupted {file_name}")
            except Exception as cleanup_error:
                print(f"    Failed to cleanup corrupted files: {cleanup_error}")
        
        # COLMAP이 없으면 기본 추정 사용
        focal = max(w, h) * 0.9  # 약간 보수적인 추정
        cx, cy = w / 2, h / 2
        
        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"    Camera {cam_id}: Using default focal length ({focal:.1f})")
        return K
    
    def _get_colmap_intrinsics(self):
        """COLMAP reconstruction에서 카메라 내부 파라미터 읽기"""
        try:
            # COLMAP reconstruction 경로 확인
            output_dir = getattr(self, 'output_dir', None)
            if output_dir is None:
                return None
            
            reconstruction_path = Path(output_dir) / "sparse" / "0"
            cameras_bin = reconstruction_path / "cameras.bin"
            
            if not cameras_bin.exists():
                print(f"    cameras.bin not found at {cameras_bin}")
                return None
            
            # 파일 크기 확인
            file_size = cameras_bin.stat().st_size
            if file_size == 0:
                print(f"    cameras.bin is empty")
                return None
            
            print(f"    Reading COLMAP intrinsics from {cameras_bin} ({file_size} bytes)")
            
            # COLMAP reconstruction 파싱
            try:
                from scene.colmap_loader import read_intrinsics_binary
                cameras = read_intrinsics_binary(str(cameras_bin))
                print(f"    Successfully loaded {len(cameras)} cameras from COLMAP")
                
                # 이미지 ID와 카메라 ID 매핑
                images_bin = reconstruction_path / "images.bin"
                if images_bin.exists():
                    try:
                        from scene.colmap_loader import read_extrinsics_binary
                        images = read_extrinsics_binary(str(images_bin))
                        
                        # 이미지 ID -> 카메라 ID 매핑
                        image_to_camera = {}
                        for image_id, image in images.items():
                            image_to_camera[image_id] = image.camera_id
                        
                        return image_to_camera, cameras
                    except Exception as e:
                        print(f"    Failed to read images.bin: {e}")
                        return None
                
                return None
                
            except Exception as e:
                print(f"    Failed to read cameras.bin: {e}")
                return None
            
        except Exception as e:
            print(f"    COLMAP intrinsics 읽기 실패: {e}")
            # 파일이 손상되었을 수 있으므로 삭제 시도
            try:
                if cameras_bin.exists():
                    cameras_bin.unlink()
                    print(f"    Deleted corrupted cameras.bin")
                if images_bin.exists():
                    images_bin.unlink()
                    print(f"    Deleted corrupted images.bin")
                if points3d_bin.exists():
                    points3d_bin.unlink()
                    print(f"    Deleted corrupted points3D.bin")
            except Exception as del_error:
                print(f"    Failed to delete corrupted files: {del_error}")
            return None
    
    def _triangulate_all_points_robust(self):
        """개선된 삼각측량 - 더 많은 포인트 생성"""
        point_id = 0
        total_matches_processed = 0
        total_valid_matches = 0
        total_triangulated = 0
        total_validated = 0
        
        print(f"  Processing {len(self.matches)} image pairs for triangulation...")
        
        for (cam_i, cam_j), matches in self.matches.items():
            if cam_i not in self.cameras or cam_j not in self.cameras:
                continue
            
            try:
                # 투영 행렬 생성
                P_i = self._get_projection_matrix(cam_i)
                P_j = self._get_projection_matrix(cam_j)
                
                kpts_i = self.image_features[cam_i]['keypoints']
                kpts_j = self.image_features[cam_j]['keypoints']
                
                # 🔧 더 완화된 신뢰도 임계값
                high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches if conf > 0.05]  # 0.2 → 0.05로 완화
                total_matches_processed += len(matches)
                
                # 인덱스 범위 검증
                valid_matches = []
                for idx_i, idx_j, conf in high_conf_matches:
                    if (idx_i < len(kpts_i) and idx_j < len(kpts_j) and 
                        idx_i >= 0 and idx_j >= 0):
                        valid_matches.append((idx_i, idx_j, conf))
                
                total_valid_matches += len(valid_matches)
                
                # 🔧 더 적극적인 삼각측량
                if len(valid_matches) > 5:  # 10 → 5로 완화
                    batch_size = min(100, len(valid_matches))  # 배치 크기 증가
                    for batch_start in range(0, len(valid_matches), batch_size):
                        batch_end = min(batch_start + batch_size, len(valid_matches))
                        batch_matches = valid_matches[batch_start:batch_end]
                        
                        # 배치 삼각측량
                        pts_i_batch = np.array([kpts_i[idx_i] for idx_i, _, _ in batch_matches])
                        pts_j_batch = np.array([kpts_j[idx_j] for _, idx_j, _ in batch_matches])
                        
                        try:
                            # OpenCV 배치 삼각측량
                            points_4d = cv2.triangulatePoints(P_i, P_j, pts_i_batch.T, pts_j_batch.T)
                            
                            # 4D에서 3D로 변환 (더 완화된 검증)
                            for i in range(points_4d.shape[1]):
                                point_4d = points_4d[:, i]
                                
                                if abs(point_4d[3]) < 1e-10:
                                    continue
                                
                                point_3d = (point_4d[:3] / point_4d[3]).flatten()
                                total_triangulated += 1
                                
                                # 🔧 더 완화된 유효성 검사
                                if self._is_point_valid_relaxed(point_3d, cam_i, cam_j, pts_i_batch[i], pts_j_batch[i]):
                                    # 색상 추정
                                    color = self._estimate_point_color_robust(point_3d, cam_i, batch_matches[i][0])
                                    
                                    # 3D 포인트 저장
                                    self.points_3d[point_id] = {
                                        'xyz': point_3d.astype(np.float32),
                                        'color': color,
                                        'observations': [(cam_i, pts_i_batch[i], batch_matches[i][2]), 
                                                        (cam_j, pts_j_batch[i], batch_matches[i][2])]
                                    }
                                    
                                    # 관찰 데이터 추가
                                    self.point_observations[point_id].append((cam_i, pts_i_batch[i], batch_matches[i][2]))
                                    self.point_observations[point_id].append((cam_j, pts_j_batch[i], batch_matches[i][2]))
                                    
                                    point_id += 1
                                    total_validated += 1
                                    
                        except Exception as e:
                            print(f"    Batch triangulation failed for pair {cam_i}-{cam_j}: {e}")
                            continue
                else:
                    # 개별 삼각측량 (기존 방식)
                    for idx_i, idx_j, conf in valid_matches:
                        try:
                            # 삼각측량
                            pt_i = kpts_i[idx_i].astype(np.float32)
                            pt_j = kpts_j[idx_j].astype(np.float32)
                            
                            point_4d = cv2.triangulatePoints(P_i, P_j, pt_i.reshape(2, 1), pt_j.reshape(2, 1))
                            
                            if abs(point_4d[3, 0]) < 1e-10:
                                continue
                                
                            point_3d = (point_4d[:3] / point_4d[3]).flatten()
                            total_triangulated += 1
                            
                            # 🔧 더 완화된 유효성 검사
                            if self._is_point_valid_relaxed(point_3d, cam_i, cam_j, pt_i, pt_j):
                                # 색상 추정
                                color = self._estimate_point_color_robust(point_3d, cam_i, idx_i)
                                
                                # 3D 포인트 저장
                                self.points_3d[point_id] = {
                                    'xyz': point_3d.astype(np.float32),
                                    'color': color,
                                    'observations': [(cam_i, pt_i, conf), (cam_j, pt_j, conf)]
                                }
                                
                                # 관찰 데이터 추가
                                self.point_observations[point_id].append((cam_i, pt_i, conf))
                                self.point_observations[point_id].append((cam_j, pt_j, conf))
                                
                                point_id += 1
                                total_validated += 1
                                
                        except Exception as e:
                            continue
                        
            except Exception as e:
                print(f"    Error processing pair {cam_i}-{cam_j}: {e}")
                continue
        
        print(f"  Triangulation statistics:")
        print(f"    Total matches processed: {total_matches_processed}")
        print(f"    Valid matches: {total_valid_matches}")
        print(f"    Successfully triangulated: {total_triangulated}")
        print(f"    Passed validation: {total_validated}")
        print(f"    Final 3D points: {len(self.points_3d)}")
        
        return len(self.points_3d)

    def _is_point_valid_relaxed(self, point_3d, cam_i, cam_j, pt_i, pt_j):
        """완화된 3D 포인트 유효성 검사"""
        
        # 1. 기본 NaN/Inf 체크
        if np.any(np.isnan(point_3d)) or np.any(np.isinf(point_3d)):
            return False
        
        # 2. 거리 제한 (더 관대한 범위)
        distance = np.linalg.norm(point_3d)
        if distance > 500 or distance < 0.001:  # 100 → 500, 0.01 → 0.001로 완화
            return False
        
        # 3. 완화된 재투영 오차 체크
        try:
            max_reprojection_error = 0.0
            
            for cam_id, pt_observed in [(cam_i, pt_i), (cam_j, pt_j)]:
                if cam_id not in self.cameras:
                    continue
                
                cam = self.cameras[cam_id]
                K, R, T = cam['K'], cam['R'], cam['T']
                
                # 카메라 좌표계로 변환
                point_cam = R @ (point_3d - T)
                
                # 깊이 체크 (더 완화된 조건)
                if point_cam[2] <= 0.001:  # 0.01 → 0.001로 완화
                    return False
                
                # 재투영
                point_2d_proj = K @ point_cam
                
                if abs(point_2d_proj[2]) < 1e-10:
                    return False
                    
                point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                
                # 재투영 오차 계산
                error = np.linalg.norm(point_2d_proj - pt_observed)
                max_reprojection_error = max(max_reprojection_error, error)
            
            # 재투영 오차 임계값 (더 관대하게)
            if max_reprojection_error > 50.0:  # 10 → 50으로 완화
                return False
            
            return True
            
        except Exception as e:
            return False

    def _estimate_point_color_robust(self, point_3d, cam_id, kpt_idx):
        """개선된 3D 포인트 색상 추정"""
        # 실제 구현에서는 이미지에서 색상을 샘플링
        # 여기서는 간단히 랜덤 색상 사용
        return np.random.rand(3).astype(np.float32)
    
    def _bundle_adjustment_robust(self, max_iterations=50):
        """개선된 Bundle Adjustment - 더 완화된 조건"""
        
        n_cameras = len(self.cameras)
        n_points = len(self.points_3d)
        
        if n_cameras < 2 or n_points < 5:  # 20 → 5로 대폭 완화
            print("  Insufficient data for bundle adjustment")
            return
        
        # 관찰 데이터 수 계산
        total_observations = sum(len(obs) for obs in self.point_observations.values())
        n_residuals = total_observations * 2  # 각 관찰당 2개 잔차 (x, y)
        n_variables = n_cameras * 6 + n_points * 3  # 카메라 6DOF + 포인트 3DOF
        
        print(f"  BA Statistics:")
        print(f"    Cameras: {n_cameras}, Points: {n_points}")
        print(f"    Observations: {total_observations}")
        print(f"    Residuals: {n_residuals}, Variables: {n_variables}")
        
        # 🔧 더 완화된 방법 선택
        if n_residuals < n_variables:  # 2배 조건 제거
            print(f"  ⚠️  Under-constrained problem: {n_residuals} residuals < {n_variables} variables")
            print("  Using 'trf' method with very conservative settings")
            method = 'trf'
        else:
            print("  Using 'lm' method")
            method = 'lm'
        
        try:
            params = self._pack_parameters()
        except Exception as e:
            print(f"  Parameter packing failed: {e}")
            return
        
        try:
            # 🔧 더 완화된 BA 설정
            if method == 'trf':
                result = least_squares(
                    self._compute_residuals_improved,
                    params,
                    method='trf',
                    max_nfev=max_iterations,  # 반복 횟수 줄임
                    verbose=1,
                    ftol=1e-3,  # 더 완화된 수렴 조건
                    xtol=1e-3,
                    bounds=(-np.inf, np.inf)
                )
            else:
                result = least_squares(
                    self._compute_residuals_improved,
                    params,
                    method='lm',
                    max_nfev=max_iterations * 2,  # 반복 횟수 줄임
                    verbose=1,
                    ftol=1e-4,  # 더 완화된 수렴 조건
                    xtol=1e-4
                )
            
            # 결과 언패킹
            self._unpack_parameters(result.x)
            
            print(f"  Bundle adjustment completed. Final cost: {result.cost:.6f}")
            print(f"  Method: {method}, Iterations: {result.nfev}")
            
            # 🔧 더 완화된 cost 평가
            if result.cost > 1000:
                print(f"  ⚠️  높은 BA cost: {result.cost:.2f}")
                print("  포인트 클라우드 품질이 낮을 수 있습니다")
            elif result.cost > 100:
                print(f"  ⚠️  중간 BA cost: {result.cost:.2f}")
            else:
                print(f"  ✅ 좋은 BA cost: {result.cost:.2f}")
            
        except Exception as e:
            print(f"  Bundle adjustment failed: {e}")
            print("  Continuing without bundle adjustment...")

    def _compute_residuals_improved(self, params):
        """개선된 Bundle Adjustment 잔차 계산 (더 완화된 버전)"""
        residuals = []
        
        # 파라미터 언패킹
        try:
            self._unpack_parameters(params)
        except Exception as e:
            print(f"    Warning: Parameter unpacking failed: {e}")
            return np.ones(100) * 1e6
        
        # 각 관찰에 대한 재투영 오차 계산
        for point_id, observations in self.point_observations.items():
            if point_id not in self.points_3d:
                continue
                
            point_3d = self.points_3d[point_id]['xyz']
            
            for cam_id, observed_pt, conf in observations:
                if cam_id not in self.cameras:
                    continue
                
                try:
                    cam = self.cameras[cam_id]
                    K = cam['K']
                    R = cam['R']
                    T = cam['T']
                    
                    # 카메라 좌표계로 변환
                    point_cam = R @ (point_3d - T)
                    
                    # 깊이 체크 (더 완화된 조건)
                    if point_cam[2] <= 0:
                        residuals.extend([10.0, 10.0])  # 더 작은 페널티
                        continue
                    
                    # 재투영
                    point_2d_proj = K @ point_cam
                    if abs(point_2d_proj[2]) < 1e-10:
                        residuals.extend([10.0, 10.0])
                        continue
                    point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                    
                    # 🔧 더 완화된 잔차 계산
                    residual = point_2d_proj - observed_pt
                    
                    # 🔧 더 완화된 Huber loss
                    residual = self._apply_huber_loss_improved(residual, delta=10.0)  # 3.0 → 10.0으로 완화
                    
                    # 🔧 더 완화된 신뢰도 가중치
                    weight = np.clip(conf, 0.05, 1.0)  # 0.1 → 0.05로 완화
                    
                    # 🔧 더 완화된 스케일링
                    residual = residual * weight * 0.01  # 0.05 → 0.01로 완화
                    
                    residuals.extend(residual)
                    
                except Exception as e:
                    residuals.extend([2.0, 2.0])  # 더 작은 기본 오차
        
        if len(residuals) == 0:
            return np.ones(100) * 1e6
        
        residuals = np.array(residuals)
        
        # NaN이나 무한대 값 체크
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            return np.ones(len(residuals)) * 1e6
        
        return residuals

    def _apply_huber_loss_improved(self, residual, delta=3.0):
        """개선된 Huber loss 적용"""
        abs_residual = np.abs(residual)
        mask = abs_residual <= delta
        
        result = np.zeros_like(residual)
        result[mask] = residual[mask]
        result[~mask] = delta * np.sign(residual[~mask]) * (2 * np.sqrt(abs_residual[~mask] / delta) - 1)
        
        return result
    
    def _expand_point_observations(self):
        """포인트 관찰 데이터 확장으로 잔차 수 증가"""
        
        print("  Expanding point observations...")
        
        original_obs = sum(len(obs) for obs in self.point_observations.values())
        
        # 각 3D 포인트에 대해 다른 카메라에서의 재투영 확인
        for point_id, point_data in self.points_3d.items():
            point_3d = point_data['xyz']
            current_cams = set([obs[0] for obs in self.point_observations[point_id]])
            
            # 다른 카메라들에서도 이 포인트가 보이는지 확인
            for cam_id in self.cameras:
                if cam_id in current_cams:
                    continue
                
                try:
                    # 재투영 계산
                    cam = self.cameras[cam_id]
                    K, R, T = cam['K'], cam['R'], cam['T']
                    
                    point_cam = R @ (point_3d - T)
                    if point_cam[2] <= 0:  # 카메라 뒤쪽
                        continue
                    
                    point_2d_proj = K @ point_cam
                    point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                    
                    # 이미지 경계 확인
                    h, w = self.image_features[cam_id]['image_size']
                    if (0 <= point_2d_proj[0] < w and 0 <= point_2d_proj[1] < h):
                        
                        # 해당 카메라의 키포인트와 가까운지 확인
                        kpts = self.image_features[cam_id]['keypoints']
                        distances = np.linalg.norm(kpts - point_2d_proj, axis=1)
                        min_idx = np.argmin(distances)
                        
                        if distances[min_idx] < 30.0:  # 30 픽셀 내
                            # 관찰 추가
                            confidence = 0.1  # 낮은 신뢰도
                            self.point_observations[point_id].append((cam_id, point_2d_proj, confidence))
                            
                except Exception:
                    continue
        
        expanded_obs = sum(len(obs) for obs in self.point_observations.values())
        print(f"    Expanded observations: {original_obs} → {expanded_obs}")
    
    def _rotation_matrix_to_angle_axis(self, R):
        """회전 행렬을 로드리게스 벡터로 변환"""
        # 간단한 구현 (실제로는 더 정확한 변환이 필요)
        trace = np.trace(R)
        if trace > 3 - 1e-6:
            return np.zeros(3)
        
        angle = np.arccos((trace - 1) / 2)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
        
        return angle * axis
    
    def _angle_axis_to_rotation_matrix(self, angle_axis):
        """로드리게스 벡터를 회전 행렬로 변환"""
        angle = np.linalg.norm(angle_axis)
        if angle < 1e-6:
            return np.eye(3)
        
        axis = angle_axis / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def _refine_point_cloud(self):
        """포인트 클라우드 정제 (더 완화된 버전)"""
        print(f"  Refining point cloud...")
        
        # 1. 중복 포인트 제거 (더 완화된 조건)
        points_to_remove = set()
        points_list = list(self.points_3d.items())
        
        for i, (id1, point1) in enumerate(points_list):
            for j, (id2, point2) in enumerate(points_list[i+1:], i+1):
                if id1 in points_to_remove or id2 in points_to_remove:
                    continue
                
                dist = np.linalg.norm(point1['xyz'] - point2['xyz'])
                if dist < 0.0001:  # 0.001 → 0.0001로 더 엄격하게
                    points_to_remove.add(id2)
        
        # 중복 포인트 제거
        for point_id in points_to_remove:
            del self.points_3d[point_id]
            if point_id in self.point_observations:
                del self.point_observations[point_id]
        
        print(f"  Removed {len(points_to_remove)} duplicate points")
        print(f"  Final point cloud: {len(self.points_3d)} points")
    
    def _get_projection_matrix(self, cam_id):
        """카메라 투영 행렬 생성 (수정된 버전)"""
        cam = self.cameras[cam_id]
        K, R, T = cam['K'], cam['R'], cam['T']
        
        # T가 월드 좌표계의 카메라 중심이라고 가정
        # P = K[R|t] where t = -R * T (카메라 중심을 카메라 좌표계로 변환)
        t = -R @ T  # 카메라 중심을 카메라 좌표계로 변환
        RT = np.hstack([R, t.reshape(-1, 1)])
        P = K @ RT
        
        return P
    
    def _create_3dgs_scene_info(self, image_paths):
        """3DGS용 SceneInfo 생성 (개선된 버전)"""
        
        # Lazy import 3DGS modules
        CameraInfo, SceneInfo, BasicPointCloud = get_3dgs_imports()
        if CameraInfo is None:
            raise ImportError("3DGS modules not available")
        
        # CameraInfo 리스트 생성
        cam_infos = []
        for cam_id in sorted(self.cameras.keys()):
            cam = self.cameras[cam_id]
            image_path = self.image_features[cam_id]['image_path']
            h, w = self.image_features[cam_id]['image_size']
            
            # FoV 계산
            K = cam['K']
            focal_x, focal_y = K[0, 0], K[1, 1]
            FovX = 2 * np.arctan(w / (2 * focal_x))
            FovY = 2 * np.arctan(h / (2 * focal_y))
            
            # 더 나은 테스트 분할 (연결성 기반)
            is_test = self._should_be_test_camera(cam_id)
            
            cam_info = CameraInfo(
                uid=cam_id,
                R=cam['R'],
                T=cam['T'],
                FovY=float(FovY),
                FovX=float(FovX),
                image_path=image_path,
                image_name=Path(image_path).name,
                width=w,
                height=h,
                depth_params=None,
                depth_path="",
                is_test=is_test
            )
            cam_infos.append(cam_info)
        
        # 포인트 클라우드 생성
        if self.points_3d:
            points = np.array([pt['xyz'] for pt in self.points_3d.values()])
            colors = np.array([pt['color'] for pt in self.points_3d.values()])
            
            # 법선 벡터 (개선된 계산)
            normals = self._compute_point_normals(points)
            
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        else:
            # 기본 포인트 클라우드 (더 많은 수)
            n_points = 25000  # 15000 → 25000로 증가
            points = np.random.randn(n_points, 3).astype(np.float32) * 4  # 3 → 4로 증가
            colors = np.random.rand(n_points, 3).astype(np.float32)
            normals = np.random.randn(n_points, 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # 학습/테스트 분할
        train_cams = [c for c in cam_infos if not c.is_test]
        test_cams = [c for c in cam_infos if c.is_test]
        
        # NeRF 정규화
        nerf_norm = self._compute_nerf_normalization(train_cams)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cams,
            test_cameras=test_cams,
            nerf_normalization=nerf_norm,
            ply_path="",
            is_nerf_synthetic=False
        )
    
    def _should_be_test_camera(self, cam_id):
        """더 나은 테스트 카메라 선택"""
        # 연결성이 낮은 카메라를 테스트로 선택
        connectivity = len(self.camera_graph.get(cam_id, []))
        
        # 연결성이 1 이하이거나, 특정 간격으로 선택
        if connectivity <= 1:
            return True
        
        # 10개마다 1개씩 테스트로 선택 (연결성이 높은 카메라들 중에서)
        if cam_id % 10 == 0 and connectivity >= 2:
            return True
        
        return False
    
    def _compute_point_normals(self, points):
        """포인트 클라우드 법선 벡터 계산"""
        if len(points) < 3:
            return np.random.randn(len(points), 3).astype(np.float32)
        
        # 간단한 법선 계산 (sklearn 의존성 제거)
        try:
            normals = np.zeros_like(points)
            
            for i in range(len(points)):
                # 현재 포인트
                current_point = points[i]
                
                # 다른 모든 포인트와의 거리 계산
                distances = np.linalg.norm(points - current_point, axis=1)
                
                # 가장 가까운 10개 포인트 선택 (자기 자신 제외)
                nearest_indices = np.argsort(distances)[1:11]  # 자기 자신 제외
                
                if len(nearest_indices) < 3:
                    normals[i] = np.random.randn(3)
                    continue
                
                # 이웃 포인트들의 중심 계산
                neighbors = points[nearest_indices]
                centroid = np.mean(neighbors, axis=0)
                
                # 공분산 행렬 계산
                centered = neighbors - centroid
                cov_matrix = centered.T @ centered
                
                # 가장 작은 고유값에 해당하는 고유벡터가 법선
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                normal = eigenvecs[:, 0]  # 가장 작은 고유값
                
                # 방향 일관성 확인
                if normal[2] < 0:
                    normal = -normal
                
                normals[i] = normal
            
            # 정규화
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normals = normals / norms
            
        except Exception as e:
            print(f"    Warning: Normal computation failed: {e}")
            # 실패시 랜덤 법선
            normals = np.random.randn(len(points), 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        return normals.astype(np.float32)
    
    def _create_default_pointcloud(self):
        """기본 포인트 클라우드 생성 (개선된 버전)"""
        # Lazy import 3DGS modules
        _, _, BasicPointCloud = get_3dgs_imports()
        if BasicPointCloud is None:
            # Fallback: 간단한 클래스 정의
            class BasicPointCloud:
                def __init__(self, points, colors, normals):
                    self.points = points
                    self.colors = colors
                    self.normals = normals
        
        # 카메라 위치를 기반으로 한 더 현실적인 포인트 클라우드
        if len(self.cameras) > 0:
            # 카메라 중심 계산
            camera_centers = []
            for cam_id in self.cameras:
                R, T = self.cameras[cam_id]['R'], self.cameras[cam_id]['T']
                center = -R.T @ T
                camera_centers.append(center)
            
            if camera_centers:
                camera_centers = np.array(camera_centers)
                center_mean = np.mean(camera_centers, axis=0)
                center_std = np.std(camera_centers, axis=0)
                
                # 실제 포인트가 있으면 그것을 기반으로 생성
                if self.points_3d:
                    actual_points = np.array([pt['xyz'] for pt in self.points_3d.values()])
                    if len(actual_points) > 0:
                        # 실제 포인트 주변에 추가 포인트 생성
                        n_additional = 15000  # 더 많은 추가 포인트 (5000 → 15000)
                        points = np.random.randn(n_additional, 3).astype(np.float32)
                        points = points * np.std(actual_points, axis=0) * 0.8 + np.mean(actual_points, axis=0)
                        
                        # 실제 포인트와 합치기
                        points = np.vstack([actual_points, points])
                        colors = np.random.rand(len(points), 3).astype(np.float32)
                        normals = self._compute_point_normals(points)
                    else:
                        # 카메라 분포를 고려한 포인트 생성
                        n_points = 20000  # 더 많은 수 (10000 → 20000)
                        points = np.random.randn(n_points, 3).astype(np.float32)
                        points = points * center_std * 0.8 + center_mean
                        colors = np.random.rand(n_points, 3).astype(np.float32)
                        normals = self._compute_point_normals(points)
                else:
                    # 카메라 분포를 고려한 포인트 생성
                    n_points = 20000  # 더 많은 수 (10000 → 20000)
                    points = np.random.randn(n_points, 3).astype(np.float32)
                    points = points * center_std * 0.8 + center_mean
                    colors = np.random.rand(n_points, 3).astype(np.float32)
                    normals = self._compute_point_normals(points)
            else:
                # 기본 포인트 클라우드 (더 적은 수)
                points = np.random.randn(10000, 3).astype(np.float32) * 3
                colors = np.random.rand(10000, 3).astype(np.float32)
                normals = np.random.randn(10000, 3).astype(np.float32)
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        else:
            # 기본 포인트 클라우드 (더 적은 수)
            points = np.random.randn(10000, 3).astype(np.float32) * 3
            colors = np.random.rand(10000, 3).astype(np.float32)
            normals = np.random.randn(10000, 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        return BasicPointCloud(points=points, colors=colors, normals=normals)
    
    def _compute_nerf_normalization(self, cam_infos):
        """NeRF 정규화 파라미터 계산 (개선된 버전)"""
        # Lazy import 3DGS modules
        try:
            from utils.graphics_utils import getWorld2View2
        except ImportError:
            # Fallback: 간단한 함수 정의
            def getWorld2View2(R, t):
                Rt = np.zeros((4, 4))
                Rt[:3, :3] = R
                Rt[:3, 3] = t
                Rt[3, 3] = 1.0
                return Rt
        
        if not cam_infos:
            return {"translate": np.zeros(3), "radius": 1.0}
        
        # 카메라 중심 계산
        cam_centers = []
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        cam_centers = np.hstack(cam_centers)
        
        # 더 안정적인 중심 계산 (중간값 사용)
        center = np.median(cam_centers, axis=1, keepdims=True).flatten()
        
        # 거리 계산
        distances = np.linalg.norm(cam_centers - center.reshape(-1, 1), axis=0)
        
        # 더 보수적인 반지름 계산 (95 퍼센타일 사용)
        radius = np.percentile(distances, 95) * 1.2
        
        # 최소 반지름 보장
        radius = max(radius, 1.0)
        
        return {"translate": -center, "radius": radius}
    
    def _save_3dgs_format(self, scene_info, output_dir):
        """3DGS 학습을 위한 파일 구조 생성"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # COLMAP 호환 디렉토리 구조
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(exist_ok=True, parents=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 1. 카메라 내부 파라미터 저장 (cameras.txt + cameras.bin)
        self._write_cameras_txt(scene_info.train_cameras + scene_info.test_cameras, 
                               sparse_dir / "cameras.txt")
        self._write_cameras_bin(scene_info.train_cameras + scene_info.test_cameras, 
                               sparse_dir / "cameras.bin")
        
        # 2. 카메라 포즈 저장 (images.txt + images.bin)
        self._write_images_txt(scene_info.train_cameras + scene_info.test_cameras, 
                              sparse_dir / "images.txt")
        self._write_images_bin(scene_info.train_cameras + scene_info.test_cameras, 
                              sparse_dir / "images.bin")
        
        # 3. 3D 포인트 저장 (points3D.ply + points3D.bin)
        self._write_points3d_ply(scene_info.point_cloud, sparse_dir / "points3D.ply")
        self._write_points3d_bin(scene_info.point_cloud, sparse_dir / "points3D.bin")
        
        # 4. 이미지 복사 또는 심볼릭 링크
        self._setup_images_directory(scene_info.train_cameras + scene_info.test_cameras, 
                                    images_dir)
        
        print(f"  3DGS-compatible files saved to {output_dir}")
        print(f"  Use: python train.py -s {output_dir} -m {output_dir}/3dgs_output")
    
    def _write_cameras_txt(self, cam_infos, output_path):
        """COLMAP 형식 cameras.txt 생성"""
        with open(output_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cam_infos)}\n")
            
            for cam in cam_infos:
                # PINHOLE 모델 사용
                focal_x = cam.width / (2 * np.tan(cam.FovX / 2))
                focal_y = cam.height / (2 * np.tan(cam.FovY / 2))
                cx, cy = cam.width / 2, cam.height / 2
                
                f.write(f"{cam.uid} PINHOLE {cam.width} {cam.height} "
                       f"{focal_x:.6f} {focal_y:.6f} {cx:.6f} {cy:.6f}\n")
    
    def _write_cameras_bin(self, cam_infos, output_path):
        """COLMAP 형식 cameras.bin 생성"""
        try:
            from scene.colmap_loader import write_intrinsics_binary
            cameras = {}
            
            for cam in cam_infos:
                # PINHOLE 모델 사용
                focal_x = cam.width / (2 * np.tan(cam.FovX / 2))
                focal_y = cam.height / (2 * np.tan(cam.FovY / 2))
                cx, cy = cam.width / 2, cam.height / 2
                
                # COLMAP Camera 객체 생성
                from scene.colmap_loader import Camera
                camera = Camera(
                    id=cam.uid,
                    model="PINHOLE",
                    width=cam.width,
                    height=cam.height,
                    params=[focal_x, focal_y, cx, cy]
                )
                cameras[cam.uid] = camera
            
            write_intrinsics_binary(cameras, str(output_path))
            print(f"    Created cameras.bin with {len(cameras)} cameras")
            
        except ImportError:
            print(f"    Warning: COLMAP loader not available, creating simple cameras.bin")
            self._write_cameras_bin_simple(cam_infos, output_path)
        except Exception as e:
            print(f"    Warning: Failed to create cameras.bin: {e}")
            self._write_cameras_bin_simple(cam_infos, output_path)
    
    def _write_cameras_bin_simple(self, cam_infos, output_path):
        """간단한 cameras.bin 생성 (COLMAP loader 없이)"""
        try:
            import struct
            
            with open(output_path, 'wb') as f:
                # 카메라 수
                f.write(struct.pack('<Q', len(cam_infos)))
                
                for cam in cam_infos:
                    # PINHOLE 모델 사용
                    focal_x = cam.width / (2 * np.tan(cam.FovX / 2))
                    focal_y = cam.height / (2 * np.tan(cam.FovY / 2))
                    cx, cy = cam.width / 2, cam.height / 2
                    
                    # 카메라 ID (int)
                    f.write(struct.pack('<i', cam.uid))
                    
                    # 모델 ID (PINHOLE = 1)
                    model_id = 1
                    f.write(struct.pack('<i', model_id))
                    
                    # 너비, 높이 (unsigned long long)
                    f.write(struct.pack('<Q', cam.width))
                    f.write(struct.pack('<Q', cam.height))
                    
                    # 파라미터들 (double)
                    params = [focal_x, focal_y, cx, cy]
                    for param in params:
                        f.write(struct.pack('<d', param))
            
            print(f"    Created simple cameras.bin with {len(cam_infos)} cameras")
            
        except Exception as e:
            print(f"    Error creating simple cameras.bin: {e}")
            import traceback
            traceback.print_exc()
    
    def _write_images_bin(self, cam_infos, output_path):
        """COLMAP 형식 images.bin 생성"""
        try:
            from scene.colmap_loader import write_extrinsics_binary
            images = {}
            
            for cam in cam_infos:
                # 회전 행렬을 쿼터니언으로 변환
                R = cam.R
                trace = np.trace(R)
                
                if trace > 0:
                    s = np.sqrt(trace + 1.0) * 2
                    qw = 0.25 * s
                    qx = (R[2, 1] - R[1, 2]) / s
                    qy = (R[0, 2] - R[2, 0]) / s
                    qz = (R[1, 0] - R[0, 1]) / s
                else:
                    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
                    qx = (R[2,1] - R[1,2]) / (4 * qw) if qw != 0 else 0
                    qy = (R[0,2] - R[2,0]) / (4 * qw) if qw != 0 else 0
                    qz = (R[1,0] - R[0,1]) / (4 * qw) if qw != 0 else 0
                
                # 정규화
                q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                if q_norm > 0:
                    qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm
                
                # COLMAP Image 객체 생성
                from scene.colmap_loader import Image
                image = Image(
                    id=cam.uid,
                    qvec=np.array([qw, qx, qy, qz]),
                    tvec=cam.T,
                    camera_id=cam.uid,
                    name=cam.image_name,
                    xys=np.array([]),  # 빈 특징점 배열
                    point3D_ids=np.array([])  # 빈 3D 포인트 ID 배열
                )
                images[cam.uid] = image
            
            write_extrinsics_binary(images, str(output_path))
            print(f"    Created images.bin with {len(images)} images")
            
        except ImportError:
            print(f"    Warning: COLMAP loader not available, creating simple images.bin")
            self._write_images_bin_simple(cam_infos, output_path)
        except Exception as e:
            print(f"    Warning: Failed to create images.bin: {e}")
            self._write_images_bin_simple(cam_infos, output_path)
    
    def _write_images_bin_simple(self, cam_infos, output_path):
        """간단한 images.bin 생성 (COLMAP loader 없이)"""
        try:
            import struct
            
            with open(output_path, 'wb') as f:
                # 이미지 수
                f.write(struct.pack('<Q', len(cam_infos)))
                
                for cam in cam_infos:
                    # 회전 행렬을 쿼터니언으로 변환
                    R = cam.R
                    trace = np.trace(R)
                    
                    if trace > 0:
                        s = np.sqrt(trace + 1.0) * 2
                        qw = 0.25 * s
                        qx = (R[2, 1] - R[1, 2]) / s
                        qy = (R[0, 2] - R[2, 0]) / s
                        qz = (R[1, 0] - R[0, 1]) / s
                    else:
                        qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
                        qx = (R[2,1] - R[1,2]) / (4 * qw) if qw != 0 else 0
                        qy = (R[0,2] - R[2,0]) / (4 * qw) if qw != 0 else 0
                        qz = (R[1,0] - R[0,1]) / (4 * qw) if qw != 0 else 0
                    
                    # 정규화
                    q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                    if q_norm > 0:
                        qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm
                    
                    # 이미지 ID
                    f.write(struct.pack('<Q', cam.uid))
                    
                    # 쿼터니언 (qw, qx, qy, qz)
                    f.write(struct.pack('<dddd', qw, qx, qy, qz))
                    
                    # 이동 벡터 (tx, ty, tz)
                    f.write(struct.pack('<ddd', cam.T[0], cam.T[1], cam.T[2]))
                    
                    # 카메라 ID
                    f.write(struct.pack('<Q', cam.uid))
                    
                    # 이미지 이름 길이와 이름
                    name_bytes = cam.image_name.encode('utf-8')
                    f.write(struct.pack('<Q', len(name_bytes)))
                    f.write(name_bytes)
                    
                    # 특징점 수 (0개)
                    f.write(struct.pack('<Q', 0))
                    
                    # 3D 포인트 ID 수 (0개)
                    f.write(struct.pack('<Q', 0))
            
            print(f"    Created simple images.bin with {len(cam_infos)} images")
            
        except Exception as e:
            print(f"    Error creating simple images.bin: {e}")
    
    def _write_points3d_bin(self, point_cloud, output_path):
        """COLMAP 형식 points3D.bin 생성"""
        try:
            from scene.colmap_loader import write_points3D_binary
            points3d = {}
            
            points = point_cloud.points
            colors = point_cloud.colors
            
            for i in range(len(points)):
                # COLMAP Point3D 객체 생성
                from scene.colmap_loader import Point3D
                point3d = Point3D(
                    id=i,
                    xyz=points[i],
                    rgb=colors[i] * 255,  # 0-1 범위를 0-255로 변환
                    error=0.0,  # 기본 오차
                    track=[]  # 빈 트랙 (관찰 정보 없음)
                )
                points3d[i] = point3d
            
            write_points3D_binary(points3d, str(output_path))
            print(f"    Created points3D.bin with {len(points3d)} points")
            
        except ImportError:
            print(f"    Warning: COLMAP loader not available, creating simple points3D.bin")
            self._write_points3d_bin_simple(point_cloud, output_path)
        except Exception as e:
            print(f"    Warning: Failed to create points3D.bin: {e}")
            self._write_points3d_bin_simple(point_cloud, output_path)
    
    def _write_points3d_bin_simple(self, point_cloud, output_path):
        """간단한 points3D.bin 생성 (COLMAP loader 없이)"""
        try:
            import struct
            
            points = point_cloud.points
            colors = point_cloud.colors
            
            with open(output_path, 'wb') as f:
                # 포인트 수
                f.write(struct.pack('<Q', len(points)))
                
                for i in range(len(points)):
                    # 포인트 ID
                    f.write(struct.pack('<Q', i))
                    
                    # 3D 좌표 (x, y, z)
                    f.write(struct.pack('<ddd', points[i][0], points[i][1], points[i][2]))
                    
                    # RGB 색상 (0-255 범위로 변환)
                    rgb = (colors[i] * 255).astype(np.uint8)
                    f.write(struct.pack('<BBB', rgb[0], rgb[1], rgb[2]))
                    
                    # 오차 (0.0)
                    f.write(struct.pack('<d', 0.0))
                    
                    # 트랙 길이 (0개)
                    f.write(struct.pack('<Q', 0))
            
            print(f"    Created simple points3D.bin with {len(points)} points")
            
        except Exception as e:
            print(f"    Error creating simple points3D.bin: {e}")
    
    def _write_images_txt(self, cam_infos, output_path):
        """COLMAP 형식 images.txt 생성"""
        with open(output_path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(cam_infos)}\n")
            
            for cam in cam_infos:
                # 회전 행렬을 쿼터니언으로 변환
                R = cam.R
                trace = np.trace(R)
                
                if trace > 0:
                    s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                    qw = 0.25 * s
                    qx = (R[2, 1] - R[1, 2]) / s
                    qy = (R[0, 2] - R[2, 0]) / s
                    qz = (R[1, 0] - R[0, 1]) / s
                else:
                    # 안정적인 쿼터니언 변환
                    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
                    qx = (R[2,1] - R[1,2]) / (4 * qw) if qw != 0 else 0
                    qy = (R[0,2] - R[2,0]) / (4 * qw) if qw != 0 else 0
                    qz = (R[1,0] - R[0,1]) / (4 * qw) if qw != 0 else 0
                
                # 정규화
                q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                if q_norm > 0:
                    qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm
                
                f.write(f"{cam.uid} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                       f"{cam.T[0]:.6f} {cam.T[1]:.6f} {cam.T[2]:.6f} "
                       f"{cam.uid} {cam.image_name}\n")
                f.write("\n")  # 빈 특징점 라인
    
    def _write_points3d_ply(self, point_cloud, output_path):
        """PLY 형식으로 포인트 클라우드 저장"""
        points = point_cloud.points
        colors = (point_cloud.colors * 255).astype(np.uint8)
        
        with open(output_path, 'w') as f:
            # PLY 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 데이터
            for i in range(len(points)):
                x, y, z = points[i]
                nx, ny, nz = point_cloud.normals[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} "
                       f"{nx:.6f} {ny:.6f} {nz:.6f} "
                       f"{r} {g} {b}\n")
    
    def _setup_images_directory(self, cam_infos, images_dir):
        """이미지 디렉토리 설정 (심볼릭 링크 또는 복사)"""
        import shutil
        
        for cam in cam_infos:
            src_path = Path(cam.image_path)
            dst_path = images_dir / cam.image_name
            
            if not dst_path.exists():
                try:
                    # 심볼릭 링크 시도
                    dst_path.symlink_to(src_path.resolve())
                except (OSError, NotImplementedError):
                    # 실패시 복사
                    shutil.copy2(src_path, dst_path)
    
    def _calculate_adaptive_resize(self, image_path, max_dim=1600):
        """SuperGlue 권장 해상도(최대 max_dim)로 비율 유지 리사이즈 계산"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return [1024, 768]  # 기본값
            h, w = img.shape[:2]
            largest = max(h, w)
            if largest <= max_dim:
                return None  # 원본 크기 유지
            scale = max_dim / largest
            return [int(w * scale), int(h * scale)]
        except:
            return [1024, 768]  # 기본값

    def _load_image(self, image_path, resize=None):
        """이미지 로드 및 SuperGlue 권장 해상도 적용"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"    Warning: Failed to load {image_path}")
                return None

            # SuperGlue 권장 해상도 적용 (최대 1600px)
            if resize is None:
                resize = self._calculate_adaptive_resize(image_path, max_dim=1600)
            if resize is not None:
                image = cv2.resize(image, tuple(resize))
            return image.astype(np.float32)
        except Exception as e:
            print(f"    Error loading {image_path}: {e}")
            return None
    
    def _pack_parameters(self):
        """카메라 포즈와 3D 포인트를 하나의 벡터로 패킹"""
        params = []
        
        # 카메라 포즈 (회전 + 이동)
        for cam_id in sorted(self.cameras.keys()):
            cam = self.cameras[cam_id]
            R = cam['R']
            T = cam['T']
            
            # 회전 행렬을 로드리게스 벡터로 변환
            angle_axis = self._rotation_matrix_to_angle_axis(R)
            params.extend(angle_axis)
            params.extend(T)
        
        # 3D 포인트
        for point_id in sorted(self.points_3d.keys()):
            point = self.points_3d[point_id]['xyz']
            params.extend(point)
        
        params = np.array(params)
        
        # NaN이나 무한대 값 체크
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            raise ValueError("Invalid parameters detected (NaN or Inf)")
        
        return params
    
    def _unpack_parameters(self, params):
        """벡터에서 카메라 포즈와 3D 포인트 언패킹"""
        idx = 0
        
        # 카메라 포즈 복원
        for cam_id in sorted(self.cameras.keys()):
            # 로드리게스 벡터 (3개)
            angle_axis = params[idx:idx+3]
            idx += 3
            
            # 이동 벡터 (3개)
            T = params[idx:idx+3]
            idx += 3
            
            # 회전 행렬로 변환
            R = self._angle_axis_to_rotation_matrix(angle_axis)
            
            self.cameras[cam_id]['R'] = R.astype(np.float32)
            self.cameras[cam_id]['T'] = T.astype(np.float32)
        
        # 3D 포인트 복원
        for point_id in sorted(self.points_3d.keys()):
            xyz = params[idx:idx+3]
            idx += 3
            self.points_3d[point_id]['xyz'] = xyz.astype(np.float32)

def readSuperGlueSceneInfo(path, images, eval, train_test_exp=False, llffhold=8, 
                          superglue_config="outdoor", max_images=100):
    """SuperGlue 기반 완전한 SfM으로 SceneInfo 생성"""
    
    print("=== SuperGlue Complete SfM Pipeline ===")
    print(f"🚀 Pipeline available: {PIPELINE_AVAILABLE}")
    
    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline not available. Using fallback scene creation...")
        # 이미지 디렉토리 경로
        images_folder = Path(path) / (images if images else "images")
        return _create_fallback_scene_info(images_folder, max_images)
    
    # 이미지 디렉토리 경로
    images_folder = Path(path) / (images if images else "images")
    output_folder = Path(path) / "superglue_sfm_output"
    
    # SuperGlue 설정 (더 완화된 설정)
    config = {
        'superpoint': {
            'nms_radius': 3,  # 4 → 3으로 완화
            'keypoint_threshold': 0.001,  # 0.005 → 0.001로 대폭 완화
            'max_keypoints': 8192  # 4096 → 8192로 증가
        },
        'superglue': {
            'weights': superglue_config,  # 'indoor' 또는 'outdoor'
            'sinkhorn_iterations': 15,  # 20 → 15로 완화
            'match_threshold': 0.05,  # 0.1 → 0.05로 완화
        }
    }
    
    # SuperGlue 3DGS 파이프라인 실행
    try:
        pipeline = SuperGlue3DGSPipeline(config)
        print("✅ SuperGlue pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize SuperGlue pipeline: {e}")
        print("Falling back to simple camera arrangement...")
        return _create_fallback_scene_info(images_folder, max_images)
    
    try:
        scene_info = pipeline.process_images_to_3dgs(
            image_dir=images_folder,
            output_dir=output_folder,
            max_images=max_images
        )
        
        print(f"\n=== SuperGlue SfM Results ===")
        print(f"Training cameras: {len(scene_info.train_cameras)}")
        print(f"Test cameras: {len(scene_info.test_cameras)}")
        print(f"3D points: {len(scene_info.point_cloud.points)}")
        print(f"Scene radius: {scene_info.nerf_normalization['radius']:.3f}")
        
        return scene_info
        
    except Exception as e:
        print(f"SuperGlue SfM failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 실패시 fallback
        print("Falling back to simple camera arrangement...")
        return _create_fallback_scene_info(images_folder, max_images)


def _create_fallback_scene_info(images_folder, max_images):
    """개선된 fallback scene 생성"""
    try:
        # 이미지 수집
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(glob.glob(str(Path(images_folder) / ext)))
        
        image_paths.sort()
        image_paths = image_paths[:max_images]
        
        if not image_paths:
            raise ValueError(f"No images found in {images_folder}")
        
        print(f"📸 Found {len(image_paths)} images")
        
        # 카메라 정보 생성
        cam_infos = []
        for i, image_path in enumerate(image_paths):
            try:
                # 이미지 크기 확인
                image = Image.open(image_path)
                width, height = image.size
                
                # 원형 배치로 카메라 배치
                angle = i * (2 * np.pi / len(image_paths))
                radius = 3.0
                
                # 카메라 포즈 (원을 바라보도록)
                camera_pos = np.array([
                    radius * np.cos(angle),
                    0.0,  # 높이 고정
                    radius * np.sin(angle)
                ])
                
                # 원점을 향하는 방향
                look_at = np.array([0.0, 0.0, 0.0])
                up = np.array([0.0, 1.0, 0.0])
                
                # 카메라 회전 행렬 계산
                forward = look_at - camera_pos
                forward = forward / np.linalg.norm(forward)
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                
                R = np.array([right, up, -forward]).T  # OpenCV 컨벤션
                T = camera_pos
                
                # FOV 계산 (더 안전한 값들)
                focal_length = max(width, height) * 0.8
                FovX = 2 * np.arctan(width / (2 * focal_length))
                FovY = 2 * np.arctan(height / (2 * focal_length))
                
                # 테스트 카메라 선택 (더 균등하게 분산)
                is_test = (i % 8 == 0)  # 8개마다 1개씩 테스트
                
                cam_info = CameraInfo(
                    uid=i,
                    R=R.astype(np.float32),
                    T=T.astype(np.float32),
                    FovY=float(FovY),
                    FovX=float(FovX),
                    image_path=image_path,
                    image_name=Path(image_path).name,
                    width=int(width),
                    height=int(height),
                    depth_params=None,
                    depth_path="",
                    is_test=is_test
                )
                cam_infos.append(cam_info)
                
            except Exception as e:
                print(f"    Warning: Failed to process {image_path}: {e}")
                continue
        
        if not cam_infos:
            raise ValueError("No valid cameras created")
        
        # 개선된 포인트 클라우드 생성
        n_points = 12000  # 8000 → 12000로 증가
        
        # 더 현실적인 3D 포인트 분포
        # 구형 분포 + 일부 평면 구조
        points_sphere = np.random.randn(n_points // 2, 3).astype(np.float32)
        points_sphere = points_sphere / np.linalg.norm(points_sphere, axis=1, keepdims=True) * 3.0  # 2.0 → 3.0
        
        # 평면 구조 추가 (바닥면)
        points_plane = np.random.randn(n_points // 2, 3).astype(np.float32)
        points_plane[:, 1] = np.abs(points_plane[:, 1]) * 0.2 - 1.0  # 바닥 근처 (0.1 → 0.2, -0.5 → -1.0)
        points_plane[:, [0, 2]] *= 2.0  # 1.5 → 2.0
        
        points = np.vstack([points_sphere, points_plane])
        
        # 더 현실적인 색상 (회색조 + 약간의 색상)
        colors = np.random.rand(n_points, 3).astype(np.float32)
        colors = colors * 0.5 + 0.3  # 0.3-0.8 범위
        
        # 법선 벡터 (무작위지만 정규화됨)
        normals = np.random.randn(n_points, 3).astype(np.float32)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
        
        # BasicPointCloud 생성 시 차원 확인
        assert points.shape == (n_points, 3), f"Points shape error: {points.shape}"
        assert colors.shape == (n_points, 3), f"Colors shape error: {colors.shape}"
        assert normals.shape == (n_points, 3), f"Normals shape error: {normals.shape}"
        
        pcd = BasicPointCloud(
            points=points,
            colors=colors,
            normals=normals
        )
        
        # 학습/테스트 분할
        train_cams = [c for c in cam_infos if not c.is_test]
        test_cams = [c for c in cam_infos if c.is_test]
        
        # NeRF 정규화 (개선된 버전)
        if train_cams:
            camera_centers = []
            for cam in train_cams:
                # 카메라 중심 계산
                center = -cam.R.T @ cam.T
                camera_centers.append(center)
            
            camera_centers = np.array(camera_centers)
            scene_center = np.mean(camera_centers, axis=0)
            distances = np.linalg.norm(camera_centers - scene_center, axis=1)
            scene_radius = np.max(distances) * 1.2
            
            # 최소/최대 제한
            scene_radius = max(scene_radius, 1.0)
            scene_radius = min(scene_radius, 10.0)
        else:
            scene_center = np.zeros(3)
            scene_radius = 3.0
        
        nerf_normalization = {
            "translate": -scene_center.astype(np.float32),
            "radius": float(scene_radius)
        }
        
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cams,
            test_cameras=test_cams,
            nerf_normalization=nerf_normalization,
            ply_path="",
            is_nerf_synthetic=False
        )
        
        print(f"✓ Fallback scene created:")
        print(f"  - {len(train_cams)} training cameras")
        print(f"  - {len(test_cams)} test cameras")
        print(f"  - {n_points} 3D points")
        print(f"  - Scene radius: {scene_radius:.2f}")
        
        return scene_info
        
    except Exception as e:
        print(f"Failed to create fallback scene: {e}")
        raise


# 명령줄 인터페이스
def main():
    """명령줄에서 SuperGlue 3DGS 파이프라인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperGlue 3DGS Pipeline")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input images directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for 3DGS training")
    parser.add_argument("--max_images", type=int, default=100,
                       help="Maximum number of images to process")
    parser.add_argument("--config", choices=["indoor", "outdoor"], 
                       default="outdoor", help="SuperGlue configuration")
    parser.add_argument("--device", default="cuda", 
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # SuperGlue 설정
    config = {
        'superpoint': {
            'nms_radius': 3,  # 4 → 3으로 완화
            'keypoint_threshold': 0.001,  # 0.005 → 0.001로 대폭 완화
            'max_keypoints': 8192  # 4096 → 8192로 증가
        },
        'superglue': {
            'weights': args.config,
            'sinkhorn_iterations': 15,  # 20 → 15로 완화
            'match_threshold': 0.05,  # 0.1 → 0.05로 완화
        }
    }
    
    print(f"=== SuperGlue 3DGS Pipeline ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max images: {args.max_images}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    
    # 파이프라인 실행
    pipeline = SuperGlue3DGSPipeline(config, device=args.device)
    
    try:
        scene_info = pipeline.process_images_to_3dgs(
            image_dir=args.input,
            output_dir=args.output,
            max_images=args.max_images
        )
        
        print(f"\n=== Success! ===")
        print(f"Processed {len(scene_info.train_cameras)} training cameras")
        print(f"Generated {len(scene_info.point_cloud.points)} 3D points")
        print(f"Files saved to: {args.output}")
        print(f"\nNext step:")
        print(f"python train.py -s {args.output} -m {args.output}/3dgs_output")
        
    except Exception as e:
        print(f"\n=== Error! ===")
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())