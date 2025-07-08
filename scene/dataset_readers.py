# scene/dataset_readers.py에 추가할 완전한 SuperGlue 통합 코드

import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path
import json
from scipy.optimize import least_squares
from PIL import Image
from typing import NamedTuple
import glob

# 기존 3DGS imports
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB

# SuperGlue 모듈 경로 수정
def import_superglue_modules():
    """SuperGlue 모듈들을 동적으로 import"""
    try:
        # 현재 디렉토리에서 SuperGlue 찾기
        current_dir = Path(__file__).parent.parent  # gaussian-splatting 루트
        
        # 가능한 SuperGlue 경로들
        possible_paths = [
            current_dir / "models",
            current_dir / "Superglue" / "models", 
            current_dir / "SuperGlue" / "models",
            current_dir,
            current_dir / "Superglue",
            current_dir / "SuperGlue"
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "matching.py").exists():
                sys.path.insert(0, str(path.parent))
                sys.path.insert(0, str(path))
                break
        
        # SuperGlue 모듈 import
        from models.matching import Matching
        from models.utils import frame2tensor
        
        return Matching, frame2tensor
        
    except ImportError as e:
        print(f"SuperGlue modules not found: {e}")
        print("Falling back to simple pose estimation...")
        return None, None

# SuperGlue 사용 가능 여부 확인
try:
    Matching, frame2tensor = import_superglue_modules()
    SUPERGLUE_AVAILABLE = (Matching is not None)
except:
    SUPERGLUE_AVAILABLE = False
    Matching, frame2tensor = None, None

print(f"SuperGlue available: {SUPERGLUE_AVAILABLE}")


class SimpleSuperGluePipeline:
    """간소화된 SuperGlue 3DGS 파이프라인 (fallback 포함)"""
    
    def __init__(self, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if SUPERGLUE_AVAILABLE and Matching is not None:
            # SuperGlue 설정
            if config is None:
                config = {
                    'superpoint': {
                        'nms_radius': 4,
                        'keypoint_threshold': 0.005,
                        'max_keypoints': 1024
                    },
                    'superglue': {
                        'weights': 'outdoor',
                        'sinkhorn_iterations': 20,
                        'match_threshold': 0.2,
                    }
                }
            
            try:
                self.matching = Matching(config).eval().to(self.device)
                self.superglue_ready = True
                print(f"SuperGlue initialized on {self.device}")
            except Exception as e:
                print(f"SuperGlue initialization failed: {e}")
                self.superglue_ready = False
        else:
            self.superglue_ready = False
            print("SuperGlue not available, using simple pose estimation")
        
        # SfM 데이터
        self.cameras = {}
        self.points_3d = {}
        self.image_features = {}
    
    def process_images_to_scene_info(self, image_dir, max_images=100):
        """이미지를 SceneInfo로 변환"""
        
        print(f"\n=== Processing {max_images} images with SuperGlue Pipeline ===")
        
        # 1. 이미지 수집
        image_paths = self._collect_images(image_dir, max_images)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        if self.superglue_ready:
            # 2. SuperGlue 파이프라인
            try:
                return self._superglue_pipeline(image_paths)
            except Exception as e:
                print(f"SuperGlue pipeline failed: {e}")
                print("Falling back to simple arrangement...")
                return self._simple_arrangement(image_paths)
        else:
            # 3. 간단한 배치
            return self._simple_arrangement(image_paths)
    
    def _collect_images(self, image_dir, max_images):
        """이미지 파일 수집"""
        image_dir = Path(image_dir)
        image_paths = []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        image_paths.sort()
        return image_paths[:max_images]
    
    def debug_superpoint_output(self, image_path):
        image = self._load_image(image_path)
        inp = frame2tensor(image, self.device)
    
        with torch.no_grad():
            pred = self.matching.superpoint({'image': inp})
    
        print(f"SuperPoint output keys: {pred.keys()}")
        for key, value in pred.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    
        return pred
    
    def _superglue_pipeline(self, image_paths):
        """SuperGlue 기반 SfM 파이프라인"""
        
        print("Running SuperGlue SfM pipeline...")
        
        # 1. 특징점 추출
        print("1. Extracting features...")
        self._extract_features(image_paths[:20])  # 처음 20장만 처리
        
        # 2. 매칭
        print("2. Matching features...")
        matches = self._match_sequential()
        
        # 3. 포즈 추정
        print("3. Estimating poses...")
        self._estimate_poses_simple(matches)
        
        # 4. SceneInfo 생성
        print("4. Creating scene info...")
        return self._create_scene_info(image_paths[:len(self.cameras)])
    
    def _extract_features(self, image_paths):
        """SuperPoint로 특징점 추출"""
        for i, image_path in enumerate(image_paths):
            print(f"  {i+1}/{len(image_paths)}: {image_path.name}")
            
            self.debug_superpoint_output(image_path)
            
            image = self._load_image(image_path)
            if image is None:
                continue
            
            # SuperPoint 특징점 추출
            inp = frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            
            self.image_features[i] = {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(),
                'scores': pred['scores'][0].cpu().numpy(),
                'image_path': str(image_path),
                'image_size': image.shape[:2]
            }
    
    def _match_sequential(self):
        """순차적 매칭"""
        matches = {}
        n_images = len(self.image_features)
        
        for i in range(n_images - 1):
            j = i + 1
            match_result = self._match_pair(i, j)
            if len(match_result) > 10:
                matches[(i, j)] = match_result
        
        print(f"  Found {len(matches)} good matches")
        return matches
    
    def _match_pair(self, i, j):
        """더 안전한 매칭 함수"""
        try:
            feat0 = self.image_features[i]
            feat1 = self.image_features[j]
        
            # 입력 데이터 확인
            if 'keypoints' not in feat0 or 'keypoints' not in feat1:
                return []
        
            # 매칭 수행
            pred = self.matching({
                'keypoints0': torch.from_numpy(feat0['keypoints']).float().to(self.device),
                'keypoints1': torch.from_numpy(feat1['keypoints']).float().to(self.device),
                'descriptors0': torch.from_numpy(feat0['descriptors']).float().to(self.device),
                'descriptors1': torch.from_numpy(feat1['descriptors']).float().to(self.device),
                'scores0': torch.from_numpy(feat0['scores']).float().to(self.device),
                'scores1': torch.from_numpy(feat1['scores']).float().to(self.device),
                'image0': torch.zeros(1, 1, *feat0['image_size']).to(self.device),
                'image1': torch.zeros(1, 1, *feat1['image_size']).to(self.device),
        })
        
        # 매칭 결과 처리
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
        
            # 유효한 매칭만 선택
            valid = matches > -1
            matches = matches[valid]
            confidence = confidence[valid]
        
            # 신뢰도 기준 필터링
            conf_mask = confidence > 0.2
            matches = matches[conf_mask]
            confidence = confidence[conf_mask]
        
            return matches
        
        except Exception as e:
            print(f"  Matching failed for pair {i}-{j}: {e}")
            return []
    
    def _estimate_poses_simple(self, matches):
        """간단한 포즈 추정"""
        # 첫 번째 카메라를 원점으로
        self.cameras[0] = {
            'R': np.eye(3, dtype=np.float32),
            'T': np.zeros(3, dtype=np.float32),
            'K': self._estimate_intrinsics(0)
        }
        
        # 순차적 포즈 추정
        for cam_id in range(1, len(self.image_features)):
            if (cam_id-1, cam_id) in matches:
                R, T = self._estimate_relative_pose(cam_id-1, cam_id, matches[(cam_id-1, cam_id)])
                if R is not None:
                    self.cameras[cam_id] = {
                        'R': R,
                        'T': T,
                        'K': self._estimate_intrinsics(cam_id)
                    }
                    continue
            
            # 실패시 기본 배치
            angle = cam_id * 0.3
            self.cameras[cam_id] = {
                'R': np.array([[np.cos(angle), 0, np.sin(angle)],
                               [0, 1, 0],
                               [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32),
                'T': np.array([3*np.sin(angle), 0, 3*(1-np.cos(angle))], dtype=np.float32),
                'K': self._estimate_intrinsics(cam_id)
            }
    
    def _estimate_relative_pose(self, cam_i, cam_j, match_list):
        """Essential Matrix로 상대 포즈 추정"""
        if len(match_list) < 8:
            return None, None
        
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        pts_i = np.array([kpts_i[idx_i] for idx_i, _, conf in match_list if conf > 0.4])
        pts_j = np.array([kpts_j[idx_j] for _, idx_j, conf in match_list if conf > 0.4])
        
        if len(pts_i) < 8:
            return None, None
        
        K_i = self.cameras[cam_i]['K']
        K_j = self._estimate_intrinsics(cam_j)
        
        try:
            E, mask = cv2.findEssentialMat(pts_i, pts_j, K_i, 
                                           method=cv2.RANSAC, 
                                           prob=0.999, threshold=1.0)
            
            if E is not None:
                _, R, T, _ = cv2.recoverPose(E, pts_i, pts_j, K_i)
                return R, T.flatten()
        except:
            pass
        
        return None, None
    
    def _estimate_intrinsics(self, cam_id):
        """카메라 내부 파라미터 추정"""
        h, w = self.image_features[cam_id]['image_size']
        focal = max(w, h) * 0.8
        
        return np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _simple_arrangement(self, image_paths):
        """간단한 원형 카메라 배치"""
        print("Using simple circular camera arrangement...")
        
        cam_infos = []
        for i, image_path in enumerate(image_paths):
            # 이미지 크기 확인
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except:
                width, height = 1920, 1080
            
            # 원형 배치
            angle = (i / len(image_paths)) * 2 * np.pi
            radius = 4.0
            
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=np.float32)
            
            T = np.array([
                radius * np.sin(angle),
                0,
                radius * (1 - np.cos(angle))
            ], dtype=np.float32)
            
            # FOV 계산
            focal = max(width, height) * 0.8
            FovX = 2 * np.arctan(width / (2 * focal))
            FovY = 2 * np.arctan(height / (2 * focal))
            
            cam_info = CameraInfo(
                uid=i,
                R=R,
                T=T,
                FovY=float(FovY),
                FovX=float(FovX),
                image_path=str(image_path),
                image_name=image_path.name,
                width=width,
                height=height,
                depth_params=None,
                depth_path="",
                is_test=(i % 8 == 0)
            )
            cam_infos.append(cam_info)
        
        # 기본 포인트 클라우드
        n_points = 10000
        points = np.random.randn(n_points, 3).astype(np.float32) * 2
        colors = np.random.rand(n_points, 3).astype(np.float32)
        normals = np.random.randn(n_points, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
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
    
    def _create_scene_info(self, image_paths):
        """SuperGlue 결과로 SceneInfo 생성"""
        
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
                is_test=(cam_id % 8 == 0)
            )
            cam_infos.append(cam_info)
        
        # 간단한 포인트 클라우드
        n_points = 5000
        points = np.random.randn(n_points, 3).astype(np.float32) * 1.5
        colors = np.random.rand(n_points, 3).astype(np.float32)
        normals = np.random.randn(n_points, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
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
    
    def _compute_nerf_normalization(self, cam_infos):
        """NeRF 정규화 파라미터 계산"""
        cam_centers = []
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        if cam_centers:
            cam_centers = np.hstack(cam_centers)
            center = np.mean(cam_centers, axis=1, keepdims=True).flatten()
            distances = np.linalg.norm(cam_centers - center.reshape(-1, 1), axis=0)
            radius = np.max(distances) * 1.1
        else:
            center = np.zeros(3)
            radius = 5.0
        
        return {"translate": -center, "radius": radius}
    
    def _load_image(self, image_path, resize=(640, 480)):
        """이미지 로드"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            if resize:
                image = cv2.resize(image, resize)
            
            return image.astype(np.float32)
        except:
            return None


# dataset_readers.py의 기존 클래스들 (CameraInfo, SceneInfo 등)은 그대로 유지

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def _emergency_fallback(images_folder, max_images):
    """완전 실패시 비상 fallback"""
    
    # 이미지 수집
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(list(Path(images_folder).glob(ext)))
    
    image_paths.sort()
    image_paths = image_paths[:max_images]
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    print(f"Emergency fallback: {len(image_paths)} images")
    
    # 매우 간단한 카메라 배치
    cam_infos = []
    for i, image_path in enumerate(image_paths):
        # 이미지 크기
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 1920, 1080
        
        # 일직선 배치
        R = np.eye(3, dtype=np.float32)
        T = np.array([0, 0, -i * 0.5], dtype=np.float32)
        
        # 기본 FOV
        FovX = FovY = np.pi / 3  # 60도
        
        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=str(image_path),
            image_name=image_path.name,
            width=width,
            height=height,
            depth_params=None,
            depth_path="",
            is_test=(i % 10 == 0)
        )
        cam_infos.append(cam_info)
    
    # 기본 포인트 클라우드
    n_points = 1000
    points = np.random.randn(n_points, 3).astype(np.float32)
    colors = np.random.rand(n_points, 3).astype(np.float32)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # 분할
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization={"translate": np.zeros(3), "radius": 3.0},
        ply_path="",
        is_nerf_synthetic=False
    )
    
def readSuperGlueSceneInfo(path, images, eval, train_test_exp=False, llffhold=8, 
                          superglue_config="outdoor", max_images=100):
    """SuperGlue 기반 SceneInfo 생성 (완전 안전 버전)"""
    
    print("=== SuperGlue Scene Loader ===")
    print(f"SuperGlue config: {superglue_config}")
    print(f"Max images: {max_images}")
    
    # 이미지 디렉토리 확인
    images_folder = Path(path) / (images if images else "images")
    if not images_folder.exists():
        images_folder = Path(path)
    
    print(f"Image folder: {images_folder}")
    
    # 이미지 수집
    image_paths = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    for ext in extensions:
        image_paths.extend(list(images_folder.glob(ext)))
    
    image_paths.sort()
    image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    # SuperGlue 시도
    try:
        # 동적 import 시도
        import sys
        import os
        
        # 가능한 경로들 시도
        possible_paths = [
            Path(__file__).parent.parent / "models",
            Path(__file__).parent.parent / "Superglue" / "models",
            Path(__file__).parent.parent / "SuperGlue" / "models",
        ]
        
        for p in possible_paths:
            if p.exists() and (p / "matching.py").exists():
                sys.path.insert(0, str(p.parent))
                break
        
        from models.matching import Matching
        from models.utils import frame2tensor
        
        print("✅ SuperGlue modules loaded successfully")
        
        # SuperGlue 설정
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.001,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': superglue_config,
                'sinkhorn_iterations': 20,
                'match_threshold': 0.1,
            }
        }
        
        # SuperGlue 초기화
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        matching = Matching(config).eval().to(device)
        
        print(f"SuperGlue initialized on {device}")
        
        # 간단한 SuperGlue 파이프라인
        return _run_simple_superglue_pipeline(image_paths, matching, device, frame2tensor)
        
    except Exception as e:
        print(f"❌ SuperGlue failed: {e}")
        print("Using simple circular camera arrangement...")
        
        # Fallback: 간단한 원형 배치
        return _create_simple_camera_arrangement(image_paths)


def _run_simple_superglue_pipeline(image_paths, matching, device, frame2tensor):
    """간소화된 SuperGlue 파이프라인"""
    
    print("Running SuperGlue pipeline...")
    
    # 최대 20장만 처리 (시간 절약)
    process_paths = image_paths[:min(20, len(image_paths))]
    
    # 1. 특징점 추출
    features = {}
    print("Extracting features...")
    
    for i, image_path in enumerate(process_paths):
        try:
            # 이미지 로드
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            image = cv2.resize(image, (640, 480))
            inp = frame2tensor(image.astype(np.float32), device)
            
            # SuperPoint 특징점 추출
            with torch.no_grad():
                pred = matching.superpoint({'image': inp})
            
            features[i] = {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(),
                'image_path': str(image_path),
                'image_size': (480, 640)  # H, W
            }
            
        except Exception as e:
            print(f"  Failed to process {image_path.name}: {e}")
            continue
    
    print(f"Extracted features from {len(features)} images")
    
    # 2. 간단한 매칭 (순차적)
    matches = {}
    for i in range(len(features) - 1):
        if i not in features or (i+1) not in features:
            continue
            
        try:
            # SuperGlue 매칭
            feat_i = features[i]
            feat_j = features[i+1]
            
            data = {
                'keypoints0': torch.from_numpy(feat_i['keypoints']).unsqueeze(0).to(device),
                'keypoints1': torch.from_numpy(feat_j['keypoints']).unsqueeze(0).to(device),
                'descriptors0': torch.from_numpy(feat_i['descriptors']).unsqueeze(0).to(device),
                'descriptors1': torch.from_numpy(feat_j['descriptors']).unsqueeze(0).to(device),
                'image0': torch.zeros(1, 1, 480, 640).to(device),
                'image1': torch.zeros(1, 1, 480, 640).to(device),
            }
            
            with torch.no_grad():
                pred = matching.superglue(data)
            
            # 매칭 결과
            match_indices = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
            valid = match_indices > -1
            good_matches = []
            
            for idx in np.where(valid)[0]:
                match_idx = match_indices[idx]
                conf = confidence[idx]
                if conf > 0.3:
                    good_matches.append((idx, match_idx, conf))
            
            if len(good_matches) > 10:
                matches[(i, i+1)] = good_matches
                
        except Exception as e:
            print(f"  Matching failed for pair {i}-{i+1}: {e}")
            continue
    
    print(f"Found {len(matches)} good image pairs")
    
    # 3. 간단한 포즈 추정
    cameras = {}
    
    # 첫 번째 카메라를 원점으로
    cameras[0] = {
        'R': np.eye(3, dtype=np.float32),
        'T': np.zeros(3, dtype=np.float32),
        'K': _estimate_camera_intrinsics(640, 480)
    }
    
    # 순차적 포즈 추정
    for i in range(1, len(features)):
        if (i-1, i) in matches:
            # Essential Matrix로 포즈 추정 시도
            R, T = _estimate_pose_from_matches(
                features[i-1], features[i], matches[(i-1, i)], 
                cameras[i-1]['K'], _estimate_camera_intrinsics(640, 480)
            )
            
            if R is not None:
                cameras[i] = {
                    'R': R,
                    'T': T,
                    'K': _estimate_camera_intrinsics(640, 480)
                }
                continue
        
        # 실패시 기본 배치
        angle = i * 0.3
        cameras[i] = {
            'R': np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32),
            'T': np.array([4*np.sin(angle), 0, 4*(1-np.cos(angle))], dtype=np.float32),
            'K': _estimate_camera_intrinsics(640, 480)
        }
    
    print(f"Estimated poses for {len(cameras)} cameras")
    
    # 4. CameraInfo 생성
    cam_infos = []
    
    for cam_id in sorted(cameras.keys()):
        if cam_id >= len(image_paths):
            break
            
        cam = cameras[cam_id]
        image_path = image_paths[cam_id]
        
        # 실제 이미지 크기 확인
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 640, 480
        
        # FoV 계산
        K = cam['K']
        focal_x, focal_y = K[0, 0], K[1, 1]
        FovX = 2 * np.arctan(width / (2 * focal_x))
        FovY = 2 * np.arctan(height / (2 * focal_y))
        
        cam_info = CameraInfo(
            uid=cam_id,
            R=cam['R'],
            T=cam['T'],
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=str(image_path),
            image_name=image_path.name,
            width=width,
            height=height,
            depth_params=None,
            depth_path="",
            is_test=(cam_id % 8 == 0)
        )
        cam_infos.append(cam_info)
    
    # 나머지 이미지들은 간단한 배치로
    for i in range(len(cameras), len(image_paths)):
        image_path = image_paths[i]
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 640, 480
        
        # 간단한 배치
        angle = i * 0.2
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
        T = np.array([3*np.sin(angle), 0, 3*(1-np.cos(angle))], dtype=np.float32)
        
        focal = max(width, height) * 0.8
        FovX = 2 * np.arctan(width / (2 * focal))
        FovY = 2 * np.arctan(height / (2 * focal))
        
        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=str(image_path),
            image_name=image_path.name,
            width=width,
            height=height,
            depth_params=None,
            depth_path="",
            is_test=(i % 8 == 0)
        )
        cam_infos.append(cam_info)
    
    # 포인트 클라우드 생성
    n_points = 8000
    points = np.random.randn(n_points, 3).astype(np.float32) * 1.5
    colors = np.random.rand(n_points, 3).astype(np.float32)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # 학습/테스트 분할
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화
    nerf_norm = _compute_scene_normalization(train_cams)
    
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path="",
        is_nerf_synthetic=False
    )


def _estimate_camera_intrinsics(width, height):
    """카메라 내부 파라미터 추정"""
    focal = max(width, height) * 0.8
    return np.array([
        [focal, 0, width/2],
        [0, focal, height/2],
        [0, 0, 1]
    ], dtype=np.float32)


def _estimate_pose_from_matches(feat_i, feat_j, matches, K_i, K_j):
    """매칭에서 상대 포즈 추정"""
    try:
        if len(matches) < 8:
            return None, None
        
        kpts_i = feat_i['keypoints']
        kpts_j = feat_j['keypoints']
        
        pts_i = np.array([kpts_i[idx_i] for idx_i, _, conf in matches if conf > 0.4])
        pts_j = np.array([kpts_j[idx_j] for _, idx_j, conf in matches if conf > 0.4])
        
        if len(pts_i) < 8:
            return None, None
        
        # Essential Matrix 추정
        E, mask = cv2.findEssentialMat(pts_i, pts_j, K_i, 
                                       method=cv2.RANSAC, 
                                       prob=0.999, threshold=1.0)
        
        if E is not None:
            _, R, T, _ = cv2.recoverPose(E, pts_i, pts_j, K_i)
            return R, T.flatten()
            
    except Exception as e:
        print(f"    Pose estimation failed: {e}")
    
    return None, None


def _create_simple_camera_arrangement(image_paths):
    """완전 fallback: 간단한 원형 카메라 배치"""
    
    print("Creating simple circular camera arrangement...")
    
    cam_infos = []
    for i, image_path in enumerate(image_paths):
        # 이미지 크기 확인
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 1920, 1080
        
        # 원형 배치
        angle = (i / len(image_paths)) * 2 * np.pi
        radius = 5.0
        
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        
        T = np.array([
            radius * np.sin(angle),
            0,
            radius * (1 - np.cos(angle))
        ], dtype=np.float32)
        
        # FOV 계산
        focal = max(width, height) * 0.8
        FovX = 2 * np.arctan(width / (2 * focal))
        FovY = 2 * np.arctan(height / (2 * focal))
        
        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=str(image_path),
            image_name=image_path.name,
            width=width,
            height=height,
            depth_params=None,
            depth_path="",
            is_test=(i % 8 == 0)
        )
        cam_infos.append(cam_info)
    
    # 기본 포인트 클라우드
    n_points = 10000
    points = np.random.randn(n_points, 3).astype(np.float32) * 2
    colors = np.random.rand(n_points, 3).astype(np.float32)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # 학습/테스트 분할
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화
    nerf_norm = _compute_scene_normalization(train_cams)
    
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path="",
        is_nerf_synthetic=False
    )


def _compute_scene_normalization(cam_infos):
    """장면 정규화 파라미터 계산"""
    try:
        cam_centers = []
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        if cam_centers:
            cam_centers = np.hstack(cam_centers)
            center = np.mean(cam_centers, axis=1, keepdims=True).flatten()
            distances = np.linalg.norm(cam_centers - center.reshape(-1, 1), axis=0)
            radius = np.max(distances) * 1.1
        else:
            center = np.zeros(3)
            radius = 5.0
        
        return {"translate": -center, "radius": radius}
        
    except:
        return {"translate": np.zeros(3), "radius": 5.0}


sceneLoadTypeCallbacks = {
    "SuperGlue": readSuperGlueSceneInfo
}

# sceneLoadTypeCallbacks에 추가
sceneLoadTypeCallbacks["SuperGlue"] = readSuperGlueSceneInfo