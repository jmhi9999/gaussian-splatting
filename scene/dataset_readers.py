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

# 기존 타입들
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


# SuperGlue 모듈 경로 수정
def import_superglue_pipeline():
    """SuperGlue 파이프라인 동적 import"""
    try:
        # 현재 디렉토리에서 SuperGlue 경로 찾기
        current_dir = Path(__file__).parent.parent  # gaussian-splatting 루트
        
        # SuperGlue 경로들
        superglue_paths = [
            current_dir / "Superglue",
            current_dir / "SuperGlue", 
            current_dir
        ]
        
        for path in superglue_paths:
            complete_sfm_file = path / "complete_superglue_sfm.py"
            if complete_sfm_file.exists():
                # 해당 경로를 sys.path에 추가
                sys.path.insert(0, str(path))
                
                # 모듈 import
                from complete_superglue_sfm import SuperGlue3DGSPipeline
                print(f"✓ SuperGlue pipeline imported from {path}")
                return SuperGlue3DGSPipeline
        
        print("✗ SuperGlue pipeline not found")
        return None
        
    except ImportError as e:
        print(f"✗ SuperGlue import failed: {e}")
        return None

def import_superglue_colmap_hybrid():
    """SuperGlue + COLMAP 하이브리드 파이프라인 동적 import"""
    try:
        # 현재 디렉토리에서 SuperGlue 경로 찾기
        current_dir = Path(__file__).parent.parent  # gaussian-splatting 루트
        
        # SuperGlue 경로들
        superglue_paths = [
            current_dir / "Superglue",
            current_dir / "SuperGlue", 
            current_dir
        ]
        
        for path in superglue_paths:
            hybrid_file = path / "superglue_colmap_hybrid.py"
            if hybrid_file.exists():
                # 해당 경로를 sys.path에 추가
                sys.path.insert(0, str(path))
                
                # 모듈 import
                from superglue_colmap_hybrid import SuperGlueCOLMAPHybrid
                print(f"✓ SuperGlue + COLMAP hybrid pipeline imported from {path}")
                return SuperGlueCOLMAPHybrid
        
        print("✗ SuperGlue + COLMAP hybrid pipeline not found")
        return None
        
    except ImportError as e:
        print(f"✗ SuperGlue + COLMAP hybrid import failed: {e}")
        return None

# SuperGlue 파이프라인 import 시도
SuperGlue3DGSPipeline = import_superglue_pipeline()
SUPERGLUE_PIPELINE_AVAILABLE = (SuperGlue3DGSPipeline is not None)

# SuperGlue + COLMAP 하이브리드 파이프라인 import 시도
SuperGlueCOLMAPHybrid = import_superglue_colmap_hybrid()
SUPERGLUE_COLMAP_HYBRID_AVAILABLE = (SuperGlueCOLMAPHybrid is not None)

# Hloc 파이프라인 import 시도
def import_hloc_pipeline():
    """Hloc 파이프라인 동적 import"""
    try:
        # 현재 디렉토리에서 SuperGlue 경로 찾기
        current_dir = Path(__file__).parent.parent  # gaussian-splatting 루트
        
        # SuperGlue 경로들
        superglue_paths = [
            current_dir / "Superglue",
            current_dir / "SuperGlue", 
            current_dir
        ]
        
        for path in superglue_paths:
            hloc_file = path / "hloc_pipeline.py"
            if hloc_file.exists():
                # 해당 경로를 sys.path에 추가
                sys.path.insert(0, str(path))
                
                # 모듈 import
                from hloc_pipeline import readHlocSceneInfo
                print(f"✓ Hloc pipeline imported from {path}")
                return readHlocSceneInfo
        
        print("✗ Hloc pipeline not found")
        return None
        
    except ImportError as e:
        print(f"✗ Hloc import failed: {e}")
        return None

# Hloc 파이프라인 import 시도
readHlocSceneInfo = import_hloc_pipeline()
HLOC_PIPELINE_AVAILABLE = (readHlocSceneInfo is not None)


def readSuperGlueSceneInfo(path, images="images", eval=False, train_test_exp=False, 
                          llffhold=8, superglue_config="outdoor", max_images=100):
    """SuperGlue 완전 파이프라인으로 SceneInfo 생성"""
    
    print("\n" + "="*60)
    print("           SUPERGLUE + 3DGS PIPELINE")
    print("="*60)
    
    print(f"📁 Source path: {path}")
    print(f"🖼️  Images folder: {images}")
    print(f"🔧 SuperGlue config: {superglue_config}")
    print(f"📊 Max images: {max_images}")
    print(f"🚀 Pipeline available: {SUPERGLUE_PIPELINE_AVAILABLE}")
    
    # 이미지 디렉토리 경로
    images_folder = Path(path) / images
    if not images_folder.exists():
        # fallback 경로들 시도
        fallback_paths = [Path(path), Path(path) / "input"]
        for fallback in fallback_paths:
            if fallback.exists():
                images_folder = fallback
                break
    
    print(f"📂 Using images folder: {images_folder}")
    
    if SUPERGLUE_PIPELINE_AVAILABLE:
        try:
            print("\n🔥 STARTING SUPERGLUE PIPELINE...")
            
            # SuperGlue 설정
            config = {
                'superpoint': {
                    'nms_radius': 3,
                    'keypoint_threshold': 0.003,
                    'max_keypoints': 4096
                },
                'superglue': {
                    'weights': superglue_config,
                    'sinkhorn_iterations': 30,
                    'match_threshold': 0.15,
                }
            }
            
            # 출력 디렉토리
            output_folder = Path(path) / "superglue_sfm_output"
            
            # 파이프라인 실행
            pipeline = SuperGlue3DGSPipeline(config)
            
            print(f"🎯 Calling process_images_to_3dgs...")
            print(f"   - Input: {images_folder}")
            print(f"   - Output: {output_folder}")
            print(f"   - Max images: {max_images}")
            
            scene_info = pipeline.process_images_to_3dgs(
                image_dir=str(images_folder),
                output_dir=str(output_folder),
                max_images=max_images
            )
            
            print("\n🎉 SUPERGLUE PIPELINE SUCCESS!")
            print(f"✓ Training cameras: {len(scene_info.train_cameras)}")
            print(f"✓ Test cameras: {len(scene_info.test_cameras)}")
            print(f"✓ Point cloud: {len(scene_info.point_cloud.points)} points")
            print(f"✓ Scene radius: {scene_info.nerf_normalization['radius']:.3f}")
            
            return scene_info
            
        except Exception as e:
            print(f"\n❌ SUPERGLUE PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("\n⚠️  Falling back to simple camera arrangement...")
            
    else:
        print("\n⚠️  SuperGlue pipeline not available, using fallback...")
    
    # Fallback: 간단한 카메라 배치
    return _create_fallback_scene_info(images_folder, max_images)

def readSuperGlueCOLMAPHybridSceneInfo(path, images="images", eval=False, train_test_exp=False, 
                                      llffhold=8, superglue_config="outdoor", max_images=100, colmap_exe="colmap"):
    """SuperGlue + COLMAP 하이브리드 파이프라인으로 SceneInfo 생성"""
    
    print("\n" + "="*60)
    print("    SUPERGLUE + COLMAP HYBRID PIPELINE")
    print("="*60)
    
    print(f"📁 Source path: {path}")
    print(f"🖼️  Images folder: {images}")
    print(f"🔧 SuperGlue config: {superglue_config}")
    print(f"📊 Max images: {max_images}")
    print(f"🔧 COLMAP exe: {colmap_exe}")
    print(f"🚀 Hybrid pipeline available: {SUPERGLUE_COLMAP_HYBRID_AVAILABLE}")
    
    # 이미지 디렉토리 경로
    images_folder = Path(path) / images
    if not images_folder.exists():
        # fallback 경로들 시도
        fallback_paths = [Path(path), Path(path) / "input"]
        for fallback in fallback_paths:
            if fallback.exists():
                images_folder = fallback
                break
    
    print(f"📂 Using images folder: {images_folder}")
    
    if SUPERGLUE_COLMAP_HYBRID_AVAILABLE:
        try:
            print("\n🔥 STARTING SUPERGLUE + COLMAP HYBRID PIPELINE...")
            
            # 출력 디렉토리
            output_folder = Path(path) / "superglue_colmap_hybrid_output"
            
            # 하이브리드 파이프라인 실행
            pipeline = SuperGlueCOLMAPHybrid(
                colmap_exe=colmap_exe,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            print(f"🎯 Calling process_images...")
            print(f"   - Input: {images_folder}")
            print(f"   - Output: {output_folder}")
            print(f"   - Max images: {max_images}")
            
            scene_info = pipeline.process_images(
                image_dir=str(images_folder),
                output_dir=str(output_folder),
                max_images=max_images
            )
            
            if scene_info:
                print("\n🎉 SUPERGLUE + COLMAP HYBRID PIPELINE SUCCESS!")
                print(f"✓ Training cameras: {len(scene_info.train_cameras)}")
                print(f"✓ Test cameras: {len(scene_info.test_cameras)}")
                print(f"✓ Point cloud: {len(scene_info.point_cloud.points)} points")
                print(f"✓ Scene radius: {scene_info.nerf_normalization['radius']:.3f}")
                
                return scene_info
            else:
                print("\n❌ Hybrid pipeline returned None")
                raise RuntimeError("Hybrid pipeline failed to create scene_info")
                
        except Exception as e:
            print(f"\n❌ SUPERGLUE + COLMAP HYBRID PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("\n⚠️  Falling back to simple camera arrangement...")
            
    else:
        print("\n⚠️  SuperGlue + COLMAP hybrid pipeline not available, using fallback...")
    
    # Fallback: 간단한 카메라 배치
    return _create_fallback_scene_info(images_folder, max_images)

def _create_fallback_scene_info(images_folder, max_images):
    """SuperGlue 실패시 fallback scene 생성"""
    
    print(f"\n📋 Creating fallback scene from {images_folder}")
    
    # 이미지 수집
    image_paths = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    for ext in extensions:
        image_paths.extend(list(Path(images_folder).glob(ext)))
    
    image_paths.sort()
    image_paths = image_paths[:max_images]
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    print(f"📸 Found {len(image_paths)} images")
    
    # CameraInfo 생성
    cam_infos = []
    for i, image_path in enumerate(image_paths):
        # 이미지 크기 확인
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 640, 480
        
        # 원형 배치 (더 realistic한 카메라 배치)
        angle = i * (2 * np.pi / len(image_paths))
        radius = 3.0
        
        # 카메라가 원점을 바라보도록 설정
        cam_pos = np.array([
            radius * np.cos(angle),
            0.0,  # Y는 고정
            radius * np.sin(angle)
        ], dtype=np.float32)
        
        # 원점을 바라보는 회전 행렬
        forward = -cam_pos / np.linalg.norm(cam_pos)  # 원점을 향함
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        R = np.column_stack([right, up, forward]).astype(np.float32)
        T = cam_pos
        
        # FOV 설정
        focal_length = max(width, height) * 0.8
        FovX = focal2fov(focal_length, width)
        FovY = focal2fov(focal_length, height)
        
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
            is_test=(i % 8 == 0)  # 8장마다 테스트용
        )
        cam_infos.append(cam_info)
    
    # 포인트 클라우드 생성 (원점 주변에 구형 분포)
    n_points = 5000
    
    # 구형 분포
    phi = np.random.uniform(0, 2*np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    u = np.random.uniform(0, 1, n_points)
    
    theta = np.arccos(costheta)
    r = 1.5 * np.cbrt(u)  # 구형 분포를 위한 반지름
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi) 
    z = r * np.cos(theta)
    
    points = np.column_stack([x, y, z]).astype(np.float32)
    
    # 컬러는 위치 기반으로 생성
    colors = np.abs(points).astype(np.float32)
    colors = colors / np.max(colors)  # 정규화
    
    # 법선벡터 (외향)
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # 학습/테스트 분할
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화
    cam_centers = []
    for cam in cam_infos:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])
    
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
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path="",
        is_nerf_synthetic=False
    )
    
    print(f"✓ Fallback scene created:")
    print(f"  - {len(train_cams)} training cameras")
    print(f"  - {len(test_cams)} test cameras") 
    print(f"  - {len(points)} 3D points")
    print(f"  - Scene radius: {radius:.2f}")
    
    return scene_info


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
    
    def visualize_matches(self, i, j, save_path=None):
        """매칭 결과 시각화"""
        if (i, j) not in self.matches:
            print(f"No matches found between images {i} and {j}")
            return
    
        # 이미지 로드
        img0 = cv2.imread(self.image_features[i]['image_path'])
        img1 = cv2.imread(self.image_features[j]['image_path'])
    
        # 매칭 포인트 추출
        matches = self.matches[(i, j)]
        kpts0 = self.image_features[i]['keypoints']
        kpts1 = self.image_features[j]['keypoints']
    
        # 매칭 시각화
        matched_img = cv2.drawMatches(
            img0, [cv2.KeyPoint(kpts0[m[0]][0], kpts0[m[0]][1], 1) for m in matches],
            img1, [cv2.KeyPoint(kpts1[m[1]][0], kpts1[m[1]][1], 1) for m in matches],
            [cv2.DMatch(idx, idx, 0) for idx in range(len(matches))],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    
        if save_path:
            cv2.imwrite(save_path, matched_img)
    
        return matched_img
    
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
            
            if i > 1:
                # visualize image match
                self.visualize_matches(i-1, i, save_path="output/match_viz/")
        
    
    
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
            
            self.visualize_matches(i, j, save_path="output/match_viz/")
        
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




sceneLoadTypeCallbacks = {
    "SuperGlue": readSuperGlueSceneInfo,
    "SuperGlueCOLMAPHybrid": readSuperGlueCOLMAPHybridSceneInfo,
    "Hloc": readHlocSceneInfo,
}

# sceneLoadTypeCallbacks에 추가
sceneLoadTypeCallbacks["SuperGlue"] = readSuperGlueSceneInfo
sceneLoadTypeCallbacks["SuperGlueCOLMAPHybrid"] = readSuperGlueCOLMAPHybridSceneInfo

# Colmap과 Blender 로더도 추가 (기존 함수들이 있다면)
try:
    from scene.colmap_loader import readColmapSceneInfo
    sceneLoadTypeCallbacks["Colmap"] = readColmapSceneInfo
except ImportError:
    print("Warning: Colmap loader not available")

try:
    from scene.blender_loader import readBlenderSceneInfo
    sceneLoadTypeCallbacks["Blender"] = readBlenderSceneInfo
except ImportError:
    print("Warning: Blender loader not available")

def test_superglue_connection():
    """SuperGlue 연결 테스트"""
    print("Testing SuperGlue connection...")
    print(f"SuperGlue3DGSPipeline available: {SUPERGLUE_PIPELINE_AVAILABLE}")
    
    if SUPERGLUE_PIPELINE_AVAILABLE:
        try:
            config = {
                'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024},
                'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
            }
            pipeline = SuperGlue3DGSPipeline(config)
            print("✓ SuperGlue pipeline instantiated successfully!")
            return True
        except Exception as e:
            print(f"✗ SuperGlue pipeline test failed: {e}")
            return False
    else:
        print("✗ SuperGlue pipeline not available")
        return False

if __name__ == "__main__":
    test_superglue_connection()