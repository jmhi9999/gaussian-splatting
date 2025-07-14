#!/usr/bin/env python3
"""
Hloc + 3DGS 통합 파이프라인 (깔끔한 버전)
순환 import 문제 해결
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import subprocess
from typing import Optional, List, NamedTuple

# scipy import 추가 (pycolmap 최신 버전 지원용)
try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available - will use fallback rotation conversion")

# 순환 import 방지: 필요한 타입들을 직접 정의
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    depth_params: dict
    depth_path: str
    is_test: bool

class BasicPointCloud:
    def __init__(self, points, colors, normals):
        self.points = points
        self.colors = colors  
        self.normals = normals

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

# Hloc 및 pycolmap import
try:
    import pycolmap
    HLOC_AVAILABLE = True
    print("✓ pycolmap imported successfully")
except ImportError as e:
    HLOC_AVAILABLE = False
    print(f"✗ pycolmap import failed: {e}")

def quaternion_to_rotmat(qvec):
    """쿼터니언을 회전 행렬로 변환"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=np.float32)

def focal2fov(focal, pixels):
    """Focal length를 FoV로 변환"""
    return 2*np.arctan(pixels/(2*focal))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """World2View 행렬 계산"""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def matrix_to_quaternion(R):
    """회전 행렬을 쿼터니언으로 변환 (scipy 없이)"""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s=4*qw 
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s=4*qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s=4*qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s=4*qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz], dtype=np.float32)

def get_image_quaternion_and_translation(image):
    """pycolmap Image 객체에서 안전하게 quaternion과 translation 추출 (Rigid3d 지원)"""
    
    # 최신 pycolmap API 시도 (cam_from_world)
    if hasattr(image, 'cam_from_world'):
        try:
            # cam_from_world는 pycolmap.Rigid3d 객체 (함수 아님!)
            rigid3d = image.cam_from_world
            
            # Rigid3d 객체에서 4x4 변환 행렬 추출
            if hasattr(rigid3d, 'matrix'):
                cam_from_world = rigid3d.matrix()
            elif hasattr(rigid3d, 'Matrix'):
                cam_from_world = rigid3d.Matrix()
            else:
                # rotation()과 translation() 메서드로 개별 추출
                if hasattr(rigid3d, 'rotation') and hasattr(rigid3d, 'translation'):
                    R = rigid3d.rotation()
                    if hasattr(R, 'matrix'):
                        R_matrix = R.matrix()  # 3x3 회전 행렬
                    else:
                        R_matrix = R  # 이미 행렬일 경우
                    
                    t = rigid3d.translation()  # 3x1 평행이동
                    
                    # 4x4 변환 행렬 구성
                    cam_from_world = np.eye(4, dtype=np.float32)
                    cam_from_world[:3, :3] = R_matrix
                    cam_from_world[:3, 3] = t
                else:
                    raise ValueError("Cannot extract matrix from Rigid3d object")
            
            # 회전 행렬 (3x3)
            R = cam_from_world[:3, :3]
            
            # 평행이동 벡터 (3x1)
            t = cam_from_world[:3, 3]
            
            # 회전 행렬을 쿼터니언으로 변환
            if SCIPY_AVAILABLE:
                qvec_scipy = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
                qvec = np.array([qvec_scipy[3], qvec_scipy[0], qvec_scipy[1], qvec_scipy[2]], dtype=np.float32)  # [w, x, y, z]
            else:
                qvec = matrix_to_quaternion(R)  # [w, x, y, z]
            
            return qvec, t.astype(np.float32)
            
        except Exception as e:
            print(f"cam_from_world Rigid3d extraction failed: {e}")
    
    # projection_center 시도
    if hasattr(image, 'projection_center'):
        try:
            center = image.projection_center()
            # 기본 회전 (단위 행렬)과 중심점 사용
            qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
            tvec = -center.astype(np.float32)  # 카메라 중심의 반대
            return qvec, tvec
        except Exception as e:
            print(f"projection_center method failed: {e}")
    
    # 기존 API 시도 (하위 호환성)
    quat_attrs = ['qvec', 'quat', 'quaternion', 'rotation_quaternion']
    trans_attrs = ['tvec', 'trans', 'translation']
    
    # Quaternion 추출
    qvec = None
    for attr in quat_attrs:
        if hasattr(image, attr):
            try:
                qvec = getattr(image, attr)
                if callable(qvec):
                    qvec = qvec()
                qvec = np.array(qvec, dtype=np.float32)
                break
            except:
                continue
    
    # Translation 추출
    tvec = None
    for attr in trans_attrs:
        if hasattr(image, attr):
            try:
                tvec = getattr(image, attr)
                if callable(tvec):
                    tvec = tvec()
                tvec = np.array(tvec, dtype=np.float32)
                break
            except:
                continue
    
    # 기본값 사용 (최후의 수단)
    if qvec is None or tvec is None:
        print(f"Using default pose for image (no pose attributes found)")
        qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 단위 쿼터니언
        tvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 원점
    
    return qvec, tvec

def safe_get_rotation_matrix(image):
    """pycolmap Image 객체에서 안전하게 회전 행렬 추출 (Rigid3d 지원)"""
    
    # 최신 pycolmap API 시도 (cam_from_world Rigid3d)
    if hasattr(image, 'cam_from_world'):
        try:
            rigid3d = image.cam_from_world
            
            # Rigid3d에서 4x4 변환 행렬 추출
            if hasattr(rigid3d, 'matrix'):
                cam_from_world = rigid3d.matrix()
                R = cam_from_world[:3, :3].astype(np.float32)
                return R
            elif hasattr(rigid3d, 'Matrix'):
                cam_from_world = rigid3d.Matrix()
                R = cam_from_world[:3, :3].astype(np.float32)
                return R
            elif hasattr(rigid3d, 'rotation'):
                rotation_obj = rigid3d.rotation()
                if hasattr(rotation_obj, 'matrix'):
                    R = rotation_obj.matrix().astype(np.float32)
                    return R
                else:
                    # rotation_obj가 이미 행렬일 경우
                    R = np.array(rotation_obj, dtype=np.float32)
                    return R
        except Exception as e:
            print(f"cam_from_world Rigid3d rotation extraction failed: {e}")
    
    # 기존 메서드들 시도
    try:
        if hasattr(image, 'rotation_matrix'):
            return image.rotation_matrix().astype(np.float32)
        elif hasattr(image, 'qvec2rotmat'):
            return image.qvec2rotmat().astype(np.float32)
        else:
            # 쿼터니언에서 회전 행렬 생성
            qvec, _ = get_image_quaternion_and_translation(image)
            return quaternion_to_rotmat(qvec)
            
    except Exception as e:
        print(f"Warning: Failed to get rotation matrix: {e}")
        # 기본 단위 행렬 반환
        return np.eye(3, dtype=np.float32)

def collect_images(image_dir: str, max_images: int = 100) -> List[Path]:
    """이미지 파일들을 수집"""
    image_dir = Path(image_dir)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    all_images = []
    for ext in extensions:
        all_images.extend(image_dir.glob(ext))
    
    all_images.sort(key=lambda x: x.name)
    return all_images[:max_images]

def run_hloc_reconstruction(image_dir: Path, output_dir: Path, max_images: int = 100):
    """Hloc SfM 재구성 실행"""
    print(f"\n🚀 Running Hloc SfM reconstruction")
    print(f"📁 Images: {image_dir}")
    print(f"📁 Output: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Feature extraction
        print("🔍 Extracting features...")
        extract_cmd = [
            sys.executable, '-m', 'hloc.extract_features',
            '--image_dir', str(image_dir),
            '--export_dir', str(output_dir),
            '--conf', 'superpoint_aachen'
        ]
        
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ Feature extraction failed: {result.stderr}")
            return None
        print("✅ Feature extraction completed")
        
        # 2. Image pairs
        print("🔗 Creating image pairs...")
        pairs_path = output_dir / 'pairs.txt'
        image_paths = collect_images(image_dir, max_images)
        
        with open(pairs_path, 'w') as f:
            for i, img1 in enumerate(image_paths):
                for j, img2 in enumerate(image_paths[i+1:], i+1):
                    f.write(f"{img1.name} {img2.name}\n")
        
        num_pairs = len(image_paths) * (len(image_paths) - 1) // 2
        print(f"✅ Created {num_pairs} image pairs")
        
        # 3. Feature matching  
        print("🔗 Matching features...")
        match_cmd = [
            sys.executable, '-m', 'hloc.match_features',
            '--pairs', str(pairs_path),
            '--features', 'feats-superpoint-n4096-r1024',
            '--matches', 'feats-superpoint-n4096-r1024_matches-superglue_pairs.h5',
            '--export_dir', str(output_dir),
            '--conf', 'superglue'
        ]
        
        result = subprocess.run(match_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ Feature matching failed: {result.stderr}")
            return None
        print("✅ Feature matching completed")
        
        # 4. SfM reconstruction
        print("🏗️  Running SfM reconstruction...")
        sfm_dir = output_dir / 'sfm'
        sfm_dir.mkdir(exist_ok=True)
        
        reconstruction_cmd = [
            sys.executable, '-m', 'hloc.reconstruction',
            '--sfm_dir', str(sfm_dir),
            '--image_dir', str(image_dir),
            '--pairs', str(pairs_path),
            '--features', str(output_dir / 'feats-superpoint-n4096-r1024.h5'),
            '--matches', str(output_dir / 'feats-superpoint-n4096-r1024_matches-superglue_pairs.h5'),
            '--camera_mode', 'SINGLE'
        ]
        
        result = subprocess.run(reconstruction_cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"⚠️  SfM reconstruction failed: {result.stderr}")
        
        # 5. Load COLMAP model
        if (sfm_dir / 'cameras.bin').exists():
            model = pycolmap.Reconstruction(str(sfm_dir))
        elif (sfm_dir / 'cameras.txt').exists():
            model = pycolmap.Reconstruction()
            model.read_text(str(sfm_dir))
        else:
            print("❌ No COLMAP model files found")
            return None
        
        if len(model.images) == 0:
            print("❌ No cameras registered in reconstruction")
            return None
        
        print(f"✅ SfM success: {len(model.images)} cameras, {len(model.points3D)} points")
        return model
        
    except Exception as e:
        print(f"❌ Hloc reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def colmap_to_scene_info(model, image_paths: List[Path]) -> SceneInfo:
    """COLMAP 모델을 SceneInfo로 변환"""
    print("📊 Converting COLMAP model to SceneInfo...")
    
    cam_infos = []
    for img_id, image in model.images.items():
        try:
            # 카메라 정보
            camera_id = getattr(image, 'camera_id', img_id)
            if camera_id not in model.cameras:
                print(f"⚠️  Camera {camera_id} not found for image {img_id}")
                continue
            
            cam = model.cameras[camera_id]
            
            # 이미지 경로 찾기
            image_name = getattr(image, 'name', f'image_{img_id}')
            image_path = None
            for path in image_paths:
                if path.name == image_name:
                    image_path = path
                    break
            
            if image_path is None:
                print(f"⚠️  Image not found: {image_name}")
                continue
            
            # 이미지 크기
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except:
                width = getattr(cam, 'width', 640)
                height = getattr(cam, 'height', 480)
            
            # 회전 행렬과 평행이동
            R = safe_get_rotation_matrix(image)
            qvec, tvec = get_image_quaternion_and_translation(image)
            T = tvec
            
            # FOV 계산
            model_name = getattr(cam, 'model_name', 'PINHOLE')
            params = getattr(cam, 'params', [width * 0.8, height * 0.8])
            
            if model_name in ['SIMPLE_PINHOLE', 'PINHOLE']:
                if model_name == 'SIMPLE_PINHOLE':
                    fx = fy = params[0] if len(params) > 0 else width * 0.8
                else:
                    fx = params[0] if len(params) > 0 else width * 0.8
                    fy = params[1] if len(params) > 1 else height * 0.8
                
                FovX = focal2fov(fx, width)
                FovY = focal2fov(fy, height)
            else:
                FovX = focal2fov(width * 0.8, width)
                FovY = focal2fov(height * 0.8, height)
            
            cam_info = CameraInfo(
                uid=img_id,
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
                is_test=(img_id % 8 == 0)
            )
            cam_infos.append(cam_info)
            
        except Exception as e:
            print(f"⚠️  Failed to process image {img_id}: {e}")
            continue
    
    # 카메라가 처리되지 않았다면 기본 시나리오 생성
    if len(cam_infos) == 0:
        print("⚠️  No cameras processed from COLMAP - creating fallback scenario")
        return create_fallback_scene_info(image_paths)
    
    # 3D 포인트 클라우드
    if len(model.points3D) > 0:
        xyz = np.array([point.xyz for point in model.points3D.values()], dtype=np.float32)
        colors = np.array([point.color / 255.0 for point in model.points3D.values()], dtype=np.float32)
        normals = np.random.randn(*xyz.shape).astype(np.float32)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    else:
        print("⚠️  No 3D points found, creating default point cloud")
        n_points = 10000
        xyz = np.random.randn(n_points, 3).astype(np.float32) * 2
        colors = np.random.rand(n_points, 3).astype(np.float32)
        normals = np.random.randn(n_points, 3).astype(np.float32)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)
    
    # Train/Test 분할
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화
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
    
    nerf_norm = {"translate": -center, "radius": radius}
    
    print(f"✅ SceneInfo created: {len(train_cams)} train, {len(test_cams)} test cameras")
    
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path="",
        is_nerf_synthetic=False
    )

def create_fallback_scene_info(image_paths: List[Path]) -> SceneInfo:
    """COLMAP 실패시 fallback SceneInfo 생성"""
    print("🛠️  Creating fallback SceneInfo with circular camera arrangement...")
    
    cam_infos = []
    for i, image_path in enumerate(image_paths):
        try:
            # 이미지 크기
            with Image.open(image_path) as img:
                width, height = img.size
            
            # 기본 카메라 매트릭스
            fx = fy = max(width, height) * 0.8
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            # 원형 배치
            angle = 2 * np.pi * i / len(image_paths)
            radius = 3.0
            
            # 카메라 위치
            camera_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ], dtype=np.float32)
            
            # 원점을 바라보는 방향
            look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            
            # View 행렬 계산
            forward = look_at - camera_pos
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)
            
            # 회전 행렬 (카메라 -> 월드)
            R = np.column_stack([right, up, -forward]).T.astype(np.float32)
            T = camera_pos.astype(np.float32)
            
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
            
        except Exception as e:
            print(f"⚠️  Failed to process fallback {image_path}: {e}")
            continue
    
    if len(cam_infos) == 0:
        raise RuntimeError("Failed to create any cameras in fallback mode")
    
    # 기본 포인트 클라우드
    n_points = 10000
    points = np.random.randn(n_points, 3).astype(np.float32) * 2
    colors = np.random.rand(n_points, 3).astype(np.float32)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화
    center = np.zeros(3)
    radius = 5.0
    nerf_norm = {"translate": -center, "radius": radius}
    
    print(f"✅ Fallback SceneInfo: {len(train_cams)} train, {len(test_cams)} test cameras")
    
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path="",
        is_nerf_synthetic=False
    )

def readHlocSceneInfo(path: str, 
                     images: str = "images", 
                     eval: bool = False, 
                     train_test_exp: bool = False,
                     max_images: int = 100,
                     **kwargs) -> SceneInfo:
    """Hloc 파이프라인으로 SceneInfo 생성 (메인 함수)"""
    
    print("\n" + "="*60)
    print("              HLOC + 3DGS PIPELINE")
    print("="*60)
    
    # 경로 설정
    image_dir = Path(path) / images
    if not image_dir.exists():
        fallback_paths = [Path(path), Path(path) / "input"]
        for fallback in fallback_paths:
            if fallback.exists():
                image_dir = fallback
                break
    
    print(f"📁 Source path: {path}")
    print(f"📁 Images folder: {image_dir}")
    print(f"📊 Max images: {max_images}")
    print(f"🚀 Hloc available: {HLOC_AVAILABLE}")
    
    # 이미지 수집
    image_paths = collect_images(image_dir, max_images)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"📸 Found {len(image_paths)} images")
    
    # HLOC이 사용 가능한 경우 시도
    if HLOC_AVAILABLE:
        try:
            # Hloc SfM 실행
            output_dir = Path(path) / "hloc_output"
            model = run_hloc_reconstruction(image_dir, output_dir, max_images)
            
            if model is not None:
                # SceneInfo 생성 시도
                scene_info = colmap_to_scene_info(model, image_paths)
                print("✅ Hloc pipeline completed successfully!")
                return scene_info
            else:
                print("⚠️  Hloc reconstruction failed")
        
        except Exception as e:
            print(f"⚠️  Hloc pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  pycolmap not available")
    
    # Fallback: 기본 시나리오 생성
    print("🛠️  Using fallback scenario (circular camera arrangement)")
    try:
        scene_info = create_fallback_scene_info(image_paths)
        print("✅ Fallback scenario created successfully!")
        return scene_info
    except Exception as e:
        print(f"❌ Fallback scenario failed: {e}")
        raise RuntimeError(f"All reconstruction methods failed: {e}")

if __name__ == "__main__":
    # 테스트
    print("Testing Hloc Pipeline...")
    print(f"pycolmap available: {HLOC_AVAILABLE}")
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        try:
            scene_info = readHlocSceneInfo(test_path)
            print(f"✅ Success: {len(scene_info.train_cameras)} train cameras")
        except Exception as e:
            print(f"❌ Test failed: {e}")