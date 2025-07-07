# superglue_scene_reader.py
# scene/dataset_readers.py에 추가할 SuperGlue 전용 함수

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from scene.dataset_readers import CameraInfo, SceneInfo, BasicPointCloud
from utils.graphics_utils import BasicPointCloud
import cv2

# 새로운 하이브리드 파이프라인 import
from superglue_colmap_pipeline import SuperGlueColmapPipeline

def readSuperGlueSceneInfo(path, images, eval, train_test_exp=False, llffhold=8):
    """SuperGlue + COLMAP 하이브리드 파이프라인으로 scene 정보 생성"""
    
    print("=== Loading scene with SuperGlue + COLMAP Pipeline ===")
    
    # 이미지 경로 수집
    images_folder = os.path.join(path, images if images else "images")
    image_paths = []
    
    # 지원하는 이미지 확장자들
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    # SuperGlue + COLMAP 하이브리드 파이프라인 실행
    print("Running SuperGlue + COLMAP pipeline...")
    
    # 임시 출력 디렉토리
    temp_output_dir = os.path.join(path, "temp_sfm_output")
    
    # 파이프라인 실행
    pipeline = SuperGlueColmapPipeline()
    try:
        success = pipeline.run_pipeline(images_folder, temp_output_dir, max_images=100)
        if not success:
            raise RuntimeError("SuperGlue + COLMAP pipeline failed")
        
        # 결과 로딩
        cameras = pipeline.load_results()
        print(f"Successfully loaded {len(cameras)} cameras from COLMAP results")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        print("Falling back to simple pose estimation...")
        cameras = _fallback_simple_poses(image_paths)
    
    # CameraInfo 리스트 생성
    cam_infos = []
    for i, camera in enumerate(cameras):
        # 이미지 정보 가져오기
        image_path = image_paths[i] if i < len(image_paths) else image_paths[0]
        image = Image.open(image_path)
        width, height = image.size
        
        # 카메라 파라미터 추출
        R = camera.rotation().toRotationMatrix().numpy()
        T = camera.position().numpy()
        
        # FOV 계산
        focal = camera.focal()
        FovY = 2 * np.arctan(height / (2 * focal))
        FovX = 2 * np.arctan(width / (2 * focal))
        
        # 테스트 이미지 분할 (evaluation용)
        is_test = False
        if eval:
            if llffhold > 0:
                is_test = (i % llffhold == 0)
        
        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T, 
            FovY=FovY,
            FovX=FovX,
            image_path=image_path,
            image_name=Path(image_path).name,
            width=width,
            height=height,
            depth_params=None,
            depth_path="",
            is_test=is_test
        )
        cam_infos.append(cam_info)
    
    # 학습/테스트 분할
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화 파라미터 계산
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # PLY 파일 경로 (COLMAP 결과에서 생성)
    ply_path = os.path.join(temp_output_dir, "sparse", "0", "points3D.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(path, "superglue_points.ply")
    
    # SceneInfo 생성
    scene_info = SceneInfo(
        point_cloud=_load_point_cloud(ply_path),
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    
    print(f"SuperGlue + COLMAP scene loaded: {len(train_cam_infos)} train, {len(test_cam_infos)} test cameras")
    return scene_info

def _fallback_simple_poses(image_paths):
    """파이프라인 실패 시 간단한 포즈 추정"""
    print("Using fallback simple pose estimation...")
    
    cameras = []
    for i, image_path in enumerate(image_paths):
        # 간단한 포즈 생성
        angle = (i / len(image_paths)) * 2 * np.pi
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        T = np.array([np.sin(angle), 0, np.cos(angle) - 1], dtype=np.float32) * 0.5
        
        # 임시 카메라 객체 생성 (실제로는 더 복잡한 구현 필요)
        cameras.append({
            'R': R,
            'T': T,
            'focal': 1000.0,  # 임시 focal length
            'width': 1920,
            'height': 1080
        })
    
    return cameras

def _load_point_cloud(ply_path):
    """PLY 파일에서 포인트 클라우드 로딩"""
    if not os.path.exists(ply_path):
        # 기본 포인트 클라우드 생성
        return _create_default_point_cloud()
    
    try:
        # PLY 파일 파싱 (간단한 버전)
        points = []
        colors = []
        
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        # 헤더 건너뛰기
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                start_idx = i + 1
                break
        
        # 포인트 데이터 파싱
        for line in lines[start_idx:]:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:  # x, y, z, r, g, b
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                    
                    points.append([x, y, z])
                    colors.append([r/255, g/255, b/255])
        
        if len(points) > 0:
            points = np.array(points, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)
            
            # 법선 벡터 (임시)
            normals = np.random.randn(len(points), 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            
            return BasicPointCloud(
                points=points,
                colors=colors,
                normals=normals
            )
    
    except Exception as e:
        print(f"Failed to load PLY file: {e}")
    
    return _create_default_point_cloud()

def _create_default_point_cloud():
    """기본 포인트 클라우드 생성"""
    num_points = 10000
    
    # 랜덤 3D 포인트들
    points = np.random.randn(num_points, 3).astype(np.float32) * 2
    
    # 랜덤 컬러
    colors = np.random.rand(num_points, 3).astype(np.float32)
    
    # 법선 벡터 (임시)
    normals = np.random.randn(num_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    return BasicPointCloud(
        points=points,
        colors=colors,
        normals=normals
    )

def getNerfppNorm(cam_infos):
    """NeRF++ 정규화 파라미터 계산"""
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_infos:
        # 카메라 중심 계산
        W2C = np.zeros((4, 4))
        W2C[:3, :3] = cam.R.transpose()
        W2C[:3, 3] = cam.T
        W2C[3, 3] = 1.0
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}