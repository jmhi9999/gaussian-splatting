# superglue_scene_reader.py
# scene/dataset_readers.py에 추가할 SuperGlue 전용 함수

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from scene.dataset_readers import CameraInfo, SceneInfo, BasicPointCloud
from utils.graphics_utils import BasicPointCloud
from superglue_matcher import SuperGlueMatcher
import cv2

def readSuperGlueSceneInfo(path, images, eval, train_test_exp=False, llffhold=8):
    """SuperGlue 기반으로 scene 정보 생성"""
    
    print("=== Loading scene with SuperGlue ===")
    
    # SuperGlue 매처 초기화
    matcher = SuperGlueMatcher()
    
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
    
    # SuperGlue로 특징점 매칭 수행
    print("Extracting features and matching with SuperGlue...")
    matching_results = matcher.match_multiple_images(image_paths)
    
    # 임시 카메라 포즈 추정 (간단한 버전)
    cameras_info = estimate_camera_poses_simple(image_paths, matching_results)
    
    # 3D 포인트 클라우드 생성 (간단한 버전)
    point_cloud = create_point_cloud_simple(matching_results, cameras_info)
    
    # CameraInfo 리스트 생성
    cam_infos = []
    for i, (image_path, cam_pose) in enumerate(zip(image_paths, cameras_info)):
        
        # 이미지 로드하여 크기 확인
        image = Image.open(image_path)
        width, height = image.size
        
        # 임시 카메라 내부 파라미터 (실제로는 calibration 필요)
        focal_length = max(width, height) * 0.7  # 대략적인 값
        FovY = 2 * np.arctan(height / (2 * focal_length))
        FovX = 2 * np.arctan(width / (2 * focal_length))
        
        # 테스트 이미지 분할 (evaluation용)
        is_test = False
        if eval:
            if llffhold > 0:
                is_test = (i % llffhold == 0)
        
        cam_info = CameraInfo(
            uid=i,
            R=cam_pose['R'],
            T=cam_pose['T'], 
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
    
    # PLY 파일 경로 (임시)
    ply_path = os.path.join(path, "superglue_points.ply")
    
    # SceneInfo 생성
    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    
    print(f"SuperGlue scene loaded: {len(train_cam_infos)} train, {len(test_cam_infos)} test cameras")
    return scene_info

def estimate_camera_poses_simple(image_paths, matching_results):
    """간단한 카메라 포즈 추정 (PnP 기반)"""
    n_images = len(image_paths)
    cameras_info = []
    
    # 첫 번째 카메라를 원점으로 설정
    for i in range(n_images):
        if i == 0:
            # 첫 번째 카메라는 항등 변환
            R = np.eye(3, dtype=np.float32)
            T = np.zeros(3, dtype=np.float32)
        else:
            # 간단한 포즈 추정 (실제로는 더 정교한 방법 필요)
            # 여기서는 임시로 랜덤한 작은 변환 적용
            angle = (i / n_images) * 2 * np.pi
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=np.float32)
            T = np.array([np.sin(angle), 0, np.cos(angle) - 1], dtype=np.float32) * 0.5
        
        cameras_info.append({'R': R, 'T': T})
    
    return cameras_info

def create_point_cloud_simple(matching_results, cameras_info):
    """간단한 3D 포인트 클라우드 생성"""
    
    # 매칭된 특징점들로부터 3D 포인트 생성 (간단한 버전)
    points_3d = []
    colors = []
    
    # 임시로 랜덤한 3D 포인트들 생성
    # 실제로는 triangulation 필요
    num_points = 10000
    
    # 랜덤 3D 포인트들
    points = np.random.randn(num_points, 3).astype(np.float32) * 2
    
    # 랜덤 컬러
    point_colors = np.random.rand(num_points, 3).astype(np.float32)
    
    # 법선 벡터 (임시)
    normals = np.random.randn(num_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    return BasicPointCloud(
        points=points,
        colors=point_colors, 
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