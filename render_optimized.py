#!/usr/bin/env python3
"""
Optimized rendering script that skips unnecessary dataset loading
Only loads trained Gaussian model and renders specified views
"""

import torch
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import Camera
import json

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def load_trained_model(model_path, iteration=-1):
    """훈련된 Gaussian 모델만 로드 (데이터셋 로드 없이)"""
    
    # 모델 파일 경로 찾기
    if iteration == -1:
        # 가장 최근 체크포인트 찾기
        checkpoints = []
        for file in os.listdir(model_path):
            if file.startswith("chkpnt") and file.endswith(".pth"):
                iteration_num = int(file.split("_")[1].split(".")[0])
                checkpoints.append((iteration_num, file))
        
        if checkpoints:
            iteration, checkpoint_file = max(checkpoints)
            model_file = os.path.join(model_path, checkpoint_file)
        else:
            print("No checkpoint found!")
            return None, None
    else:
        model_file = os.path.join(model_path, f"chkpnt_{iteration}.pth")
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None, None
    
    print(f"Loading trained model from: {model_file}")
    
    # 모델 파라미터 로드
    checkpoint = torch.load(model_file)
    
    # GaussianModel 초기화
    gaussians = GaussianModel(checkpoint.get('sh_degree', 3))
    
    # 훈련된 파라미터 복원
    gaussians.restore(checkpoint, None)
    
    return gaussians, iteration


def create_camera_from_json(cam_info, resolution_scale=1.0):
    """JSON 카메라 정보로부터 Camera 객체 생성"""
    
    # 내부 파라미터
    FovY = cam_info["FovY"]
    FovX = cam_info["FovX"]
    
    # 외부 파라미터
    R = np.array(cam_info["R"])
    T = np.array(cam_info["T"])
    
    # 이미지 크기
    width = int(cam_info["width"] * resolution_scale)
    height = int(cam_info["height"] * resolution_scale)
    
    # Camera 객체 생성
    camera = Camera(
        colmap_id=cam_info["id"],
        R=R,
        T=T,
        FovX=FovX,
        FovY=FovY,
        image=torch.zeros(3, height, width),  # 더미 이미지
        gt_alpha_mask=None,
        image_name=cam_info["img_name"],
        uid=cam_info["id"],
        depth_params=None,
        depth_path="",
        is_test=False
    )
    
    return camera


def load_cameras_from_transforms(model_path):
    """transforms.json에서 카메라 정보 로드"""
    
    # transforms.json 파일들 찾기
    transforms_files = []
    for file in ["transforms_train.json", "transforms_test.json", "transforms.json"]:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            transforms_files.append((file, file_path))
    
    if not transforms_files:
        print("No transforms.json files found!")
        return [], []
    
    train_cameras = []
    test_cameras = []
    
    for file_name, file_path in transforms_files:
        print(f"Loading cameras from: {file_path}")
        
        with open(file_path, 'r') as f:
            transforms = json.load(f)
        
        is_test = "test" in file_name
        
        for frame in transforms.get("frames", []):
            camera = create_camera_from_json(frame)
            
            if is_test:
                test_cameras.append(camera)
            else:
                train_cameras.append(camera)
    
    return train_cameras, test_cameras


def render_custom_view(gaussians, pipeline, camera_pose, intrinsics, image_size, background):
    """사용자 정의 뷰포인트에서 렌더링"""
    
    # 카메라 파라미터로부터 Camera 객체 생성
    R, T = camera_pose
    FovX, FovY = intrinsics
    width, height = image_size
    
    camera = Camera(
        colmap_id=0,
        R=R,
        T=T,
        FovX=FovX,
        FovY=FovY,
        image=torch.zeros(3, height, width),
        gt_alpha_mask=None,
        image_name="custom_view",
        uid=0,
        depth_params=None,
        depth_path="",
        is_test=False
    )
    
    # 렌더링 수행
    with torch.no_grad():
        rendered_image = render(camera, gaussians, pipeline, background)["render"]
    
    return rendered_image


def render_optimized(model_path, iteration=-1, output_dir=None, custom_cameras=None):
    """최적화된 렌더링 함수"""
    
    print("🚀 Starting optimized rendering...")
    
    # 1. 훈련된 모델만 로드 (데이터셋 로드 없이)
    gaussians, loaded_iteration = load_trained_model(model_path, iteration)
    if gaussians is None:
        return False
    
    print(f"✅ Loaded model from iteration {loaded_iteration}")
    
    # 2. 파이프라인 설정
    pipeline = PipelineParams(ArgumentParser()).extract(ArgumentParser().parse_args([]))
    
    # 3. 배경 설정
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 4. 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join(model_path, "optimized_renders")
    makedirs(output_dir, exist_ok=True)
    
    # 5. 카메라 정보 로드 (사용자 정의 카메라가 없으면)
    if custom_cameras is None:
        print("📷 Loading camera information...")
        train_cameras, test_cameras = load_cameras_from_transforms(model_path)
        all_cameras = train_cameras + test_cameras
        
        if not all_cameras:
            print("❌ No cameras found!")
            return False
    else:
        all_cameras = custom_cameras
    
    # 6. 렌더링 수행
    print(f"🎬 Rendering {len(all_cameras)} views...")
    
    with torch.no_grad():
        for idx, camera in enumerate(tqdm(all_cameras, desc="Rendering")):
            # 렌더링
            rendered_image = render(camera, gaussians, pipeline, background)["render"]
            
            # 저장
            output_path = os.path.join(output_dir, f"render_{idx:05d}.png")
            torchvision.utils.save_image(rendered_image, output_path)
    
    print(f"✅ Rendering completed! Output saved to: {output_dir}")
    return True


def create_circular_cameras(radius=3.0, n_views=36, height=1.0):
    """원형 궤도 카메라 생성"""
    cameras = []
    
    for i in range(n_views):
        angle = i * (2 * np.pi / n_views)
        
        # 카메라 위치
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height
        
        # 카메라 포즈
        camera_pos = np.array([x, y, z])
        look_at = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        # 회전 행렬 계산
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        R = np.array([right, up, -forward]).T
        T = camera_pos
        
        # FOV 설정
        FovX = FovY = np.pi / 3  # 60도
        
        # Camera 객체 생성
        camera = Camera(
            colmap_id=i,
            R=R.astype(np.float32),
            T=T.astype(np.float32),
            FovX=FovX,
            FovY=FovY,
            image=torch.zeros(3, 800, 800),
            gt_alpha_mask=None,
            image_name=f"circular_{i:03d}",
            uid=i,
            depth_params=None,
            depth_path="",
            is_test=False
        )
        
        cameras.append(camera)
    
    return cameras


if __name__ == "__main__":
    parser = ArgumentParser(description="Optimized rendering script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--iteration", type=int, default=-1, help="Model iteration to load")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--circular", action="store_true", help="Use circular camera path")
    parser.add_argument("--n_views", type=int, default=36, help="Number of views for circular path")
    parser.add_argument("--radius", type=float, default=3.0, help="Radius for circular path")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    # 시스템 상태 초기화
    safe_state(args.quiet)
    
    # 사용자 정의 카메라 생성 (원형 궤도)
    custom_cameras = None
    if args.circular:
        print(f"🔄 Creating circular camera path with {args.n_views} views")
        custom_cameras = create_circular_cameras(args.radius, args.n_views)
    
    # 최적화된 렌더링 수행
    success = render_optimized(
        model_path=args.model_path,
        iteration=args.iteration,
        output_dir=args.output_dir,
        custom_cameras=custom_cameras
    )
    
    if success:
        print("🎉 Rendering completed successfully!")
    else:
        print("❌ Rendering failed!") 