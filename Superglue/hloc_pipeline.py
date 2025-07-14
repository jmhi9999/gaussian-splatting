#!/usr/bin/env python3
"""
Hloc + 3DGS 통합 파이프라인 (수정된 버전)
Command line 방식으로 Hloc 실행하여 API 호환성 문제 해결
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import cv2
from PIL import Image
import json
import subprocess
from typing import Optional, List, Tuple, Dict, Any

# 3DGS imports
from utils.graphics_utils import BasicPointCloud, focal2fov, getWorld2View2
from scene.dataset_readers import CameraInfo, SceneInfo

try:
    # Hloc imports (SfM에 필요한 것만)
    from hloc import extract_features, match_features, reconstruction
    import pycolmap
    HLOC_AVAILABLE = True
    print("✓ Hloc SfM modules imported successfully")
except ImportError as e:
    HLOC_AVAILABLE = False
    print(f"✗ Hloc import failed: {e}")

class HlocPipeline:
    """Hloc 기반 SfM 파이프라인 (Command Line 실행)"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # SuperPoint 가용성 확인
        self.superpoint_available = self._check_superpoint()
        
        if not HLOC_AVAILABLE:
            print("⚠️  Hloc not available, falling back to simple pipeline")
    
    def _check_superpoint(self):
        """SuperPoint 사용 가능성 확인"""
        try:
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, '-c', 
                'from SuperGluePretrainedNetwork.models import superpoint'
            ], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
            
    def process_images(self, 
                      image_dir: str, 
                      output_dir: str, 
                      max_images: int = 100) -> Optional[SceneInfo]:
        """이미지들을 처리하여 SceneInfo 생성"""
        
        print(f"\n🚀 Starting Hloc Pipeline")
        print(f"📁 Input: {image_dir}")
        print(f"📁 Output: {output_dir}")
        print(f"📊 Max images: {max_images}")
        
        try:
            # 출력 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. 이미지 수집
            image_paths = self._collect_images(image_dir, max_images)
            if len(image_paths) == 0:
                raise ValueError(f"No images found in {image_dir}")
            
            print(f"📸 Found {len(image_paths)} images")
            
            if HLOC_AVAILABLE:
                # 2. Hloc SfM 파이프라인 실행 (Command Line)
                scene_info = self._run_hloc_command_line(image_paths, output_path)
                if scene_info:
                    return scene_info
                else:
                    print("⚠️  Hloc pipeline failed, falling back...")
            
            # 3. Fallback: 간단한 카메라 배치 (개발용만)
            print("❌ Hloc failed - using fallback is NOT recommended for production!")
            return self._create_fallback_scene(image_paths)
            
        except Exception as e:
            print(f"❌ Hloc pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _collect_images(self, image_dir: str, max_images: int) -> List[Path]:
        """이미지 파일 수집"""
        image_dir = Path(image_dir)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        image_paths.sort()
        return image_paths[:max_images]
    
    def _run_hloc_command_line(self, image_paths: List[Path], output_path: Path) -> Optional[SceneInfo]:
        """Command Line으로 Hloc SfM 파이프라인 실행"""
        
        print("\n📊 Running Hloc SfM pipeline (Command Line)...")
        
        try:
            image_dir = image_paths[0].parent
            
            # 1. 특징점 추출
            print("🔍 Extracting features...")
            
            # SuperPoint 사용 가능하면 SuperPoint, 아니면 SIFT
            if self.superpoint_available:
                feature_conf = 'superpoint_aachen'
                feature_file = 'feats-superpoint-n4096-r1024'  # .h5 확장자 제거 (Hloc이 자동으로 추가)
                print("Using SuperPoint extractor")
            else:
                feature_conf = 'sift'
                feature_file = 'feats-sift'  # .h5 확장자 제거 (Hloc이 자동으로 추가)
                print("Using SIFT extractor (SuperPoint not available)")
            
            extract_cmd = [
                sys.executable, '-m', 'hloc.extract_features',
                '--image_dir', str(image_dir),
                '--export_dir', str(output_path),
                '--conf', feature_conf
            ]
            
            print(f"Command: {' '.join(extract_cmd)}")
            result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"❌ Feature extraction failed: {result.stderr}")
                return None
            
            print("✅ Feature extraction completed")
            
            # 2. 매칭 페어 생성
            print("🔗 Creating image pairs...")
            pairs_path = output_path / 'pairs.txt'
            
            # 순차 매칭 + 추가 연결
            with open(pairs_path, 'w') as f:
                # 순차 연결
                for i in range(len(image_paths) - 1):
                    name_i = image_paths[i].name
                    name_j = image_paths[i + 1].name
                    f.write(f"{name_i} {name_j}\n")
                
                # 추가 연결 (안정성)
                for i in range(len(image_paths)):
                    for j in range(i + 2, min(i + 4, len(image_paths))):
                        name_i = image_paths[i].name
                        name_j = image_paths[j].name
                        f.write(f"{name_i} {name_j}\n")
            
            print(f"✅ Created {sum(1 for line in open(pairs_path))} image pairs")
            
            # 3. 특징점 매칭
            print("🔗 Matching features...")
            
            # SuperPoint면 SuperGlue, SIFT면 NN-mutual
            if self.superpoint_available:
                matcher_conf = 'superglue'
                matcher_file = f'{feature_file}_matches-superglue_pairs.h5'  # Hloc의 실제 파일명 형식
                print("Using SuperGlue matcher")
            else:
                matcher_conf = 'NN-mutual'
                matcher_file = f'{feature_file}_matches-NN-mutual_pairs.h5'  # Hloc의 실제 파일명 형식
                print("Using NN-mutual matcher")
            
            # 매칭 명령어 수정 (경로 문제 해결)
            match_cmd = [
                sys.executable, '-m', 'hloc.match_features',
                '--pairs', str(pairs_path),
                '--features', feature_file,  # 파일명만
                '--matches', matcher_file,   # 파일명만  
                '--export_dir', str(output_path),
                '--conf', matcher_conf
            ]
            
            print(f"Command: {' '.join(match_cmd)}")
            result = subprocess.run(match_cmd, capture_output=True, text=True, timeout=1200)
            
            if result.returncode != 0:
                print(f"❌ Feature matching failed: {result.stderr}")
                return None
            
            print("✅ Feature matching completed")
            
            # 4. SfM 재구성
            print("🏗️  Running SfM reconstruction...")
            sfm_dir = output_path / 'sfm'
            sfm_dir.mkdir(exist_ok=True)
            
            reconstruction_cmd = [
                sys.executable, '-m', 'hloc.reconstruction',
                '--sfm_dir', str(sfm_dir),
                '--image_dir', str(image_dir),
                '--pairs', str(pairs_path),
                '--features', str(output_path / (feature_file + '.h5')),  # .h5 확장자 추가
                '--matches', str(output_path / matcher_file),   # Hloc의 실제 파일명 형식
                '--camera_mode', 'SINGLE'
            ]
            
            print(f"Command: {' '.join(reconstruction_cmd)}")
            result = subprocess.run(reconstruction_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                print(f"⚠️  SfM reconstruction command failed: {result.stderr}")
                print("Trying to load existing reconstruction...")
            
            # 5. COLMAP 모델 로드
            try:
                model_files = list(sfm_dir.glob('*'))
                print(f"SfM output files: {[f.name for f in model_files]}")
                
                # COLMAP 모델 로드 시도
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
                return self._create_scene_info_from_colmap(model, image_paths)
                
            except Exception as e:
                print(f"❌ Failed to load COLMAP model: {e}")
                return None
            
        except subprocess.TimeoutExpired:
            print("❌ Hloc pipeline timeout")
            return None
        except Exception as e:
            print(f"❌ Hloc command line pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_scene_info_from_colmap(self, model, image_paths: List[Path]) -> SceneInfo:
        """COLMAP 모델에서 SceneInfo 생성"""
        
        print("📊 Converting COLMAP model to SceneInfo...")
        
        # 카메라 정보 추출
        cam_infos = []
        for img_id, image in model.images.items():
            cam = model.cameras[image.camera_id]
            
            # 이미지 경로 찾기
            image_name = image.name
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
                width, height = cam.width, cam.height
            
            # 회전 행렬과 평행이동 (COLMAP format)
            # pycolmap.Image는 qvec를 사용하여 회전 행렬을 계산해야 함
            R = image.qvec2rotmat().astype(np.float32)  # rotmat() → qvec2rotmat()
            T = image.tvec.astype(np.float32)
            
            # 카메라 내부 파라미터에서 FOV 계산
            if cam.model_name in ['SIMPLE_PINHOLE', 'PINHOLE']:
                if cam.model_name == 'SIMPLE_PINHOLE':
                    fx = fy = cam.params[0]
                else:
                    fx, fy = cam.params[0], cam.params[1]
                
                FovX = focal2fov(fx, width)
                FovY = focal2fov(fy, height)
            else:
                # 기본값
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
                is_test=(img_id % 8 == 0)  # 8번째마다 테스트
            )
            cam_infos.append(cam_info)
        
        print(f"✓ Created {len(cam_infos)} camera infos")
        
        # 3D 포인트 추출
        points_3d = []
        colors_3d = []
        
        for point_id, point in model.points3D.items():
            points_3d.append(point.xyz)
            colors_3d.append(point.color / 255.0)  # 0-1로 정규화
        
        if len(points_3d) == 0:
            print("⚠️  No 3D points found, creating default point cloud")
            points_3d = np.random.randn(1000, 3).astype(np.float32)
            colors_3d = np.random.rand(1000, 3).astype(np.float32)
        else:
            points_3d = np.array(points_3d, dtype=np.float32)
            colors_3d = np.array(colors_3d, dtype=np.float32)
        
        # 법선 벡터 (간단히 0으로)
        normals_3d = np.zeros_like(points_3d)
        
        pcd = BasicPointCloud(
            points=points_3d,
            colors=colors_3d,
            normals=normals_3d
        )
        
        print(f"✓ Created point cloud with {len(points_3d)} points")
        
        # 학습/테스트 분할
        train_cams = [c for c in cam_infos if not c.is_test]
        test_cams = [c for c in cam_infos if c.is_test]
        
        # NeRF 정규화
        nerf_norm = self._compute_nerf_normalization(cam_infos)
        
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cams,
            test_cameras=test_cams,
            nerf_normalization=nerf_norm,
            ply_path="",
            is_nerf_synthetic=False
        )
        
        print(f"✅ SceneInfo created successfully!")
        print(f"   - Training cameras: {len(train_cams)}")
        print(f"   - Test cameras: {len(test_cams)}")
        print(f"   - 3D points: {len(points_3d)}")
        print(f"   - Scene radius: {nerf_norm['radius']:.3f}")
        
        return scene_info
    
    def _compute_nerf_normalization(self, cam_infos: List[CameraInfo]) -> Dict[str, Any]:
        """NeRF 정규화 파라미터 계산 (COLMAP 스타일)"""
        
        cam_centers = []
        for cam in cam_infos:
            # COLMAP 스타일에서 camera center 복원: camera_center = -R^T @ T
            camera_center = -cam.R.T @ cam.T
            cam_centers.append(camera_center)
        
        if len(cam_centers) > 0:
            cam_centers = np.array(cam_centers)
            center = np.mean(cam_centers, axis=0)
            distances = np.linalg.norm(cam_centers - center, axis=1)
            radius = np.max(distances)
            
            # 최소 반지름 보장
            if radius < 1e-6:
                radius = 5.0
                print(f"⚠️  Computed radius too small, setting to {radius}")
            else:
                radius *= 1.1  # 10% 여유분
        else:
            center = np.zeros(3)
            radius = 5.0
            print(f"⚠️  No cameras found, using default radius: {radius}")
        
        return {"translate": -center, "radius": float(radius)}
    
    def _create_fallback_scene(self, image_paths: List[Path]) -> SceneInfo:
        """Fallback: 개발용만 사용 - 실제 품질 목표에는 부적합"""
        
        print("⚠️⚠️⚠️  FALLBACK SCENE - NOT FOR PRODUCTION USE ⚠️⚠️⚠️")
        print("This will NOT achieve SSIM 0.9+ target!")
        
        # [이전과 동일한 fallback 코드...]
        # 간단한 원형 배치
        cam_infos = []
        for i, image_path in enumerate(image_paths):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except:
                width, height = 1920, 1080
            
            angle = (i / len(image_paths)) * 2 * np.pi
            radius = 5.0
            
            camera_center = np.array([
                radius * np.cos(angle),
                0.0,
                radius * np.sin(angle)
            ], dtype=np.float32)
            
            z_axis = -camera_center / np.linalg.norm(camera_center)
            world_up = np.array([0, 1, 0], dtype=np.float32)
            x_axis = np.cross(world_up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
            T = -R @ camera_center
            
            focal = max(width, height) * 0.8
            FovX = focal2fov(focal, width)
            FovY = focal2fov(focal, height)
            
            cam_info = CameraInfo(
                uid=i, R=R, T=T, FovY=float(FovY), FovX=float(FovX),
                image_path=str(image_path), image_name=image_path.name,
                width=width, height=height, depth_params=None, depth_path="",
                is_test=(i % 8 == 0)
            )
            cam_infos.append(cam_info)
        
        # 간단한 포인트 클라우드
        n_points = 10000
        points = np.random.randn(n_points, 3).astype(np.float32) * 2
        colors = np.random.rand(n_points, 3).astype(np.float32)
        normals = np.random.randn(n_points, 3).astype(np.float32)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        train_cams = [c for c in cam_infos if not c.is_test]
        test_cams = [c for c in cam_infos if c.is_test]
        nerf_norm = self._compute_nerf_normalization(cam_infos)
        
        return SceneInfo(
            point_cloud=pcd, train_cameras=train_cams, test_cameras=test_cams,
            nerf_normalization=nerf_norm, ply_path="", is_nerf_synthetic=False
        )


def readHlocSceneInfo(path: str, 
                     images: str = "images", 
                     eval: bool = False, 
                     train_test_exp: bool = False,
                     max_images: int = 100,
                     feature_extractor: str = 'superpoint_aachen',
                     matcher: str = 'superglue') -> SceneInfo:
    """Hloc 파이프라인으로 SceneInfo 생성"""
    
    print("\n" + "="*60)
    print("              HLOC + 3DGS PIPELINE")
    print("="*60)
    
    # 이미지 폴더 경로
    image_dir = Path(path) / images
    if not image_dir.exists():
        fallback_paths = [Path(path), Path(path) / "input"]
        for fallback in fallback_paths:
            if fallback.exists():
                image_dir = fallback
                break
    
    print(f"📁 Source path: {path}")
    print(f"📁 Images folder: {image_dir}")
    print(f"🔧 Feature extractor: {feature_extractor}")
    print(f"🔧 Matcher: {matcher}")
    print(f"📊 Max images: {max_images}")
    print(f"🚀 Hloc available: {HLOC_AVAILABLE}")
    
    # 출력 디렉토리
    output_dir = Path(path) / "hloc_output"
    
    # Hloc 파이프라인 실행
    pipeline = HlocPipeline(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    scene_info = pipeline.process_images(
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        max_images=max_images
    )
    
    if scene_info is None:
        raise RuntimeError("Failed to create scene info with Hloc pipeline")
    
    return scene_info


if __name__ == "__main__":
    # 테스트
    print("Testing Hloc Pipeline...")
    print(f"Hloc available: {HLOC_AVAILABLE}")