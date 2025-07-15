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
from utils import frame2tensor
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
    """SuperGlue 파이프라인 동적 import - 개선된 버전"""
    try:
        # 현재 디렉토리에서 SuperGlue 경로 찾기
        current_dir = Path(__file__).parent.parent  # gaussian-splatting 루트
        
        # SuperGlue 경로들 (더 많은 경로 추가)
        superglue_paths = [
            current_dir / "Superglue",
            current_dir / "SuperGlue", 
            current_dir,
            Path.cwd() / "Superglue",
            Path.cwd() / "SuperGlue",
            Path.cwd()
        ]
        
        print("🔍 Searching for SuperGlue pipeline...")
        for path in superglue_paths:
            complete_sfm_file = path / "complete_superglue_sfm.py"
            print(f"  Checking: {complete_sfm_file}")
            
            if complete_sfm_file.exists():
                print(f"  ✓ Found SuperGlue pipeline at: {path}")
                
                # 해당 경로를 sys.path에 추가
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
                    print(f"  ✓ Added {path} to Python path")
                
                try:
                    # 모듈 import 시도
                    from complete_superglue_sfm import SuperGlue3DGSPipeline
                    print(f"✓ SuperGlue pipeline imported successfully from {path}")
                    return SuperGlue3DGSPipeline
                except ImportError as e:
                    print(f"  ✗ Import failed: {e}")
                    continue
        
        print("✗ SuperGlue pipeline not found in any of the searched paths")
        return None
        
    except Exception as e:
        print(f"✗ SuperGlue import failed with exception: {e}")
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
            current_dir,
            Path.cwd(),  # 현재 작업 디렉토리 추가
            Path.cwd() / "Superglue",
            Path.cwd() / "SuperGlue"
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
            'nms_radius': 2,                # 3 → 2 (더 밀집된 특징점)
            'keypoint_threshold': 0.001,    # 0.003 → 0.001 (더 많은 특징점)
            'max_keypoints': 6144           # 4096 → 6144 (더 많은 특징점)
            },
            'superglue': {
            'weights': superglue_config,    # 그대로 유지
            'sinkhorn_iterations': 50,      # 30 → 50 (더 정교한 매칭)
            'match_threshold': 0.1,         # 0.15 → 0.1 (더 관대한 매칭)
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
    return False

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
    return False


sceneLoadTypeCallbacks = {
    "SuperGlue": readSuperGlueSceneInfo,
    "SuperGlueCOLMAPHybrid": readSuperGlueCOLMAPHybridSceneInfo,
    "Hloc": readHlocSceneInfo,
}

# sceneLoadTypeCallbacks에 추가
sceneLoadTypeCallbacks["SuperGlue"] = readSuperGlueSceneInfo
sceneLoadTypeCallbacks["SuperGlueCOLMAPHybrid"] = readSuperGlueCOLMAPHybridSceneInfo


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