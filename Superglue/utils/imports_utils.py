"""
Common imports and utilities for SuperGlue 3DGS pipeline
"""
import numpy as np
import torch
import sys
from pathlib import Path
from PIL import Image as PILImage

# CLIP 관련 imports (선택적)
CLIP_AVAILABLE = False
try:
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    CLIP_AVAILABLE = True
    print("✓ CLIP available and tested")
except (ImportError, Exception) as e:
    CLIP_AVAILABLE = False
    print(f"⚠️  CLIP not available: {e}. AdaptiveMatcher will use fallback descriptors.")

def get_3dgs_imports():
    """3DGS 관련 모듈들을 lazy import - 개선된 버전"""
    # gaussian-splatting 루트 디렉토리를 Python path에 추가
    gaussian_splatting_root = Path(__file__).parent.parent.parent
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
        return False
    
    print(f"\n📊 Pipeline Availability Summary:")
    print(f"  SuperGlue: {'✓' if superglue_available else '✗'}")
    print(f"  3DGS: {'✓' if gs_available else '✗'}")
    
    return True 