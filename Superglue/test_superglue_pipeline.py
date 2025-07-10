#!/usr/bin/env python3
"""
SuperGlue 3DGS Pipeline Test Script
"""

import sys
import os
from pathlib import Path

# Add the gaussian-splatting root to Python path
gaussian_splatting_root = Path(__file__).parent.parent
sys.path.insert(0, str(gaussian_splatting_root))

# Add Superglue directory to path
superglue_dir = Path(__file__).parent
sys.path.insert(0, str(superglue_dir))

from complete_superglue_sfm import SuperGlue3DGSPipeline

def test_pipeline():
    """Test the SuperGlue 3DGS pipeline"""
    
    # Test configuration
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 4096
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    
    # Initialize pipeline
    pipeline = SuperGlue3DGSPipeline(config, device='cuda')
    
    # Test with a small number of images
    input_dir = "../ImageInputs/images"  # Adjust path as needed
    output_dir = "test_output"
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available directories:")
        for item in os.listdir(".."):
            print(f"  - {item}")
        return False
    
    try:
        # Process images
        scene_info = pipeline.process_images_to_3dgs(
            image_dir=input_dir,
            output_dir=output_dir,
            max_images=10  # Start with small number
        )
        
        print(f"\n=== Test Results ===")
        print(f"Training cameras: {len(scene_info.train_cameras)}")
        print(f"Test cameras: {len(scene_info.test_cameras)}")
        print(f"3D points: {len(scene_info.point_cloud.points)}")
        print(f"Scene radius: {scene_info.nerf_normalization['radius']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n✅ Test passed!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 