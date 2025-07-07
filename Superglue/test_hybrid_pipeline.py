# test_hybrid_pipeline.py
# SuperGlue + COLMAP 하이브리드 파이프라인 테스트

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from Superglue.superglue_colmap_pipeline import SuperGlueColmapPipeline
from Superglue.superglue_scene_reader import readSuperGlueSceneInfo

def test_hybrid_pipeline():
    """하이브리드 파이프라인 테스트"""
    
    # 테스트 이미지 디렉토리 (ImageInputs/images 사용)
    image_dir = "ImageInputs/images"
    output_dir = "test_output"
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} not found!")
        print("Please make sure you have images in ImageInputs/images/")
        return False
    
    print("=== Testing SuperGlue + COLMAP Hybrid Pipeline ===")
    print(f"Input directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    
    # 1. 직접 파이프라인 테스트
    print("\n1. Testing direct pipeline...")
    try:
        pipeline = SuperGlueColmapPipeline()
        success = pipeline.run_pipeline(image_dir, output_dir, max_images=20)
        
        if success:
            print("✓ Direct pipeline test passed!")
            
            # 결과 로딩 테스트
            try:
                cameras = pipeline.load_results()
                print(f"✓ Successfully loaded {len(cameras)} cameras")
            except Exception as e:
                print(f"✗ Failed to load results: {e}")
        else:
            print("✗ Direct pipeline test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        return False
    
    # 2. Scene reader 통합 테스트
    print("\n2. Testing scene reader integration...")
    try:
        # 임시 scene 정보 생성
        scene_info = readSuperGlueSceneInfo(
            path="test_output",
            images="images",
            eval=True,
            train_test_exp=False,
            llffhold=8
        )
        
        print(f"✓ Scene reader test passed!")
        print(f"  - Train cameras: {len(scene_info.train_cameras)}")
        print(f"  - Test cameras: {len(scene_info.test_cameras)}")
        print(f"  - Point cloud: {len(scene_info.point_cloud.points)} points")
        
    except Exception as e:
        print(f"✗ Scene reader error: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True

def test_with_custom_images(image_dir, output_dir, max_images=20):
    """사용자 지정 이미지로 테스트"""
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} not found!")
        return False
    
    print(f"=== Testing with custom images ===")
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Max images: {max_images}")
    
    try:
        pipeline = SuperGlueColmapPipeline()
        success = pipeline.run_pipeline(image_dir, output_dir, max_images)
        
        if success:
            cameras = pipeline.load_results()
            print(f"✓ Success! Loaded {len(cameras)} cameras")
            return True
        else:
            print("✗ Pipeline failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SuperGlue + COLMAP Hybrid Pipeline")
    parser.add_argument("--image_dir", default="ImageInputs/images", 
                       help="Input image directory")
    parser.add_argument("--output_dir", default="test_output", 
                       help="Output directory")
    parser.add_argument("--max_images", type=int, default=20, 
                       help="Maximum number of images to process")
    parser.add_argument("--test_mode", choices=["default", "custom"], default="default",
                       help="Test mode: default uses built-in test, custom uses specified images")
    
    args = parser.parse_args()
    
    if args.test_mode == "default":
        success = test_hybrid_pipeline()
    else:
        success = test_with_custom_images(args.image_dir, args.output_dir, args.max_images)
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 