from pathlib import Path
import os
import sys

def readHlocSceneInfo(path, images="images", eval=False, train_test_exp=False, llffhold=8, max_images=100, hloc_options=None):
    """
    Hloc 기반 SfM 파이프라인을 실행하고 3DGS용 SceneInfo로 변환
    """
    print("\n" + "="*60)
    print("           HLOC + 3DGS PIPELINE")
    print("="*60)
    
    print(f"📁 Source path: {path}")
    print(f"🖼️  Images folder: {images}")
    print(f"📊 Max images: {max_images}")
    
    try:
        # 1. hloc import
        from hloc import extract_features, match_features, reconstruction
        import pycolmap
        print("✓ Hloc modules imported successfully")
        
        # 2. 이미지 경로 수집
        images_dir = Path(path) / images
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        image_list = sorted([str(p) for p in images_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])[:max_images]
        print(f"✓ Found {len(image_list)} images")
        
        if len(image_list) == 0:
            raise ValueError("No images found in directory")
        
        # 3. Feature 추출
        print("\n[1/4] Extracting features...")
        features_path = Path(path) / "hloc_features.h5"
        
        # hloc의 올바른 API 사용
        extract_features.main(
            conf=extract_features.confs['superpoint_aachen'],
            image_dir=images_dir,
            export_dir=Path(path),
            image_list=image_list
        )
        
        # 4. Pair 생성 (exhaustive)
        print("\n[2/4] Creating image pairs...")
        pairs_path = Path(path) / "hloc_pairs.txt"
        with open(pairs_path, "w") as f:
            for i in range(len(image_list)):
                for j in range(i+1, len(image_list)):
                    f.write(f"{Path(image_list[i]).name} {Path(image_list[j]).name}\n")
        
        # 5. Matching
        print("\n[3/4] Matching features...")
        matches_path = Path(path) / "hloc_matches.h5"
        
        # hloc의 올바른 API 사용
        match_features.main(
            conf=match_features.confs['superglue'],
            pairs=pairs_path,
            features=features_path,
            export_dir=Path(path)
        )
        
        # 6. SfM 재구성
        print("\n[4/4] Running SfM reconstruction...")
        sfm_dir = Path(path) / "hloc_sfm"
        sfm_dir.mkdir(exist_ok=True)
        
        # hloc의 올바른 API 사용
        reconstruction.main(
            sfm_dir=sfm_dir,
            image_dir=images_dir,
            pairs=pairs_path,
            features=features_path,
            matches=matches_path,
            camera_model='PINHOLE',
            single_camera=True
        )
        
        print("\n🎉 HLOC PIPELINE SUCCESS!")
        
        # 7. COLMAP 결과를 3DGS용 SceneInfo로 변환
        print("\n[5/4] Converting to 3DGS format...")
        from scene.colmap_loader import readColmapSceneInfo
        return readColmapSceneInfo(str(sfm_dir), images, eval, train_test_exp)
        
    except ImportError as e:
        print(f"\n❌ HLOC IMPORT FAILED: {e}")
        print("Falling back to SuperGlue pipeline...")
        
        # Fallback to SuperGlue
        from complete_superglue_sfm import readSuperGlueSceneInfo
        return readSuperGlueSceneInfo(path, images, eval, train_test_exp, llffhold, "outdoor", max_images)
        
    except Exception as e:
        print(f"\n❌ HLOC PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Falling back to SuperGlue pipeline...")
        
        # Fallback to SuperGlue
        from complete_superglue_sfm import readSuperGlueSceneInfo
        return readSuperGlueSceneInfo(path, images, eval, train_test_exp, llffhold, "outdoor", max_images) 