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
       from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
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
       print("\n[1/5] Extracting features...")
      
       # hloc 1.5의 올바른 API 사용
       extract_features.main(
           extract_features.confs['superpoint_aachen'],
           images_dir,
           Path(path)
       )
      
       # 4. Pair 생성 (exhaustive)
       print("\n[2/5] Creating image pairs...")
      
       # 이미지 리스트 파일 생성
       image_list_path = Path(path) / 'image_list.txt'
       with open(image_list_path, 'w') as f:
           for img_path in image_list:
               f.write(f"{Path(img_path).name}\n")
      
       pairs_path = Path(path) / 'pairs-exhaustive.txt'
       pairs_from_exhaustive.main(
           pairs_path,
           image_list_path
       )
      
       # 5. Matching
       print("\n[3/5] Matching features...")
      
       # 동적으로 features와 matches 파일명 찾기
       features_dir = Path(path)
       features_files = list(features_dir.glob("feats-*.h5"))
       if not features_files:
           raise FileNotFoundError("No feature files found. Expected pattern: feats-*.h5")
      
       features_path = features_files[0]  # 첫 번째 features 파일 사용
       print(f"✓ Using features file: {features_path.name}")
      
       # matches 파일명은 features 파일명을 기반으로 생성
       matches_name = features_path.name.replace("feats-", "matches-")
       matches_path = features_dir / matches_name
      
       print(f"✓ Using matches file: {matches_name}")
      
       match_features.main(
           match_features.confs['superglue'],
           pairs_path,
           features_path, 
           None,
           matches_path
       )
      
       # 6. SfM 재구성
       print("\n[4/5] Running SfM reconstruction...")
       sfm_dir = Path(path) / "hloc_sfm"
       sfm_dir.mkdir(exist_ok=True)
      
       # hloc 1.5의 올바른 API 사용
       reconstruction.main(
           str(sfm_dir),
           str(images_dir),
           str(pairs_path),
           str(features_path),
           str(matches_path)
       )
      
       print("\n🎉 HLOC PIPELINE SUCCESS!")
      
       # 7. COLMAP 결과를 3DGS용 SceneInfo로 변환
       print("\n[5/5] Converting to 3DGS format...")
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



