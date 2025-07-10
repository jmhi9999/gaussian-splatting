from pathlib import Path
import os

def readHlocSceneInfo(path, images="images", eval=False, train_test_exp=False, llffhold=8, max_images=100, hloc_options=None):
    """
    Hloc 기반 SfM 파이프라인을 실행하고 3DGS용 SceneInfo로 변환
    """
    # 1. hloc import
    from hloc import extract_features, match_features, reconstruction
    import pycolmap

    # 2. 이미지 경로 수집
    images_dir = Path(path) / images
    image_list = sorted([str(p) for p in images_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])[:max_images]

    # 3. Feature 추출
    features_path = Path(path) / "hloc_features.h5"
    extract_features.main(
        {
            "output": features_path,
            "image_dir": images_dir,
            "feature_conf": "superpoint_aachen",
        }
    )

    # 4. Pair 생성 (exhaustive)
    pairs_path = Path(path) / "hloc_pairs.txt"
    with open(pairs_path, "w") as f:
        for i in range(len(image_list)):
            for j in range(i+1, len(image_list)):
                f.write(f"{os.path.basename(image_list[i])} {os.path.basename(image_list[j])}\n")

    # 5. Matching
    matches_path = Path(path) / "hloc_matches.h5"
    match_features.main(
        {
            "output": matches_path,
            "pairs": pairs_path,
            "features": features_path,
            "matcher_conf": "superglue"
        }
    )

    # 6. SfM 재구성
    sfm_dir = Path(path) / "hloc_sfm"
    sfm_dir.mkdir(exist_ok=True)
    reconstruction.main(
        {
            "output": sfm_dir,
            "image_dir": images_dir,
            "pairs": pairs_path,
            "features": features_path,
            "matches": matches_path,
            "camera_model": "PINHOLE",
            "single_camera": True,
        }
    )

    # 7. COLMAP 결과를 3DGS용 SceneInfo로 변환
    from scene.colmap_loader import readColmapSceneInfo
    return readColmapSceneInfo(str(sfm_dir), images, eval, train_test_exp) 