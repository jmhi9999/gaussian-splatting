# hloc_pipeline.py
# hloc 1.5 (PyPI) 버전에 맞는 파이프라인
# hloc 설치 필요: pip install hloc

from pathlib import Path
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction

def run_hloc_pipeline(
    image_dir="ImageInputs/images",
    output_dir="ImageInputs/hloc_outputs",
    feature_conf_name="superpoint_aachen",
    matcher_conf_name="superglue"
):
    images = Path(image_dir)
    outputs = Path(output_dir)
    outputs.mkdir(exist_ok=True)

    # 1. 특징점 추출
    feature_conf = extract_features.confs[feature_conf_name]
    print(f"[hloc] Extracting features with {feature_conf_name}...")
    extract_features.main(feature_conf, images, outputs)
    features_name = feature_conf['output']  # 예: 'feats-superpoint-n4096-r1024'
    features_path = outputs / f"{features_name}.h5"

    # 2. 이미지 파일명 리스트 생성
    file_names = [img.name for img in sorted(images.iterdir()) if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    # 3. 쌍 목록 생성 (파일명 리스트를 넘김)
    pairs_path = outputs / "pairs.txt"
    print("[hloc] Generating exhaustive pairs...")
    pairs_from_exhaustive.main(pairs_path, image_list=file_names)

    # 4. 매칭
    matcher_conf = match_features.confs[matcher_conf_name]
    matches_name = f"matches-{matcher_conf_name}_{features_name}"
    matches_path = outputs / f"{matches_name}.h5"
    print(f"[hloc] Matching features with {matcher_conf_name}...")
    match_features.main(
        matcher_conf,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path
    )

    # 5. COLMAP sparse mapping
    print("[hloc] Running COLMAP sparse mapping...")
    reconstruction.main(
        images=images,
        image_list=None,  # 모든 이미지 사용
        features=features_path,
        matches=matches_path,
        sfm_dir=outputs / 'sfm',
        database_path=outputs / 'database.db',
        skip_geometric_verification=False,
    )
    print(f"[hloc] Done! Results in {outputs / 'sfm'}")

if __name__ == "__main__":
    run_hloc_pipeline() 