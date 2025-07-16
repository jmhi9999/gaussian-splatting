# hloc_pipeline.py
# 이 파일은 SuperPoint + SuperGlue + COLMAP sparse mapping을 hloc 기반으로 한 번에 수행하는 간단한 파이프라인입니다.
# hloc 설치 필요: pip install git+https://github.com/cvg/Hierarchical-Localization.git

from pathlib import Path
from hloc import extract_features, match_features, reconstruction

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

    # 2. 매칭
    matcher_conf = match_features.confs[matcher_conf_name]
    print(f"[hloc] Matching features with {matcher_conf_name}...")
    match_features.main(matcher_conf, outputs, outputs)

    # 3. COLMAP sparse mapping
    print("[hloc] Running COLMAP sparse mapping...")
    reconstruction.main(
        images=images,
        image_list=None,  # 모든 이미지 사용
        features=outputs / 'features.h5',
        matches=outputs / 'matches.h5',
        sfm_dir=outputs / 'sfm',
        database_path=outputs / 'database.db',
        skip_geometric_verification=False,
    )
    print(f"[hloc] Done! Results in {outputs / 'sfm'}")

if __name__ == "__main__":
    run_hloc_pipeline() 