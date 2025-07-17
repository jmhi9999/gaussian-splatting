# hloc_pipeline.py
# hloc 1.5 (PyPI) 버전에 맞는 파이프라인
# hloc 설치 필요: pip install hloc

from pathlib import Path
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
import pycolmap

def step1_extract_features(images, outputs, feature_conf_name):
    feature_conf = extract_features.confs[feature_conf_name]
    print(f"[hloc] Extracting features with {feature_conf_name}...")
    extract_features.main(feature_conf, images, outputs)
    features_name = feature_conf['output']
    features_path = outputs / f"{features_name}.h5"
    return features_name, features_path

def step2_make_file_names(images):
    return [img.name for img in sorted(images.iterdir()) if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

def step3_generate_pairs(outputs, file_names):
    pairs_path = outputs / "pairs.txt"
    print("[hloc] Generating exhaustive pairs...")
    pairs_from_exhaustive.main(pairs_path, image_list=file_names)
    return pairs_path

def step4_matching(outputs, matcher_conf_name, features_name, pairs_path, features_path):
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
    return matches_path

def step5_reconstruction(outputs, images, pairs_path, features_path, matches_path):
    print("[hloc] Running COLMAP sparse mapping...")
    reconstruction.main(
        sfm_dir=outputs / 'sfm',
        image_dir=images,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        skip_geometric_verification=False,
        image_options={'camera_model': 'PINHOLE'},
    )
    print(f"[hloc] Done! Results in {outputs / 'sfm'}")

def run_hloc_pipeline(
    image_dir="ImageInputs/images",
    output_dir="ImageInputs/hloc_outputs",
    feature_conf_name="superpoint_max",
    matcher_conf_name="superpoint+lightglue",
    start_step=1
):
    images = Path(image_dir)
    outputs = Path(output_dir)
    outputs.mkdir(exist_ok=True)

    features_name = None
    features_path = None
    file_names = None
    pairs_path = None
    matches_path = None

    if start_step <= 1:
        features_name, features_path = step1_extract_features(images, outputs, feature_conf_name)
    else:
        feature_conf = extract_features.confs[feature_conf_name]
        features_name = feature_conf['output']
        features_path = outputs / f"{features_name}.h5"

    if start_step <= 2:
        file_names = step2_make_file_names(images)
    else:
        file_names = [img.name for img in sorted(images.iterdir()) if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if start_step <= 3:
        pairs_path = step3_generate_pairs(outputs, file_names)
    else:
        pairs_path = outputs / "pairs.txt"

    if start_step <= 4:
        matches_path = step4_matching(outputs, matcher_conf_name, features_name, pairs_path, features_path)
    else:
        matches_name = f"matches-{matcher_conf_name}_{features_name}"
        matches_path = outputs / f"{matches_name}.h5"

    if start_step <= 5:
        step5_reconstruction(outputs, images, pairs_path, features_path, matches_path)

if __name__ == "__main__":
    run_hloc_pipeline(start_step=1) 