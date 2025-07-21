from pathlib import Path
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
import argparse

class HlocPipelineConfig:
    def __init__(self, image_dir, output_dir,  feature_conf_name="superpoint_inloc", matcher_conf_name="superpoint+lightglue", start_step=1):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.feature_conf_name = feature_conf_name
        self.matcher_conf_name = matcher_conf_name
        self.start_step = start_step

def step1_extract_features(images, outputs, feature_conf_name):
    print(f"[Hloc] Step 1: Extracting features with {feature_conf_name}...")
    feature_conf = extract_features.confs[feature_conf_name]
    extract_features.main(feature_conf, images, outputs)
    features_name = feature_conf['output']
    features_path = outputs / f"{features_name}.h5"
    return features_name, features_path

def step2_make_file_names(images):
    print("[Hloc] Step 2: Making file names...")
    return [img.name for img in sorted(images.iterdir()) if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

def step3_generate_pairs(outputs, file_names):
    print("[Hloc] Step 3: Generating exhaustive pairs...")
    pairs_path = outputs / "pairs.txt"
    pairs_from_exhaustive.main(pairs_path, image_list=file_names)
    return pairs_path

def step4_matching(outputs, matcher_conf_name, features_name, pairs_path, features_path):
    matcher_conf = match_features.confs[matcher_conf_name]
    matches_name = f"matches-{matcher_conf_name}_{features_name}"
    matches_path = outputs / f"{matches_name}.h5"
    print(f"[Hloc] Step 4: Matching features with {matcher_conf_name}...")
    match_features.main(
        matcher_conf,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path
    )
    return matches_path

def step5_reconstruction(outputs, images, pairs_path, features_path, matches_path):
    print("[Hloc] Step 5: Running COLMAP sparse mapping...")
    reconstruction.main(
        sfm_dir=outputs / 'sfm',
        image_dir=images,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        skip_geometric_verification=False,
        image_options={'camera_model': 'PINHOLE'},
    )
    print(f"[Hloc] Step 5: Done! Results in {outputs / 'sfm'}")

def run_hloc_pipeline(config=None):    
    if config is None:
        return ValueError("Configuration must be provided.")
    else:
        if not isinstance(config, HlocPipelineConfig):
            return ValueError("Invalid configuration type. Expected HlocPipelineConfig.")

        images = config.image_dir
        outputs = config.output_dir
        outputs.mkdir(exist_ok=True)
        feature_conf_name = config.feature_conf_name
        matcher_conf_name = config.matcher_conf_name
        start_step = config.start_step

    features_name = None
    features_path = None
    file_names = None
    pairs_path = None
    matches_path = None

    # Step 1: Feature extraction
    if start_step == 1:
        features_name, features_path = step1_extract_features(images, outputs, feature_conf_name)
    else:
        features_name = extract_features.confs[feature_conf_name]['output']
        features_path = outputs / f"{features_name}.h5"

    # Step 2: Make file names
    file_names = step2_make_file_names(images)

    # Step 3: Generate pairs
    pairs_path = outputs / "pairs.txt"
    if start_step == 3:
        pairs_path = step3_generate_pairs(outputs, file_names)

    # Step 4: Matching
    matches_name = f"matches-{matcher_conf_name}_{features_name}"
    matches_path = outputs / f"{matches_name}.h5"
    if start_step == 4:
        matches_path = step4_matching(outputs, matcher_conf_name, features_name, pairs_path, features_path)

    # Step 5: Reconstruction
    if start_step == 5:
        step5_reconstruction(outputs, images, pairs_path, features_path, matches_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HLOC pipeline for feature extraction and matching.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--feature_conf_name", type=str, default="superpoint_inloc", help="Feature configuration name.")
    parser.add_argument("--matcher_conf_name", type=str, default="superpoint+lightglue", help="Matcher configuration name.")
    parser.add_argument("--start_step", type=int, default=1, help="Step to start the pipeline from (default: 1).")
    args = parser.parse_args()
    config = HlocPipelineConfig(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        feature_conf_name=args.feature_conf_name,
        matcher_conf_name=args.matcher_conf_name,
        start_step=args.start_step
    )
    print(f"Running HLOC pipeline with configuration: {config.__dict__}")
    run_hloc_pipeline(config)