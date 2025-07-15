import os
import subprocess
import torch
import numpy as np
from PIL import Image
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import random
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def extract_superpoint_features(image_dir, output_path, config=None):
    print("Step 1: SuperPoint feature extraction (direct, GPU 지원)")
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config is None:
        config = {}
    model = SuperPoint(config).to(device)
    model.eval()
    all_keypoints = {}
    all_descriptors = {}
    all_scores = {}
    image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    for img_name in image_list:
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('L')
        img_tensor = torch.from_numpy(np.array(img)).float()[None, None] / 255.0
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            result = model({'image': img_tensor})
        all_keypoints[img_name] = result['keypoints'][0].cpu().numpy()
        all_descriptors[img_name] = result['descriptors'][0].cpu().numpy()
        all_scores[img_name] = result['scores'][0].cpu().numpy()
        print(f"Keypoints saved for {img_name} - {all_keypoints[img_name].shape[0]} keypoints")
    np.savez(output_path, keypoints=all_keypoints, descriptors=all_descriptors, scores=all_scores)
    print(f"SuperPoint features saved to {output_path}")

def generate_image_pairs(image_list, max_skip=8, random_pairs=2000):
    pairs = []
    n = len(image_list)
    # Sequential + skip
    for i in range(n):
        for skip in range(1, max_skip+1):
            j = i + skip
            if j < n:
                pairs.append((image_list[i], image_list[j]))
    # 랜덤 일부 쌍 추가 (long-range)
    all_possible = [(image_list[i], image_list[j]) for i in range(n) for j in range(i+max_skip+1, n)]
    if len(all_possible) > 0 and random_pairs > 0:
        pairs += random.sample(all_possible, min(random_pairs, len(all_possible)))
    return pairs

def match_superglue(features_path, image_dir, output_path, superglue_config=None, partial_gap=5):
    print("Step 2: SuperGlue feature matching (direct, GPU 지원)")
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if superglue_config is None:
        superglue_config = {'match_threshold': 0.05}
    else:
        if 'match_threshold' not in superglue_config:
            superglue_config['match_threshold'] = 0.05
    model = SuperGlue(superglue_config).to(device)
    model.eval()
    data = np.load(features_path, allow_pickle=True)
    keypoints = data['keypoints'].item()
    descriptors = data['descriptors'].item()
    scores = data['scores'].item()
    image_list = sorted(keypoints.keys())
    pairs = generate_image_pairs(image_list, max_skip=8, random_pairs=2000)
    matches_dict = {}
    for img0, img1 in pairs:
        kpts0 = torch.from_numpy(keypoints[img0])[None].to(device)
        kpts1 = torch.from_numpy(keypoints[img1])[None].to(device)
        desc0 = torch.from_numpy(descriptors[img0])[None].to(device)
        desc1 = torch.from_numpy(descriptors[img1])[None].to(device)
        scores0 = torch.from_numpy(scores[img0])[None].to(device)
        scores1 = torch.from_numpy(scores[img1])[None].to(device)
        img0_tensor = torch.zeros((1, 1, 240, 320), device=device)
        img1_tensor = torch.zeros((1, 1, 240, 320), device=device)
        with torch.no_grad():
            result = model({
                'keypoints0': kpts0,
                'keypoints1': kpts1,
                'descriptors0': desc0,
                'descriptors1': desc1,
                'scores0': scores0,
                'scores1': scores1,
                'image0': img0_tensor,
                'image1': img1_tensor,
            })
        matches0 = result['matches0'][0].cpu().numpy()
        matches_dict[(img0, img1)] = matches0
        print(f"Matches saved for {img0} and {img1} - {matches0.shape[0]} matches")
    np.savez(output_path, matches=matches_dict)
    print(f"SuperGlue matches saved to {output_path}")

def export_superglue2colmap_format(features_path, matches_path, colmap_desc_dir, matches_txt_path, image_dir, min_matches_per_pair=15):
    """
    features_path: superpoint_features.npz
    matches_path: superglue_matches.npz
    colmap_desc_dir: COLMAP keypoint txt 저장 폴더 (colmap_desc/)
    matches_txt_path: COLMAP matches.txt 저장 경로
    image_dir: 실제 이미지가 있는 폴더 경로
    min_matches_per_pair: matches.txt에 기록할 최소 매칭 수
    """
    import numpy as np
    import os
    os.makedirs(colmap_desc_dir, exist_ok=True)
    features = np.load(features_path, allow_pickle=True)
    matches_data = np.load(matches_path, allow_pickle=True)
    keypoints = features['keypoints'].item()
    matches_dict = matches_data['matches'].item()

    # 실제 이미지 파일명 리스트 (정확한 대소문자, 확장자 포함)
    image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    # 1. 각 이미지별 COLMAP keypoint txt 생성 (실제 이미지명 기준)
    for img_name in image_list:
        kps = keypoints.get(img_name)
        if kps is None:
            continue
        with open(os.path.join(colmap_desc_dir, f"{img_name}.txt"), 'w') as f:
            f.write(f"{kps.shape[0]} 128\n")
            for r in range(kps.shape[0]):
                f.write(f"{kps[r,0]} {kps[r,1]} 0.00 0.00\n")

    # 2. 전체 쌍에 대해 matches.txt 생성 (쌍별로 실제 이미지명 사용)
    with open(matches_txt_path, 'w') as f:
        for (im1, im2), matches in matches_dict.items():
            valid_matches = [(i, int(m)) for i, m in enumerate(matches) if m != -1]
            if len(valid_matches) < min_matches_per_pair:
                continue  # 매칭 min_matches_per_pair개 미만 쌍은 기록하지 않음
            f.write(f"{im1} {im2}\n")
            for i, m in valid_matches:
                f.write(f"{i} {m}\n")
            f.write("\n\n")

def validate_data(features_path, matches_path, image_dir, desc_dir, num_visualize=3):
    print("\n[Validation] Checking data consistency and match quality...")
    features = np.load(features_path, allow_pickle=True)
    matches_data = np.load(matches_path, allow_pickle=True)
    keypoints = features['keypoints'].item()
    descriptors = features['descriptors'].item()
    scores = features['scores'].item()
    matches_dict = matches_data['matches'].item()
    image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    image_list_clean = [img.strip().lower() for img in image_list]
    keypoints_imgs = set([k.strip().lower() for k in keypoints.keys()])
    desc_imgs = set([os.path.splitext(f)[0] for f in os.listdir(desc_dir) if f.endswith('.txt')])
    matches_imgs = set()
    for k in matches_dict.keys():
        matches_imgs.add(k[0].strip().lower())
        matches_imgs.add(k[1].strip().lower())
    print(f"- Images in folder: {len(image_list_clean)}")
    print(f"- Images in keypoints: {len(keypoints_imgs)}")
    print(f"- Images in desc_dir: {len(desc_imgs)}")
    print(f"- Images in matches: {len(matches_imgs)}")
    missing_in_kp = set(image_list_clean) - keypoints_imgs
    missing_in_desc = set(image_list_clean) - desc_imgs
    missing_in_matches = set(image_list_clean) - matches_imgs
    if missing_in_kp:
        print(f"[WARNING] Images missing in keypoints: {missing_in_kp}")
    if missing_in_desc:
        print(f"[WARNING] Images missing in desc_dir: {missing_in_desc}")
    if missing_in_matches:
        print(f"[WARNING] Images missing in matches: {missing_in_matches}")
    print(f"- Total keypoints files: {len(keypoints)}")
    print(f"- Total matches pairs: {len(matches_dict)}")
    match_counts = [np.sum(np.array(m) != -1) for m in matches_dict.values()]
    if match_counts:
        print(f"- Match count per pair: min={np.min(match_counts)}, max={np.max(match_counts)}, mean={np.mean(match_counts):.1f}")
    else:
        print("[WARNING] No matches found!")
    # 4. 랜덤 매칭 시각화 (실제 매칭된 점을 선으로 연결)
    if num_visualize > 0 and len(matches_dict) > 0:
        pairs = list(matches_dict.keys())
        for i in range(min(num_visualize, len(pairs))):
            pair = random.choice(pairs)
            img0, img1 = pair
            img0_clean = img0.strip().lower()
            img1_clean = img1.strip().lower()
            img0_path = os.path.join(image_dir, img0)
            img1_path = os.path.join(image_dir, img1)
            if not (os.path.exists(img0_path) and os.path.exists(img1_path)):
                continue
            kpts0 = keypoints[img0]
            kpts1 = keypoints[img1]
            matches = matches_dict[pair]
            img0_pil = Image.open(img0_path).convert('RGB')
            img1_pil = Image.open(img1_path).convert('RGB')
            fig, axes = plt.subplots(1,2,figsize=(12,6))
            axes[0].imshow(img0_pil)
            axes[0].scatter(kpts0[:,0], kpts0[:,1], s=2, c='r')
            axes[0].set_title(img0)
            axes[1].imshow(img1_pil)
            axes[1].scatter(kpts1[:,0], kpts1[:,1], s=2, c='b')
            axes[1].set_title(img1)
            # 매칭된 점을 선으로 연결
            for idx0, idx1 in [(i, int(m)) for i, m in enumerate(matches) if m != -1]:
                x0, y0 = kpts0[idx0]
                x1, y1 = kpts1[idx1]
                con = ConnectionPatch(xyA=(x1, y1), xyB=(x0, y0), coordsA="data", coordsB="data",
                                     axesA=axes[1], axesB=axes[0], color="lime", linewidth=1, alpha=0.7)
                axes[1].add_artist(con)
            plt.suptitle(f"Sample match: {img0} <-> {img1} (matches: {np.sum(np.array(matches)!=-1)})")
            plt.tight_layout()
            out_path = f"validation_match_{i}_{img0_clean}_{img1_clean}.png"
            plt.savefig(out_path)
            plt.close()
            print(f"  - Saved match visualization: {out_path}")
    print("[Validation] Done.\n")

def run_colmap_mapper(database_path, image_path, output_path):
    print("Step 4: COLMAP sparse reconstruction (mapper)")
    os.makedirs(output_path, exist_ok=True)
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", output_path,
        "--Mapper.init_min_num_inliers", "8",
        "--log_to_stderr", "1"
    ], check=True)

def run_train_py():
    print("Step 5: Run train.py")
    subprocess.run([
        "python", "train.py"
    ], check=True)

def run_colmap_feature_importer(database_path, image_path, desc_path):
    print("Step 3-1: COLMAP feature_importer (keypoints)")
    subprocess.run([
        "colmap", "feature_importer",
        "--database_path", database_path,
        "--image_path", image_path,
        "--import_path", desc_path
    ], check=True)

def run_colmap_matches_importer(database_path, matches_path):
    print("Step 3-2: COLMAP matches_importer")
    subprocess.run([
        "colmap", "matches_importer",
        "--database_path", database_path,
        "--match_list_path", matches_path,
        "--match_type", "raw",
        "--SiftMatching.guided_matching", "1"
    ], check=True)

if __name__ == "__main__":
    # 파라미터 쉽게 조정
    superpoint_config = {
        'nms_radius': 3,
        'keypoint_threshold': 0.002,
        'max_keypoints': 4096
    }
    superglue_config = {'match_threshold': 0.05}
    min_matches_per_pair = 15  # 원하는 값으로 조정
    extract_superpoint_features("ImageInputs/images", "ImageInputs/superpoint_features.npz", config=superpoint_config)
    match_superglue("ImageInputs/superpoint_features.npz", "ImageInputs/images", "ImageInputs/superglue_matches.npz", superglue_config=superglue_config)
    export_superglue2colmap_format(
        features_path="ImageInputs/superpoint_features.npz",
        matches_path="ImageInputs/superglue_matches.npz",
        colmap_desc_dir="ImageInputs/colmap_desc",
        matches_txt_path="ImageInputs/superglue_matches.txt",
        image_dir="ImageInputs/images",
        min_matches_per_pair=min_matches_per_pair
    )
    validate_data(
        "ImageInputs/superpoint_features.npz",
        "ImageInputs/superglue_matches.npz",
        "ImageInputs/images",
        "ImageInputs/colmap_desc",
        num_visualize=3
    )
    run_colmap_feature_importer("ImageInputs/database.db", "ImageInputs/images", "ImageInputs/colmap_desc")
    run_colmap_matches_importer("ImageInputs/database.db", "ImageInputs/superglue_matches.txt")
    run_colmap_mapper("ImageInputs/database.db", "ImageInputs/images", "ImageInputs/sparse")
    #run_train_py() 