import os
import subprocess
import torch
import numpy as np
from PIL import Image
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import random
import matplotlib.pyplot as plt

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
    np.savez(output_path, keypoints=all_keypoints, descriptors=all_descriptors, scores=all_scores)
    print(f"SuperPoint features saved to {output_path}")

def generate_image_pairs(image_list, partial_gap=5):
    pairs = []
    # Sequential pairs
    for i in range(len(image_list) - 1):
        pairs.append((image_list[i], image_list[i+1]))
    # Partial exhaustive: every partial_gap-th image와 모든 이미지 쌍
    for i in range(0, len(image_list), partial_gap):
        for j in range(i+1, len(image_list)):
            pairs.append((image_list[i], image_list[j]))
    return pairs

def match_superglue(features_path, image_dir, output_path, superglue_config=None, partial_gap=5):
    print("Step 2: SuperGlue feature matching (direct, GPU 지원)")
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if superglue_config is None:
        superglue_config = {'match_threshold': 0.1}
    else:
        if 'match_threshold' not in superglue_config:
            superglue_config['match_threshold'] = 0.1
    model = SuperGlue(superglue_config).to(device)
    model.eval()
    data = np.load(features_path, allow_pickle=True)
    keypoints = data['keypoints'].item()
    descriptors = data['descriptors'].item()
    scores = data['scores'].item()
    image_list = sorted(keypoints.keys())
    pairs = generate_image_pairs(image_list, partial_gap=partial_gap)
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
    np.savez(output_path, matches=matches_dict)
    print(f"SuperGlue matches saved to {output_path}")

def write_colmap_files(features_path, matches_path, desc_dir, matches_txt_path, img_format='.jpg'):
    print("Step 3: Convert SuperGlue results to COLMAP .txt format (내장 구현)")
    os.makedirs(desc_dir, exist_ok=True)
    features = np.load(features_path, allow_pickle=True)
    matches_data = np.load(matches_path, allow_pickle=True)
    keypoints = features['keypoints'].item()
    matches_dict = matches_data['matches'].item()
    # 1. 각 이미지별 COLMAP keypoint 텍스트 파일 생성
    for img_name, kps in keypoints.items():
        # 이미지 이름을 소문자 및 strip 처리 (DB와 일치 보장)
        img_name_clean = img_name.strip().lower()
        desc_file = os.path.join(desc_dir, f"{img_name_clean}.txt")
        with open(desc_file, 'w') as f:
            f.write(f"{kps.shape[0]} 128\n")
            for r in range(kps.shape[0]):
                # scale=1.0, orientation=0.0으로 저장 (COLMAP SIFT txt 포맷)
                f.write(f"{kps[r,0]} {kps[r,1]} 1.0 0.0\n")
    # 2. 쌍별 matches.txt 생성
    with open(matches_txt_path, 'w') as f:
        for (im1, im2), matches in matches_dict.items():
            im1_clean = im1.strip().lower()
            im2_clean = im2.strip().lower()
            f.write(f"{im1_clean} {im2_clean}\n")
            for i, m in enumerate(matches):
                if m != -1:
                    f.write(f"{i} {int(m)}\n")
            f.write("\n\n")
    print(f"COLMAP keypoints saved to {desc_dir}, matches saved to {matches_txt_path}")

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
    # 1. 이미지 이름 일치 체크
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
    # 2. keypoint/matches 개수
    print(f"- Total keypoints files: {len(keypoints)}")
    print(f"- Total matches pairs: {len(matches_dict)}")
    # 3. 매칭 통계
    match_counts = [np.sum(np.array(m) != -1) for m in matches_dict.values()]
    if match_counts:
        print(f"- Match count per pair: min={np.min(match_counts)}, max={np.max(match_counts)}, mean={np.mean(match_counts):.1f}")
    else:
        print("[WARNING] No matches found!")
    # 4. 랜덤 매칭 시각화
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
            # Draw matches
            fig, ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].imshow(img0_pil)
            ax[0].scatter(kpts0[:,0], kpts0[:,1], s=2, c='r')
            ax[0].set_title(img0)
            ax[1].imshow(img1_pil)
            ax[1].scatter(kpts1[:,0], kpts1[:,1], s=2, c='b')
            ax[1].set_title(img1)
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
        "--output_path", output_path
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
        "--match_type", "raw"
    ], check=True)

if __name__ == "__main__":
    extract_superpoint_features("ImageInputs/images", "ImageInputs/superpoint_features.npz")
    match_superglue("ImageInputs/superpoint_features.npz", "ImageInputs/images", "ImageInputs/superglue_matches.npz")
    write_colmap_files(
        "ImageInputs/superpoint_features.npz",
        "ImageInputs/superglue_matches.npz",
        "ImageInputs/colmap_desc",
        "ImageInputs/superglue_matches.txt",
        img_format='.jpg'
    )
    # 데이터 유효성 검증
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
    run_train_py() 