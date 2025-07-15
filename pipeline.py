import os
import subprocess

# 1. SuperPoint feature extraction
print("Step 1: SuperPoint feature extraction")
subprocess.run([
    "python", "SuperGluePretrainedNetwork/extract_superpoint.py",
    "--input_dir", "ImageInputs/images",
    "--output", "ImageInputs/superpoint_features.npz"
], check=True)

# 2. SuperGlue feature matching (sequential + partial exhaustive)
print("Step 2: SuperGlue feature matching")
subprocess.run([
    "python", "SuperGluePretrainedNetwork/match_superglue.py",
    "--input_dir", "ImageInputs/images",
    "--features", "ImageInputs/superpoint_features.npz",
    "--output", "ImageInputs/superglue_matches.npz",
    "--matching", "sequential+partial_exhaustive"
], check=True)

# 3. Convert to COLMAP format
print("Step 3: Convert SuperGlue results to COLMAP format")
subprocess.run([
    "python", "Superglue/superglue2colmap.py",
    "--features", "ImageInputs/superpoint_features.npz",
    "--matches", "ImageInputs/superglue_matches.npz",
    "--database", "ImageInputs/database.db"
], check=True)

# 4. COLMAP sparse reconstruction (mapper)
print("Step 4: COLMAP sparse reconstruction")
os.makedirs("ImageInputs/sparse/0", exist_ok=True)
subprocess.run([
    "colmap", "mapper",
    "--database_path", "ImageInputs/database.db",
    "--image_path", "ImageInputs/images",
    "--output_path", "ImageInputs/sparse"
], check=True)

# 5. Run train.py
print("Step 5: Run train.py")
subprocess.run([
    "python", "train.py"
], check=True) 