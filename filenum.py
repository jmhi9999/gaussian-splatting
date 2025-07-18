import os
import argparse

def rename_files(start_num, directory):
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # List only .jpg files and sort them
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith('.jpg'))

    if not files:
        print("No .jpg files found in the directory.")
        return

    print(f"Renaming {len(files)} files in '{directory}' starting from {start_num:04d}...")

    for i, filename in enumerate(files):
        new_name = f"{start_num + i:04d}.jpg"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)

        # Avoid overwriting files unintentionally
        if os.path.exists(dst):
            print(f"Warning: {dst} already exists! Skipping.")
            continue

        os.rename(src, dst)
        print(f"Renamed '{filename}' -> '{new_name}'")

    print("Renaming completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename image files to start from a specific number.")
    parser.add_argument("-s", "--start", type=int, required=True, help="Starting number for filenames (e.g., 100)")
    parser.add_argument("-d", "--dir", required=True, help="Directory containing image files")

    args = parser.parse_args()
    rename_files(args.start, args.dir)

