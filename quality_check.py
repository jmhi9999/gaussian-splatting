def quick_check():
    from scene.colmap_loader import read_extrinsics_binary
    
    # 전체 이미지 수
    import os
    all_images = [f for f in os.listdir("ImageInputs/images") 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"전체 이미지: {len(all_images)}장")
    
    # register된 이미지 수
    cam_extrinsics = read_extrinsics_binary("ImageInputs/sparse/0/images.bin")
    registered_images = [cam.name for cam in cam_extrinsics.values()]
    print(f"Register된 이미지: {len(registered_images)}장")
    
    # 실패한 이미지들
    failed_images = set(all_images) - set(registered_images)
    print(f"실패한 이미지: {len(failed_images)}장")
    
    if failed_images:
        print("\n재촬영 필요한 이미지들:")
        for img in sorted(failed_images):
            print(f"- {img}")

quick_check()