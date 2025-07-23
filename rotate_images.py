#!/usr/bin/env python3
"""
rotate_images.py

폴더 내 모든 이미지 파일을 시계방향으로 90도 회전합니다.
사용법:
    python rotate_images.py --input_folder /path/to/folder [--output_folder /path/to/output]
만약 output_folder를 지정하지 않으면 input_folder 내에서 파일이 덮어쓰기 됩니다.
"""
import os
import argparse
import cv2

def rotate_images(input_folder, output_folder=None):
    # 출력 폴더 설정
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(supported_ext):
            continue

        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)

        img = cv2.imread(src_path)
        if img is None:
            print(f'[경고] 이미지 읽기 실패: {src_path}')
            continue

        # 시계 방향으로 90도 회전
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # 저장
        cv2.imwrite(dst_path, rotated)
        print(f'[완료] 회전된 이미지 저장: {dst_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='폴더 내 이미지 일괄 90° 시계 방향 회전')
    parser.add_argument('--input_folder', '-i', required=True, help='이미지 폴더 경로')
    parser.add_argument('--output_folder', '-o', help='회전된 이미지를 저장할 폴더 (없으면 덮어쓰기)')
    args = parser.parse_args()

    rotate_images(args.input_folder, args.output_folder)