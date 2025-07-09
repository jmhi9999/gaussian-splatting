#!/usr/bin/env python3
"""
개선된 SuperGlue 매칭 테스트 스크립트
고해상도 이미지에서 매칭 품질을 확인합니다.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# SuperGlue import
sys.path.append('Superglue')
from superglue_matcher import SuperGlueMatcher

def test_matching_quality(image_dir, output_dir="matching_results"):
    """매칭 품질 테스트"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 경로 수집
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(Path(image_dir).glob(ext))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    if len(image_paths) < 2:
        print("Need at least 2 images for testing")
        return
    
    # SuperGlue 매처 초기화
    matcher = SuperGlueMatcher()
    
    # 테스트 1: 기본 매칭
    print("\n=== Test 1: Basic Matching ===")
    result1 = matcher.match_image_pair(image_paths[0], image_paths[1])
    if result1:
        print(f"  Basic matching: {len(result1['matched_kpts0'])} matches")
        print(f"  Resize used: {result1.get('resize_used', 'None')}")
    else:
        print("  Basic matching failed")
    
    # 테스트 2: 품질 체크 매칭
    print("\n=== Test 2: Quality Check Matching ===")
    result2 = matcher.match_with_quality_check(image_paths[0], image_paths[1], min_matches=10)
    if result2:
        print(f"  Quality matching: {len(result2['matched_kpts0'])} matches")
        print(f"  Resize used: {result2.get('resize_used', 'None')}")
    else:
        print("  Quality matching failed")
    
    # 테스트 3: 다중 이미지 매칭
    print("\n=== Test 3: Multiple Image Matching ===")
    if len(image_paths) >= 3:
        results3 = matcher.match_multiple_images_with_quality(image_paths[:3], min_matches=5)
        print(f"  Multiple matching: {len(results3)} successful pairs")
        
        for pair_key, result in results3.items():
            print(f"    Pair {pair_key}: {len(result['matched_kpts0'])} matches")
    
    # 매칭 결과 시각화
    if result1 and len(result1['matched_kpts0']) > 0:
        visualize_matches(image_paths[0], image_paths[1], result1, 
                        os.path.join(output_dir, "basic_matching.jpg"))
    
    if result2 and len(result2['matched_kpts0']) > 0:
        visualize_matches(image_paths[0], image_paths[1], result2, 
                        os.path.join(output_dir, "quality_matching.jpg"))

def visualize_matches(img_path1, img_path2, result, output_path):
    """매칭 결과 시각화"""
    try:
        # 이미지 로드
        img1 = cv2.imread(str(img_path1))
        img2 = cv2.imread(str(img_path2))
        
        if img1 is None or img2 is None:
            print(f"  Failed to load images for visualization")
            return
        
        # 매칭 포인트 추출
        kpts1 = result['matched_kpts0']
        kpts2 = result['matched_kpts1']
        
        # 이미지 크기 조정 (시각화용)
        max_height = 600
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 > max_height:
            scale1 = max_height / h1
            new_w1 = int(w1 * scale1)
            new_h1 = int(h1 * scale1)
            img1 = cv2.resize(img1, (new_w1, new_h1))
            kpts1 = kpts1 * scale1
        
        if h2 > max_height:
            scale2 = max_height / h2
            new_w2 = int(w2 * scale2)
            new_h2 = int(h2 * scale2)
            img2 = cv2.resize(img2, (new_w2, new_h2))
            kpts2 = kpts2 * scale2
        
        # 매칭 라인 그리기
        h1, w1 = img1.shape[:2]
        combined_img = np.hstack([img1, img2])
        
        # 매칭 포인트 그리기
        for i in range(min(len(kpts1), len(kpts2))):
            pt1 = tuple(map(int, kpts1[i]))
            pt2 = tuple(map(int, (kpts2[i][0] + w1, kpts2[i][1])))
            
            # 랜덤 색상
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            
            # 포인트 그리기
            cv2.circle(combined_img, pt1, 3, color, -1)
            cv2.circle(combined_img, pt2, 3, color, -1)
            
            # 라인 그리기
            cv2.line(combined_img, pt1, pt2, color, 1)
        
        # 결과 저장
        cv2.imwrite(output_path, combined_img)
        print(f"  Visualization saved to {output_path}")
        
    except Exception as e:
        print(f"  Visualization failed: {e}")

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("Usage: python test_improved_matching.py <image_directory>")
        print("Example: python test_improved_matching.py ImageInputs/images/")
        return
    
    image_dir = sys.argv[1]
    
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} does not exist")
        return
    
    print(f"Testing improved SuperGlue matching on {image_dir}")
    test_matching_quality(image_dir)

if __name__ == "__main__":
    main() 