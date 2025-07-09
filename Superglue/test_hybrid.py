#!/usr/bin/env python3
"""
SuperGlue + COLMAP 하이브리드 파이프라인 테스트 스크립트
"""

import os
import sys
from pathlib import Path
import subprocess

def test_requirements():
    """필수 요구사항 확인"""
    print("🔍 필수 요구사항 확인 중...")
    
    # Python 패키지 확인
    required_packages = ['torch', 'cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} (누락)")
    
    if missing_packages:
        print(f"\n❌ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print("pip install torch torchvision opencv-python numpy")
        return False
    
    # COLMAP 확인
    try:
        result = subprocess.run(['colmap', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✓ COLMAP")
        else:
            print("  ✗ COLMAP (설치 필요)")
            return False
    except Exception:
        print("  ✗ COLMAP (설치 필요)")
        print("Ubuntu/Debian: sudo apt-get install colmap")
        print("macOS: brew install colmap")
        return False
    
    # CUDA 확인
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠ CUDA (CPU 사용)")
    except:
        print("  ⚠ CUDA (확인 불가)")
    
    print("✅ 모든 요구사항 충족")
    return True

def test_superglue_models():
    """SuperGlue 모델 확인"""
    print("\n🔍 SuperGlue 모델 확인 중...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ✗ models 디렉토리 없음")
        return False
    
    required_files = [
        "models/matching.py",
        "models/superglue.py", 
        "models/superpoint.py",
        "models/utils.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}")
            return False
    
    # 가중치 파일 확인
    weights_dir = Path("models/weights")
    if weights_dir.exists():
        weight_files = list(weights_dir.glob("*.pth"))
        if weight_files:
            print(f"  ✓ 가중치 파일 {len(weight_files)}개")
        else:
            print("  ⚠ 가중치 파일 없음 (자동 다운로드 예정)")
    else:
        print("  ⚠ weights 디렉토리 없음 (자동 다운로드 예정)")
    
    print("✅ SuperGlue 모델 준비 완료")
    return True

def test_image_directory(image_dir):
    """이미지 디렉토리 확인"""
    print(f"\n🔍 이미지 디렉토리 확인: {image_dir}")
    
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"  ✗ 디렉토리 없음: {image_dir}")
        return False
    
    # 이미지 파일 찾기
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_path.glob(ext)))
    
    if not image_files:
        print(f"  ✗ 이미지 파일 없음")
        return False
    
    print(f"  ✓ 이미지 {len(image_files)}개 발견")
    
    # 샘플 이미지 정보
    sample_image = image_files[0]
    try:
        import cv2
        img = cv2.imread(str(sample_image))
        if img is not None:
            h, w = img.shape[:2]
            print(f"  ✓ 샘플 이미지 크기: {w}x{h}")
        else:
            print(f"  ⚠ 이미지 로드 실패")
    except Exception as e:
        print(f"  ⚠ 이미지 확인 실패: {e}")
    
    return True

def run_quick_test(image_dir, output_dir):
    """빠른 테스트 실행"""
    print(f"\n🚀 빠른 테스트 실행...")
    
    # 테스트용 작은 이미지 수
    max_images = min(10, len(list(Path(image_dir).glob("*.jpg"))))
    
    cmd = [
        sys.executable, "superglue_colmap_hybrid.py",
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--max_images", str(max_images)
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("✅ 테스트 성공!")
            return True
        else:
            print("❌ 테스트 실패")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 테스트 타임아웃 (30분)")
        return False
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 SuperGlue + COLMAP 하이브리드 파이프라인 테스트")
    print("=" * 60)
    
    # 1. 요구사항 확인
    if not test_requirements():
        sys.exit(1)
    
    # 2. SuperGlue 모델 확인
    if not test_superglue_models():
        sys.exit(1)
    
    # 3. 명령행 인수 확인
    if len(sys.argv) < 2:
        print("\n📝 사용법:")
        print("python test_hybrid.py <이미지_디렉토리> [출력_디렉토리]")
        print("\n예시:")
        print("python test_hybrid.py ./ImageInputs/images ./test_output")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"
    
    # 4. 이미지 디렉토리 확인
    if not test_image_directory(image_dir):
        sys.exit(1)
    
    # 5. 빠른 테스트 실행 (선택사항)
    if len(sys.argv) > 3 and sys.argv[3] == "--run-test":
        if not run_quick_test(image_dir, output_dir):
            sys.exit(1)
    
    print(f"\n🎉 모든 테스트 통과!")
    print(f"이제 다음 명령으로 실행하세요:")
    print(f"python superglue_colmap_hybrid.py --image_dir {image_dir} --output_dir {output_dir}")

if __name__ == "__main__":
    main() 