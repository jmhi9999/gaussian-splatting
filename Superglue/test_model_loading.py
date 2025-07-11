#!/usr/bin/env python3
"""
SuperPoint 모델 로드 테스트 스크립트
"""

import sys
import os
from pathlib import Path
import torch

def test_superpoint_loading():
    """SuperPoint 모델 로드 테스트"""
    print("🔍 SuperPoint 모델 로드 테스트 시작")
    
    # 현재 디렉토리 확인
    current_dir = Path(__file__).parent
    print(f"현재 디렉토리: {current_dir}")
    
    # models 디렉토리 확인
    models_dir = current_dir / "models"
    print(f"models 디렉토리: {models_dir}")
    print(f"models 디렉토리 존재: {models_dir.exists()}")
    
    if models_dir.exists():
        print("models 디렉토리 내용:")
        for item in models_dir.iterdir():
            print(f"  - {item.name}")
    
    # 가중치 파일 확인
    weights_dir = models_dir / "weights"
    if weights_dir.exists():
        print(f"weights 디렉토리 내용:")
        for item in weights_dir.iterdir():
            print(f"  - {item.name} ({item.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Python 경로 확인
    print(f"Python 경로:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # SuperPoint import 시도
    print("\n🔧 SuperPoint import 시도...")
    
    try:
        # 방법 1: 직접 경로 import
        print("방법 1: 직접 경로 import")
        import importlib.util
        
        superpoint_path = models_dir / "superpoint.py"
        print(f"SuperPoint 파일 경로: {superpoint_path}")
        print(f"SuperPoint 파일 존재: {superpoint_path.exists()}")
        
        if superpoint_path.exists():
            spec = importlib.util.spec_from_file_location("superpoint", superpoint_path)
            superpoint_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(superpoint_module)
            
            SuperPoint = superpoint_module.SuperPoint
            print("✓ SuperPoint 클래스 import 성공")
            
            # 모델 인스턴스 생성 테스트
            print("SuperPoint 모델 인스턴스 생성 테스트...")
            config = {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            }
            
            model = SuperPoint(config)
            print("✓ SuperPoint 모델 인스턴스 생성 성공")
            
            # 테스트 추론
            print("SuperPoint 테스트 추론...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            test_tensor = torch.zeros(1, 1, 480, 640).to(device)
            with torch.no_grad():
                result = model({'image': test_tensor})
            
            print("✓ SuperPoint 테스트 추론 성공")
            print(f"결과 키: {list(result.keys())}")
            
            return True
            
        else:
            print("✗ SuperPoint 파일이 존재하지 않습니다")
            return False
            
    except Exception as e:
        print(f"✗ SuperPoint import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_superglue_loading():
    """SuperGlue 모델 로드 테스트"""
    print("\n🔧 SuperGlue import 시도...")
    
    try:
        # SuperGlue import
        import importlib.util
        
        superglue_path = Path(__file__).parent / "models" / "superglue.py"
        print(f"SuperGlue 파일 경로: {superglue_path}")
        
        if superglue_path.exists():
            spec = importlib.util.spec_from_file_location("superglue", superglue_path)
            superglue_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(superglue_module)
            
            SuperGlue = superglue_module.SuperGlue
            print("✓ SuperGlue 클래스 import 성공")
            
            # 모델 인스턴스 생성 테스트
            print("SuperGlue 모델 인스턴스 생성 테스트...")
            config = {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
            
            model = SuperGlue(config)
            print("✓ SuperGlue 모델 인스턴스 생성 성공")
            
            return True
            
        else:
            print("✗ SuperGlue 파일이 존재하지 않습니다")
            return False
            
    except Exception as e:
        print(f"✗ SuperGlue import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 SuperPoint/SuperGlue 모델 로드 테스트")
    print("=" * 50)
    
    # SuperPoint 테스트
    superpoint_success = test_superpoint_loading()
    
    # SuperGlue 테스트
    superglue_success = test_superglue_loading()
    
    print("\n" + "=" * 50)
    print("📊 테스트 결과:")
    print(f"SuperPoint: {'✅ 성공' if superpoint_success else '❌ 실패'}")
    print(f"SuperGlue: {'✅ 성공' if superglue_success else '❌ 실패'}")
    
    if superpoint_success and superglue_success:
        print("\n🎉 모든 모델 로드 성공!")
    else:
        print("\n⚠️  일부 모델 로드 실패") 