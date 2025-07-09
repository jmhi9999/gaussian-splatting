#!/usr/bin/env python3
"""
SuperGlue + COLMAP 하이브리드 파이프라인 실행 스크립트
3DGS train.py와 통합된 파이프라인을 실행합니다.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="SuperGlue + COLMAP 하이브리드 파이프라인")
    parser.add_argument("--source_path", type=str, required=True,
                       help="입력 이미지 디렉토리")
    parser.add_argument("--model_path", type=str, default=None,
                       help="출력 모델 디렉토리 (기본값: ./output/auto)")
    parser.add_argument("--max_images", type=int, default=100,
                       help="최대 처리 이미지 수 (기본값: 100)")
    parser.add_argument("--superglue_config", type=str, default="outdoor",
                       help="SuperGlue 설정 (indoor/outdoor, 기본값: outdoor)")
    parser.add_argument("--iterations", type=int, default=30000,
                       help="3DGS 학습 반복 수 (기본값: 30000)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="GPU 디바이스 (기본값: cuda)")
    parser.add_argument("--colmap_exe", type=str, default="colmap",
                       help="COLMAP 실행 파일 경로 (기본값: colmap)")
    parser.add_argument("--skip_training", action="store_true",
                       help="SfM만 실행하고 3DGS 학습은 건너뛰기")
    
    args = parser.parse_args()
    
    # 경로 확인
    source_path = Path(args.source_path)
    if not source_path.exists():
        print(f"❌ 오류: 입력 경로가 존재하지 않습니다: {source_path}")
        sys.exit(1)
    
    # 모델 경로 설정
    if args.model_path is None:
        import uuid
        unique_str = str(uuid.uuid4())[:10]
        model_path = Path(f"./output/{unique_str}")
    else:
        model_path = Path(args.model_path)
    
    print("🚀 SuperGlue + COLMAP 하이브리드 파이프라인 시작")
    print("=" * 60)
    print(f"📁 입력 경로: {source_path}")
    print(f"📁 출력 경로: {model_path}")
    print(f"🖼️  최대 이미지: {args.max_images}")
    print(f"🔧 SuperGlue 설정: {args.superglue_config}")
    print(f"🎯 3DGS 반복 수: {args.iterations}")
    print(f"💻 디바이스: {args.device}")
    
    # 1단계: SuperGlue + COLMAP 하이브리드 SfM
    print("\n📊 1단계: SuperGlue + COLMAP 하이브리드 SfM 실행")
    print("-" * 40)
    
    # train.py 명령 구성
    train_cmd = [
        sys.executable, "train.py",
        "--source_path", str(source_path),
        "--model_path", str(model_path),
        "--scene_type", "SuperGlueCOLMAPHybrid",
        "--superglue_config", args.superglue_config,
        "--max_images", str(args.max_images),
        "--data_device", args.device,
        "--iterations", str(args.iterations),
        "--colmap_exe", args.colmap_exe,  # COLMAP 경로 추가
        "--quiet"  # 출력 줄이기
    ]
    
    print(f"실행 명령: {' '.join(train_cmd)}")
    
    try:
        # train.py 실행 (SfM 단계만)
        print("\n🔥 SfM 파이프라인 실행 중...")
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ SfM 파이프라인 성공!")
            
            if not args.skip_training:
                print("\n📊 2단계: 3DGS 학습 시작")
                print("-" * 40)
                
                # 학습 명령 (SfM 결과 사용)
                train_cmd = [
                    sys.executable, "train.py",
                    "--source_path", str(source_path),
                    "--model_path", str(model_path),
                    "--scene_type", "SuperGlueCOLMAPHybrid",
                    "--iterations", str(args.iterations),
                    "--data_device", args.device
                ]
                
                print(f"학습 명령: {' '.join(train_cmd)}")
                
                # 학습 실행
                print("\n🔥 3DGS 학습 실행 중...")
                result = subprocess.run(train_cmd, timeout=7200)  # 2시간 타임아웃
                
                if result.returncode == 0:
                    print("✅ 3DGS 학습 완료!")
                    print(f"\n🎉 모든 파이프라인 성공!")
                    print(f"📁 결과 위치: {model_path}")
                    print(f"📊 다음 명령으로 결과 확인:")
                    print(f"   python train.py --source_path {source_path} --model_path {model_path} --scene_type SuperGlueCOLMAPHybrid")
                else:
                    print("❌ 3DGS 학습 실패")
                    sys.exit(1)
            else:
                print("⏭️  3DGS 학습 건너뛰기 (--skip_training)")
                print(f"📁 SfM 결과 위치: {model_path}")
        else:
            print("❌ SfM 파이프라인 실패")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("❌ 타임아웃: 파이프라인이 너무 오래 걸렸습니다")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 