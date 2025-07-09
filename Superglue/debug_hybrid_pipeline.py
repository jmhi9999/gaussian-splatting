#!/usr/bin/env python3
"""
SuperGlue + COLMAP 하이브리드 파이프라인 디버깅 스크립트
"""


import os
import sys
import subprocess
import sqlite3
from pathlib import Path


def check_environment():
   """환경 설정 확인"""
   print("🔍 환경 설정 확인 중...")
  
   # 1. COLMAP 설치 확인
   try:
       result = subprocess.run(["colmap", "--help"], capture_output=True, text=True)
       if result.returncode == 0:
           print("  ✓ COLMAP 설치됨")
       else:
           print("  ✗ COLMAP 실행 실패")
           return False
   except FileNotFoundError:
       print("  ✗ COLMAP이 설치되지 않음")
       print("    해결: sudo apt install colmap  또는  conda install colmap")
       return False
  
   # 2. Python 모듈 확인
   required_modules = ["torch", "cv2", "numpy"]
   for module in required_modules:
       try:
           __import__(module)
           print(f"  ✓ {module} 모듈 확인")
       except ImportError:
           print(f"  ✗ {module} 모듈 없음")
           return False
  
   # 3. 3DGS 모듈 경로 확인
   current_dir = Path.cwd()
   scene_path = current_dir / "scene"
   utils_path = current_dir / "utils"
  
   if scene_path.exists() and utils_path.exists():
       print("  ✓ 3DGS 모듈 경로 확인")
   else:
       print("  ⚠️  3DGS 모듈 경로 없음 (기본 구현 사용)")
  
   # 4. 이미지 디렉토리 확인
   image_dir = Path("ImageInputs/images")
   if image_dir.exists():
       image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
       print(f"  ✓ 이미지 디렉토리 확인 ({len(image_files)}장)")
       if len(image_files) == 0:
           print("  ⚠️  이미지 파일이 없음")
   else:
       print("  ✗ 이미지 디렉토리 없음")
       return False
  
   return True


def check_database_schema(db_path):
   """데이터베이스 스키마 확인"""
   print("\n🗄️ 데이터베이스 스키마 확인...")
  
   if not db_path.exists():
       print("  ✗ 데이터베이스 파일 없음")
       return False
  
   try:
       conn = sqlite3.connect(str(db_path))
       cursor = conn.cursor()
      
       # 테이블 목록 확인
       cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
       tables = [row[0] for row in cursor.fetchall()]
      
       expected_tables = ['cameras', 'images', 'keypoints', 'descriptors', 'matches']
      
       for table in expected_tables:
           if table in tables:
               cursor.execute(f"SELECT COUNT(*) FROM {table}")
               count = cursor.fetchone()[0]
               print(f"  ✓ {table}: {count}개 레코드")
           else:
               print(f"  ✗ {table} 테이블 없음")
      
       # 스키마 상세 확인
       cursor.execute("PRAGMA table_info(cameras)")
       camera_schema = cursor.fetchall()
       print(f"  카메라 스키마: {len(camera_schema)}개 컬럼")
      
       conn.close()
       return True
      
   except Exception as e:
       print(f"  ✗ 데이터베이스 확인 실패: {e}")
       return False


def fix_database_permissions(db_path):
   """데이터베이스 권한 수정"""
   print("\n🔧 데이터베이스 권한 수정...")
  
   try:
       # 파일 권한 변경
       os.chmod(db_path, 0o666)
      
       # 디렉토리 권한 변경
       os.chmod(db_path.parent, 0o777)
      
       print("  ✓ 권한 수정 완료")
       return True
      
   except Exception as e:
       print(f"  ✗ 권한 수정 실패: {e}")
       return False


def clean_output_directory():
   """출력 디렉토리 정리"""
   print("\n🧹 출력 디렉토리 정리...")
  
   output_path = Path("ImageInputs/superglue_colmap_hybrid_output")
  
   if output_path.exists():
       import shutil
       shutil.rmtree(output_path)
       print("  ✓ 기존 출력 디렉토리 삭제")
  
   output_path.mkdir(parents=True, exist_ok=True)
   print("  ✓ 새 출력 디렉토리 생성")
  
   return output_path


def test_colmap_basic():
   """기본 COLMAP 기능 테스트"""
   print("\n🧪 COLMAP 기본 기능 테스트...")
  
   # 임시 디렉토리 생성
   test_dir = Path("temp_colmap_test")
   test_dir.mkdir(exist_ok=True)
  
   try:
       # 빈 데이터베이스 생성 테스트
       db_path = test_dir / "test.db"
      
       conn = sqlite3.connect(str(db_path))
       cursor = conn.cursor()
      
       # 간단한 테이블 생성
       cursor.execute('''
           CREATE TABLE test_table (
               id INTEGER PRIMARY KEY,
               name TEXT
           )
       ''')
      
       cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test",))
       conn.commit()
       conn.close()
      
       print("  ✓ SQLite 기본 동작 확인")
      
       # COLMAP database_creator 테스트
       cmd = ["colmap", "database_creator", "--database_path", str(db_path)]
       result = subprocess.run(cmd, capture_output=True, text=True)
      
       if result.returncode == 0:
           print("  ✓ COLMAP database_creator 동작 확인")
       else:
           print(f"  ⚠️  COLMAP database_creator 실패: {result.stderr}")
      
       return True
      
   except Exception as e:
       print(f"  ✗ COLMAP 테스트 실패: {e}")
       return False
  
   finally:
       # 정리
       if test_dir.exists():
           import shutil
           shutil.rmtree(test_dir)


def create_minimal_database(image_paths, db_path):
   """최소한의 동작하는 데이터베이스 생성"""
   print("\n🔨 최소 데이터베이스 생성...")
  
   try:
       # 기존 파일 삭제
       if db_path.exists():
           db_path.unlink()
      
       # COLMAP의 database_creator 사용
       cmd = ["colmap", "database_creator", "--database_path", str(db_path)]
       result = subprocess.run(cmd, capture_output=True, text=True)
      
       if result.returncode != 0:
           print(f"  ✗ database_creator 실패: {result.stderr}")
           return False
      
       print("  ✓ COLMAP database_creator 성공")
      
       # 이미지 정보 추가
       conn = sqlite3.connect(str(db_path))
       cursor = conn.cursor()
      
       # 기본 카메라 추가 (SIMPLE_PINHOLE 모델)
       import cv2
       import numpy as np
      
       sample_img = cv2.imread(str(image_paths[0]))
       height, width = sample_img.shape[:2]
      
       # SIMPLE_PINHOLE 모델 (model=0): [f, cx, cy]
       focal = max(width, height) * 1.2
       params = np.array([focal, width/2, height/2], dtype=np.float64)
      
       cursor.execute(
           "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
           (0, width, height, params.tobytes(), focal)
       )
      
       camera_id = cursor.lastrowid
      
       # 이미지 정보 추가
       for i, img_path in enumerate(image_paths[:10]):  # 처음 10장만
           image_name = f"image_{i:04d}.jpg"
           cursor.execute(
               "INSERT INTO images (name, camera_id) VALUES (?, ?)",
               (image_name, camera_id)
           )
      
       conn.commit()
       conn.close()
      
       print(f"  ✓ {len(image_paths[:10])}장 이미지 정보 추가")
       return True
      
   except Exception as e:
       print(f"  ✗ 최소 데이터베이스 생성 실패: {e}")
       return False


def run_simplified_pipeline():
   """단순화된 파이프라인 실행"""
   print("\n🚀 단순화된 파이프라인 실행...")
  
   # 1. 환경 확인
   if not check_environment():
       return False
  
   # Qt GUI 문제 해결을 위한 환경 변수 설정
   env = os.environ.copy()
   env["QT_QPA_PLATFORM"] = "offscreen"
   env["DISPLAY"] = ":0"
  
   # xvfb가 사용 가능한지 확인하고 사용
   try:
       xvfb_result = subprocess.run(["which", "xvfb-run"], capture_output=True, text=True)
       if xvfb_result.returncode == 0:
           print("  ✓ xvfb-run 사용 가능")
           use_xvfb = True
       else:
           print("  ⚠️  xvfb-run 없음, offscreen 모드 사용")
           use_xvfb = False
   except:
       use_xvfb = False
  
   # 2. 출력 디렉토리 정리
   output_path = clean_output_directory()
  
   # 3. 이미지 수집
   image_dir = Path("ImageInputs/images")
   image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
   image_paths = sorted(image_paths)[:20]  # 처음 20장만
  
   print(f"  선택된 이미지: {len(image_paths)}장")
  
   # 4. 입력 이미지 복사
   input_dir = output_path / "input"
   input_dir.mkdir(exist_ok=True)
  
   import shutil
   for i, src in enumerate(image_paths):
       dst = input_dir / f"image_{i:04d}.jpg"
       shutil.copy2(src, dst)
  
   print(f"  ✓ 이미지 복사 완료")
  
   # 5. 최소 데이터베이스 생성
   db_path = output_path / "database.db"
   if not create_minimal_database(image_paths, db_path):
       return False
  
   # 6. 데이터베이스 스키마 확인
   check_database_schema(db_path)
  
   # 7. 권한 수정
   fix_database_permissions(db_path)
  
   # 8. COLMAP 특징점 추출
   print("\n🔍 COLMAP 특징점 추출...")
   base_cmd = [
       "colmap", "feature_extractor",
       "--database_path", str(db_path),
       "--image_path", str(input_dir),
       "--ImageReader.single_camera", "1",
       "--SiftExtraction.max_num_features", "1000"
   ]
  
   if use_xvfb:
       cmd = ["xvfb-run", "-a"] + base_cmd
   else:
       cmd = base_cmd
  
   result = subprocess.run(cmd, capture_output=True, text=True, env=env)
   if result.returncode == 0:
       print("  ✓ 특징점 추출 성공")
   else:
       print(f"  ✗ 특징점 추출 실패: {result.stderr}")
       return False
  
   # 9. COLMAP 매칭
   print("\n🔗 COLMAP 매칭...")
   base_cmd = [
       "colmap", "exhaustive_matcher",
       "--database_path", str(db_path)
   ]
  
   if use_xvfb:
       cmd = ["xvfb-run", "-a"] + base_cmd
   else:
       cmd = base_cmd
  
   result = subprocess.run(cmd, capture_output=True, text=True, env=env)
   if result.returncode == 0:
       print("  ✓ 매칭 성공")
   else:
       print(f"  ✗ 매칭 실패: {result.stderr}")
       return False
  
   # 10. COLMAP 매퍼 (관대한 설정)
   print("\n📐 COLMAP 매퍼...")
   sparse_dir = output_path / "sparse"
   sparse_dir.mkdir(exist_ok=True)
  
   base_cmd = [
       "colmap", "mapper",
       "--database_path", str(db_path),
       "--image_path", str(input_dir),
       "--output_path", str(sparse_dir),
       "--Mapper.min_num_matches", "4",
       "--Mapper.init_min_num_inliers", "8",
       "--Mapper.abs_pose_min_num_inliers", "4",
       "--Mapper.filter_max_reproj_error", "16.0"
   ]
  
   if use_xvfb:
       cmd = ["xvfb-run", "-a"] + base_cmd
   else:
       cmd = base_cmd
  
   result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
   if result.returncode == 0:
       print("  ✓ 매퍼 성공")
      
       # 결과 확인
       recon_dirs = list(sparse_dir.glob("*/"))
       if recon_dirs:
           print(f"  생성된 reconstruction: {len(recon_dirs)}개")
           for recon_dir in recon_dirs:
               bin_files = list(recon_dir.glob("*.bin"))
               print(f"    {recon_dir.name}: {len(bin_files)}개 파일")
      
       return True
   else:
       print(f"  ✗ 매퍼 실패")
       print(f"  stdout: {result.stdout}")
       print(f"  stderr: {result.stderr}")
       return False


def main():
   """메인 함수"""
   print("🔧 SuperGlue + COLMAP 하이브리드 파이프라인 디버거")
   print("=" * 60)
  
   # 기본 환경 테스트
   if not check_environment():
       print("\n❌ 환경 설정에 문제가 있습니다.")
       return
  
   # COLMAP 기본 기능 테스트
   if not test_colmap_basic():
       print("\n❌ COLMAP 기본 기능에 문제가 있습니다.")
       return
  
   # 단순화된 파이프라인 실행
   if run_simplified_pipeline():
       print("\n✅ 단순화된 파이프라인 성공!")
       print("\n다음 단계:")
       print("1. 수정된 superglue_colmap_hybrid_fixed.py 사용")
       print("2. python superglue_colmap_hybrid_fixed.py --source_path ImageInputs/images")
   else:
       print("\n❌ 단순화된 파이프라인 실패")
       print("\n문제 해결 방안:")
       print("1. COLMAP 재설치: sudo apt install colmap")
       print("2. 이미지 품질 확인")
       print("3. 디스크 공간 확인")
       print("4. 권한 문제: sudo chmod -R 777 ImageInputs/")


if __name__ == "__main__":
   main()


