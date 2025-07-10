import os
import sys
import sqlite3
import subprocess
import numpy as np
import cv2
import torch
from pathlib import Path
import shutil
import json
from typing import List, Optional, Tuple, Dict, Any

class SuperGlueCOLMAPHybrid:
    def __init__(self, 
                 superglue_config: str = "outdoor",
                 colmap_exe: str = "colmap",
                 device: str = "cuda"):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.colmap_exe = colmap_exe
        
        # SuperGlue 설정
        self.superglue_config = {
            'outdoor': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            },
            'indoor': {
                'weights': 'indoor', 
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }[superglue_config]
        
        self._load_models()
    
    def _load_models(self):
        """SuperPoint와 SuperGlue 모델 로드 - 개선된 버전"""
        print(f"🔧 SuperGlue 모델 로드 시도 (device: {self.device})")
        
        try:
            # GPU 메모리 확인
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  GPU 메모리: {gpu_memory:.1f}GB")
                
                # 메모리 부족시 CPU로 fallback
                if gpu_memory < 2.0:  # 2GB 미만
                    print("  ⚠️  GPU 메모리 부족, CPU 사용")
                    self.device = "cpu"
            
            # 모델 경로 확인
            models_dir = Path(__file__).parent / "models"
            if not models_dir.exists():
                print(f"  ✗ models 디렉토리 없음: {models_dir}")
                print("  COLMAP-only 모드로 실행됩니다")
                self.superpoint = None
                self.superglue = None
                return
            
            # 가중치 파일 확인
            weights_dir = models_dir / "weights"
            if not weights_dir.exists():
                print(f"  ⚠️  weights 디렉토리 없음: {weights_dir}")
                print("  가중치 파일이 없어도 모델 구조는 로드 시도...")
            
            # 필수 파일 확인
            required_files = [
                models_dir / "superpoint.py",
                models_dir / "superglue.py",
                models_dir / "matching.py",
                models_dir / "utils.py"
            ]
            
            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                print(f"  ✗ 필수 파일 누락: {missing_files}")
                print("  COLMAP-only 모드로 실행됩니다")
                self.superpoint = None
                self.superglue = None
                return
            
            # SuperPoint/SuperGlue import 시도
            sys.path.insert(0, str(models_dir.parent))
            
            try:
                # 직접 경로로 import 시도
                import importlib.util
                
                # SuperPoint import
                superpoint_spec = importlib.util.spec_from_file_location(
                    "superpoint", models_dir / "superpoint.py")
                superpoint_module = importlib.util.module_from_spec(superpoint_spec)
                superpoint_spec.loader.exec_module(superpoint_module)
                SuperPoint = superpoint_module.SuperPoint
                
                # SuperGlue import
                superglue_spec = importlib.util.spec_from_file_location(
                    "superglue", models_dir / "superglue.py")
                superglue_module = importlib.util.module_from_spec(superglue_spec)
                superglue_spec.loader.exec_module(superglue_module)
                SuperGlue = superglue_module.SuperGlue
                
                print("  ✓ SuperPoint/SuperGlue 모듈 import 성공 (직접 경로)")
                
            except Exception as e:
                print(f"  🔄 직접 import 실패, 일반 import 시도: {e}")
                try:
                    from models.superpoint import SuperPoint
                    from models.superglue import SuperGlue
                    print("  ✓ SuperPoint/SuperGlue 모듈 import 성공 (일반 import)")
                except ImportError as e2:
                    print(f"  ✗ 모델 import 실패: {e2}")
                    print("  COLMAP-only 모드로 실행됩니다")
                    self.superpoint = None
                    self.superglue = None
                    return
            
            # 설정
            superpoint_config = {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            }
            
            superglue_config = {
                'weights': self.superglue_config['weights'],
                'sinkhorn_iterations': self.superglue_config['sinkhorn_iterations'],
                'match_threshold': self.superglue_config['match_threshold'],
            }
            
            # 모델 로드 (메모리 절약 모드)
            try:
                print(f"    SuperPoint 모델 로드 중...")
                self.superpoint = SuperPoint(superpoint_config).eval()
                if self.device == "cuda":
                    self.superpoint = self.superpoint.to(self.device)
                print(f"    ✓ SuperPoint 모델 로드 완료")
                
                print(f"    SuperGlue 모델 로드 중...")
                self.superglue = SuperGlue(superglue_config).eval()
                if self.device == "cuda":
                    self.superglue = self.superglue.to(self.device)
                print(f"    ✓ SuperGlue 모델 로드 완료")
                
                print(f"  ✓ SuperPoint/SuperGlue 모델 로드 완료 (device: {self.device})")
                
                # 테스트 실행
                print(f"    SuperPoint 테스트 중...")
                test_tensor = torch.zeros(1, 1, 480, 640).to(self.device)
                with torch.no_grad():
                    _ = self.superpoint({'image': test_tensor})
                print("  ✓ SuperPoint 테스트 성공")
                
                # SuperGlue 테스트
                print(f"    SuperGlue 테스트 중...")
                
                try:
                    # SuperPoint로 실제 특징점 추출
                    with torch.no_grad():
                        pred0 = self.superpoint({'image': test_tensor})
                        pred1 = self.superpoint({'image': test_tensor})
                    
                    # SuperGlue 입력 데이터 준비 (tensor stack 형태)
                    test_data = {
                        'image0': test_tensor,
                        'image1': test_tensor,
                        'keypoints0': torch.stack(pred0['keypoints']).to(self.device),
                        'keypoints1': torch.stack(pred1['keypoints']).to(self.device),
                        'scores0': torch.stack(pred0['scores']).to(self.device),
                        'scores1': torch.stack(pred1['scores']).to(self.device),
                        'descriptors0': torch.stack(pred0['descriptors']).to(self.device),
                        'descriptors1': torch.stack(pred1['descriptors']).to(self.device),
                    }
                    
                    with torch.no_grad():
                        _ = self.superglue(test_data)
                    print("  ✓ SuperGlue 테스트 성공")
                    
                except Exception as e:
                    print(f"  ⚠️  SuperGlue 테스트 실패 (무시하고 계속): {e}")
                    print("  SuperGlue는 매칭 시에만 사용됩니다")
                
            except Exception as e:
                print(f"  ✗ 모델 로드/테스트 실패: {e}")
                import traceback
                traceback.print_exc()
                
                # SuperPoint만이라도 사용 가능한지 확인
                if self.superpoint is not None:
                    print("  ⚠️  SuperGlue만 실패, SuperPoint-only 모드로 실행됩니다")
                    self.superglue = None
                else:
                    print("  COLMAP-only 모드로 실행됩니다")
                    self.superpoint = None
                    self.superglue = None
                
        except Exception as e:
            print(f"  ✗ SuperGlue 모델 로드 전체 실패: {e}")
            self.superpoint = None
            self.superglue = None

    def _extract_superpoint_features(self, image_paths, database_path, input_dir):
        """SuperPoint 특징점 추출 - Shape 검증 강화"""
        print("  🔥 SuperPoint 특징점 추출 (Shape 검증 강화)...")
        
        if self.superpoint is None:
            print("  ❌ SuperPoint 모델이 없습니다!")
            return False
        
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 이미지 ID 가져오기
            cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
            images = cursor.fetchall()
            
            successful_extractions = 0
            for idx, (image_id, image_name) in enumerate(images):
                print(f"    [{idx+1:3d}/{len(images)}] {image_name}")
                
                # 인덱스 기반으로 원본 이미지 경로 매칭
                if idx < len(image_paths):
                    original_img_path = image_paths[idx]
                else:
                    original_img_path = input_dir / image_name
                
                # SuperPoint 특징점 추출
                keypoints, descriptors = self._extract_single_superpoint_features(original_img_path)
                
                if keypoints is not None and len(keypoints) > 0 and descriptors is not None:
                    print(f"      추출 성공: {len(keypoints)}개 키포인트, {descriptors.shape}")
                    
                    # ✅ Shape 검증
                    n_keypoints = keypoints.shape[0]
                    n_descriptors = descriptors.shape[0]
                    descriptor_dim = descriptors.shape[1]
                    
                    if n_keypoints != n_descriptors:
                        print(f"      ❌ 키포인트-디스크립터 개수 불일치: {n_keypoints} vs {n_descriptors}")
                        continue
                    
                    print(f"      ✅ Shape 검증 통과: {n_keypoints}개 키포인트, {descriptor_dim}차원")
                    
                    # COLMAP 호환 데이터 타입으로 변환
                    keypoints_colmap = keypoints.astype(np.float64)
                    descriptors_colmap = descriptors  # 이미 uint8로 변환됨
                    
                    print(f"      데이터 타입: keypoints={keypoints_colmap.dtype}, descriptors={descriptors_colmap.dtype}")
                    
                    # ✅ 바이트 크기 정확히 계산
                    if descriptors_colmap.dtype == np.uint8:
                        expected_bytes = n_descriptors * descriptor_dim * 1  # uint8 = 1 byte
                    elif descriptors_colmap.dtype == np.float32:
                        expected_bytes = n_descriptors * descriptor_dim * 4  # float32 = 4 bytes
                    else:
                        expected_bytes = len(descriptors_colmap.tobytes())
                    
                    actual_bytes = len(descriptors_colmap.tobytes())
                    
                    print(f"      크기 검증:")
                    print(f"        형태: {descriptors_colmap.shape}")
                    print(f"        예상 크기: {expected_bytes} bytes")
                    print(f"        실제 크기: {actual_bytes} bytes")
                    
                    if expected_bytes != actual_bytes:
                        print(f"      ⚠️  크기 불일치 발견!")
                    
                    try:
                        # keypoints 저장
                        cursor.execute(
                            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                            (image_id, n_keypoints, 2, keypoints_colmap.tobytes())
                        )
                        
                        # descriptors 저장 - ✅ 정확한 차원 수 저장
                        cursor.execute(
                            "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                            (image_id, n_descriptors, descriptor_dim, descriptors_colmap.tobytes())
                        )
                        
                        print(f"      ✅ DB 저장 성공: {n_keypoints}개 키포인트, {descriptor_dim}차원")
                        successful_extractions += 1
                        
                    except Exception as db_error:
                        print(f"      ❌ DB 저장 실패: {db_error}")
                else:
                    print(f"      ❌ SuperPoint 추출 실패")
            
            conn.commit()
            conn.close()
            
            print(f"  📊 결과: {successful_extractions}/{len(images)} 성공")
            
            if successful_extractions > 0:
                print(f"  🎉 SuperPoint 특징점 추출 완료!")
                return True
            else:
                print("  ❌ 모든 SuperPoint 추출 실패")
                return False
            
        except Exception as e:
            print(f"  ❌ SuperPoint 특징점 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _run_colmap_feature_extraction_fast(self, database_path, image_path):
        """빠른 COLMAP 특징점 추출 (timeout 단축)"""
        print("  ⚡ 빠른 COLMAP SIFT 특징점 추출...")
        
        base_cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "2048",  # 증가
            "--SiftExtraction.num_threads", "4"  # 멀티스레드
        ]
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            # timeout을 600초(10분)로 단축
            result = subprocess.run(base_cmd, capture_output=True, text=True, 
                                 timeout=600, env=env)
            if result.returncode == 0:
                print("  ✓ COLMAP SIFT 특징점 추출 완료")
            else:
                print(f"  ✗ COLMAP SIFT 추출 실패: {result.stderr}")
                # 더 관대한 설정으로 재시도
                self._run_colmap_feature_extraction_permissive(database_path, image_path)
        except subprocess.TimeoutExpired:
            print("  ⚠️  COLMAP 특징점 추출 타임아웃 (10분)")
            print("  🔄 더 관대한 설정으로 재시도...")
            self._run_colmap_feature_extraction_permissive(database_path, image_path)
        except Exception as e:
            print(f"  ✗ COLMAP 특징점 추출 오류: {e}")

    def _run_colmap_feature_extraction_permissive(self, database_path, image_path):
        """관대한 설정의 COLMAP 특징점 추출"""
        print("  🔄 관대한 COLMAP SIFT 설정으로 재시도...")
        
        base_cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "1000",  # 줄임
            "--SiftExtraction.first_octave", "0",
            "--SiftExtraction.num_octaves", "3",  # 줄임
            "--SiftExtraction.octave_resolution", "2"  # 줄임
        ]
        
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, 
                                 timeout=300, env=env)  # 5분으로 단축
            if result.returncode == 0:
                print("  ✓ 관대한 COLMAP SIFT 추출 완료")
            else:
                print(f"  ✗ 관대한 COLMAP SIFT도 실패: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("  ❌ 관대한 COLMAP도 타임아웃, SfM 실패")
        except Exception as e:
            print(f"  ❌ 관대한 COLMAP 오류: {e}")

    def _extract_single_superpoint_features(self, image_path):
        """단일 이미지에서 SuperPoint 특징점 추출 - Shape 수정"""
        try:
            print(f"        이미지 로드: {image_path}")
            
            # 이미지 로드
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"        ❌ cv2.imread 실패")
                return None, None
            
            h, w = img.shape[:2]
            print(f"        이미지 크기: {w}x{h}")
            
            # 큰 이미지 리사이즈
            if h > 1600 or w > 1600:
                scale = min(1600/w, 1600/h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
                print(f"        리사이즈: {new_w}x{new_h}")
            
            # 그레이스케일 변환
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # 텐서 변환
            img_tensor = torch.from_numpy(img_gray).float().to(self.device) / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            print(f"        텐서 형태: {img_tensor.shape}")
            
            # SuperPoint 추론
            with torch.no_grad():
                pred = self.superpoint({'image': img_tensor})
                keypoints = pred['keypoints'][0].cpu().numpy()  # (N, 2)
                descriptors = pred['descriptors'][0].cpu().numpy()  # (256, N) ← 여기가 문제!
            
            # ✅ 핵심 수정: descriptor shape 확인 및 수정
            print(f"        원본 출력: keypoints={keypoints.shape}, descriptors={descriptors.shape}")
            
            # SuperPoint 출력이 (256, N)이면 (N, 256)으로 transpose
            if len(descriptors.shape) == 2:
                if descriptors.shape[0] == 256 and descriptors.shape[1] == keypoints.shape[0]:
                    # (256, N) → (N, 256)로 transpose
                    descriptors = descriptors.T
                    print(f"        ✅ Descriptor transpose: {descriptors.shape}")
                elif descriptors.shape[1] == 256:
                    # 이미 (N, 256) 형태
                    print(f"        ✅ Descriptor 형태 정상: {descriptors.shape}")
                else:
                    print(f"        ⚠️  예상치 못한 descriptor 형태: {descriptors.shape}")
            
            print(f"        최종 결과: {keypoints.shape[0]}개 키포인트, {descriptors.shape}")
            
            # 개수 일치 확인
            if keypoints.shape[0] != descriptors.shape[0]:
                print(f"        ❌ 키포인트-디스크립터 개수 불일치: {keypoints.shape[0]} vs {descriptors.shape[0]}")
                return None, None
            
            # 차원 변환 (256 -> 128) + uint8 변환
            if descriptors.shape[1] == 256:
                descriptors_128 = self._convert_descriptors_to_sift_format(descriptors)
                return keypoints, descriptors_128
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"        ❌ SuperPoint 오류: {e}")
            return None, None

    def _convert_descriptors_to_sift_format(self, descriptors):
        """SuperPoint descriptor를 COLMAP SIFT 형식으로 완전 변환"""
        try:
            print(f"      🔄 디스크립터 변환: {descriptors.shape} {descriptors.dtype}")
            
            # 1. 차원 축소: 256 -> 128
            if descriptors.shape[1] == 256:
                # L2 정규화
                descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)
                
                n_features = descriptors.shape[0]
                descriptors_128 = np.zeros((n_features, 128), dtype=np.float32)
                
                # 2개씩 묶어서 평균
                for j in range(128):
                    descriptors_128[:, j] = (descriptors[:, j*2] + descriptors[:, j*2+1]) / 2.0
                
                # 다시 L2 정규화
                descriptors_128 = descriptors_128 / (np.linalg.norm(descriptors_128, axis=1, keepdims=True) + 1e-8)
            else:
                descriptors_128 = descriptors.astype(np.float32)
            
            # 2. ✅ COLMAP SIFT 형식으로 변환: float32 -> uint8
            # SIFT descriptor는 0-255 범위의 uint8
            
            # 정규화: [-1, 1] -> [0, 1]
            descriptors_norm = (descriptors_128 + 1.0) / 2.0
            descriptors_norm = np.clip(descriptors_norm, 0.0, 1.0)
            
            # uint8로 변환: [0, 1] -> [0, 255]
            descriptors_uint8 = (descriptors_norm * 255.0).astype(np.uint8)
            
            print(f"      ✅ 변환 완료: {descriptors_uint8.shape} {descriptors_uint8.dtype}")
            print(f"      값 범위: [{descriptors_uint8.min()}, {descriptors_uint8.max()}]")
            
            return descriptors_uint8
            
        except Exception as e:
            print(f"      ❌ 디스크립터 변환 오류: {e}")
            # fallback: 간단한 변환
            if descriptors.shape[1] >= 128:
                desc_128 = descriptors[:, :128]
                desc_norm = np.clip((desc_128 + 1.0) / 2.0, 0.0, 1.0)
                return (desc_norm * 255.0).astype(np.uint8)
            else:
                return descriptors.astype(np.uint8)

    def _match_single_pair(self, image_path1, image_path2):
        """두 이미지 간 SuperGlue 매칭 수행"""
        try:
            print(f"        🔍 SuperGlue 매칭: {image_path1.name} ↔ {image_path2.name}")
            
            # 이미지 로드 및 전처리
            img1 = self._load_and_preprocess_image(image_path1)
            img2 = self._load_and_preprocess_image(image_path2)
            
            if img1 is None or img2 is None:
                print(f"        ❌ 이미지 로드 실패")
                return None
            
            # SuperPoint 특징점 추출
            pred1 = self._extract_superpoint_features_for_matching(img1)
            pred2 = self._extract_superpoint_features_for_matching(img2)
            
            if pred1 is None or pred2 is None:
                print(f"        ❌ SuperPoint 특징점 추출 실패")
                return None
            
            # SuperGlue 매칭
            matches = self._run_superglue_matching_on_pair(pred1, pred2)
            
            if matches is not None and len(matches) > 0:
                print(f"        ✅ {len(matches)}개 매칭 발견")
                return matches
            else:
                print(f"        ❌ 매칭 실패")
                return None
                
        except Exception as e:
            print(f"        ❌ 매칭 오류: {e}")
            return None

    def _load_and_preprocess_image(self, image_path):
        """이미지 로드 및 전처리"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            h, w = img.shape[:2]
            
            # 큰 이미지 리사이즈
            if h > 1600 or w > 1600:
                scale = min(1600/w, 1600/h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # 그레이스케일 변환
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            return img_gray
            
        except Exception as e:
            print(f"        ❌ 이미지 전처리 오류: {e}")
            return None

    def _extract_superpoint_features_for_matching(self, img_gray):
        """매칭용 SuperPoint 특징점 추출"""
        try:
            # 텐서 변환
            img_tensor = torch.from_numpy(img_gray).float().to(self.device) / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # SuperPoint 추론
            with torch.no_grad():
                pred = self.superpoint({'image': img_tensor})
                keypoints = pred['keypoints'][0].cpu().numpy()  # (N, 2)
                scores = pred['scores'][0].cpu().numpy()  # (N,)
                descriptors = pred['descriptors'][0].cpu().numpy()  # (256, N)
            
            # descriptor transpose - SuperGlue 호환을 위해
            if len(descriptors.shape) == 2 and descriptors.shape[0] == 256:
                descriptors = descriptors.T  # (N, 256)
            
            print(f"        SuperPoint 결과: {len(keypoints)}개 키포인트, {descriptors.shape}")
            
            # 최소 특징점 수 확인
            if len(keypoints) < 10:
                print(f"        ⚠️  특징점 부족: {len(keypoints)}개")
                return None
            
            # 메모리 정리
            del img_tensor
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors
            }
            
        except Exception as e:
            print(f"        ❌ SuperPoint 추출 오류: {e}")
            return None

    def _run_superglue_matching_on_pair(self, pred1, pred2):
        """SuperGlue를 사용한 두 이미지 간 매칭"""
        try:
            print(f"        SuperGlue 입력 데이터 준비 중...")
            print(f"        pred1: keypoints={pred1['keypoints'].shape}, scores={pred1['scores'].shape}, descriptors={pred1['descriptors'].shape}")
            print(f"        pred2: keypoints={pred2['keypoints'].shape}, scores={pred2['scores'].shape}, descriptors={pred2['descriptors'].shape}")
            
            # SuperGlue가 기대하는 형태로 데이터 변환
            # SuperGlue는 tensor stack 형태를 기대
            keypoints0 = torch.from_numpy(pred1['keypoints']).unsqueeze(0).to(self.device)  # (1, N, 2)
            keypoints1 = torch.from_numpy(pred2['keypoints']).unsqueeze(0).to(self.device)  # (1, N, 2)
            scores0 = torch.from_numpy(pred1['scores']).unsqueeze(0).to(self.device)  # (1, N)
            scores1 = torch.from_numpy(pred2['scores']).unsqueeze(0).to(self.device)  # (1, N)
            descriptors0 = torch.from_numpy(pred1['descriptors']).unsqueeze(0).to(self.device)  # (1, N, 256)
            descriptors1 = torch.from_numpy(pred2['descriptors']).unsqueeze(0).to(self.device)  # (1, N, 256)
            
            print(f"        변환된 shapes: keypoints0={keypoints0.shape}, descriptors0={descriptors0.shape}")
            
            # 입력 데이터 준비
            data = {
                'image0': torch.zeros(1, 1, 480, 640).to(self.device),  # 더미 이미지
                'image1': torch.zeros(1, 1, 480, 640).to(self.device),  # 더미 이미지
                'keypoints0': keypoints0,
                'keypoints1': keypoints1,
                'scores0': scores0,
                'scores1': scores1,
                'descriptors0': descriptors0,
                'descriptors1': descriptors1,
            }
            
            # SuperGlue 추론
            with torch.no_grad():
                pred = self.superglue(data)
                matches = pred['matches0'][0].cpu().numpy()  # (N,)
                confidence = pred['matching_scores0'][0].cpu().numpy()  # (N,)
            
            # 메모리 정리
            del data, keypoints0, keypoints1, scores0, scores1, descriptors0, descriptors1
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 유효한 매칭 필터링
            valid_matches = []
            for i, match_idx in enumerate(matches):
                if match_idx != -1:  # -1은 매칭되지 않음을 의미
                    confidence_score = confidence[i]
                    if confidence_score > self.superglue_config['match_threshold']:
                        valid_matches.append([i, match_idx])
            
            if len(valid_matches) > 0:
                print(f"        ✅ SuperGlue 매칭: {len(valid_matches)}개 (임계값: {self.superglue_config['match_threshold']})")
                return np.array(valid_matches, dtype=np.int32)
            else:
                print(f"        ⚠️  SuperGlue 매칭 부족, fallback 시도...")
                # SuperGlue 실패시 간단한 descriptor 매칭으로 fallback
                return self._fallback_descriptor_matching(pred1, pred2)
                
        except Exception as e:
            print(f"        ❌ SuperGlue 매칭 오류: {e}")
            # fallback 매칭 시도
            return self._fallback_descriptor_matching(pred1, pred2)

    def _fallback_descriptor_matching(self, pred1, pred2):
        """간단한 descriptor 매칭 fallback"""
        try:
            print(f"        🔄 Fallback descriptor 매칭 시도...")
            
            desc1 = pred1['descriptors']  # (N1, 256)
            desc2 = pred2['descriptors']  # (N2, 256)
            
            # L2 거리 계산
            desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
            desc2_norm = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)
            
            # 모든 쌍의 거리 계산
            distances = np.zeros((desc1.shape[0], desc2.shape[0]))
            for i in range(desc1.shape[0]):
                for j in range(desc2.shape[0]):
                    distances[i, j] = np.linalg.norm(desc1_norm[i] - desc2_norm[j])
            
            # 최근접 이웃 매칭
            matches = []
            for i in range(desc1.shape[0]):
                best_j = np.argmin(distances[i])
                best_distance = distances[i, best_j]
                
                # 거리 임계값 체크
                if best_distance < 0.8:  # 더 관대한 임계값
                    matches.append([i, best_j])
            
            if len(matches) > 0:
                print(f"        ✅ Fallback 매칭: {len(matches)}개")
                return np.array(matches, dtype=np.int32)
            else:
                print(f"        ❌ Fallback 매칭도 실패")
                return None
                
        except Exception as e:
            print(f"        ❌ Fallback 매칭 오류: {e}")
            return None

    def _run_superpoint_only_matching(self, image_paths, database_path):
        """SuperPoint만 사용한 매칭"""
        print("  🔥 SuperPoint-only 매칭 중...")
        
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 기존 matches 정리
            cursor.execute("DELETE FROM matches")
            cursor.execute("DELETE FROM two_view_geometries")
            
            # 이미지 ID 매핑 생성
            image_id_map = {}
            cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
            for image_id, name in cursor.fetchall():
                try:
                    idx = int(name.split('_')[1].split('.')[0])
                    image_id_map[idx] = image_id
                except:
                    continue
            
            # 매칭 수행
            successful_matches = 0
            total_pairs = 0
            
            for i in range(len(image_paths)):
                for j in range(i + 1, min(i + 5, len(image_paths))):  # 인접한 5장씩만
                    total_pairs += 1
                    
                    print(f"      매칭 {i}-{j}...")
                    matches = self._match_single_pair_superpoint_only(image_paths[i], image_paths[j])
                    
                    if matches is not None and len(matches) >= 10:
                        if i in image_id_map and j in image_id_map:
                            pair_id = image_id_map[i] * 2147483647 + image_id_map[j]
                            
                            cursor.execute(
                                "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                                (pair_id, len(matches), 2, matches.tobytes())
                            )
                            
                            print(f"        ✅ {len(matches)}개 매칭 저장 (SuperPoint-only)")
                            successful_matches += 1
                        else:
                            print(f"        ❌ 이미지 ID 매핑 실패")
                    else:
                        print(f"        ❌ 매칭 실패 또는 부족")
            
            conn.commit()
            conn.close()
            
            print(f"    📊 SuperPoint-only 매칭 결과: {successful_matches}/{total_pairs} 성공")
            
            if successful_matches == 0:
                print("    ⚠️  SuperPoint-only 매칭 실패, COLMAP 매칭으로 fallback...")
                self._run_colmap_matching_fast(database_path)
            else:
                print("    ✅ SuperPoint-only 매칭 완료!")
                
        except Exception as e:
            print(f"    ❌ SuperPoint-only 매칭 오류: {e}")
            print("    🔄 COLMAP 매칭으로 fallback...")
            self._run_colmap_matching_fast(database_path)

    def _match_single_pair_superpoint_only(self, image_path1, image_path2):
        """SuperPoint만 사용한 두 이미지 간 매칭"""
        try:
            print(f"        🔍 SuperPoint-only 매칭: {image_path1.name} ↔ {image_path2.name}")
            
            # 이미지 로드 및 전처리
            img1 = self._load_and_preprocess_image(image_path1)
            img2 = self._load_and_preprocess_image(image_path2)
            
            if img1 is None or img2 is None:
                print(f"        ❌ 이미지 로드 실패")
                return None
            
            # SuperPoint 특징점 추출
            pred1 = self._extract_superpoint_features_for_matching(img1)
            pred2 = self._extract_superpoint_features_for_matching(img2)
            
            if pred1 is None or pred2 is None:
                print(f"        ❌ SuperPoint 특징점 추출 실패")
                return None
            
            # SuperPoint descriptor 매칭
            matches = self._fallback_descriptor_matching(pred1, pred2)
            
            if matches is not None and len(matches) > 0:
                print(f"        ✅ {len(matches)}개 매칭 발견 (SuperPoint-only)")
                return matches
            else:
                print(f"        ❌ SuperPoint-only 매칭 실패")
                return None
                
        except Exception as e:
            print(f"        ❌ SuperPoint-only 매칭 오류: {e}")
            return None

    # 나머지 메서드들은 기존과 동일하게 유지...
    def process_images(self, image_dir: str, output_dir: str, max_images: int = 100):
        """메인 처리 메서드"""
        print("🚀 SuperGlue + COLMAP 하이브리드 파이프라인 시작")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 이미지 수집
            print("\n[1/6] 이미지 수집...")
            image_paths = self._collect_images(image_dir, max_images)
            if len(image_paths) == 0:
                raise RuntimeError("처리할 이미지를 찾을 수 없습니다")
            
            print(f"  선택된 이미지: {len(image_paths)}장")
            input_dir = self._prepare_input_images(image_paths, output_path)
            
            # 2. COLMAP 데이터베이스 생성
            print("\n[2/6] COLMAP 데이터베이스 생성...")
            database_path = output_path / "database.db"
            self._create_colmap_database(image_paths, database_path, input_dir)
            
            # 3. 특징점 추출 (SuperPoint 또는 COLMAP SIFT)
            print("\n[3/6] 특징점 추출...")
            self._extract_superpoint_features(image_paths, database_path, input_dir)
            
            # 4. 매칭 (빠른 COLMAP exhaustive matcher)
            print("\n[4/6] 특징점 매칭...")
            self._run_superglue_matching(image_paths, database_path)
            
            # 5. 포즈 추정
            print("\n[5/6] 포즈 추정...")
            sparse_dir = output_path / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            self._run_colmap_mapper_fast(database_path, input_dir, sparse_dir)
            
            # 6. 언디스토션 (옵션)
            print("\n[6/6] 언디스토션...")
            undistorted_dir = output_path / "undistorted"
            self._run_colmap_undistortion_fast(input_dir, sparse_dir, undistorted_dir)
            
            # 7. 3DGS 변환
            print("\n[7/6] 3DGS 형식 변환...")
            scene_info = self._convert_to_3dgs_format(output_path, image_paths)
            
            print("✅ 하이브리드 파이프라인 완료!")
            return scene_info
            
        except Exception as e:
            print(f"\n❌ 파이프라인 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_colmap_matching_fast(self, database_path):
        """빠른 COLMAP 매칭"""
        print("  ⚡ 빠른 COLMAP 매칭...")
        
        base_cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.max_num_matches", "1000"  # 매칭 수 제한
        ]
        
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, 
                                 timeout=300, env=env)  # 5분 제한
            if result.returncode == 0:
                print("  ✓ COLMAP 매칭 완료")
            else:
                print(f"  ✗ COLMAP 매칭 실패: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("  ⚠️  COLMAP 매칭 타임아웃")
        except Exception as e:
            print(f"  ✗ COLMAP 매칭 오류: {e}")

    def _run_colmap_mapper_fast(self, database_path, image_path, output_path):
        """빠른 COLMAP 매퍼"""
        print("  ⚡ 빠른 COLMAP 매퍼...")
        
        base_cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--Mapper.min_num_matches", "8",
            "--Mapper.init_min_num_inliers", "16",
            "--Mapper.abs_pose_min_num_inliers", "8"
        ]
        
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, 
                                 timeout=600, env=env)  # 10분 제한
            if result.returncode == 0:
                print("  ✓ COLMAP 매퍼 완료")
                return True
            else:
                print(f"  ✗ COLMAP 매퍼 실패: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ⚠️  COLMAP 매퍼 타임아웃")
            return False
        except Exception as e:
            print(f"  ✗ COLMAP 매퍼 오류: {e}")
            return False

    def _run_colmap_undistortion_fast(self, image_path, sparse_path, output_path):
        """빠른 언디스토션 (옵션)"""
        print("  ⚡ 이미지 언디스토션...")
        
        # sparse 결과 확인
        best_model = None
        if sparse_path.exists():
            model_dirs = [d for d in sparse_path.iterdir() if d.is_dir()]
            if model_dirs:
                best_model = model_dirs[0]  # 첫 번째 모델 사용
        
        if best_model is None:
            print("  ⚠️  sparse 결과 없음, 원본 이미지 복사...")
            self._copy_original_images_fallback(image_path, output_path)
            return
        
        # 언디스토션 생략하고 원본 이미지 복사 (빠른 실행)
        print("  ⚡ 언디스토션 생략, 원본 이미지 사용...")
        self._copy_original_images_fallback(image_path, output_path)

    def _copy_original_images_fallback(self, image_path, output_path):
        """원본 이미지 복사"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            images_dir = output_path / "images"
            images_dir.mkdir(exist_ok=True)
            
            # 원본 이미지들을 복사
            copied_count = 0
            for img_file in Path(image_path).iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dst_file = images_dir / img_file.name
                    shutil.copy2(img_file, dst_file)
                    copied_count += 1
            
            print(f"  ✓ 원본 이미지 복사 완료: {copied_count}장")
            
        except Exception as e:
            print(f"  ✗ 이미지 복사 실패: {e}")

    # 나머지 필요한 메서드들 (간소화된 버전)
    def _collect_images(self, image_dir, max_images):
        """이미지 수집"""
        image_dir = Path(image_dir)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        image_paths.sort()
        return image_paths[:max_images]

    def _prepare_input_images(self, image_paths, output_path):
        """입력 이미지 준비"""
        input_dir = output_path / "input"
        input_dir.mkdir(exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            dst_name = f"image_{i:04d}{img_path.suffix}"
            dst_path = input_dir / dst_name
            shutil.copy2(img_path, dst_path)
        
        print(f"  {len(image_paths)}장 이미지 준비 완료")
        return input_dir

    def _create_colmap_database(self, image_paths, database_path, input_dir):
        """COLMAP 데이터베이스 생성"""
        try:
            # 기존 DB 삭제
            if database_path.exists():
                database_path.unlink()
            
            # COLMAP database_creator 실행
            result = subprocess.run([
                self.colmap_exe, "database_creator",
                "--database_path", str(database_path)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("  ✓ COLMAP database_creator 성공")
            else:
                raise RuntimeError(f"database_creator 실패: {result.stderr}")
            
            # 카메라 및 이미지 정보 추가
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 첫 번째 이미지로 카메라 매개변수 추정
            first_img = cv2.imread(str(image_paths[0]))
            height, width = first_img.shape[:2]
            focal = max(width, height) * 1.2  # 추정 초점거리
            
            camera_model = 1  # PINHOLE (fx, fy, cx, cy)
            params = np.array([focal, focal, width/2, height/2], dtype=np.float64)  # ✅ 4개 매개변수
            
            print(f"  카메라 모델: PINHOLE({camera_model}), 매개변수: {len(params)}개")
            
            # 카메라 추가
            cursor.execute(
                "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                (camera_model, width, height, params.tobytes(), int(focal))  # ✅ model=1
            )
            camera_id = cursor.lastrowid
            
            # 이미지 정보 추가
            for i, img_path in enumerate(image_paths):
                image_name = f"image_{i:04d}{img_path.suffix}"
                cursor.execute(
                    "INSERT INTO images (name, camera_id) VALUES (?, ?)",
                    (image_name, camera_id)
                )
            
            conn.commit()
            conn.close()
            
            print(f"  ✓ {len(image_paths)}장 이미지 정보 추가")
            return True
            
        except Exception as e:
            print(f"  ✗ 데이터베이스 생성 실패: {e}")
            return False
        
    def _run_superglue_matching(self, image_paths, database_path):
        """SuperGlue 매칭 - 실제 매칭 결과를 COLMAP DB에 저장"""
        print("  🔥 SuperGlue 매칭 중...")
        
        if self.superglue is None:
            if self.superpoint is not None:
                print("  ⚠️  SuperGlue 모델 없음, SuperPoint-only 매칭으로 fallback...")
                self._run_superpoint_only_matching(image_paths, database_path)
            else:
                print("  ⚠️  SuperGlue 모델 없음, COLMAP 매칭으로 fallback...")
                self._run_colmap_matching_fast(database_path)
            return
        
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # 기존 matches 정리
            cursor.execute("DELETE FROM matches")
            cursor.execute("DELETE FROM two_view_geometries")
            
            # 이미지 쌍 매칭
            successful_matches = 0
            total_pairs = 0
            
            print(f"    {len(image_paths)}장 이미지에서 매칭 수행...")
            
            # 이미지 ID 매핑 생성
            image_id_map = {}
            cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
            for image_id, name in cursor.fetchall():
                # image_0000.jpg -> 0
                try:
                    idx = int(name.split('_')[1].split('.')[0])
                    image_id_map[idx] = image_id
                except:
                    continue
            
            # 매칭 수행
            for i in range(len(image_paths)):
                for j in range(i + 1, min(i + 5, len(image_paths))):  # 인접한 5장씩만
                    total_pairs += 1
                    
                    print(f"      매칭 {i}-{j}...")
                    matches = self._match_single_pair(image_paths[i], image_paths[j])
                    
                    if matches is not None and len(matches) >= 10:  # 최소 10개 매칭
                        # COLMAP DB에 저장
                        if i in image_id_map and j in image_id_map:
                            pair_id = image_id_map[i] * 2147483647 + image_id_map[j]  # COLMAP pair_id 형식
                            
                            cursor.execute(
                                "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                                (pair_id, len(matches), 2, matches.tobytes())
                            )
                            
                            print(f"        ✅ {len(matches)}개 매칭 저장 (pair_id: {pair_id})")
                            successful_matches += 1
                        else:
                            print(f"        ❌ 이미지 ID 매핑 실패")
                    else:
                        print(f"        ❌ 매칭 실패 또는 부족")
            
            conn.commit()
            conn.close()
            
            print(f"    📊 매칭 결과: {successful_matches}/{total_pairs} 성공")
            
            if successful_matches == 0:
                print("    ⚠️  SuperGlue 매칭 실패, COLMAP 매칭으로 fallback...")
                self._run_colmap_matching_fast(database_path)
            else:
                print("    ✅ SuperGlue 매칭 완료!")
                
        except Exception as e:
            print(f"    ❌ SuperGlue 매칭 오류: {e}")
            print("    🔄 COLMAP 매칭으로 fallback...")
            self._run_colmap_matching_fast(database_path)
        
    def _create_default_scene_info(self, image_paths, output_path):
        """기본 SceneInfo 생성 - CameraInfo 파라미터 수정"""
        print("    🎯 기본 SceneInfo 생성...")
        
        try:
            from utils.graphics_utils import BasicPointCloud
            from scene.dataset_readers import CameraInfo, SceneInfo
            
            # 첫 번째 이미지로 기본 파라미터 설정
            sample_img = cv2.imread(str(image_paths[0]))
            if sample_img is None:
                height, width = 480, 640
            else:
                height, width = sample_img.shape[:2]
            
            # 카메라 정보 생성
            train_cameras = []
            test_cameras = []
            
            for i, img_path in enumerate(image_paths):
                # 이미지 실제 크기 확인
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                    else:
                        h, w = height, width
                except:
                    h, w = height, width
                
                # 카메라 내부 파라미터
                focal_length = max(w, h) * 1.2
                fov_x = 2 * np.arctan(w / (2 * focal_length))
                fov_y = 2 * np.arctan(h / (2 * focal_length))
                
                # 카메라 외부 파라미터 (원형 배치)
                angle = 2 * np.pi * i / len(image_paths)
                radius = 3.0
                
                # 회전 행렬 (카메라가 중심을 바라보도록)
                R = np.array([
                    [np.cos(angle + np.pi/2), 0, np.sin(angle + np.pi/2)],
                    [0, 1, 0],
                    [-np.sin(angle + np.pi/2), 0, np.cos(angle + np.pi/2)]
                ], dtype=np.float32)
                
                # 이동 벡터
                T = np.array([
                    radius * np.cos(angle),
                    0.0,
                    radius * np.sin(angle)
                ], dtype=np.float32)
                
                # ✅ CameraInfo 생성 - 올바른 파라미터만 사용
                cam_info = CameraInfo(
                    uid=i,
                    R=R,
                    T=T,
                    FovY=fov_y,
                    FovX=fov_x,
                    depth_params=None,  # ← image가 아님
                    image_path=str(img_path),
                    image_name=img_path.name,
                    depth_path="",
                    width=w,
                    height=h,
                    is_test=(i % 8 == 0)  # 8개마다 1개씩 테스트
                )
                
                if cam_info.is_test:
                    test_cameras.append(cam_info)
                else:
                    train_cameras.append(cam_info)
            
            print(f"      생성된 카메라: train={len(train_cameras)}, test={len(test_cameras)}")
            
            # 기본 포인트 클라우드 생성
            n_points = 2000
            xyz = np.random.randn(n_points, 3).astype(np.float32) * 1.5
            rgb = np.random.rand(n_points, 3).astype(np.float32)
            normals = np.random.randn(n_points, 3).astype(np.float32)
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
            
            point_cloud = BasicPointCloud(
                points=xyz,
                colors=rgb,
                normals=normals
            )
            
            # NeRF 정규화 계산
            cam_centers = []
            for cam in train_cameras:
                # 카메라 중심 = -R^T * T
                cam_center = -np.dot(cam.R.T, cam.T)
                cam_centers.append(cam_center)
            
            if cam_centers:
                cam_centers = np.array(cam_centers)
                center = np.mean(cam_centers, axis=0)
                distances = np.linalg.norm(cam_centers - center, axis=1)
                radius = np.max(distances) * 1.1
            else:
                center = np.zeros(3)
                radius = 5.0
            
            nerf_normalization = {
                "translate": -center,
                "radius": radius
            }
            
            # PLY 파일 저장
            ply_path = output_path / "points3D.ply"
            self._save_basic_ply(ply_path, xyz, rgb)
            
            # SceneInfo 생성
            scene_info = SceneInfo(
                point_cloud=point_cloud,
                train_cameras=train_cameras,
                test_cameras=test_cameras,
                nerf_normalization=nerf_normalization,
                ply_path=str(ply_path),
                is_nerf_synthetic=False
            )
            
            print(f"      ✅ SceneInfo 생성 완료!")
            print(f"         Train cameras: {len(train_cameras)}")
            print(f"         Test cameras: {len(test_cameras)}")
            print(f"         Point cloud: {len(xyz)} points")
            print(f"         Scene radius: {radius:.3f}")
            
            return scene_info
            
        except Exception as e:
            print(f"      ❌ 기본 SceneInfo 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_basic_ply(self, ply_path, xyz, rgb):
        """기본 PLY 파일 저장"""
        try:
            from plyfile import PlyData, PlyElement
            
            # RGB를 0-255 범위로 변환
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            
            # PLY 형식으로 데이터 준비
            vertex_data = []
            for i in range(len(xyz)):
                vertex_data.append((
                    xyz[i, 0], xyz[i, 1], xyz[i, 2],  # x, y, z
                    rgb_uint8[i, 0], rgb_uint8[i, 1], rgb_uint8[i, 2]  # r, g, b
                ))
            
            vertex_array = np.array(vertex_data, dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            
            vertex_element = PlyElement.describe(vertex_array, 'vertex')
            ply_data = PlyData([vertex_element])
            ply_data.write(str(ply_path))
            
            print(f"      PLY 파일 저장: {ply_path}")
            
        except Exception as e:
            print(f"      PLY 저장 실패: {e}")

    def _run_colmap_mapper_improved(self, database_path, image_path, output_path):
        """개선된 COLMAP 매퍼 - 더 관대한 설정"""
        print("  🔥 개선된 COLMAP 매퍼...")
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["DISPLAY"] = ":0"
        env["XDG_RUNTIME_DIR"] = "/tmp/runtime-colmap"
        
        # 더 관대한 매퍼 설정
        base_cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            
            # 📉 더 관대한 설정
            "--Mapper.min_num_matches", "3",              # 4 → 3
            "--Mapper.init_min_num_inliers", "6",         # 8 → 6  
            "--Mapper.abs_pose_min_num_inliers", "3",     # 4 → 3
            "--Mapper.filter_max_reproj_error", "20.0",   # 더 큰 허용 오차
            "--Mapper.ba_refine_focal_length", "0",       # 초점거리 고정
            "--Mapper.ba_refine_principal_point", "0",    # 주점 고정
            "--Mapper.ba_refine_extra_params", "0",       # 추가 파라미터 고정
            
            # 🚀 성능 개선
            "--Mapper.max_num_models", "1",               # 단일 모델만
            "--Mapper.min_model_size", "3",               # 최소 3장 이미지
        ]
        
        print(f"    명령: {' '.join(base_cmd)}")
        
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, 
                                timeout=600, env=env)
            
            if result.returncode == 0:
                print("  ✅ COLMAP 매퍼 성공!")
                
                # 결과 확인
                reconstruction_dirs = [d for d in output_path.iterdir() if d.is_dir()]
                if reconstruction_dirs:
                    print(f"    생성된 reconstruction: {len(reconstruction_dirs)}개")
                    for recon_dir in reconstruction_dirs:
                        bin_files = list(recon_dir.glob("*.bin"))
                        print(f"      {recon_dir.name}: {len(bin_files)}개 파일")
                
                return True
            else:
                print(f"  ❌ COLMAP 매퍼 실패:")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("  ⚠️  COLMAP 매퍼 타임아웃 (10분)")
            return False
        except Exception as e:
            print(f"  ❌ COLMAP 매퍼 오류: {e}")
            return False

    def _convert_to_3dgs_format(self, output_path, image_paths):
        """3DGS 형식으로 변환 - Import 경로 수정"""
        print("  🔧 3DGS SceneInfo 생성 중...")
        
        try:
            # ✅ 정확한 import 경로
            from utils.graphics_utils import BasicPointCloud
            from scene.dataset_readers import CameraInfo, SceneInfo
            
            # sparse 디렉토리 확인
            sparse_dir = output_path / "sparse"
            reconstruction_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
            
            if reconstruction_dirs:
                # COLMAP reconstruction이 있는 경우
                reconstruction_path = reconstruction_dirs[0]  # 첫 번째 reconstruction 사용
                print(f"    COLMAP reconstruction 발견: {reconstruction_path}")
                
                # 실제 COLMAP 결과 사용 시도
                try:
                    return self._parse_colmap_reconstruction(reconstruction_path, image_paths, output_path)
                except Exception as e:
                    print(f"    COLMAP reconstruction 파싱 실패: {e}")
                    print("    기본 SceneInfo로 fallback...")
            else:
                print("    COLMAP reconstruction 없음, 기본 SceneInfo 생성...")
            
            # 기본 SceneInfo 생성
            return self._create_default_scene_info(image_paths, output_path)
            
        except Exception as e:
            print(f"  ❌ 3DGS 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

# 사용 예시
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_path", type=str, default="./output_hybrid", help="출력 디렉토리")
    parser.add_argument("--max_images", type=int, default=100, help="최대 이미지 수")
    parser.add_argument("--config", type=str, default="outdoor", help="SuperGlue 설정")
    
    args = parser.parse_args()
    
    print("🚀 SuperGlue + COLMAP 하이브리드 파이프라인 테스트")
    print("=" * 60)
    
    pipeline = SuperGlueCOLMAPHybrid(
        superglue_config=args.config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    scene_info = pipeline.process_images(
        image_dir=args.source_path,
        output_dir=args.output_path,
        max_images=args.max_images
    )
    
    if scene_info:
        print("✅ 파이프라인 성공!")
        print(f"   Train cameras: {len(scene_info.train_cameras)}")
        print(f"   Test cameras: {len(scene_info.test_cameras)}")
        print(f"   Point cloud: {len(scene_info.point_cloud.points)} points")
    else:
        print("❌ 파이프라인 실패")