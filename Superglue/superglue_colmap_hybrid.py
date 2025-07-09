# superglue_colmap_hybrid.py
# SuperGlue feature extraction + COLMAP pose estimation 하이브리드 파이프라인

import numpy as np
import cv2
import torch
import sqlite3
import struct
from pathlib import Path
import subprocess
import os
import tempfile
from collections import defaultdict
import argparse
import sys

class SuperGlueCOLMAPHybrid:
    """SuperGlue 특징점 + COLMAP SfM 하이브리드 시스템"""
    
    def __init__(self, colmap_exe="colmap", device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.colmap_exe = colmap_exe
        
        # SuperGlue 설정 (모션 블러에 최적화)
        config = {
            'superpoint': {
                'nms_radius': 3,
                'keypoint_threshold': 0.003,  # 낮춰서 더 많은 특징점
                'max_keypoints': 4096,
                'remove_borders': 8
            },
            'superglue': {
                'weights': 'outdoor',  # 더 robust
                'sinkhorn_iterations': 50,
                'match_threshold': 0.1,  # 관대하게
            }
        }
        
        # SuperGlue 초기화
        try:
            from models.matching import Matching
            from models.utils import frame2tensor
            self.matching = Matching(config).eval().to(self.device)
            self.frame2tensor = frame2tensor
            self.superglue_ready = True
            print(f"✓ SuperGlue 초기화 완료 on {self.device}")
        except Exception as e:
            print(f"✗ SuperGlue 초기화 실패: {e}")
            self.superglue_ready = False
        
        # COLMAP 확인
        self.colmap_ready = self._check_colmap()
    
    def _check_colmap(self):
        """COLMAP 설치 확인"""
        try:
            result = subprocess.run([self.colmap_exe, '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ COLMAP 준비 완료")
                return True
        except Exception as e:
            print(f"✗ COLMAP 확인 실패: {e}")
        return False
    
    def process_images(self, image_dir, output_dir, max_images=100):
        """메인 처리 파이프라인"""
        
        if not self.superglue_ready or not self.colmap_ready:
            raise RuntimeError("SuperGlue 또는 COLMAP가 준비되지 않음")
        
        print(f"\n=== SuperGlue + COLMAP 하이브리드 파이프라인 ===")
        print(f"입력: {image_dir}")
        print(f"출력: {output_dir}")
        print(f"최대 이미지: {max_images}장")
        
        # 출력 디렉토리 설정
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 이미지 수집 및 복사
        print("\n[1/5] 이미지 수집...")
        image_paths = self._collect_images(image_dir, max_images)
        if len(image_paths) == 0:
            raise RuntimeError("처리할 이미지를 찾을 수 없습니다")
        
        input_dir = self._prepare_input_images(image_paths, output_path)
        
        # 2. SuperGlue 특징점 추출 및 매칭
        print("\n[2/5] SuperGlue 특징점 추출 및 매칭...")
        database_path = output_path / "database.db"
        self._create_colmap_database(image_paths, database_path)
        
        # 3. COLMAP으로 포즈 추정
        print("\n[3/5] COLMAP 포즈 추정...")
        sparse_dir = output_path / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        self._run_colmap_mapper(database_path, input_dir, sparse_dir)
        
        # 4. 이미지 언디스토션
        print("\n[4/5] 이미지 언디스토션...")
        undistorted_dir = output_path / "undistorted"
        self._run_colmap_undistortion(input_dir, sparse_dir, undistorted_dir)
        
        # 5. 3DGS 형식으로 변환
        print("\n[5/5] 3DGS 형식 변환...")
        scene_info = self._convert_to_3dgs_format(undistorted_dir)
        
        print(f"\n✓ 완료! 결과: {output_dir}")
        return scene_info
    
    def _collect_images(self, image_dir, max_images):
        """이미지 수집 및 품질 필터링"""
        image_dir = Path(image_dir)
        
        # 이미지 파일 찾기
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(list(image_dir.glob(ext)))
        
        # 시간순 정렬
        image_paths.sort()
        
        # 품질 기반 필터링 (간단한 블러 검출)
        if len(image_paths) > max_images * 1.5:
            print(f"  {len(image_paths)}장 중 품질 필터링...")
            quality_scores = []
            
            for path in image_paths:
                score = self._evaluate_image_sharpness(path)
                quality_scores.append((score, path))
            
            # 상위 품질 이미지 선택
            quality_scores.sort(reverse=True)
            selected = [path for _, path in quality_scores[:max_images]]
            print(f"  품질 필터링 후: {len(selected)}장")
            return selected
        
        return image_paths[:max_images]
    
    def _evaluate_image_sharpness(self, image_path):
        """이미지 선명도 평가"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            
            # 라플라시안 분산으로 선명도 측정
            return cv2.Laplacian(image, cv2.CV_64F).var()
        except:
            return 0.0
    
    def _prepare_input_images(self, image_paths, output_path):
        """COLMAP용 입력 이미지 준비"""
        input_dir = output_path / "input"
        input_dir.mkdir(exist_ok=True)
        
        # 이미지 복사 (필요시 리사이즈)
        for i, src_path in enumerate(image_paths):
            dst_path = input_dir / f"image_{i:04d}{src_path.suffix}"
            
            # 이미지 로드 및 전처리
            image = cv2.imread(str(src_path))
            if image is None:
                continue
            
            # 크기 제한 (메모리 절약)
            h, w = image.shape[:2]
            if max(h, w) > 2048:
                scale = 2048 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # 약간의 선명화 (모션 블러 완화)
            image = self._enhance_image(image)
            
            cv2.imwrite(str(dst_path), image)
        
        print(f"  {len(image_paths)}장 이미지 준비 완료")
        return input_dir
    
    def _enhance_image(self, image):
        """이미지 선명화"""
        # 언샤프 마스킹
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        sharpened = cv2.addWeighted(image, 1.3, blurred, -0.3, 0)
        
        # 대비 향상
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _create_colmap_database(self, image_paths, database_path):
        """SuperGlue 특징점으로 COLMAP 데이터베이스 생성"""
        
        # 데이터베이스 초기화
        self._initialize_colmap_database(database_path)
        
        # 각 이미지에서 특징점 추출
        features = {}
        for i, image_path in enumerate(image_paths):
            print(f"  특징점 추출: {i+1}/{len(image_paths)} - {image_path.name}")
            feat = self._extract_superpoint_features(image_path)
            if feat is not None:
                features[i] = feat
                self._add_features_to_database(database_path, i, feat)
        
        # SuperGlue 매칭
        print(f"  SuperGlue 매칭...")
        matches_added = 0
        total_pairs = len(image_paths) * (len(image_paths) - 1) // 2
        
        # 순차적 매칭 + 선택적 매칭
        for i in range(len(image_paths)):
            for j in range(i+1, min(i+10, len(image_paths))):  # 인접 10장
                if i in features and j in features:
                    matches = self._match_superglue(features[i], features[j])
                    if len(matches) > 20:  # 최소 매칭 수
                        self._add_matches_to_database(database_path, i, j, matches)
                        matches_added += 1
        
        print(f"  매칭 완료: {matches_added}개 이미지 쌍")
    
    def _initialize_colmap_database(self, database_path):
        """COLMAP 데이터베이스 초기화"""
        if database_path.exists():
            database_path.unlink()
        
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        
        # 테이블 생성
        cursor.execute('''
            CREATE TABLE cameras (
                camera_id INTEGER PRIMARY KEY,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE keypoints (
                image_id INTEGER NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE descriptors (
                image_id INTEGER NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE matches (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _extract_superpoint_features(self, image_path):
        """SuperPoint로 특징점 추출"""
        try:
            # 이미지 로드
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # 전처리
            if max(image.shape) > 1600:  # 크기 제한
                scale = 1600 / max(image.shape)
                new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, new_size)
            
            # SuperPoint 특징점 추출
            inp = self.frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            
            # 결과 정리
            keypoints = pred['keypoints'][0].cpu().numpy()
            descriptors = pred['descriptors'][0].cpu().numpy()
            scores = pred['scores'][0].cpu().numpy()
            
            # 고품질 특징점만 선택
            mask = scores > np.percentile(scores, 60)  # 상위 40%
            
            return {
                'keypoints': keypoints[mask],
                'descriptors': descriptors[:, mask],
                'scores': scores[mask],
                'image_size': image.shape[:2]
            }
            
        except Exception as e:
            print(f"    특징점 추출 실패: {e}")
            return None
    
    def _add_features_to_database(self, database_path, image_id, features):
        """데이터베이스에 특징점 추가"""
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        
        # 카메라 정보 추가 (Simple Pinhole 모델)
        h, w = features['image_size']
        focal = max(w, h) * 0.8  # 보수적 추정
        
        camera_params = np.array([focal, w/2, h/2], dtype=np.float64)
        cursor.execute(
            "INSERT OR REPLACE INTO cameras VALUES (?, ?, ?, ?, ?)",
            (image_id, 0, w, h, camera_params.tobytes())  # 0 = SIMPLE_PINHOLE
        )
        
        # 이미지 정보 추가
        cursor.execute(
            "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, f"image_{image_id:04d}.jpg", image_id, 1, 0, 0, 0, 0, 0, 0)
        )
        
        # 키포인트 추가
        kpts = features['keypoints'].astype(np.float32)
        cursor.execute(
            "INSERT OR REPLACE INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, len(kpts), 2, kpts.tobytes())
        )
        
        # 디스크립터 추가
        desc = features['descriptors'].T.astype(np.float32)  # (N, 256)
        cursor.execute(
            "INSERT OR REPLACE INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id, len(desc), 256, desc.tobytes())
        )
        
        conn.commit()
        conn.close()
    
    def _match_superglue(self, features1, features2):
        """SuperGlue로 두 이미지 매칭"""
        try:
            # 텐서 준비
            data = {
                'keypoints0': torch.from_numpy(features1['keypoints']).float().unsqueeze(0).to(self.device),
                'keypoints1': torch.from_numpy(features2['keypoints']).float().unsqueeze(0).to(self.device),
                'descriptors0': torch.from_numpy(features1['descriptors']).float().unsqueeze(0).to(self.device),
                'descriptors1': torch.from_numpy(features2['descriptors']).float().unsqueeze(0).to(self.device),
                'scores0': torch.from_numpy(features1['scores']).float().unsqueeze(0).to(self.device),
                'scores1': torch.from_numpy(features2['scores']).float().unsqueeze(0).to(self.device),
            }
            
            # SuperGlue 매칭
            with torch.no_grad():
                pred = self.matching.superglue(data)
            
            # 매칭 결과 처리
            matches0 = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
            # 유효한 매칭만 추출
            valid = matches0 > -1
            matches = np.column_stack([
                np.where(valid)[0],
                matches0[valid]
            ])
            
            # 기하학적 검증
            if len(matches) > 8:
                matches = self._geometric_verification_matches(
                    matches, features1['keypoints'], features2['keypoints']
                )
            
            return matches
            
        except Exception as e:
            print(f"    매칭 실패: {e}")
            return np.array([]).reshape(0, 2)
    
    def _geometric_verification_matches(self, matches, kpts1, kpts2):
        """RANSAC으로 기하학적 검증"""
        if len(matches) < 8:
            return matches
        
        try:
            pts1 = kpts1[matches[:, 0]]
            pts2 = kpts2[matches[:, 1]]
            
            # Fundamental Matrix로 검증
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, cv2.FM_RANSAC,
                ransacReprojThreshold=3.0,  # 관대한 임계값
                confidence=0.99
            )
            
            if F is not None and mask is not None:
                return matches[mask.ravel().astype(bool)]
                
        except:
            pass
        
        return matches
    
    def _add_matches_to_database(self, database_path, img1_id, img2_id, matches):
        """데이터베이스에 매칭 결과 추가"""
        if len(matches) == 0:
            return
        
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        
        # COLMAP pair_id 계산
        if img1_id > img2_id:
            img1_id, img2_id = img2_id, img1_id
            matches = matches[:, [1, 0]]  # 순서 바꿈
        
        pair_id = img1_id * 2147483647 + img2_id  # COLMAP 공식
        
        # 매칭 데이터 변환
        matches_data = matches.astype(np.uint32)
        
        cursor.execute(
            "INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?)",
            (pair_id, len(matches), 2, matches_data.tobytes())
        )
        
        conn.commit()
        conn.close()
    
    def _run_colmap_mapper(self, database_path, image_path, output_path):
        """COLMAP으로 SfM 수행"""
        cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--Mapper.ba_global_function_tolerance", "0.000001",
            "--Mapper.ba_global_max_num_iterations", "100",
            "--Mapper.ba_local_max_num_iterations", "50",
            "--Mapper.min_num_matches", "15",  # 낮춤
            "--Mapper.init_min_num_inliers", "30",  # 낮춤
            "--Mapper.abs_pose_min_num_inliers", "15",  # 낮춤
            "--Mapper.filter_max_reproj_error", "8.0",  # 높임 (블러 고려)
        ]
        
        print(f"  COLMAP Mapper 실행...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                print(f"  경고: COLMAP Mapper 오류")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
            else:
                print(f"  ✓ COLMAP SfM 완료")
        except subprocess.TimeoutExpired:
            print(f"  경고: COLMAP Mapper 타임아웃")
        except Exception as e:
            print(f"  오류: COLMAP Mapper 실패: {e}")
    
    def _run_colmap_undistortion(self, image_path, sparse_path, output_path):
        """COLMAP 언디스토션"""
        # 가장 큰 reconstruction 찾기
        sparse_models = list(sparse_path.glob("*/"))
        if not sparse_models:
            print("  경고: Sparse reconstruction 없음")
            return
        
        # 가장 많은 이미지를 가진 모델 선택
        best_model = max(sparse_models, 
                        key=lambda x: len(list(x.glob("images.bin"))))
        
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(best_model),
            "--output_path", str(output_path),
            "--output_type", "COLMAP"
        ]
        
        print(f"  COLMAP 언디스토션 실행...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print(f"  ✓ 언디스토션 완료")
            else:
                print(f"  경고: 언디스토션 오류")
        except Exception as e:
            print(f"  오류: 언디스토션 실패: {e}")
    
    def _convert_to_3dgs_format(self, colmap_path):
        """COLMAP 결과를 3DGS SceneInfo로 변환"""
        try:
            # COLMAP 데이터 읽기
            from scene.dataset_readers import readColmapSceneInfo
            scene_info = readColmapSceneInfo(str(colmap_path), "images", eval=False)
            
            print(f"  ✓ 3DGS 변환 완료")
            print(f"    - 학습 카메라: {len(scene_info.train_cameras)}개")
            print(f"    - 테스트 카메라: {len(scene_info.test_cameras)}개")
            
            return scene_info
            
        except Exception as e:
            print(f"  오류: 3DGS 변환 실패: {e}")
            return None

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SuperGlue + COLMAP 하이브리드 파이프라인")
    parser.add_argument("--image_dir", type=str, required=True, 
                       help="입력 이미지 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="출력 디렉토리")
    parser.add_argument("--max_images", type=int, default=100,
                       help="최대 처리 이미지 수 (기본값: 100)")
    parser.add_argument("--colmap_exe", type=str, default="colmap",
                       help="COLMAP 실행 파일 경로 (기본값: colmap)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="GPU 디바이스 (기본값: cuda)")
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    try:
        pipeline = SuperGlueCOLMAPHybrid(
            colmap_exe=args.colmap_exe,
            device=args.device
        )
    except Exception as e:
        print(f"✗ 파이프라인 초기화 실패: {e}")
        sys.exit(1)
    
    # 이미지 처리
    try:
        scene_info = pipeline.process_images(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            max_images=args.max_images
        )
        
        if scene_info:
            print("\n🎉 성공! 3DGS 학습 준비 완료")
            print(f"결과 디렉토리: {args.output_dir}")
            print("\n다음 명령으로 3DGS 학습:")
            print(f"python train.py -s {args.output_dir}")
        else:
            print("\n❌ 실패: 3DGS 변환에 실패했습니다")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()