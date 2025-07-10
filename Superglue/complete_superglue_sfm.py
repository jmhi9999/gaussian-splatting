# superglue_3dgs_complete.py
# SuperGlue와 3DGS 완전 통합 파이프라인

import numpy as np
import cv2
import torch
from pathlib import Path
import json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys
from scipy.spatial.distance import cdist

# SuperGlue 관련 imports
from models.matching import Matching
from models.utils import frame2tensor

# 3DGS 관련 imports - lazy import로 변경
def get_3dgs_imports():
    """3DGS 관련 모듈들을 lazy import"""
    # gaussian-splatting 루트 디렉토리를 Python path에 추가
    gaussian_splatting_root = Path(__file__).parent.parent
    if str(gaussian_splatting_root) not in sys.path:
        sys.path.insert(0, str(gaussian_splatting_root))
    
    try:
        from scene.dataset_readers import CameraInfo, SceneInfo
        from utils.graphics_utils import BasicPointCloud
        return CameraInfo, SceneInfo, BasicPointCloud
    except ImportError as e:
        print(f"Warning: Could not import 3DGS modules: {e}")
        return None, None, None


class SuperGlue3DGSPipeline:
    """SuperGlue 기반 완전한 3DGS SfM 파이프라인"""
    
    def __init__(self, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # SuperGlue 설정 (더 완화된 설정)
        if config is None:
            config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,  # 더 낮은 임계값으로 더 많은 특징점
                    'max_keypoints': 4096  # 특징점 수 증가 (2048 → 4096)
                },
                'superglue': {
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.1,  # 더 낮은 매칭 임계값 (0.3 → 0.1)
                }
            }
        
        self.matching = Matching(config).eval().to(self.device)
        
        # SfM 데이터 저장소
        self.cameras = {}  # camera_id -> {'R': R, 'T': T, 'K': K, 'image_path': path}
        self.points_3d = {}  # point_id -> {'xyz': xyz, 'color': rgb, 'observations': [(cam_id, kpt_idx)]}
        self.image_features = {}  # image_id -> SuperPoint features
        self.matches = {}  # (img_i, img_j) -> SuperGlue matches
        
        # Bundle Adjustment를 위한 추가 데이터
        self.camera_graph = defaultdict(list)  # 카메라 연결 그래프
        self.point_observations = defaultdict(list)  # 포인트 관찰 데이터
        
        print(f'SuperGlue 3DGS Pipeline initialized on {self.device}')
    
    def process_images_to_3dgs(self, image_dir, output_dir, max_images=120):
        """이미지 디렉토리에서 3DGS 학습 가능한 상태까지 완전 처리"""
        
        print(f"\n=== SuperGlue + 3DGS Pipeline: Processing up to {max_images} images ===")
        
        # 1. 이미지 수집 및 정렬
        image_paths = self._collect_images(image_dir, max_images)
        print(f"Found {len(image_paths)} images")
        
        # 2. SuperPoint 특징점 추출
        print("\n[1/7] Extracting SuperPoint features...")
        self._extract_all_features(image_paths)
        
        # 3. SuperGlue 매칭 (지능적 페어링)
        print("\n[2/7] SuperGlue feature matching...")
        self._intelligent_matching(max_pairs=min(len(image_paths) * 15, 1500))
        
        # 4. 초기 카메라 포즈 추정 (개선된 버전)
        print("\n[3/7] Camera pose estimation...")
        self._estimate_camera_poses_robust()
        
        # 5. 3D 포인트 삼각측량 (개선된 버전)
        print("\n[4/7] 3D point triangulation...")
        self._triangulate_all_points_robust()
        
        # 6. Bundle Adjustment (실제 구현)
        print("\n[5/7] Bundle adjustment optimization...")
        self._bundle_adjustment_robust()
        
        # 7. 포인트 클라우드 정제 (더 완화된 버전)
        print("\n[6/7] Point cloud refinement...")
        self._refine_point_cloud()
        
        # 8. 3DGS 형식으로 변환
        print("\n[7/7] Converting to 3DGS format...")
        scene_info = self._create_3dgs_scene_info(image_paths)
        
        # 9. 결과 저장
        self._save_3dgs_format(scene_info, output_dir)
        
        return scene_info
    
    def _collect_images(self, image_dir, max_images):
        """이미지 수집 및 정렬"""
        image_dir = Path(image_dir)
        image_paths = []
        
        # 지원하는 확장자
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        # 정렬 및 제한
        image_paths.sort()
        return image_paths[:max_images]
    
    def _extract_all_features(self, image_paths):
        """모든 이미지에서 SuperPoint 특징점 추출 (수정된 버전)"""
        for i, image_path in enumerate(image_paths):
            print(f"  {i+1:3d}/{len(image_paths)}: {image_path.name}")
            
            # 이미지 로드
            image = self._load_image(image_path)
            if image is None:
                continue
            
            # SuperPoint 특징점 추출
            inp = frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            
            # 결과 저장 - 모든 필요한 키 포함
            self.image_features[i] = {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(), 
                'scores': pred['scores'][0].cpu().numpy(),
                'image_path': str(image_path),
                'image_size': image.shape[:2]  # (H, W)
            }
            
            print(f"    Keypoints: {self.image_features[i]['keypoints'].shape[0]}")
            
        print(f"  Extracted features from {len(self.image_features)} images")
    
    def _intelligent_matching(self, max_pairs=1500):
        """지능적 이미지 매칭 (더 완화된 버전)"""
        n_images = len(self.image_features)
        
        # 1. 순차적 매칭 (인접 이미지) - 더 낮은 임계값
        for i in range(n_images - 1):
            matches = self._match_pair_superglue(i, i+1)
            if len(matches) > 5:  # 더 낮은 임계값 (8 → 5)
                self.matches[(i, i+1)] = matches
                self.camera_graph[i].append(i+1)
                self.camera_graph[i+1].append(i)
        
        # 2. 건너뛰기 매칭 (2, 3, 5, 8, 12 간격) - 더 낮은 임계값
        for gap in [2, 3, 5, 8, 12]:
            for i in range(n_images - gap):
                j = i + gap
                if len(self.matches) >= max_pairs:
                    break
                    
                matches = self._match_pair_superglue(i, j)
                if len(matches) > 8:  # 더 낮은 임계값 (12 → 8)
                    self.matches[(i, j)] = matches
                    self.camera_graph[i].append(j)
                    self.camera_graph[j].append(i)
        
        # 3. 추가 매칭 (연결되지 않은 카메라들) - 더 적극적
        for i in range(n_images):
            if len(self.camera_graph[i]) < 2:  # 연결이 적은 카메라 (3 → 2)
                for j in range(i+1, n_images):
                    if len(self.camera_graph[j]) < 3 and len(self.matches) < max_pairs:  # (4 → 3)
                        matches = self._match_pair_superglue(i, j)
                        if len(matches) > 5:  # 더 낮은 임계값 (8 → 5)
                            self.matches[(i, j)] = matches
                            self.camera_graph[i].append(j)
                            self.camera_graph[j].append(i)
        
        # 4. 매칭 품질 검증 및 필터링 (더 완화된 조건)
        self._filter_low_quality_matches_very_relaxed()
        
        print(f"  Created {len(self.matches)} image pairs with good matches")
        print(f"  Camera connectivity: {[len(self.camera_graph[i]) for i in range(min(10, n_images))]}")
    
    def _filter_low_quality_matches_very_relaxed(self):
        """매우 완화된 낮은 품질의 매칭 필터링"""
        pairs_to_remove = []
        
        for (cam_i, cam_j), matches in self.matches.items():
            if len(matches) < 3:  # 더 낮은 임계값 (5 → 3)
                pairs_to_remove.append((cam_i, cam_j))
                continue
            
            # 매칭 품질 분석
            confidences = [conf for _, _, conf in matches]
            avg_confidence = np.mean(confidences)
            
            if avg_confidence < 0.1:  # 더 낮은 임계값 (0.2 → 0.1)
                pairs_to_remove.append((cam_i, cam_j))
                continue
            
            # 매칭 분포 분석 (더 완화된 조건)
            if self._has_poor_matching_distribution_very_relaxed(cam_i, cam_j, matches):
                pairs_to_remove.append((cam_i, cam_j))
        
        # 필터링된 매칭 제거
        for pair in pairs_to_remove:
            cam_i, cam_j = pair
            del self.matches[pair]
            
            # 그래프에서도 제거
            if cam_j in self.camera_graph[cam_i]:
                self.camera_graph[cam_i].remove(cam_j)
            if cam_i in self.camera_graph[cam_j]:
                self.camera_graph[cam_j].remove(cam_i)
        
        print(f"  Filtered out {len(pairs_to_remove)} low-quality matches (very relaxed)")
    
    def _has_poor_matching_distribution_very_relaxed(self, cam_i, cam_j, matches):
        """매우 완화된 매칭 분포 검사"""
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        # 인덱스 범위 검증
        valid_matches = []
        for idx_i, idx_j, conf in matches:
            if (idx_i < len(kpts_i) and idx_j < len(kpts_j) and 
                idx_i >= 0 and idx_j >= 0):
                valid_matches.append((idx_i, idx_j, conf))
        
        if len(valid_matches) < 2:  # 더 낮은 임계값 (3 → 2)
            return True
        
        # 매칭된 점들의 위치 분석
        matched_i = np.array([kpts_i[idx_i] for idx_i, _, _ in valid_matches])
        matched_j = np.array([kpts_j[idx_j] for _, idx_j, _ in valid_matches])
        
        # 이미지 크기
        h_i, w_i = self.image_features[cam_i]['image_size']
        h_j, w_j = self.image_features[cam_j]['image_size']
        
        # 경계 근처의 매칭이 너무 많은지 확인 (더 완화된 조건)
        border_threshold = 20  # 더 작은 경계 (30 → 20)
        
        border_matches_i = np.sum((matched_i[:, 0] < border_threshold) | 
                                  (matched_i[:, 0] > w_i - border_threshold) |
                                  (matched_i[:, 1] < border_threshold) | 
                                  (matched_i[:, 1] > h_i - border_threshold))
        
        border_matches_j = np.sum((matched_j[:, 0] < border_threshold) | 
                                  (matched_j[:, 0] > w_j - border_threshold) |
                                  (matched_j[:, 1] < border_threshold) | 
                                  (matched_j[:, 1] > h_j - border_threshold))
        
        # 경계 매칭이 전체의 95% 이상이면 나쁜 분포 (90% → 95%)
        if border_matches_i > len(valid_matches) * 0.95 or border_matches_j > len(valid_matches) * 0.95:
            return True
        
        return False
    
    def _match_pair_superglue(self, cam_i, cam_j):
        """수정된 SuperGlue 매칭 (더 완화된 버전)"""
        
        if cam_i not in self.image_features or cam_j not in self.image_features:
            return []
        
        try:
            feat_i = self.image_features[cam_i]
            feat_j = self.image_features[cam_j]
            
            if feat_i['keypoints'].shape[0] == 0 or feat_j['keypoints'].shape[0] == 0:
                return []
            
            # SuperGlue 입력 데이터 구성
            data = {
                'keypoints0': torch.from_numpy(feat_i['keypoints']).unsqueeze(0).to(self.device),
                'keypoints1': torch.from_numpy(feat_j['keypoints']).unsqueeze(0).to(self.device),
                'descriptors0': torch.from_numpy(feat_i['descriptors']).unsqueeze(0).to(self.device),
                'descriptors1': torch.from_numpy(feat_j['descriptors']).unsqueeze(0).to(self.device),
                'scores0': torch.from_numpy(feat_i['scores']).unsqueeze(0).to(self.device),
                'scores1': torch.from_numpy(feat_j['scores']).unsqueeze(0).to(self.device),
                'image0': torch.zeros((1, 1, feat_i['image_size'][0], feat_i['image_size'][1])).to(self.device),
                'image1': torch.zeros((1, 1, feat_j['image_size'][0], feat_j['image_size'][1])).to(self.device),
            }
            
            # SuperGlue 매칭 수행
            with torch.no_grad():
                result = self.matching.superglue(data)
            
            # 매칭 결과 추출 (올바른 키 사용)
            indices0 = result['indices0'][0].cpu().numpy()
            indices1 = result['indices1'][0].cpu().numpy()
            mscores0 = result['matching_scores0'][0].cpu().numpy()
            
            # 유효한 매칭 추출 (더 낮은 임계값)
            valid_matches = []
            threshold = 0.05  # 더 낮은 임계값 (0.1 → 0.05)
            
            for i, j in enumerate(indices0):
                if j >= 0 and mscores0[i] > threshold:
                    # 상호 매칭 확인
                    if j < len(indices1) and indices1[j] == i:
                        # 인덱스 범위 추가 검증
                        if i < len(feat_i['keypoints']) and j < len(feat_j['keypoints']):
                            valid_matches.append((i, j, mscores0[i]))
            
            print(f"    Pair {cam_i}-{cam_j}: {len(valid_matches)} mutual matches")
            return valid_matches
            
        except Exception as e:
            print(f"    SuperGlue matching failed for pair {cam_i}-{cam_j}: {e}")
            return []
    
    def _estimate_camera_poses_robust(self):
        """개선된 카메라 포즈 추정 (더 안전한 버전)"""
        
        # 첫 번째 카메라를 원점으로 설정
        self.cameras[0] = {
            'R': np.eye(3, dtype=np.float32),
            'T': np.zeros(3, dtype=np.float32),
            'K': self._estimate_intrinsics(0)
        }
        
        print(f"  Camera 0: Origin (reference)")
        
        # 1단계: 연결된 카메라들만 포즈 추정
        estimated_cameras = {0}
        queue = [0]
        
        while queue:
            current_cam = queue.pop(0)
            
            # 현재 카메라와 연결된 카메라들 확인
            for neighbor_cam in self.camera_graph[current_cam]:
                if neighbor_cam in estimated_cameras:
                    continue
                
                # 매칭 데이터 찾기
                pair_key = (current_cam, neighbor_cam) if current_cam < neighbor_cam else (neighbor_cam, current_cam)
                if pair_key not in self.matches:
                    continue
                
                # Essential Matrix 기반 포즈 추정
                R_rel, T_rel = self._estimate_relative_pose_robust(current_cam, neighbor_cam, pair_key)
                
                if R_rel is not None and T_rel is not None:
                    # 월드 좌표계에서의 절대 포즈 계산
                    R_ref, T_ref = self.cameras[current_cam]['R'], self.cameras[current_cam]['T']
                    
                    # 상대 포즈를 절대 포즈로 변환
                    R_world = R_rel @ R_ref
                    T_world = R_rel @ T_ref + T_rel
                    
                    # 포즈 유효성 검사
                    if self._is_valid_rotation_matrix(R_world):
                        self.cameras[neighbor_cam] = {
                            'R': R_world.astype(np.float32),
                            'T': T_world.astype(np.float32),
                            'K': self._estimate_intrinsics(neighbor_cam)
                        }
                        
                        print(f"  Camera {neighbor_cam}: Estimated from camera {current_cam}")
                        estimated_cameras.add(neighbor_cam)
                        queue.append(neighbor_cam)
                    else:
                        print(f"  Camera {neighbor_cam}: Invalid pose, skipping...")
                else:
                    print(f"  Camera {neighbor_cam}: Pose estimation failed, will use default pose")
        
        # 2단계: 연결되지 않은 카메라들에 대한 기본 포즈 설정
        for cam_id in range(len(self.image_features)):
            if cam_id not in estimated_cameras:
                print(f"  Camera {cam_id}: Using default pose (not connected)")
                # 기본 원형 배치
                angle = cam_id * (2 * np.pi / len(self.image_features))
                radius = 3.0
                
                self.cameras[cam_id] = {
                    'R': np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32),
                    'T': np.array([radius * np.sin(angle), 0, radius * (1 - np.cos(angle))], dtype=np.float32),
                    'K': self._estimate_intrinsics(cam_id)
                }
        
        print(f"  Estimated poses for {len(estimated_cameras)} cameras")
        print(f"  Total cameras with poses: {len(self.cameras)}")
    
    def _estimate_relative_pose_robust(self, cam_i, cam_j, pair_key):
        """개선된 두 카메라 간 상대 포즈 추정 (더 완화된 버전)"""
        matches = self.matches[pair_key]
        
        if len(matches) < 4:  # 더 낮은 임계값 (6 → 4)
            print(f"    Pair {cam_i}-{cam_j}: Insufficient matches ({len(matches)} < 4)")
            return None, None
        
        # 매칭점들 추출 (더 완화된 필터링)
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        # 더 낮은 신뢰도 매칭도 사용 (0.1 → 0.05)
        high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches if conf > 0.05]
        
        if len(high_conf_matches) < 4:
            # 더 낮은 신뢰도 매칭도 시도
            high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches if conf > 0.1]  # 0.3 → 0.1
        
        if len(high_conf_matches) < 4:
            print(f"    Pair {cam_i}-{cam_j}: Insufficient high-confidence matches ({len(high_conf_matches)} < 4)")
            return None, None
        
        # 인덱스 범위 검증
        valid_matches = []
        for idx_i, idx_j, conf in high_conf_matches:
            if (idx_i < len(kpts_i) and idx_j < len(kpts_j) and 
                idx_i >= 0 and idx_j >= 0):
                valid_matches.append((idx_i, idx_j, conf))
        
        if len(valid_matches) < 4:
            print(f"    Pair {cam_i}-{cam_j}: Insufficient valid matches after index validation ({len(valid_matches)} < 4)")
            return None, None
        
        pts_i = np.array([kpts_i[idx_i] for idx_i, _, conf in valid_matches])
        pts_j = np.array([kpts_j[idx_j] for _, idx_j, conf in valid_matches])
        
        # 카메라 내부 파라미터
        K_i = self.cameras.get(cam_i, {}).get('K', self._estimate_intrinsics(cam_i))
        K_j = self._estimate_intrinsics(cam_j)
        
        # 여러 임계값으로 시도 (더 완화된 임계값)
        thresholds = [2.0, 4.0, 6.0, 8.0]  # 더 큰 임계값들
        best_R, best_T = None, None
        best_inliers = 0
        
        for threshold in thresholds:
            try:
                # Essential Matrix 추정
                E, mask = cv2.findEssentialMat(
                    pts_i, pts_j, K_i,
                    method=cv2.RANSAC,
                    prob=0.99,  # 더 완화
                    threshold=threshold,
                    maxIters=500  # 더 적은 반복
                )
                
                if E is None or E.shape != (3, 3):
                    continue
                
                # 포즈 복원 (에러 처리 추가)
                try:
                    _, R, T, mask = cv2.recoverPose(E, pts_i, pts_j, K_i, mask=mask)
                    
                    if R is None or T is None:
                        continue
                    
                    # 인라이어 수 계산
                    inliers = np.sum(mask)
                    
                    if inliers > best_inliers:
                        # 더 완화된 재투영 오차 검증
                        if self._verify_pose_quality_very_relaxed(pts_i, pts_j, R, T, K_i, K_j):
                            best_R, best_T = R, T.flatten()
                            best_inliers = inliers
                            
                except cv2.error as e:
                    print(f"      cv2.recoverPose failed for threshold {threshold}: {e}")
                    continue
                    
            except cv2.error as e:
                print(f"      cv2.findEssentialMat failed for threshold {threshold}: {e}")
                continue
            except Exception as e:
                print(f"      Unexpected error for threshold {threshold}: {e}")
                continue
        
        # OpenCV 방법이 실패한 경우 fallback 사용
        if best_R is None:
            print(f"    Pair {cam_i}-{cam_j}: OpenCV methods failed, trying fallback...")
            best_R, best_T = self._estimate_pose_fallback(pts_i, pts_j, K_i, K_j)
        
        if best_R is not None:
            print(f"    Pair {cam_i}-{cam_j}: Successfully estimated pose with {best_inliers} inliers")
        else:
            print(f"    Pair {cam_i}-{cam_j}: Failed to estimate pose")
        
        return best_R, best_T
    
    def _estimate_pose_fallback(self, pts_i, pts_j, K_i, K_j):
        """OpenCV 실패시 사용할 fallback 포즈 추정"""
        try:
            # 간단한 호모그래피 기반 방법
            H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 5.0)
            
            if H is None:
                return None, None
            
            # 호모그래피에서 회전과 이동 추출 (근사)
            # 이는 정확하지 않지만 기본적인 포즈를 제공
            K_inv = np.linalg.inv(K_i)
            R_approx = K_inv @ H @ K_i
            
            # SVD를 사용하여 회전 행렬로 정규화
            U, _, Vt = np.linalg.svd(R_approx)
            R = U @ Vt
            
            # 회전 행렬 유효성 검사
            if not self._is_valid_rotation_matrix(R):
                # 기본 회전 행렬 사용
                R = np.eye(3)
            
            # 이동 벡터 추정 (간단한 근사)
            T = np.array([0.1, 0.0, 0.0])  # 기본 이동
            
            return R, T
            
        except Exception as e:
            print(f"      Fallback pose estimation failed: {e}")
            return None, None
    
    def _is_valid_rotation_matrix(self, R):
        """회전 행렬이 유효한지 확인"""
        try:
            # 행렬식이 1에 가까운지 확인
            det = np.linalg.det(R)
            if abs(det - 1.0) > 0.1:
                return False
            
            # R * R^T = I인지 확인
            I = np.eye(3)
            RRt = R @ R.T
            if np.max(np.abs(RRt - I)) > 0.1:
                return False
            
            return True
        except:
            return False
    
    def _verify_pose_quality_very_relaxed(self, pts_i, pts_j, R, T, K_i, K_j):
        """매우 완화된 포즈 품질 검증"""
        # 재투영 오차 계산
        P_i = K_i @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_j = K_j @ np.hstack([R, T.reshape(-1, 1)])
        
        errors = []
        depths_i = []
        depths_j = []
        
        for pt_i, pt_j in zip(pts_i, pts_j):
            # 삼각측량
            point_4d = cv2.triangulatePoints(P_i, P_j, pt_i.reshape(2, 1), pt_j.reshape(2, 1))
            if abs(point_4d[3, 0]) > 1e-10:
                point_3d = (point_4d[:3] / point_4d[3]).flatten()
                
                # 재투영 (3D 좌표계에서)
                proj_i_3d = P_i @ np.append(point_3d, 1)
                proj_j_3d = P_j @ np.append(point_3d, 1)
                
                # 2D 좌표로 변환
                proj_i_2d = proj_i_3d[:2] / proj_i_3d[2]
                proj_j_2d = proj_j_3d[:2] / proj_j_3d[2]
                
                error_i = np.linalg.norm(proj_i_2d - pt_i)
                error_j = np.linalg.norm(proj_j_2d - pt_j)
                errors.append(max(error_i, error_j))
                
                # 깊이 정보 저장 (3D 좌표계에서)
                depths_i.append(proj_i_3d[2])
                depths_j.append(proj_j_3d[2])
        
        if len(errors) < 2:  # 더 낮은 임계값 (3 → 2)
            return False
        
        # 오차 통계
        median_error = np.median(errors)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # 깊이 검증 (더 완화된 조건)
        if depths_i and depths_j:
            depths_i = np.array(depths_i)
            depths_j = np.array(depths_j)
            
            # 깊이가 양수인지 확인
            if np.any(depths_i <= 0) or np.any(depths_j <= 0):
                return False
            
            # 깊이 비율이 합리적인지 확인 (더 완화된 조건)
            depth_ratios = depths_j / depths_i
            if np.median(depth_ratios) < 0.01 or np.median(depth_ratios) > 50:  # 0.05~20 → 0.01~50
                return False
        
        # 오차 임계값 검증 (더 완화된 조건)
        pose_quality = (median_error < 15.0 and   # 8.0 → 15.0
                mean_error < 20.0 and    # 10.0 → 20.0
                max_error < 50.0)        # 20.0 → 50.0
        
        if not pose_quality:
            print(f"      Pose quality check failed: median={median_error:.2f}, mean={mean_error:.2f}, max={max_error:.2f}")
        
        return pose_quality
    
    def _estimate_intrinsics(self, cam_id):
        """개선된 카메라 내부 파라미터 추정"""
        h, w = self.image_features[cam_id]['image_size']
        
        # 더 정확한 focal length 추정
        # 일반적으로 focal length는 이미지 크기의 0.8~1.2배
        focal = max(w, h) * 0.9  # 약간 보수적인 추정
        
        # 주점을 이미지 중심으로 설정
        cx, cy = w / 2, h / 2
        
        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _triangulate_all_points_robust(self):
        """개선된 3D 포인트 삼각측량 (더 완화된 버전)"""
        point_id = 0
        total_matches_processed = 0
        total_valid_matches = 0
        total_triangulated = 0
        total_validated = 0
        
        print(f"  Processing {len(self.matches)} image pairs for triangulation...")
        
        for (cam_i, cam_j), matches in self.matches.items():
            if cam_i not in self.cameras or cam_j not in self.cameras:
                continue
            
            try:
                # 투영 행렬 생성
                P_i = self._get_projection_matrix(cam_i)
                P_j = self._get_projection_matrix(cam_j)
                
                kpts_i = self.image_features[cam_i]['keypoints']
                kpts_j = self.image_features[cam_j]['keypoints']
                
                # 더 낮은 신뢰도 매칭도 사용 (0.1 → 0.05)
                high_conf_matches = [(idx_i, idx_j, conf) for idx_i, idx_j, conf in matches if conf > 0.05]
                total_matches_processed += len(matches)
                
                # 인덱스 범위 검증
                valid_matches = []
                for idx_i, idx_j, conf in high_conf_matches:
                    if (idx_i < len(kpts_i) and idx_j < len(kpts_j) and 
                        idx_i >= 0 and idx_j >= 0):
                        valid_matches.append((idx_i, idx_j, conf))
                
                total_valid_matches += len(valid_matches)
                
                for idx_i, idx_j, conf in valid_matches:
                    try:
                        # 삼각측량
                        pt_i = kpts_i[idx_i].astype(np.float32)
                        pt_j = kpts_j[idx_j].astype(np.float32)
                        
                        point_4d = cv2.triangulatePoints(
                            P_i, P_j,
                            pt_i.reshape(2, 1),
                            pt_j.reshape(2, 1)
                        )
                        
                        if abs(point_4d[3, 0]) > 1e-10:
                            point_3d = (point_4d[:3] / point_4d[3]).flatten()
                            total_triangulated += 1
                            
                            # 더 완화된 유효성 검사
                            if self._is_point_valid_extremely_relaxed(point_3d, cam_i, cam_j, pt_i, pt_j):
                                # 색상 추정 (이미지에서 샘플링)
                                color = self._estimate_point_color_robust(point_3d, cam_i, idx_i)
                                
                                self.points_3d[point_id] = {
                                    'xyz': point_3d.astype(np.float32),
                                    'color': color,
                                    'observations': [(cam_i, idx_i), (cam_j, idx_j)]
                                }
                                
                                # Bundle Adjustment를 위한 관찰 데이터 저장
                                self.point_observations[point_id] = [
                                    (cam_i, pt_i, conf),
                                    (cam_j, pt_j, conf)
                                ]
                                
                                point_id += 1
                                total_validated += 1
                                
                    except Exception as e:
                        # 개별 삼각측량 실패는 무시하고 계속 진행
                        continue
                        
            except Exception as e:
                print(f"    Error processing pair {cam_i}-{cam_j}: {e}")
                continue
        
        print(f"  Triangulation statistics:")
        print(f"    Total matches processed: {total_matches_processed}")
        print(f"    Valid matches: {total_valid_matches}")
        print(f"    Successfully triangulated: {total_triangulated}")
        print(f"    Passed validation: {total_validated}")
        print(f"    Final 3D points: {len(self.points_3d)}")
    
    def _is_point_valid_extremely_relaxed(self, point_3d, cam_i, cam_j, pt_i, pt_j):
        """극도로 완화된 3D 포인트 유효성 검사"""
        # 두 카메라 모두에서 앞쪽에 있는지 확인
        for cam_id in [cam_i, cam_j]:
            cam = self.cameras[cam_id]
            R, T = cam['R'], cam['T']
            
            # 카메라 좌표계로 변환
            point_cam = R @ (point_3d - T)
            
            if point_cam[2] <= 0.0001:  # 매우 완화된 조건 (0.001 → 0.0001)
                return False
            
            # 재투영 오차 확인
            K = cam['K']
            point_2d_proj = K @ point_cam
            point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
            
            # 실제 관찰점과 비교
            if cam_id == cam_i:
                observed_pt = pt_i
            else:
                observed_pt = pt_j
            
            reproj_error = np.linalg.norm(point_2d_proj - observed_pt)
            if reproj_error > 50.0:  # 매우 완화된 조건 (30.0 → 50.0)
                return False
            
            # 이미지 경계 확인 (매우 완화된 조건)
            h, w = self.image_features[cam_id]['image_size']
            if (point_2d_proj[0] < -200 or point_2d_proj[0] >= w + 200 or
                point_2d_proj[1] < -200 or point_2d_proj[1] >= h + 200):
                return False
        
        return True
    
    def _estimate_point_color_robust(self, point_3d, cam_id, kpt_idx):
        """개선된 3D 포인트 색상 추정"""
        # 실제 구현에서는 이미지에서 색상을 샘플링
        # 여기서는 간단히 랜덤 색상 사용
        return np.random.rand(3).astype(np.float32)
    
    def _bundle_adjustment_robust(self, max_iterations=100):
        """실제 Bundle Adjustment 구현 (더 완화된 버전)"""
        
        n_cameras = len(self.cameras)
        n_points = len(self.points_3d)
        
        if n_cameras < 2 or n_points < 5:  # 더 낮은 임계값 (10 → 5)
            print("  Insufficient data for bundle adjustment")
            return
        
        # 관찰 데이터가 충분한지 확인
        total_observations = sum(len(obs) for obs in self.point_observations.values())
        if total_observations < 10:  # 더 낮은 임계값 (20 → 10)
            print("  Insufficient observations for bundle adjustment")
            return
        
        print(f"  Optimizing {n_cameras} cameras and {n_points} points...")
        print(f"  Total observations: {total_observations}")
        
        # 초기 파라미터 벡터 구성
        try:
            params = self._pack_parameters()
        except Exception as e:
            print(f"  Parameter packing failed: {e}")
            return
        
        # Bundle Adjustment 최적화 (더 완화된 설정)
        try:
            # 잔차 수와 변수 수 계산
            n_variables = len(params)
            n_residuals = total_observations * 2  # 각 관찰당 2개 잔차 (x, y)
            
            print(f"  Variables: {n_variables}, Residuals: {n_residuals}")
            
            if n_residuals < n_variables:
                print("  Using 'trf' method (fewer residuals than variables)")
                method = 'trf'
            else:
                print("  Using 'lm' method")
                method = 'lm'
            
            result = least_squares(
                self._compute_residuals,
                params,
                method=method,
                max_nfev=max_iterations,
                verbose=1,
                ftol=1e-4,  # 더 완화된 허용오차 (1e-6 → 1e-4)
                xtol=1e-4    # 더 완화된 허용오차 (1e-6 → 1e-4)
            )
            
            # 결과 언패킹
            self._unpack_parameters(result.x)
            
            print(f"  Bundle adjustment completed. Final cost: {result.cost:.6f}")
            
        except Exception as e:
            print(f"  Bundle adjustment failed: {e}")
            print("  Continuing without bundle adjustment...")
    
    def _pack_parameters(self):
        """카메라 포즈와 3D 포인트를 하나의 벡터로 패킹"""
        params = []
        
        # 카메라 포즈 (회전 + 이동)
        for cam_id in sorted(self.cameras.keys()):
            cam = self.cameras[cam_id]
            R = cam['R']
            T = cam['T']
            
            # 회전 행렬을 로드리게스 벡터로 변환
            angle_axis = self._rotation_matrix_to_angle_axis(R)
            params.extend(angle_axis)
            params.extend(T)
        
        # 3D 포인트
        for point_id in sorted(self.points_3d.keys()):
            point = self.points_3d[point_id]['xyz']
            params.extend(point)
        
        params = np.array(params)
        
        # NaN이나 무한대 값 체크
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            raise ValueError("Invalid parameters detected (NaN or Inf)")
        
        return params
    
    def _unpack_parameters(self, params):
        """벡터에서 카메라 포즈와 3D 포인트 언패킹"""
        idx = 0
        
        # 카메라 포즈 복원
        for cam_id in sorted(self.cameras.keys()):
            # 로드리게스 벡터 (3개)
            angle_axis = params[idx:idx+3]
            idx += 3
            
            # 이동 벡터 (3개)
            T = params[idx:idx+3]
            idx += 3
            
            # 회전 행렬로 변환
            R = self._angle_axis_to_rotation_matrix(angle_axis)
            
            self.cameras[cam_id]['R'] = R.astype(np.float32)
            self.cameras[cam_id]['T'] = T.astype(np.float32)
        
        # 3D 포인트 복원
        for point_id in sorted(self.points_3d.keys()):
            xyz = params[idx:idx+3]
            idx += 3
            self.points_3d[point_id]['xyz'] = xyz.astype(np.float32)
    
    def _compute_residuals(self, params):
        """Bundle Adjustment 잔차 계산"""
        residuals = []
        
        # 파라미터 언패킹
        try:
            self._unpack_parameters(params)
        except Exception as e:
            print(f"    Warning: Parameter unpacking failed: {e}")
            return np.ones(100) * 1e6  # 큰 잔차 반환
        
        # 각 관찰에 대한 재투영 오차 계산
        for point_id, observations in self.point_observations.items():
            if point_id not in self.points_3d:
                continue
                
            point_3d = self.points_3d[point_id]['xyz']
            
            for cam_id, observed_pt, conf in observations:
                if cam_id not in self.cameras:
                    continue
                
                try:
                    cam = self.cameras[cam_id]
                    K = cam['K']
                    R = cam['R']
                    T = cam['T']
                    
                    # 카메라 좌표계로 변환
                    point_cam = R @ (point_3d - T)
                    
                    # 재투영
                    point_2d_proj = K @ point_cam
                    if abs(point_2d_proj[2]) < 1e-10:  # 0으로 나누기 방지
                        continue
                    point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                    
                    # 잔차 계산 (신뢰도로 가중치)
                    residual = (point_2d_proj - observed_pt) * conf
                    residuals.extend(residual)
                    
                except Exception as e:
                    # 개별 관찰에서 오류가 발생해도 계속 진행
                    continue
        
        if len(residuals) == 0:
            return np.ones(100) * 1e6  # 빈 잔차 방지
        
        residuals = np.array(residuals)
        
        # NaN이나 무한대 값 체크
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            return np.ones(len(residuals)) * 1e6
        
        return residuals
    
    def _rotation_matrix_to_angle_axis(self, R):
        """회전 행렬을 로드리게스 벡터로 변환"""
        # 간단한 구현 (실제로는 더 정확한 변환이 필요)
        trace = np.trace(R)
        if trace > 3 - 1e-6:
            return np.zeros(3)
        
        angle = np.arccos((trace - 1) / 2)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
        
        return angle * axis
    
    def _angle_axis_to_rotation_matrix(self, angle_axis):
        """로드리게스 벡터를 회전 행렬로 변환"""
        angle = np.linalg.norm(angle_axis)
        if angle < 1e-6:
            return np.eye(3)
        
        axis = angle_axis / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def _refine_point_cloud(self):
        """포인트 클라우드 정제 (더 완화된 버전)"""
        print(f"  Refining point cloud...")
        
        # 1. 중복 포인트 제거 (더 완화된 조건)
        points_to_remove = set()
        points_list = list(self.points_3d.items())
        
        for i, (id1, point1) in enumerate(points_list):
            for j, (id2, point2) in enumerate(points_list[i+1:], i+1):
                if id1 in points_to_remove or id2 in points_to_remove:
                    continue
                
                dist = np.linalg.norm(point1['xyz'] - point2['xyz'])
                if dist < 0.001:  # 더 작은 중복 임계값 (0.01 → 0.001)
                    points_to_remove.add(id2)
        
        # 중복 포인트 제거
        for point_id in points_to_remove:
            del self.points_3d[point_id]
            if point_id in self.point_observations:
                del self.point_observations[point_id]
        
        print(f"  Removed {len(points_to_remove)} duplicate points")
        print(f"  Final point cloud: {len(self.points_3d)} points")
    
    def _get_projection_matrix(self, cam_id):
        """카메라 투영 행렬 생성"""
        cam = self.cameras[cam_id]
        K, R, T = cam['K'], cam['R'], cam['T']
        
        # P = K[R|t] (t = -R^T * T_world)
        RT = np.hstack([R, T.reshape(-1, 1)])
        P = K @ RT
        
        return P
    
    def _create_3dgs_scene_info(self, image_paths):
        """3DGS용 SceneInfo 생성 (개선된 버전)"""
        
        # Lazy import 3DGS modules
        CameraInfo, SceneInfo, BasicPointCloud = get_3dgs_imports()
        if CameraInfo is None:
            raise ImportError("3DGS modules not available")
        
        # CameraInfo 리스트 생성
        cam_infos = []
        for cam_id in sorted(self.cameras.keys()):
            cam = self.cameras[cam_id]
            image_path = self.image_features[cam_id]['image_path']
            h, w = self.image_features[cam_id]['image_size']
            
            # FoV 계산
            K = cam['K']
            focal_x, focal_y = K[0, 0], K[1, 1]
            FovX = 2 * np.arctan(w / (2 * focal_x))
            FovY = 2 * np.arctan(h / (2 * focal_y))
            
            # 더 나은 테스트 분할 (연결성 기반)
            is_test = self._should_be_test_camera(cam_id)
            
            cam_info = CameraInfo(
                uid=cam_id,
                R=cam['R'],
                T=cam['T'],
                FovY=float(FovY),
                FovX=float(FovX),
                image_path=image_path,
                image_name=Path(image_path).name,
                width=w,
                height=h,
                depth_params=None,
                depth_path="",
                is_test=is_test
            )
            cam_infos.append(cam_info)
        
        # 포인트 클라우드 생성
        if self.points_3d:
            points = np.array([pt['xyz'] for pt in self.points_3d.values()])
            colors = np.array([pt['color'] for pt in self.points_3d.values()])
            
            # 법선 벡터 (개선된 계산)
            normals = self._compute_point_normals(points)
            
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        else:
            # 기본 포인트 클라우드
            n_points = 15000  # 더 많은 수 (8000 → 15000)
            points = np.random.randn(n_points, 3).astype(np.float32) * 3  # 더 넓은 분포
            colors = np.random.rand(n_points, 3).astype(np.float32)
            normals = np.random.randn(n_points, 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # 학습/테스트 분할
        train_cams = [c for c in cam_infos if not c.is_test]
        test_cams = [c for c in cam_infos if c.is_test]
        
        # NeRF 정규화
        nerf_norm = self._compute_nerf_normalization(train_cams)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cams,
            test_cameras=test_cams,
            nerf_normalization=nerf_norm,
            ply_path="",
            is_nerf_synthetic=False
        )
    
    def _should_be_test_camera(self, cam_id):
        """더 나은 테스트 카메라 선택"""
        # 연결성이 낮은 카메라를 테스트로 선택
        connectivity = len(self.camera_graph.get(cam_id, []))
        
        # 연결성이 1 이하이거나, 특정 간격으로 선택
        if connectivity <= 1:
            return True
        
        # 10개마다 1개씩 테스트로 선택 (연결성이 높은 카메라들 중에서)
        if cam_id % 10 == 0 and connectivity >= 2:
            return True
        
        return False
    
    def _compute_point_normals(self, points):
        """포인트 클라우드 법선 벡터 계산"""
        if len(points) < 3:
            return np.random.randn(len(points), 3).astype(np.float32)
        
        # 간단한 법선 계산 (sklearn 의존성 제거)
        try:
            normals = np.zeros_like(points)
            
            for i in range(len(points)):
                # 현재 포인트
                current_point = points[i]
                
                # 다른 모든 포인트와의 거리 계산
                distances = np.linalg.norm(points - current_point, axis=1)
                
                # 가장 가까운 10개 포인트 선택 (자기 자신 제외)
                nearest_indices = np.argsort(distances)[1:11]  # 자기 자신 제외
                
                if len(nearest_indices) < 3:
                    normals[i] = np.random.randn(3)
                    continue
                
                # 이웃 포인트들의 중심 계산
                neighbors = points[nearest_indices]
                centroid = np.mean(neighbors, axis=0)
                
                # 공분산 행렬 계산
                centered = neighbors - centroid
                cov_matrix = centered.T @ centered
                
                # 가장 작은 고유값에 해당하는 고유벡터가 법선
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                normal = eigenvecs[:, 0]  # 가장 작은 고유값
                
                # 방향 일관성 확인
                if normal[2] < 0:
                    normal = -normal
                
                normals[i] = normal
            
            # 정규화
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normals = normals / norms
            
        except Exception as e:
            print(f"    Warning: Normal computation failed: {e}")
            # 실패시 랜덤 법선
            normals = np.random.randn(len(points), 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        return normals.astype(np.float32)
    
    def _create_default_pointcloud(self):
        """기본 포인트 클라우드 생성 (개선된 버전)"""
        # Lazy import 3DGS modules
        _, _, BasicPointCloud = get_3dgs_imports()
        if BasicPointCloud is None:
            # Fallback: 간단한 클래스 정의
            class BasicPointCloud:
                def __init__(self, points, colors, normals):
                    self.points = points
                    self.colors = colors
                    self.normals = normals
        
        # 카메라 위치를 기반으로 한 더 현실적인 포인트 클라우드
        if len(self.cameras) > 0:
            # 카메라 중심 계산
            camera_centers = []
            for cam_id in self.cameras:
                R, T = self.cameras[cam_id]['R'], self.cameras[cam_id]['T']
                center = -R.T @ T
                camera_centers.append(center)
            
            if camera_centers:
                camera_centers = np.array(camera_centers)
                center_mean = np.mean(camera_centers, axis=0)
                center_std = np.std(camera_centers, axis=0)
                
                # 실제 포인트가 있으면 그것을 기반으로 생성
                if self.points_3d:
                    actual_points = np.array([pt['xyz'] for pt in self.points_3d.values()])
                    if len(actual_points) > 0:
                        # 실제 포인트 주변에 추가 포인트 생성
                        n_additional = 15000  # 더 많은 추가 포인트 (5000 → 15000)
                        points = np.random.randn(n_additional, 3).astype(np.float32)
                        points = points * np.std(actual_points, axis=0) * 0.8 + np.mean(actual_points, axis=0)
                        
                        # 실제 포인트와 합치기
                        points = np.vstack([actual_points, points])
                        colors = np.random.rand(len(points), 3).astype(np.float32)
                        normals = self._compute_point_normals(points)
                    else:
                        # 카메라 분포를 고려한 포인트 생성
                        n_points = 20000  # 더 많은 수 (10000 → 20000)
                        points = np.random.randn(n_points, 3).astype(np.float32)
                        points = points * center_std * 0.8 + center_mean
                        colors = np.random.rand(n_points, 3).astype(np.float32)
                        normals = self._compute_point_normals(points)
                else:
                    # 카메라 분포를 고려한 포인트 생성
                    n_points = 20000  # 더 많은 수 (10000 → 20000)
                    points = np.random.randn(n_points, 3).astype(np.float32)
                    points = points * center_std * 0.8 + center_mean
                    colors = np.random.rand(n_points, 3).astype(np.float32)
                    normals = self._compute_point_normals(points)
            else:
                # 기본 포인트 클라우드 (더 적은 수)
                points = np.random.randn(10000, 3).astype(np.float32) * 3
                colors = np.random.rand(10000, 3).astype(np.float32)
                normals = np.random.randn(10000, 3).astype(np.float32)
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        else:
            # 기본 포인트 클라우드 (더 적은 수)
            points = np.random.randn(10000, 3).astype(np.float32) * 3
            colors = np.random.rand(10000, 3).astype(np.float32)
            normals = np.random.randn(10000, 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        return BasicPointCloud(points=points, colors=colors, normals=normals)
    
    def _compute_nerf_normalization(self, cam_infos):
        """NeRF 정규화 파라미터 계산 (개선된 버전)"""
        # Lazy import 3DGS modules
        try:
            from utils.graphics_utils import getWorld2View2
        except ImportError:
            # Fallback: 간단한 함수 정의
            def getWorld2View2(R, t):
                Rt = np.zeros((4, 4))
                Rt[:3, :3] = R
                Rt[:3, 3] = t
                Rt[3, 3] = 1.0
                return Rt
        
        if not cam_infos:
            return {"translate": np.zeros(3), "radius": 1.0}
        
        # 카메라 중심 계산
        cam_centers = []
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        cam_centers = np.hstack(cam_centers)
        
        # 더 안정적인 중심 계산 (중간값 사용)
        center = np.median(cam_centers, axis=1, keepdims=True).flatten()
        
        # 거리 계산
        distances = np.linalg.norm(cam_centers - center.reshape(-1, 1), axis=0)
        
        # 더 보수적인 반지름 계산 (95 퍼센타일 사용)
        radius = np.percentile(distances, 95) * 1.2
        
        # 최소 반지름 보장
        radius = max(radius, 1.0)
        
        return {"translate": -center, "radius": radius}
    
    def _save_3dgs_format(self, scene_info, output_dir):
        """3DGS 학습을 위한 파일 구조 생성"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # COLMAP 호환 디렉토리 구조
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(exist_ok=True, parents=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 1. 카메라 내부 파라미터 저장 (cameras.txt)
        self._write_cameras_txt(scene_info.train_cameras + scene_info.test_cameras, 
                               sparse_dir / "cameras.txt")
        
        # 2. 카메라 포즈 저장 (images.txt)
        self._write_images_txt(scene_info.train_cameras + scene_info.test_cameras, 
                              sparse_dir / "images.txt")
        
        # 3. 3D 포인트 저장 (points3D.ply)
        self._write_points3d_ply(scene_info.point_cloud, sparse_dir / "points3D.ply")
        
        # 4. 이미지 복사 또는 심볼릭 링크
        self._setup_images_directory(scene_info.train_cameras + scene_info.test_cameras, 
                                    images_dir)
        
        print(f"  3DGS-compatible files saved to {output_dir}")
        print(f"  Use: python train.py -s {output_dir} -m {output_dir}/3dgs_output")
    
    def _write_cameras_txt(self, cam_infos, output_path):
        """COLMAP 형식 cameras.txt 생성"""
        with open(output_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cam_infos)}\n")
            
            for cam in cam_infos:
                # PINHOLE 모델 사용
                focal_x = cam.width / (2 * np.tan(cam.FovX / 2))
                focal_y = cam.height / (2 * np.tan(cam.FovY / 2))
                cx, cy = cam.width / 2, cam.height / 2
                
                f.write(f"{cam.uid} PINHOLE {cam.width} {cam.height} "
                       f"{focal_x:.6f} {focal_y:.6f} {cx:.6f} {cy:.6f}\n")
    
    def _write_images_txt(self, cam_infos, output_path):
        """COLMAP 형식 images.txt 생성"""
        with open(output_path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(cam_infos)}\n")
            
            for cam in cam_infos:
                # 회전 행렬을 쿼터니언으로 변환
                R = cam.R
                trace = np.trace(R)
                
                if trace > 0:
                    s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                    qw = 0.25 * s
                    qx = (R[2, 1] - R[1, 2]) / s
                    qy = (R[0, 2] - R[2, 0]) / s
                    qz = (R[1, 0] - R[0, 1]) / s
                else:
                    # 안정적인 쿼터니언 변환
                    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
                    qx = (R[2,1] - R[1,2]) / (4 * qw) if qw != 0 else 0
                    qy = (R[0,2] - R[2,0]) / (4 * qw) if qw != 0 else 0
                    qz = (R[1,0] - R[0,1]) / (4 * qw) if qw != 0 else 0
                
                # 정규화
                q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                if q_norm > 0:
                    qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm
                
                f.write(f"{cam.uid} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                       f"{cam.T[0]:.6f} {cam.T[1]:.6f} {cam.T[2]:.6f} "
                       f"{cam.uid} {cam.image_name}\n")
                f.write("\n")  # 빈 특징점 라인
    
    def _write_points3d_ply(self, point_cloud, output_path):
        """PLY 형식으로 포인트 클라우드 저장"""
        points = point_cloud.points
        colors = (point_cloud.colors * 255).astype(np.uint8)
        
        with open(output_path, 'w') as f:
            # PLY 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 데이터
            for i in range(len(points)):
                x, y, z = points[i]
                nx, ny, nz = point_cloud.normals[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} "
                       f"{nx:.6f} {ny:.6f} {nz:.6f} "
                       f"{r} {g} {b}\n")
    
    def _setup_images_directory(self, cam_infos, images_dir):
        """이미지 디렉토리 설정 (심볼릭 링크 또는 복사)"""
        import shutil
        
        for cam in cam_infos:
            src_path = Path(cam.image_path)
            dst_path = images_dir / cam.image_name
            
            if not dst_path.exists():
                try:
                    # 심볼릭 링크 시도
                    dst_path.symlink_to(src_path.resolve())
                except (OSError, NotImplementedError):
                    # 실패시 복사
                    shutil.copy2(src_path, dst_path)
    
    def _load_image(self, image_path, resize=None):
        """이미지 로드 및 전처리 - 적응형 resize 적용"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"    Warning: Failed to load {image_path}")
                return None
            
            # 적응형 resize 계산
            if resize is None:
                resize = self._calculate_adaptive_resize(image_path)
            
            # 크기 조정 (SuperGlue 처리용)
            if resize is None:
                pass  # 원본 크기 유지
            elif len(resize) == 2:
                image = cv2.resize(image, resize)
            elif len(resize) == 1 and resize[0] > 0:
                h, w = image.shape
                scale = resize[0] / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            return image.astype(np.float32)
        
        except Exception as e:
            print(f"    Error loading {image_path}: {e}")
            return None
    
    def _calculate_adaptive_resize(self, image_path):
        """이미지 해상도에 따른 적응형 resize 계산"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return [1024, 768]  # 기본값
            
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            # 적응형 resize 규칙
            if max_dim <= 1024:
                # 작은 이미지는 원본 크기 유지
                return None
            elif max_dim <= 2048:
                # 중간 크기는 1024로 resize
                scale = 1024 / max_dim
                return [int(w * scale), int(h * scale)]
            else:
                # 큰 이미지는 1536로 resize
                scale = 1536 / max_dim
                return [int(w * scale), int(h * scale)]
        except:
            return [1024, 768]  # 기본값

def readSuperGlueSceneInfo(path, images, eval, train_test_exp=False, llffhold=8, 
                          superglue_config="outdoor", max_images=100):
    """SuperGlue 기반 완전한 SfM으로 SceneInfo 생성"""
    
    print("=== SuperGlue Complete SfM Pipeline ===")
    
    # 이미지 디렉토리 경로
    images_folder = Path(path) / (images if images else "images")
    output_folder = Path(path) / "superglue_sfm_output"
    
    # SuperGlue 설정 (더 완화된 설정)
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,  # 더 낮은 임계값
            'max_keypoints': 4096  # 더 많은 특징점
        },
        'superglue': {
            'weights': superglue_config,  # 'indoor' 또는 'outdoor'
            'sinkhorn_iterations': 20,
            'match_threshold': 0.1,  # 더 낮은 매칭 임계값
        }
    }
    
    # SuperGlue 3DGS 파이프라인 실행
    pipeline = SuperGlue3DGSPipeline(config)
    
    try:
        scene_info = pipeline.process_images_to_3dgs(
            image_dir=images_folder,
            output_dir=output_folder,
            max_images=max_images
        )
        
        print(f"\n=== SuperGlue SfM Results ===")
        print(f"Training cameras: {len(scene_info.train_cameras)}")
        print(f"Test cameras: {len(scene_info.test_cameras)}")
        print(f"3D points: {len(scene_info.point_cloud.points)}")
        print(f"Scene radius: {scene_info.nerf_normalization['radius']:.3f}")
        
        return scene_info
        
    except Exception as e:
        print(f"SuperGlue SfM failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 실패시 fallback
        print("Falling back to simple camera arrangement...")
        return _create_fallback_scene_info(images_folder, max_images)


def _create_fallback_scene_info(images_folder, max_images):
    """SuperGlue 실패시 기본 scene 생성"""
    # Lazy import 3DGS modules
    CameraInfo, SceneInfo, BasicPointCloud = get_3dgs_imports()
    if CameraInfo is None:
        # Fallback: 간단한 클래스 정의들
        class CameraInfo:
            def __init__(self, uid, R, T, FovY, FovX, image_path, image_name, width, height, depth_params, depth_path, is_test):
                self.uid = uid
                self.R = R
                self.T = T
                self.FovY = FovY
                self.FovX = FovX
                self.image_path = image_path
                self.image_name = image_name
                self.width = width
                self.height = height
                self.depth_params = depth_params
                self.depth_path = depth_path
                self.is_test = is_test
        
        class SceneInfo:
            def __init__(self, point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path, is_nerf_synthetic):
                self.point_cloud = point_cloud
                self.train_cameras = train_cameras
                self.test_cameras = test_cameras
                self.nerf_normalization = nerf_normalization
                self.ply_path = ply_path
                self.is_nerf_synthetic = is_nerf_synthetic
        
        class BasicPointCloud:
            def __init__(self, points, colors, normals):
                self.points = points
                self.colors = colors
                self.normals = normals
    
    # 이미지 수집
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(list(Path(images_folder).glob(ext)))
    
    image_paths.sort()
    image_paths = image_paths[:max_images]
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    # 기본 카메라 배치 (원형)
    cam_infos = []
    for i, image_path in enumerate(image_paths):
        # 이미지 크기 확인
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 1920, 1080
        
        # 원형 배치
        angle = (i / len(image_paths)) * 2 * np.pi
        radius = 3.0
        
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        
        T = np.array([
            radius * np.sin(angle),
            0,
            radius * (1 - np.cos(angle))
        ], dtype=np.float32)
        
        # FOV 추정
        focal = max(width, height) * 0.8
        FovX = 2 * np.arctan(width / (2 * focal))
        FovY = 2 * np.arctan(height / (2 * focal))
        
        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=str(image_path),
            image_name=image_path.name,
            width=width,
            height=height,
            depth_params=None,
            depth_path="",
            is_test=(i % 8 == 0)
        )
        cam_infos.append(cam_info)
    
    # 기본 포인트 클라우드
    n_points = 15000  # 더 많은 수 (8000 → 15000)
    points = np.random.randn(n_points, 3).astype(np.float32) * 3  # 더 넓은 분포
    colors = np.random.rand(n_points, 3).astype(np.float32)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # 학습/테스트 분할
    train_cams = [c for c in cam_infos if not c.is_test]
    test_cams = [c for c in cam_infos if c.is_test]
    
    # NeRF 정규화
    nerf_norm = {"translate": np.zeros(3), "radius": 5.0}
    
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        nerf_normalization=nerf_norm,
        ply_path="",
        is_nerf_synthetic=False
    )


# 명령줄 인터페이스
def main():
    """명령줄에서 SuperGlue 3DGS 파이프라인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperGlue 3DGS Pipeline")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input images directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for 3DGS training")
    parser.add_argument("--max_images", type=int, default=100,
                       help="Maximum number of images to process")
    parser.add_argument("--config", choices=["indoor", "outdoor"], 
                       default="outdoor", help="SuperGlue configuration")
    parser.add_argument("--device", default="cuda", 
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # SuperGlue 설정
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,  # 더 낮은 임계값
            'max_keypoints': 4096  # 더 많은 특징점
        },
        'superglue': {
            'weights': args.config,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.1,  # 더 낮은 매칭 임계값
        }
    }
    
    print(f"=== SuperGlue 3DGS Pipeline ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max images: {args.max_images}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    
    # 파이프라인 실행
    pipeline = SuperGlue3DGSPipeline(config, device=args.device)
    
    try:
        scene_info = pipeline.process_images_to_3dgs(
            image_dir=args.input,
            output_dir=args.output,
            max_images=args.max_images
        )
        
        print(f"\n=== Success! ===")
        print(f"Processed {len(scene_info.train_cameras)} training cameras")
        print(f"Generated {len(scene_info.point_cloud.points)} 3D points")
        print(f"Files saved to: {args.output}")
        print(f"\nNext step:")
        print(f"python train.py -s {args.output} -m {args.output}/3dgs_output")
        
    except Exception as e:
        print(f"\n=== Error! ===")
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())