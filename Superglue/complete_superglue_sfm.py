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

# SuperGlue 관련 imports
from models.matching import Matching
from models.utils import frame2tensor

# 3DGS 관련 imports
from scene.dataset_readers import CameraInfo, SceneInfo
from utils.graphics_utils import BasicPointCloud


class SuperGlue3DGSPipeline:
    """SuperGlue 기반 완전한 3DGS SfM 파이프라인"""
    
    def __init__(self, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # SuperGlue 설정 (고해상도 이미지 처리에 최적화)
        if config is None:
            config = {
                'superpoint': {
                    'nms_radius': 3,  # 증가
                    'keypoint_threshold': 0.001,
                    'max_keypoints': 4096  # 더 많은 특징점
                },
                'superglue': {
                    'weights': 'outdoor',  # indoor 가중치 사용
                    'sinkhorn_iterations': 100,  # 증가
                    'match_threshold': 0.1,  # 증가
                }
            }
        
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        # SfM 데이터 저장소
        self.cameras = {}  # camera_id -> {'R': R, 'T': T, 'K': K, 'image_path': path}
        self.points_3d = {}  # point_id -> {'xyz': xyz, 'color': rgb, 'observations': [(cam_id, kpt_idx)]}
        self.image_features = {}  # image_id -> SuperPoint features
        self.matches = {}  # (img_i, img_j) -> SuperGlue matches
        
        print(f'SuperGlue 3DGS Pipeline initialized on {self.device}')
    
    def process_images_to_3dgs(self, image_dir, output_dir, max_images=120):
        """이미지 디렉토리에서 3DGS 학습 가능한 상태까지 완전 처리"""
        
        print(f"\n=== SuperGlue + 3DGS Pipeline: Processing up to {max_images} images ===")
        
        # 1. 이미지 수집 및 정렬
        image_paths = self._collect_images(image_dir, max_images)
        print(f"Found {len(image_paths)} images")
        
        # 2. SuperPoint 특징점 추출
        print("\n[1/6] Extracting SuperPoint features...")
        self._extract_all_features(image_paths)
        
        # 3. SuperGlue 매칭 (지능적 페어링)
        print("\n[2/6] SuperGlue feature matching...")
        self._intelligent_matching(max_pairs=min(len(image_paths) * 10, 1000))
        
        # 4. 초기 카메라 포즈 추정
        print("\n[3/6] Camera pose estimation...")
        self._estimate_camera_poses()
        
        # 5. 3D 포인트 삼각측량
        print("\n[4/6] 3D point triangulation...")
        self._triangulate_all_points()
        
        # 6. Bundle Adjustment
        print("\n[5/6] Bundle adjustment optimization...")
        self._bundle_adjustment()
        
        # 7. 3DGS 형식으로 변환
        print("\n[6/6] Converting to 3DGS format...")
        scene_info = self._create_3dgs_scene_info(image_paths)
        
        # 8. 결과 저장
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
        """모든 이미지에서 SuperPoint 특징점 추출"""
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
            
            # 결과 저장
            self.image_features[i] = {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(),
                'scores': pred['scores'][0].cpu().numpy(),
                'image_path': str(image_path),
                'image_size': image.shape[:2]  # (H, W)
            }
            
            print(f"{i+1}th image's keypoints extracted: {self.image_features[i]['keypoints']}")
            
        print(f"  Extracted features from {len(self.image_features)} images")
    
    def _intelligent_matching(self, max_pairs=1000):
        """지능적 이미지 매칭 (순차적 + 선택적)"""
        n_images = len(self.image_features)
        
        # 1. 순차적 매칭 (인접 이미지)
        for i in range(n_images - 1):
            matches = self._match_pair_superglue(i, i+1)
            if len(matches) > 20:
                self.matches[(i, i+1)] = matches
        
        # 2. 건너뛰기 매칭 (2, 3, 5, 10 간격)
        for gap in [2, 3, 5, 10]:
            for i in range(n_images - gap):
                j = i + gap
                if len(self.matches) >= max_pairs:
                    break
                    
                matches = self._match_pair_superglue(i, j)
                if len(matches) > 30:  # 더 높은 임계값
                    self.matches[(i, j)] = matches
        
        print(f"  Created {len(self.matches)} image pairs with good matches")
    
    def _match_pair_superglue(self, cam_i, cam_j):
        """수정된 SuperGlue 매칭"""
    
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
                'image0': torch.zeros((1, 1, 480, 640)).to(self.device),
                'image1': torch.zeros((1, 1, 480, 640)).to(self.device),
            }
        
        # SuperGlue 매칭 수행
            with torch.no_grad():
                result = self.matching.superglue(data)
        
            # **핵심 수정: indices0/indices1 사용 (matches0/matches1 아님)**
            indices0 = result['indices0'][0].cpu().numpy()
            indices1 = result['indices1'][0].cpu().numpy()
            mscores0 = result['matching_scores0'][0].cpu().numpy()
        
        # 유효한 매칭 추출
            valid_matches = []
            threshold = self.matching.superglue.config['match_threshold']
        
            for i, j in enumerate(indices0):
                if j >= 0 and mscores0[i] > threshold:
                # 상호 매칭 확인
                    if j < len(indices1) and indices1[j] == i:
                        valid_matches.append((i, j, mscores0[i]))
        
            print(f"  Pair {cam_i}-{cam_j}: {len(valid_matches)} mutual matches")
            return valid_matches
        
        except Exception as e:
            print(f"SuperGlue matching failed for pair {cam_i}-{cam_j}: {e}")
            return []
        
        
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
        
            # *** 디버깅: SuperPoint 출력 확인 ***
            if i == 0:  # 첫 번째 이미지에서만 출력
                print(f"  SuperPoint output keys: {list(pred.keys())}")
                for k, v in pred.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")
        
        # 결과 저장 - 모든 필요한 키 포함
            self.image_features[i] = {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(), 
                'scores': pred['scores'][0].cpu().numpy(),
                'image_path': str(image_path),
                'image_size': image.shape[:2]  # (H, W)
            }
        
        print(f"  Extracted features from {len(self.image_features)} images")


# 추가: SuperGlue 매칭 디버깅 함수
    def debug_superglue_matching(self, i, j):
        """SuperGlue 매칭 과정 디버깅"""
    
        print(f"\n=== Debugging SuperGlue matching for pair {i}-{j} ===")
    
    # 특징점 확인
        feat0 = self.image_features[i] 
        feat1 = self.image_features[j]
    
        print(f"Image {i}: {feat0['keypoints'].shape[0]} keypoints")
        print(f"Image {j}: {feat1['keypoints'].shape[0]} keypoints")
    
    # 이미지 로드
        img0 = self._load_image(feat0['image_path'])
        img1 = self._load_image(feat1['image_path'])
    
    # 텐서 변환
        inp0 = frame2tensor(img0, self.device)
        inp1 = frame2tensor(img1, self.device)
    
    # SuperPoint 재추출
        with torch.no_grad():
            pred0 = self.matching.superpoint({'image': inp0})
            pred1 = self.matching.superpoint({'image': inp1})
    
        print(f"SuperPoint pred0 keys: {list(pred0.keys())}")
        print(f"SuperPoint pred1 keys: {list(pred1.keys())}")
    
        # 데이터 준비
        data = {
            'image0': inp0,
            'image1': inp1,
            'keypoints0': pred0['keypoints'],
            'keypoints1': pred1['keypoints'],
            'descriptors0': pred0['descriptors'], 
            'descriptors1': pred1['descriptors'],
            'scores0': pred0['scores'],
            'scores1': pred1['scores']
        }
    
        print(f"Input data keys: {list(data.keys())}")
    
    # SuperGlue 매칭
        try:
            with torch.no_grad():
                result = self.matching(data)
            print(f"SuperGlue result keys: {list(result.keys())}")
            print(f"Matches found: {(result['matches0'][0] > -1).sum().item()}")
            return True
        
        except Exception as e:
            print(f"SuperGlue matching failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def _estimate_camera_poses(self):
        """순차적 카메라 포즈 추정"""
        
        # 첫 번째 카메라를 원점으로 설정
        self.cameras[0] = {
            'R': np.eye(3, dtype=np.float32),
            'T': np.zeros(3, dtype=np.float32),
            'K': self._estimate_intrinsics(0)
        }
        
        print(f"  Camera 0: Origin (reference)")
        
        # 순차적으로 다른 카메라들의 포즈 추정
        for cam_id in range(1, len(self.image_features)):
            success = False
            
            # 이미 등록된 카메라와의 매칭을 이용해 포즈 추정
            for ref_cam in range(cam_id):
                if ref_cam not in self.cameras:
                    continue
                
                # 매칭 데이터 찾기
                pair_key = (ref_cam, cam_id) if ref_cam < cam_id else (cam_id, ref_cam)
                if pair_key not in self.matches:
                    continue
                
                # Essential Matrix 기반 포즈 추정
                R_rel, T_rel = self._estimate_relative_pose(ref_cam, cam_id, pair_key)
                
                if R_rel is not None and T_rel is not None:
                    # 월드 좌표계에서의 절대 포즈 계산
                    R_ref, T_ref = self.cameras[ref_cam]['R'], self.cameras[ref_cam]['T']
                    
                    # 상대 포즈를 절대 포즈로 변환
                    R_world = R_rel @ R_ref
                    T_world = R_rel @ T_ref + T_rel
                    
                    self.cameras[cam_id] = {
                        'R': R_world.astype(np.float32),
                        'T': T_world.astype(np.float32),
                        'K': self._estimate_intrinsics(cam_id)
                    }
                    
                    print(f"  Camera {cam_id}: Estimated from camera {ref_cam}")
                    success = True
                    break
            
            if not success:
                print(f"  Camera {cam_id}: Failed to estimate, using default pose")
                # 실패시 원형 배치로 기본 포즈 설정
                angle = cam_id * (2 * np.pi / len(self.image_features))
                radius = 3.0
                
                self.cameras[cam_id] = {
                    'R': np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32),
                    'T': np.array([radius * np.sin(angle), 0, radius * (1 - np.cos(angle))], dtype=np.float32),
                    'K': self._estimate_intrinsics(cam_id)
                }
    
    def _estimate_relative_pose(self, cam_i, cam_j, pair_key):
        """두 카메라 간 상대 포즈 추정"""
        matches = self.matches[pair_key]
        
        if len(matches) < 8:
            return None, None
        
        # 매칭점들 추출
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        pts_i = np.array([kpts_i[idx_i] for idx_i, _, conf in matches if conf > 0.4])
        pts_j = np.array([kpts_j[idx_j] for _, idx_j, conf in matches if conf > 0.4])
        
        if len(pts_i) < 8:
            return None, None
        
        # 카메라 내부 파라미터
        K_i = self.cameras.get(cam_i, {}).get('K', self._estimate_intrinsics(cam_i))
        K_j = self._estimate_intrinsics(cam_j)
        
        # Essential Matrix 추정
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, K_i,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
            maxIters=1000
        )
        
        if E is None:
            return None, None
        
        # 포즈 복원
        _, R, T, mask = cv2.recoverPose(E, pts_i, pts_j, K_i, mask=mask)
        
        return R, T.flatten()
    
    def _estimate_intrinsics(self, cam_id):
        """카메라 내부 파라미터 추정"""
        # 이미지 크기 기반 focal length 추정
        h, w = self.image_features[cam_id]['image_size']
        
        # 일반적인 가정: focal length ≈ max(w,h) * 0.8~1.2
        focal = max(w, h) * 1.0
        
        K = np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _triangulate_all_points(self):
        """모든 매칭 쌍에 대해 3D 포인트 삼각측량"""
        point_id = 0
        
        for (cam_i, cam_j), matches in self.matches.items():
            if cam_i not in self.cameras or cam_j not in self.cameras:
                continue
            
            # 투영 행렬 생성
            P_i = self._get_projection_matrix(cam_i)
            P_j = self._get_projection_matrix(cam_j)
            
            kpts_i = self.image_features[cam_i]['keypoints']
            kpts_j = self.image_features[cam_j]['keypoints']
            
            for idx_i, idx_j, conf in matches:
                if conf < 0.5:  # 높은 신뢰도만 사용
                    continue
                
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
                    
                    # 유효성 검사
                    if self._is_point_valid(point_3d, cam_i, cam_j):
                        # 색상 추정 (이미지에서 샘플링)
                        color = self._estimate_point_color(point_3d, cam_i, idx_i)
                        
                        self.points_3d[point_id] = {
                            'xyz': point_3d.astype(np.float32),
                            'color': color,
                            'observations': [(cam_i, idx_i), (cam_j, idx_j)]
                        }
                        
                        point_id += 1
        
        print(f"  Triangulated {len(self.points_3d)} 3D points")
    
    def _get_projection_matrix(self, cam_id):
        """카메라 투영 행렬 생성"""
        cam = self.cameras[cam_id]
        K, R, T = cam['K'], cam['R'], cam['T']
        
        # P = K[R|t] (t = -R^T * T_world)
        RT = np.hstack([R, T.reshape(-1, 1)])
        P = K @ RT
        
        return P
    
    def _is_point_valid(self, point_3d, cam_i, cam_j):
        """3D 포인트 유효성 검사"""
        # 두 카메라 모두에서 앞쪽에 있는지 확인
        for cam_id in [cam_i, cam_j]:
            cam = self.cameras[cam_id]
            R, T = cam['R'], cam['T']
            
            # 카메라 좌표계로 변환
            point_cam = R @ (point_3d - T)
            
            if point_cam[2] <= 0.1:  # 너무 가까우면 제외
                return False
            
            # 재투영 오차 확인
            K = cam['K']
            point_2d_proj = K @ point_cam
            point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
            
            # 이미지 경계 확인
            h, w = self.image_features[cam_id]['image_size']
            if (point_2d_proj[0] < 0 or point_2d_proj[0] >= w or
                point_2d_proj[1] < 0 or point_2d_proj[1] >= h):
                return False
        
        return True
    
    def _estimate_point_color(self, point_3d, cam_id, kpt_idx):
        """3D 포인트의 색상 추정"""
        # 실제 구현에서는 이미지에서 색상을 샘플링
        # 여기서는 간단히 랜덤 색상 사용
        return np.random.rand(3).astype(np.float32)
    
    def _bundle_adjustment(self, max_iterations=100):
        """Bundle Adjustment 최적화"""
        
        n_cameras = len(self.cameras)
        n_points = len(self.points_3d)
        
        if n_cameras < 2 or n_points < 10:
            print("  Insufficient data for bundle adjustment")
            return
        
        print(f"  Optimizing {n_cameras} cameras and {n_points} points...")
        
        total_observations = sum(len(point['observations']) for point in self.points_3d.values())
    
        # 변수 수 계산
        n_variables = 6 * (n_cameras - 1) + 3 * n_points
        n_residuals = total_observations * 2  
    
        if n_residuals <= n_variables:
            print(f"  Insufficient observations: {n_residuals} residuals < {n_variables} variables")
            return
        
        # 파라미터 벡터 구성
        camera_params = []
        for cam_id in sorted(self.cameras.keys())[1:]:  # 첫 번째 카메라 고정
            cam = self.cameras[cam_id]
            rvec, _ = cv2.Rodrigues(cam['R'])
            camera_params.extend(rvec.flatten())
            camera_params.extend(cam['T'])
        
        point_params = []
        for point_id in sorted(self.points_3d.keys()):
            point_params.extend(self.points_3d[point_id]['xyz'])
        
        if len(camera_params) == 0 or len(point_params) == 0:
            print("  No parameters to optimize")
            return
        
        initial_params = np.array(camera_params + point_params)
        
        # 최적화 실행
        try:
            result = least_squares(
                self._bundle_adjustment_residual,
                initial_params,
                args=(n_cameras, n_points),
                method='lm',
                max_nfev=max_iterations
            )
            
            # 결과 적용
            self._update_from_bundle_adjustment(result.x, n_cameras, n_points)
            print(f"  Bundle adjustment completed. Final cost: {result.cost:.6f}")
            
        except Exception as e:
            print(f"  Bundle adjustment failed: {e}")
    
    def _bundle_adjustment_residual(self, params, n_cameras, n_points):
        """Bundle Adjustment 잔차 함수"""
        residuals = []
        
        # 파라미터 분리
        camera_params = params[:6*(n_cameras-1)]
        point_params = params[6*(n_cameras-1):]
        
        # 임시 카메라 파라미터 업데이트
        temp_cameras = self.cameras.copy()
        for i, cam_id in enumerate(sorted(self.cameras.keys())[1:]):
            start_idx = i * 6
            rvec = camera_params[start_idx:start_idx+3]
            tvec = camera_params[start_idx+3:start_idx+6]
            
            R, _ = cv2.Rodrigues(rvec)
            temp_cameras[cam_id]['R'] = R
            temp_cameras[cam_id]['T'] = tvec
        
        # 재투영 오차 계산
        point_idx = 0
        for point_id in sorted(self.points_3d.keys()):
            if point_idx >= len(point_params) // 3:
                break
            
            point_3d = point_params[point_idx*3:(point_idx+1)*3]
            observations = self.points_3d[point_id]['observations']
            
            for cam_id, kpt_idx in observations:
                # 재투영
                projected = self._project_point(point_3d, temp_cameras[cam_id])
                
                # 실제 관측값
                observed = self.image_features[cam_id]['keypoints'][kpt_idx]
                
                # 잔차
                residuals.extend(projected - observed)
            
            point_idx += 1
        
        return np.array(residuals)
    
    def _project_point(self, point_3d, camera):
        """3D 포인트를 2D로 재투영"""
        R, T, K = camera['R'], camera['T'], camera['K']
        
        # 카메라 좌표계로 변환
        point_cam = R @ (point_3d - T)
        
        if point_cam[2] > 0:
            point_2d = K @ point_cam
            return point_2d[:2] / point_2d[2]
        else:
            return np.array([0.0, 0.0])
    
    def _update_from_bundle_adjustment(self, params, n_cameras, n_points):
        """Bundle Adjustment 결과 적용"""
        camera_params = params[:6*(n_cameras-1)]
        point_params = params[6*(n_cameras-1):]
        
        # 카메라 업데이트
        for i, cam_id in enumerate(sorted(self.cameras.keys())[1:]):
            start_idx = i * 6
            rvec = camera_params[start_idx:start_idx+3]
            tvec = camera_params[start_idx+3:start_idx+6]
            
            R, _ = cv2.Rodrigues(rvec)
            self.cameras[cam_id]['R'] = R
            self.cameras[cam_id]['T'] = tvec
        
        # 포인트 업데이트
        point_idx = 0
        for point_id in sorted(self.points_3d.keys()):
            if point_idx >= len(point_params) // 3:
                break
            self.points_3d[point_id]['xyz'] = point_params[point_idx*3:(point_idx+1)*3]
            point_idx += 1
    
    def _create_3dgs_scene_info(self, image_paths):
        """3DGS용 SceneInfo 생성"""
        
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
                is_test=(cam_id % 8 == 0)  # 8개마다 1개씩 테스트용
            )
            cam_infos.append(cam_info)
        
        # 포인트 클라우드 생성
        if self.points_3d:
            points = np.array([pt['xyz'] for pt in self.points_3d.values()])
            colors = np.array([pt['color'] for pt in self.points_3d.values()])
            
            # 법선 벡터 (임시)
            normals = np.random.randn(len(points), 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        else:
            # 기본 포인트 클라우드
            pcd = self._create_default_pointcloud()
        
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
    
    def _create_default_pointcloud(self):
        """기본 포인트 클라우드 생성"""
        points = np.random.randn(5000, 3).astype(np.float32) * 2
        colors = np.random.rand(5000, 3).astype(np.float32)
        normals = np.random.randn(5000, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        return BasicPointCloud(points=points, colors=colors, normals=normals)
    
    def _compute_nerf_normalization(self, cam_infos):
        """NeRF 정규화 파라미터 계산"""
        from utils.graphics_utils import getWorld2View2
        
        cam_centers = []
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        if cam_centers:
            cam_centers = np.hstack(cam_centers)
            center = np.mean(cam_centers, axis=1, keepdims=True).flatten()
            distances = np.linalg.norm(cam_centers - center.reshape(-1, 1), axis=0)
            radius = np.max(distances) * 1.1
        else:
            center = np.zeros(3)
            radius = 1.0
        
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
    
    # SuperGlue 설정
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 2048
        },
        'superglue': {
            'weights': superglue_config,  # 'indoor' 또는 'outdoor'
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
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
    from scene.dataset_readers import CameraInfo, SceneInfo
    from utils.graphics_utils import BasicPointCloud
    
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
    n_points = 10000
    points = np.random.randn(n_points, 3).astype(np.float32) * 2
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
            'keypoint_threshold': 0.005,
            'max_keypoints': 2048
        },
        'superglue': {
            'weights': args.config,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
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