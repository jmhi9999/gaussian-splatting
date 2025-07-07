# complete_superglue_sfm.py
# SuperGlue 기반 완전한 SfM 파이프라인 구현

import numpy as np
import cv2
import torch
from pathlib import Path
import json
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from collections import defaultdict

from models.matching import Matching
from models.utils import frame2tensor
from scene.dataset_readers import CameraInfo, SceneInfo
from utils.graphics_utils import BasicPointCloud

class SuperGlueSfMPipeline:
    """SuperGlue 기반 완전한 SfM 파이프라인"""
    
    def __init__(self, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # SuperGlue 설정
        if config is None:
            config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 2048  # 더 많은 특징점 사용
                },
                'superglue': {
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }
            }
        
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        # SfM 상태
        self.cameras = {}  # camera_id -> {'R': R, 'T': T, 'K': K}
        self.points_3d = {}  # point_id -> {'xyz': xyz, 'observations': [(cam_id, kpt_idx), ...]}
        self.observations = {}  # (cam_id, kpt_idx) -> point_id
        self.image_features = {}  # image_id -> {'keypoints': kpts, 'descriptors': desc}
        self.matches = {}  # (img_i, img_j) -> [(kpt_i, kpt_j, confidence), ...]
        
        print(f'SuperGlue SfM Pipeline initialized on {self.device}')
    
    def run_complete_sfm(self, image_paths, output_path=None, max_images=100):
        """완전한 SfM 파이프라인 실행"""
        
        print(f"=== SuperGlue SfM Pipeline: Processing {len(image_paths)} images ===")
        
        # 1. 모든 이미지에서 특징점 추출
        print("1. Extracting features from all images...")
        self.extract_all_features(image_paths[:max_images])
        
        # 2. 이미지 간 매칭 수행 (전체 조합)
        print("2. Matching features between image pairs...")
        self.match_all_pairs()
        
        # 3. 초기 카메라 포즈 추정 (Sequential approach)
        print("3. Estimating camera poses...")
        self.estimate_initial_poses()
        
        # 4. 3D 포인트 Triangulation
        print("4. Triangulating 3D points...")
        self.triangulate_points()
        
        # 5. Bundle Adjustment
        print("5. Bundle adjustment...")
        self.bundle_adjustment()
        
        # 6. 결과 정리 및 저장
        print("6. Finalizing results...")
        scene_info = self.create_scene_info(image_paths[:max_images])
        
        if output_path:
            self.save_results(output_path)
        
        return scene_info
    
    def extract_all_features(self, image_paths):
        """모든 이미지에서 특징점 추출"""
        for i, image_path in enumerate(image_paths):
            print(f"  Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            # 이미지 로드
            image = self._load_image(image_path)
            if image is None:
                continue
            
            # SuperPoint로 특징점 추출
            inp = frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            
            # 결과 저장
            kpts = pred['keypoints'][0].cpu().numpy()
            desc = pred['descriptors'][0].cpu().numpy()
            scores = pred['scores'][0].cpu().numpy()
            
            self.image_features[i] = {
                'keypoints': kpts,
                'descriptors': desc,
                'scores': scores,
                'image_path': image_path,
                'image': image
            }
    
    def match_all_pairs(self, max_pairs=None):
        """모든 이미지 쌍에 대해 매칭 수행"""
        n_images = len(self.image_features)
        total_pairs = n_images * (n_images - 1) // 2
        
        if max_pairs and total_pairs > max_pairs:
            print(f"  Too many pairs ({total_pairs}), using sequential matching strategy")
            self._sequential_matching()
        else:
            self._exhaustive_matching()
    
    def _exhaustive_matching(self):
        """전체 이미지 쌍 매칭"""
        n_images = len(self.image_features)
        
        for i in range(n_images):
            for j in range(i+1, n_images):
                matches = self._match_pair(i, j)
                if len(matches) > 10:  # 최소 매칭 수 체크
                    self.matches[(i, j)] = matches
    
    def _sequential_matching(self):
        """순차적 매칭 (인접 이미지 우선)"""
        n_images = len(self.image_features)
        
        # 순차적 매칭
        for i in range(n_images - 1):
            matches = self._match_pair(i, i+1)
            if len(matches) > 10:
                self.matches[(i, i+1)] = matches
        
        # 추가 매칭 (일정 간격)
        for gap in [2, 3, 5, 10]:
            for i in range(n_images - gap):
                j = i + gap
                matches = self._match_pair(i, j)
                if len(matches) > 20:  # 더 높은 임계값
                    self.matches[(i, j)] = matches
    
    def _match_pair(self, i, j):
        """두 이미지 간 매칭"""
        feat_i = self.image_features[i]
        feat_j = self.image_features[j]
        
        # SuperGlue 매칭
        data = {
            'keypoints0': torch.from_numpy(feat_i['keypoints']).unsqueeze(0).to(self.device),
            'keypoints1': torch.from_numpy(feat_j['keypoints']).unsqueeze(0).to(self.device),
            'descriptors0': torch.from_numpy(feat_i['descriptors']).unsqueeze(0).to(self.device),
            'descriptors1': torch.from_numpy(feat_j['descriptors']).unsqueeze(0).to(self.device),
            'scores0': torch.from_numpy(feat_i['scores']).unsqueeze(0).to(self.device),
            'scores1': torch.from_numpy(feat_j['scores']).unsqueeze(0).to(self.device),
            'image0': torch.zeros(1, 1, 480, 640).to(self.device),  # 더미 이미지
            'image1': torch.zeros(1, 1, 480, 640).to(self.device),
        }
        
        with torch.no_grad():
            pred = self.matching.superglue(data)
        
        # 유효한 매칭 추출
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        
        valid = matches > -1
        matches_list = []
        
        for idx in np.where(valid)[0]:
            match_idx = matches[idx]
            conf = confidence[idx]
            matches_list.append((idx, match_idx, conf))
        
        return matches_list
    
    def estimate_initial_poses(self):
        """초기 카메라 포즈 추정"""
        
        # 첫 번째 카메라를 기준으로 설정
        self.cameras[0] = {
            'R': np.eye(3),
            'T': np.zeros(3),
            'K': self._estimate_intrinsics(0)  # 내부 파라미터 추정
        }
        
        # 순차적으로 다른 카메라들의 포즈 추정
        for cam_id in range(1, len(self.image_features)):
            success = False
            
            # 이미 추정된 카메라들과의 매칭을 찾아서 포즈 추정
            for ref_cam in range(cam_id):
                if ref_cam not in self.cameras:
                    continue
                
                pair_key = (ref_cam, cam_id) if ref_cam < cam_id else (cam_id, ref_cam)
                if pair_key not in self.matches:
                    continue
                
                # Essential Matrix로부터 포즈 추정
                R, T = self._estimate_pose_from_essential(ref_cam, cam_id, pair_key)
                
                if R is not None and T is not None:
                    # 참조 카메라의 포즈와 결합
                    R_ref, T_ref = self.cameras[ref_cam]['R'], self.cameras[ref_cam]['T']
                    
                    # 월드 좌표계에서의 포즈 계산
                    R_world = R @ R_ref
                    T_world = R @ T_ref + T
                    
                    self.cameras[cam_id] = {
                        'R': R_world,
                        'T': T_world,
                        'K': self._estimate_intrinsics(cam_id)
                    }
                    success = True
                    break
            
            if not success:
                print(f"  Warning: Could not estimate pose for camera {cam_id}")
                # 기본 포즈 설정
                angle = cam_id * 0.1
                self.cameras[cam_id] = {
                    'R': np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]]),
                    'T': np.array([np.sin(angle), 0, 0]) * 2.0,
                    'K': self._estimate_intrinsics(cam_id)
                }
    
    def _estimate_pose_from_essential(self, cam_i, cam_j, pair_key):
        """Essential Matrix로부터 포즈 추정"""
        matches = self.matches[pair_key]
        
        if len(matches) < 8:
            return None, None
        
        # 매칭점 추출
        kpts_i = self.image_features[cam_i]['keypoints']
        kpts_j = self.image_features[cam_j]['keypoints']
        
        pts_i = []
        pts_j = []
        
        for idx_i, idx_j, conf in matches:
            if conf > 0.3:  # 신뢰도 임계값
                pts_i.append(kpts_i[idx_i])
                pts_j.append(kpts_j[idx_j])
        
        if len(pts_i) < 8:
            return None, None
        
        pts_i = np.array(pts_i, dtype=np.float32)
        pts_j = np.array(pts_j, dtype=np.float32)
        
        # 내부 파라미터
        K_i = self.cameras.get(cam_i, {}).get('K', self._estimate_intrinsics(cam_i))
        K_j = self._estimate_intrinsics(cam_j)
        
        # Essential Matrix 추정
        E, mask = cv2.findEssentialMat(pts_i, pts_j, K_i, 
                                       method=cv2.RANSAC, 
                                       prob=0.999, threshold=1.0)
        
        if E is None:
            return None, None
        
        # 포즈 복원
        _, R, T, mask = cv2.recoverPose(E, pts_i, pts_j, K_i)
        
        return R, T.flatten()
    
    def _estimate_intrinsics(self, cam_id):
        """카메라 내부 파라미터 추정 (간단한 버전)"""
        # 실제로는 calibration이나 더 정교한 방법 필요
        image_path = self.image_features[cam_id]['image_path']
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        # 일반적인 가정: focal length = max(w,h) * 0.8
        focal = max(w, h) * 0.8
        
        K = np.array([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=np.float32)
        
        return K
    
    def triangulate_points(self):
        """3D 포인트 Triangulation"""
        point_id = 0
        
        # 각 매칭 쌍에 대해 triangulation 수행
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
                
                # 이미 triangulate된 점인지 확인
                if (cam_i, idx_i) in self.observations or (cam_j, idx_j) in self.observations:
                    continue
                
                # Triangulation
                pt_i = kpts_i[idx_i]
                pt_j = kpts_j[idx_j]
                
                point_4d = cv2.triangulatePoints(P_i, P_j, 
                                                pt_i.reshape(2, 1), 
                                                pt_j.reshape(2, 1))
                
                if point_4d[3] != 0:
                    point_3d = point_4d[:3] / point_4d[3]
                    
                    # 유효성 검사 (카메라 앞쪽에 있는지)
                    if self._is_point_valid(point_3d.flatten(), cam_i, cam_j):
                        self.points_3d[point_id] = {
                            'xyz': point_3d.flatten(),
                            'observations': [(cam_i, idx_i), (cam_j, idx_j)]
                        }
                        
                        self.observations[(cam_i, idx_i)] = point_id
                        self.observations[(cam_j, idx_j)] = point_id
                        
                        point_id += 1
    
    def _get_projection_matrix(self, cam_id):
        """카메라의 투영 행렬 생성"""
        cam = self.cameras[cam_id]
        K = cam['K']
        R = cam['R']
        T = cam['T']
        
        # P = K[R|T]
        RT = np.hstack([R, T.reshape(-1, 1)])
        P = K @ RT
        
        return P
    
    def _is_point_valid(self, point_3d, cam_i, cam_j):
        """3D 포인트가 유효한지 검사"""
        # 두 카메라 모두에서 앞쪽에 있는지 확인
        for cam_id in [cam_i, cam_j]:
            cam = self.cameras[cam_id]
            R, T = cam['R'], cam['T']
            
            # 카메라 좌표계로 변환
            point_cam = R @ (point_3d - T)
            
            if point_cam[2] <= 0:  # Z <= 0이면 카메라 뒤쪽
                return False
        
        return True
    
    def bundle_adjustment(self, max_iterations=50):
        """Bundle Adjustment로 최적화"""
        
        # 파라미터 벡터 구성 (cameras + points)
        n_cameras = len(self.cameras)
        n_points = len(self.points_3d)
        
        if n_cameras < 2 or n_points < 10:
            print("  Not enough data for bundle adjustment")
            return
        
        # 초기 파라미터 (첫 번째 카메라는 고정)
        camera_params = []
        for cam_id in sorted(self.cameras.keys())[1:]:  # 첫 번째 카메라 제외
            cam = self.cameras[cam_id]
            # 회전(로드리게스) + 평행이동
            rvec, _ = cv2.Rodrigues(cam['R'])
            camera_params.extend(rvec.flatten())
            camera_params.extend(cam['T'])
        
        point_params = []
        for point_id in sorted(self.points_3d.keys()):
            point_params.extend(self.points_3d[point_id]['xyz'])
        
        initial_params = np.array(camera_params + point_params)
        
        # 최적화 수행
        print(f"  Optimizing {len(camera_params)//6} cameras and {len(point_params)//3} points...")
        
        result = least_squares(
            self._bundle_adjustment_residual,
            initial_params,
            args=(n_cameras, n_points),
            method='lm',
            max_nfev=max_iterations * len(initial_params)
        )
        
        # 결과 업데이트
        self._update_from_bundle_adjustment(result.x, n_cameras, n_points)
        
        print(f"  Bundle adjustment completed. Final cost: {result.cost:.6f}")
    
    def _bundle_adjustment_residual(self, params, n_cameras, n_points):
        """Bundle Adjustment 잔차 함수"""
        residuals = []
        
        # 파라미터 분리
        camera_params = params[:6*(n_cameras-1)]  # 첫 번째 카메라 제외
        point_params = params[6*(n_cameras-1):]
        
        # 카메라 파라미터 업데이트 (임시)
        temp_cameras = self.cameras.copy()
        for i, cam_id in enumerate(sorted(self.cameras.keys())[1:]):
            start_idx = i * 6
            rvec = camera_params[start_idx:start_idx+3]
            tvec = camera_params[start_idx+3:start_idx+6]
            
            R, _ = cv2.Rodrigues(rvec)
            temp_cameras[cam_id]['R'] = R
            temp_cameras[cam_id]['T'] = tvec
        
        # 각 관측에 대해 재투영 오차 계산
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
        """3D 포인트를 2D로 투영"""
        R, T, K = camera['R'], camera['T'], camera['K']
        
        # 카메라 좌표계로 변환
        point_cam = R @ (point_3d - T)
        
        # 투영
        if point_cam[2] > 0:
            point_2d = K @ point_cam
            return point_2d[:2] / point_2d[2]
        else:
            return np.array([0.0, 0.0])  # 뒤쪽 포인트
    
    def _update_from_bundle_adjustment(self, params, n_cameras, n_points):
        """Bundle Adjustment 결과로 카메라와 포인트 업데이트"""
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
    
    def create_scene_info(self, image_paths):
        """3DGS용 SceneInfo 생성"""
        from scene.dataset_readers import CameraInfo, SceneInfo
        from utils.graphics_utils import BasicPointCloud
        
        # CameraInfo 리스트 생성
        cam_infos = []
        for cam_id in sorted(self.cameras.keys()):
            cam = self.cameras[cam_id]
            image_path = self.image_features[cam_id]['image_path']
            
            # 이미지 크기
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            
            # FoV 계산
            K = cam['K']
            focal_x, focal_y = K[0, 0], K[1, 1]
            FovX = 2 * np.arctan(width / (2 * focal_x))
            FovY = 2 * np.arctan(height / (2 * focal_y))
            
            cam_info = CameraInfo(
                uid=cam_id,
                R=cam['R'].astype(np.float32),
                T=cam['T'].astype(np.float32),
                FovY=float(FovY),
                FovX=float(FovX),
                image_path=str(image_path),
                image_name=Path(image_path).name,
                width=width,
                height=height,
                depth_params=None,
                depth_path="",
                is_test=(cam_id % 8 == 0)  # 8개 중 1개를 테스트용
            )
            cam_infos.append(cam_info)
        
        # 포인트 클라우드 생성
        if self.points_3d:
            points = np.array([pt['xyz'] for pt in self.points_3d.values()])
            # 색상은 랜덤으로 (실제로는 이미지에서 추출 가능)
            colors = np.random.rand(len(points), 3).astype(np.float32)
            normals = np.random.randn(len(points), 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        else:
            # 기본 포인트 클라우드
            points = np.random.randn(1000, 3).astype(np.float32)
            colors = np.random.rand(1000, 3).astype(np.float32)
            normals = np.random.randn(1000, 3).astype(np.float32)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        # 학습/테스트 분할
        train_cams = [c for c in cam_infos if not c.is_test]
        test_cams = [c for c in cam_infos if c.is_test]
        
        # 정규화 계산
        nerf_norm = self._compute_nerf_normalization(train_cams)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cams,
            test_cameras=test_cams,
            nerf_normalization=nerf_norm,
            ply_path="",
            is_nerf_synthetic=False
        )
    
    def _compute_nerf_normalization(self, cam_infos):
        """NeRF 정규화 파라미터 계산"""
        from utils.graphics_utils import getWorld2View2
        
        cam_centers = []
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        cam_centers = np.hstack(cam_centers)
        center = np.mean(cam_centers, axis=1, keepdims=True).flatten()
        distances = np.linalg.norm(cam_centers - center.reshape(-1, 1), axis=0)
        radius = np.max(distances) * 1.1
        
        return {"translate": -center, "radius": radius}
    
    def save_results(self, output_path):
        """결과를 COLMAP 형식으로 저장"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # 카메라 정보 저장
        cameras_file = output_path / "cameras.json"
        images_file = output_path / "images.json"
        points_file = output_path / "points3D.json"
        
        # JSON 형식으로 저장
        cameras_data = {}
        for cam_id, cam in self.cameras.items():
            cameras_data[str(cam_id)] = {
                'R': cam['R'].tolist(),
                'T': cam['T'].tolist(),
                'K': cam['K'].tolist()
            }
        
        with open(cameras_file, 'w') as f:
            json.dump(cameras_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def _load_image(self, image_path, resize=(640, 480)):
        """이미지 로드 및 전처리"""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        if resize:
            image = cv2.resize(image, resize)
        
        return image.astype(np.float32)

# scene/dataset_readers.py에 추가할 함수
def readSuperGlueSceneInfo(path, images, eval, train_test_exp=False, llffhold=8):
    """완전한 SuperGlue SfM 파이프라인 사용"""
    
    print("=== SuperGlue Complete SfM Pipeline ===")
    
    # 이미지 경로 수집
    images_folder = Path(path) / (images if images else "images")
    image_paths = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(list(images_folder.glob(ext)))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    # SuperGlue SfM 파이프라인 실행
    sfm_pipeline = SuperGlueSfMPipeline()
    scene_info = sfm_pipeline.run_complete_sfm(
        image_paths, 
        output_path=Path(path) / "superglue_sfm_output",
        max_images=100
    )
    
    return scene_info


# 실제 사용을 위한 통합 클래스
class SuperGlue3DGSInterface:
    """SuperGlue와 3DGS를 완전히 통합하는 인터페이스"""
    
    def __init__(self, config=None):
        self.sfm_pipeline = SuperGlueSfMPipeline(config)
        
    def process_images_to_3dgs(self, image_dir, output_dir, max_images=100):
        """이미지 디렉토리로부터 3DGS 학습 가능한 상태까지 완전 처리"""
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. 이미지 수집
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(list(image_dir.glob(ext)))
        
        image_paths.sort()
        image_paths = image_paths[:max_images]
        
        print(f"Processing {len(image_paths)} images for 3DGS training")
        
        # 2. SuperGlue SfM 실행
        scene_info = self.sfm_pipeline.run_complete_sfm(
            image_paths, 
            output_path=output_dir / "sfm_results"
        )
        
        # 3. 3DGS 호환 형식으로 저장
        self._save_for_3dgs(scene_info, output_dir)
        
        return scene_info
    
    def _save_for_3dgs(self, scene_info, output_dir):
        """3DGS 학습을 위한 파일 구조 생성"""
        
        # sparse 디렉토리 생성 (COLMAP 형식 모방)
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(exist_ok=True, parents=True)
        
        # images 디렉토리에 이미지 복사 또는 링크
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 카메라 정보를 간단한 텍스트 형식으로 저장
        self._write_cameras_txt(scene_info.train_cameras + scene_info.test_cameras, 
                               sparse_dir / "cameras.txt")
        
        self._write_images_txt(scene_info.train_cameras + scene_info.test_cameras, 
                              sparse_dir / "images.txt")
        
        # 포인트 클라우드 저장
        if scene_info.point_cloud:
            self._write_points3d_ply(scene_info.point_cloud, sparse_dir / "points3D.ply")
        
        print(f"3DGS-compatible files saved to {output_dir}")
    
    def _write_cameras_txt(self, cam_infos, output_path):
        """카메라 내부 파라미터를 COLMAP 형식으로 저장"""
        with open(output_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cam_infos)}\n")
            
            for cam in cam_infos:
                # PINHOLE 모델 가정
                focal_x = cam.width / (2 * np.tan(cam.FovX / 2))
                focal_y = cam.height / (2 * np.tan(cam.FovY / 2))
                cx, cy = cam.width / 2, cam.height / 2
                
                f.write(f"{cam.uid} PINHOLE {cam.width} {cam.height} "
                       f"{focal_x} {focal_y} {cx} {cy}\n")
    
    def _write_images_txt(self, cam_infos, output_path):
        """카메라 포즈를 COLMAP 형식으로 저장"""
        with open(output_path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(cam_infos)}\n")
            
            for cam in cam_infos:
                # 회전 행렬을 쿼터니언으로 변환
                R = cam.R
                qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
                qx = (R[2,1] - R[1,2]) / (4 * qw) if qw != 0 else 0
                qy = (R[0,2] - R[2,0]) / (4 * qw) if qw != 0 else 0
                qz = (R[1,0] - R[0,1]) / (4 * qw) if qw != 0 else 0
                
                f.write(f"{cam.uid} {qw} {qx} {qy} {qz} "
                       f"{cam.T[0]} {cam.T[1]} {cam.T[2]} "
                       f"{cam.uid} {cam.image_name}\n")
                f.write("\n")  # 빈 관측 데이터 라인
    
    def _write_points3d_ply(self, point_cloud, output_path):
        """포인트 클라우드를 PLY 형식으로 저장"""
        from scene.dataset_readers import storePly
        
        # RGB 값을 0-255 범위로 변환
        colors_255 = (point_cloud.colors * 255).astype(np.uint8)
        
        storePly(str(output_path), point_cloud.points, colors_255)


# 사용 예시 및 테스트 코드
def test_superglue_sfm():
    """SuperGlue SfM 파이프라인 테스트"""
    
    # 테스트용 가상 이미지 데이터 생성
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        
        # 가상 이미지 파일들 생성 (실제로는 실제 이미지 사용)
        for i in range(5):
            # 더미 이미지 파일 생성
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"image_{i:03d}.jpg"), dummy_image)
        
        # SuperGlue 3DGS 인터페이스 사용
        interface = SuperGlue3DGSInterface()
        
        try:
            scene_info = interface.process_images_to_3dgs(
                image_dir=images_dir,
                output_dir=temp_dir / "output",
                max_images=5
            )
            
            print(f"Successfully processed {len(scene_info.train_cameras)} training cameras")
            print(f"Generated {len(scene_info.point_cloud.points)} 3D points")
            
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()


# train.py 수정을 위한 추가 코드
def modify_train_script_for_superglue():
    """train.py에 SuperGlue 지원을 추가하는 방법"""
    
    modification_guide = """
    # train.py에 다음 수정사항 적용:
    
    1. arguments/__init__.py의 ModelParams 클래스에 scene_type 파라미터 추가:
       self.scene_type = "Colmap"  # 기본값
       
    2. scene/__init__.py의 Scene 클래스에서 SuperGlue 지원:
       if args.scene_type == "SuperGlue":
           scene_info = sceneLoadTypeCallbacks["SuperGlue"](args.source_path, args.images, args.eval)
       elif os.path.exists(os.path.join(args.source_path, "sparse")):
           scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
       # ... 기존 코드
    
    3. scene/dataset_readers.py에 SuperGlue 콜백 추가:
       sceneLoadTypeCallbacks = {
           "Colmap": readColmapSceneInfo,
           "Blender": readNerfSyntheticInfo,
           "SuperGlue": readSuperGlueSceneInfo
       }
    
    4. 사용법:
       python train.py -s /path/to/images --scene_type SuperGlue -m output/superglue_scene
    """
    
    return modification_guide


# 실제 이미지로 테스트하는 함수
def run_real_test(image_directory, output_directory):
    """실제 이미지로 SuperGlue SfM 테스트"""
    
    print("=== Running SuperGlue SfM on Real Images ===")
    
    # SuperGlue 3DGS 인터페이스 생성
    interface = SuperGlue3DGSInterface()
    
    try:
        # 처리 실행
        scene_info = interface.process_images_to_3dgs(
            image_dir=image_directory,
            output_dir=output_directory,
            max_images=100
        )
        
        # 결과 통계
        print(f"\n=== Results ===")
        print(f"Training cameras: {len(scene_info.train_cameras)}")
        print(f"Test cameras: {len(scene_info.test_cameras)}")
        print(f"3D points: {len(scene_info.point_cloud.points)}")
        print(f"Scene radius: {scene_info.nerf_normalization['radius']:.3f}")
        
        # 3DGS 학습 명령어 출력
        print(f"\n=== Next Steps ===")
        print(f"Run 3DGS training with:")
        print(f"python train.py -s {output_directory} --scene_type SuperGlue -m {output_directory}/3dgs_output")
        
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 커맨드라인 인터페이스
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperGlue SfM for 3DGS")
    parser.add_argument("--input", "-i", required=True, help="Input images directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--max_images", type=int, default=100, help="Maximum number of images to process")
    parser.add_argument("--test", action="store_true", help="Run test with dummy data")
    
    args = parser.parse_args()
    
    if args.test:
        test_superglue_sfm()
    else:
        success = run_real_test(args.input, args.output)
        if success:
            print("\nSuperGlue SfM completed successfully!")
        else:
            print("\nSuperGlue SfM failed!")
            exit(1)