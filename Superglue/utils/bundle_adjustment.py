"""
Bundle Adjustment optimization for camera poses and 3D points (Advanced Implementation)
"""
import numpy as np
from scipy.optimize import least_squares
import cv2
import time


class BundleAdjuster:
    """정교한 Bundle Adjustment 최적화"""
    
    def __init__(self, loss_type='adaptive_huber', max_iter=100):
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.quality_metrics = {}
        
        # 적응적 파라미터
        self.adaptive_params = {
            'huber_delta': 3.0,
            'cauchy_sigma': 1.0,
            'trust_region_radius': 1.0,
            'damping_factor': 1e-6
        }
        
        # 수렴 기준
        self.convergence_criteria = {
            'ftol': 1e-6,
            'xtol': 1e-6,
            'gtol': 1e-6,
            'max_nfev': max_iter * 10
        }
    
    def run(self, cameras, points_3d, point_observations, callback=None):
        """정교한 Bundle Adjustment 실행"""
        print("  Running Advanced Bundle Adjustment...")
        
        # 전처리 및 검증
        if not self._validate_input_data(cameras, points_3d, point_observations):
            return False
        
        # 적응적 파라미터 초기화
        self._initialize_adaptive_parameters(cameras, points_3d, point_observations)
        
        try:
            # 초기 비용 계산
            initial_params = self._pack_parameters(cameras, points_3d)
            initial_cost = self._compute_total_cost(initial_params, cameras, points_3d, point_observations)
            
            print(f"    Initial cost: {initial_cost:.6f}")
            
            # 다단계 최적화
            result = self._multi_stage_optimization(cameras, points_3d, point_observations, callback)
            
            if result.success:
                # 최종 품질 메트릭 계산
                final_cost = result.cost
                self.quality_metrics = {
                    'initial_cost': initial_cost,
                    'final_cost': final_cost,
                    'cost_reduction': initial_cost - final_cost,
                    'iterations': result.nfev,
                    'success': True,
                    'optimization_time': getattr(result, 'optimization_time', 0),
                    'convergence_reason': self._get_convergence_reason(result)
                }
                
                print(f"    BA completed successfully:")
                print(f"      Final cost: {final_cost:.6f}")
                print(f"      Cost reduction: {self.quality_metrics['cost_reduction']:.6f}")
                print(f"      Iterations: {result.nfev}")
                print(f"      Convergence: {self.quality_metrics['convergence_reason']}")
                
                if callback:
                    callback(self.quality_metrics)
                
                return True
            else:
                print(f"    BA failed to converge: {result.message}")
                return False
                
        except Exception as e:
            print(f"    Bundle adjustment failed: {e}")
            return False
    
    def _validate_input_data(self, cameras, points_3d, point_observations):
        """입력 데이터 검증"""
        # 카메라 수 검증
        if len(cameras) < 2:
            print("    Error: Need at least 2 cameras")
            return False
        
        # 3D 포인트 수 검증
        if len(points_3d) < 10:
            print("    Error: Need at least 10 3D points")
            return False
        
        # 관찰 데이터 검증
        total_observations = sum(len(obs) for obs in point_observations.values())
        if total_observations < 20:
            print("    Error: Need at least 20 observations")
            return False
        
        # 카메라 포즈 유효성 검증
        invalid_cameras = []
        for cam_id, cam_data in cameras.items():
            if not self._is_valid_camera_pose(cam_data):
                invalid_cameras.append(cam_id)
        
        if invalid_cameras:
            print(f"    Warning: Invalid camera poses detected: {invalid_cameras}")
            # 유효하지 않은 카메라 수정
            self._fix_invalid_cameras(cameras, invalid_cameras)
        
        # 3D 포인트 유효성 검증
        invalid_points = []
        for point_id, point_data in points_3d.items():
            if not self._is_valid_3d_point(point_data['xyz']):
                invalid_points.append(point_id)
        
        if invalid_points:
            print(f"    Warning: {len(invalid_points)} invalid 3D points will be filtered")
            for point_id in invalid_points:
                del points_3d[point_id]
                if point_id in point_observations:
                    del point_observations[point_id]
        
        return True
    
    def _initialize_adaptive_parameters(self, cameras, points_3d, point_observations):
        """적응적 파라미터 초기화"""
        # 전체 관찰 수
        total_observations = sum(len(obs) for obs in point_observations.values())
        
        # 재투영 오차 분포 분석
        sample_errors = self._sample_reprojection_errors(cameras, points_3d, point_observations)
        
        if len(sample_errors) > 0:
            error_median = np.median(sample_errors)
            error_std = np.std(sample_errors)
            
            # 적응적 Huber delta 설정
            self.adaptive_params['huber_delta'] = max(2.0, error_median + 2 * error_std)
            
            # 적응적 Cauchy sigma 설정
            self.adaptive_params['cauchy_sigma'] = max(1.0, error_median + error_std)
            
            print(f"    Adaptive parameters:")
            print(f"      Huber delta: {self.adaptive_params['huber_delta']:.3f}")
            print(f"      Cauchy sigma: {self.adaptive_params['cauchy_sigma']:.3f}")
            print(f"      Sample errors: median={error_median:.3f}, std={error_std:.3f}")
    
    def _multi_stage_optimization(self, cameras, points_3d, point_observations, callback):
        """다단계 최적화"""
        
        # Stage 1: 카메라 포즈만 최적화 (3D 포인트 고정)
        print("    Stage 1: Camera pose optimization...")
        result1 = self._optimize_cameras_only(cameras, points_3d, point_observations)
        
        # Stage 2: 3D 포인트만 최적화 (카메라 포즈 고정)
        print("    Stage 2: 3D point optimization...")
        result2 = self._optimize_points_only(cameras, points_3d, point_observations)
        
        # Stage 3: 전체 최적화 (카메라 포즈 + 3D 포인트)
        print("    Stage 3: Full bundle adjustment...")
        result3 = self._optimize_full_bundle(cameras, points_3d, point_observations, callback)
        
        return result3
    
    def _optimize_cameras_only(self, cameras, points_3d, point_observations):
        """카메라 포즈만 최적화"""
        # 카메라 파라미터만 패킹
        camera_params = []
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            angle_axis = self._rotation_matrix_to_angle_axis(cam['R'])
            camera_params.extend(angle_axis)
            camera_params.extend(cam['T'])
        
        camera_params = np.array(camera_params)
        
        def camera_residuals(params):
            # 카메라 파라미터 언패킹
            self._unpack_camera_parameters(params, cameras)
            return self._compute_camera_residuals(cameras, points_3d, point_observations)
        
        try:
            result = least_squares(
                camera_residuals,
                camera_params,
                method='lm',
                max_nfev=self.max_iter // 3,
                ftol=self.convergence_criteria['ftol'],
                xtol=self.convergence_criteria['xtol']
            )
            
            print(f"      Camera optimization: cost={result.cost:.6f}, iterations={result.nfev}")
            return result
            
        except Exception as e:
            print(f"      Camera optimization failed: {e}")
            return None
    
    def _optimize_points_only(self, cameras, points_3d, point_observations):
        """3D 포인트만 최적화"""
        # 3D 포인트 파라미터만 패킹
        point_params = []
        for point_id in sorted(points_3d.keys()):
            point_params.extend(points_3d[point_id]['xyz'])
        
        point_params = np.array(point_params)
        
        def point_residuals(params):
            # 3D 포인트 파라미터 언패킹
            self._unpack_point_parameters(params, points_3d)
            return self._compute_point_residuals(cameras, points_3d, point_observations)
        
        try:
            result = least_squares(
                point_residuals,
                point_params,
                method='lm',
                max_nfev=self.max_iter // 3,
                ftol=self.convergence_criteria['ftol'],
                xtol=self.convergence_criteria['xtol']
            )
            
            print(f"      Point optimization: cost={result.cost:.6f}, iterations={result.nfev}")
            return result
            
        except Exception as e:
            print(f"      Point optimization failed: {e}")
            return None
    
    def _optimize_full_bundle(self, cameras, points_3d, point_observations, callback):
        """전체 Bundle Adjustment"""
        start_time = time.time()
        
        # 전체 파라미터 패킹
        params = self._pack_parameters(cameras, points_3d)
        
        def full_residuals(params):
            # 파라미터 언패킹
            self._unpack_parameters(params, cameras, points_3d)
            return self._compute_full_residuals(cameras, points_3d, point_observations)
        
        try:
            result = least_squares(
                full_residuals,
                params,
                method='trf',  # Trust Region Reflective 알고리즘
                max_nfev=self.convergence_criteria['max_nfev'],
                ftol=self.convergence_criteria['ftol'],
                xtol=self.convergence_criteria['xtol'],
                gtol=self.convergence_criteria['gtol'],
                verbose=1
            )
            
            optimization_time = time.time() - start_time
            result.optimization_time = optimization_time
            
            print(f"      Full optimization: cost={result.cost:.6f}, iterations={result.nfev}, time={optimization_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"      Full optimization failed: {e}")
            return None
    
    def _compute_full_residuals(self, cameras, points_3d, point_observations):
        """전체 잔차 계산 (정교한 구현)"""
        residuals = []
        
        for point_id, observations in point_observations.items():
            if point_id not in points_3d:
                continue
                
            point_3d = points_3d[point_id]['xyz']
            
            for cam_id, observed_pt, confidence in observations:
                if cam_id not in cameras:
                    continue
                
                try:
                    cam = cameras[cam_id]
                    K = cam['K']
                    R = cam['R']
                    T = cam['T']
                    
                    # 카메라 좌표계로 변환
                    point_cam = R @ (point_3d - T)
                    
                    # 깊이 검사
                    if point_cam[2] <= 0.01:
                        residuals.extend([50.0, 50.0])  # 큰 페널티
                        continue
                    
                    # 재투영
                    point_2d_proj = K @ point_cam
                    if abs(point_2d_proj[2]) < 1e-10:
                        residuals.extend([50.0, 50.0])
                        continue
                    
                    point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                    
                    # 잔차 계산
                    residual = point_2d_proj - observed_pt
                    
                    # 적응적 손실 함수 적용
                    residual = self._apply_adaptive_loss(residual, confidence)
                    
                    residuals.extend(residual)
                    
                except Exception as e:
                    residuals.extend([10.0, 10.0])  # 중간 페널티
        
        residuals = np.array(residuals)
        
        # NaN이나 무한대 값 처리
        residuals = np.nan_to_num(residuals, nan=10.0, posinf=50.0, neginf=-50.0)
        
        return residuals
    
    def _apply_adaptive_loss(self, residual, confidence):
        """적응적 손실 함수 적용"""
        residual_norm = np.linalg.norm(residual)
        
        if self.loss_type == 'adaptive_huber':
            # 적응적 Huber 손실
            delta = self.adaptive_params['huber_delta']
            if residual_norm <= delta:
                return residual * confidence
            else:
                scale = delta / residual_norm
                return residual * scale * confidence * 0.5
        
        elif self.loss_type == 'adaptive_cauchy':
            # 적응적 Cauchy 손실
            sigma = self.adaptive_params['cauchy_sigma']
            scale = sigma / (sigma + residual_norm)
            return residual * scale * confidence
        
        elif self.loss_type == 'adaptive_tukey':
            # 적응적 Tukey 손실
            c = self.adaptive_params['huber_delta'] * 2
            if residual_norm <= c:
                factor = (1 - (residual_norm / c)**2)**2
                return residual * factor * confidence
            else:
                return residual * 0.0  # 완전히 거부
        
        elif self.loss_type == 'huber':
            # 기본 Huber 손실
            delta = 3.0
            if residual_norm <= delta:
                return residual * confidence
            else:
                scale = delta / residual_norm
                return residual * scale * confidence
        
        else:
            # L2 손실 (기본값)
            return residual * confidence
    
    def _sample_reprojection_errors(self, cameras, points_3d, point_observations, max_samples=1000):
        """재투영 오차 샘플링"""
        errors = []
        sample_count = 0
        
        for point_id, observations in point_observations.items():
            if point_id not in points_3d or sample_count >= max_samples:
                break
                
            point_3d = points_3d[point_id]['xyz']
            
            for cam_id, observed_pt, _ in observations:
                if cam_id not in cameras or sample_count >= max_samples:
                    break
                
                try:
                    cam = cameras[cam_id]
                    K, R, T = cam['K'], cam['R'], cam['T']
                    
                    # 재투영 계산
                    point_cam = R @ (point_3d - T)
                    if point_cam[2] > 0:
                        point_2d_proj = K @ point_cam
                        point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                        
                        error = np.linalg.norm(point_2d_proj - observed_pt)
                        errors.append(error)
                        sample_count += 1
                        
                except Exception:
                    continue
        
        return np.array(errors)
    
    def _compute_camera_residuals(self, cameras, points_3d, point_observations):
        """카메라 포즈 최적화용 잔차"""
        return self._compute_full_residuals(cameras, points_3d, point_observations)
    
    def _compute_point_residuals(self, cameras, points_3d, point_observations):
        """3D 포인트 최적화용 잔차"""
        return self._compute_full_residuals(cameras, points_3d, point_observations)
    
    def _compute_total_cost(self, params, cameras, points_3d, point_observations):
        """전체 비용 계산"""
        # 파라미터 언패킹
        self._unpack_parameters(params, cameras, points_3d)
        
        # 잔차 계산
        residuals = self._compute_full_residuals(cameras, points_3d, point_observations)
        
        # 총 비용 (L2 norm)
        return 0.5 * np.sum(residuals**2)
    
    def _unpack_camera_parameters(self, params, cameras):
        """카메라 파라미터만 언패킹"""
        idx = 0
        for cam_id in sorted(cameras.keys()):
            # 로드리게스 벡터 (3개)
            angle_axis = params[idx:idx+3]
            idx += 3
            
            # 이동 벡터 (3개)
            T = params[idx:idx+3]
            idx += 3
            
            # 회전 행렬로 변환
            R = self._angle_axis_to_rotation_matrix(angle_axis)
            
            cameras[cam_id]['R'] = R.astype(np.float32)
            cameras[cam_id]['T'] = T.astype(np.float32)
    
    def _unpack_point_parameters(self, params, points_3d):
        """3D 포인트 파라미터만 언패킹"""
        idx = 0
        for point_id in sorted(points_3d.keys()):
            xyz = params[idx:idx+3]
            idx += 3
            points_3d[point_id]['xyz'] = xyz.astype(np.float32)
    
    def monitor_quality(self, metrics):
        """품질 모니터링 콜백"""
        if metrics:
            print(f"    Quality metrics updated: cost_reduction={metrics.get('cost_reduction', 0):.6f}")
        else:
            print("    Quality metrics updated: No metrics available")

    def _is_valid_camera_pose(self, cam_data):
        """카메라 포즈 유효성 검증"""
        try:
            R, T = cam_data['R'], cam_data['T']
            
            # 회전 행렬 검증
            if not self._is_valid_rotation_matrix(R):
                return False
            
            # 이동 벡터 검증
            if np.any(np.isnan(T)) or np.any(np.isinf(T)):
                return False
            
            # 합리적인 이동 범위 검증
            if np.linalg.norm(T) > 1000:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_valid_3d_point(self, point_3d):
        """3D 포인트 유효성 검증"""
        try:
            # NaN/Inf 검사
            if np.any(np.isnan(point_3d)) or np.any(np.isinf(point_3d)):
                return False
            
            # 합리적인 거리 범위 검증
            if np.linalg.norm(point_3d) > 10000:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _fix_invalid_cameras(self, cameras, invalid_camera_ids):
        """유효하지 않은 카메라 수정"""
        for cam_id in invalid_camera_ids:
            if cam_id in cameras:
                # 기본값으로 초기화
                cameras[cam_id]['R'] = np.eye(3, dtype=np.float32)
                cameras[cam_id]['T'] = np.zeros(3, dtype=np.float32)
    
    def _is_valid_rotation_matrix(self, R):
        """회전 행렬 유효성 검증"""
        try:
            # 행렬식 검증
            det = np.linalg.det(R)
            if abs(det - 1.0) > 0.1:
                return False
            
            # 직교성 검증
            RRT = R @ R.T
            I = np.eye(3)
            if np.max(np.abs(RRT - I)) > 0.1:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_convergence_reason(self, result):
        """수렴 이유 분석"""
        if hasattr(result, 'message'):
            return result.message
        elif hasattr(result, 'success') and result.success:
            return "Successfully converged"
        else:
            return "Unknown convergence status"
    
    def _pack_parameters(self, cameras, points_3d):
        """카메라 포즈와 3D 포인트를 하나의 벡터로 패킹"""
        params = []
        
        # 카메라 포즈 (회전 + 이동)
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            R = cam['R']
            T = cam['T']
            
            # 회전 행렬을 로드리게스 벡터로 변환
            angle_axis = self._rotation_matrix_to_angle_axis(R)
            params.extend(angle_axis)
            params.extend(T)
        
        # 3D 포인트
        for point_id in sorted(points_3d.keys()):
            if isinstance(points_3d[point_id], dict):
                point = points_3d[point_id]['xyz']
            else:
                point = points_3d[point_id]
            params.extend(point)
        
        params = np.array(params)
        
        # NaN이나 무한대 값 체크
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            raise ValueError("Invalid parameters detected (NaN or Inf)")
        
        return params
    
    def _unpack_parameters(self, params, cameras, points_3d):
        """벡터에서 카메라 포즈와 3D 포인트 언패킹"""
        idx = 0
        
        # 카메라 포즈 복원
        for cam_id in sorted(cameras.keys()):
            # 로드리게스 벡터 (3개)
            angle_axis = params[idx:idx+3]
            idx += 3
            
            # 이동 벡터 (3개)
            T = params[idx:idx+3]
            idx += 3
            
            # 회전 행렬로 변환
            R = self._angle_axis_to_rotation_matrix(angle_axis)
            
            cameras[cam_id]['R'] = R.astype(np.float32)
            cameras[cam_id]['T'] = T.astype(np.float32)
        
        # 3D 포인트 복원
        for point_id in sorted(points_3d.keys()):
            xyz = params[idx:idx+3]
            idx += 3
            if isinstance(points_3d[point_id], dict):
                points_3d[point_id]['xyz'] = xyz.astype(np.float32)
            else:
                points_3d[point_id] = xyz.astype(np.float32)

    def _rotation_matrix_to_angle_axis(self, R):
        """회전 행렬을 로드리게스 벡터로 변환 (기존 메서드 유지)"""
        try:
            # OpenCV의 Rodrigues 함수 사용
            rvec, _ = cv2.Rodrigues(R)
            return rvec.flatten().astype(np.float32)
        except:
            # 수동 계산 fallback
            trace = np.trace(R)
            if trace > 3.0 - 1e-6:
                return np.zeros(3, dtype=np.float32)
            
            angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
            if angle < 1e-6:
                return np.zeros(3, dtype=np.float32)
            
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2.0 * np.sin(angle))
            
            return (axis * angle).astype(np.float32)
    
    def _angle_axis_to_rotation_matrix(self, angle_axis):
        """로드리게스 벡터를 회전 행렬로 변환 (기존 메서드 유지)"""
        try:
            # OpenCV의 Rodrigues 함수 사용
            R, _ = cv2.Rodrigues(angle_axis)
            return R.astype(np.float32)
        except:
            # 수동 계산 fallback
            angle = np.linalg.norm(angle_axis)
            if angle < 1e-8:
                return np.eye(3, dtype=np.float32)
            
            axis = angle_axis / angle
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            one_minus_cos = 1.0 - cos_angle
            
            outer = np.outer(axis, axis)
            cross = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            R = (cos_angle * np.eye(3) + 
                 sin_angle * cross + 
                 one_minus_cos * outer)
            
            return R.astype(np.float32) 