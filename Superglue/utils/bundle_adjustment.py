"""
Bundle Adjustment optimization for camera poses and 3D points
"""
import numpy as np
from scipy.optimize import least_squares


class BundleAdjuster:
    """Bundle Adjustment 최적화"""
    
    def __init__(self, loss_type='huber', max_iter=50):
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.quality_metrics = {}
    
    def run(self, cameras, points_3d, point_observations, callback=None):
        """Bundle Adjustment 실행"""
        print("  Running Bundle Adjustment...")
        
        if len(cameras) < 2 or len(points_3d) < 5:
            print("    Insufficient data for bundle adjustment")
            return False
        
        try:
            # 파라미터 패킹
            params = self._pack_parameters(cameras, points_3d)
            
            # 최적화 실행
            result = least_squares(
                self._compute_residuals,
                params,
                args=(cameras, points_3d, point_observations),
                method='lm',
                max_nfev=self.max_iter,
                verbose=1
            )
            
            # 결과 언패킹
            self._unpack_parameters(result.x, cameras, points_3d)
            
            # 품질 메트릭 계산
            self.quality_metrics = {
                'final_cost': result.cost,
                'iterations': result.nfev,
                'success': result.success
            }
            
            if callback:
                callback(self.quality_metrics)
            
            print(f"    BA completed: cost={result.cost:.6f}, iterations={result.nfev}")
            return True
            
        except Exception as e:
            print(f"    Bundle adjustment failed: {e}")
            return False
    
    def _pack_parameters(self, cameras, points_3d):
        """파라미터를 하나의 벡터로 패킹"""
        params = []
        
        # 카메라 파라미터 (회전 + 이동)
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            R, T = cam['R'], cam['T']
            
            # 회전 행렬을 로드리게스 벡터로 변환
            angle_axis = self._rotation_matrix_to_angle_axis(R)
            params.extend(angle_axis)
            params.extend(T)
        
        # 3D 포인트
        for point_id in sorted(points_3d.keys()):
            point = points_3d[point_id]['xyz']
            params.extend(point)
        
        return np.array(params)
    
    def _unpack_parameters(self, params, cameras, points_3d):
        """벡터에서 파라미터 언패킹"""
        idx = 0
        
        # 카메라 파라미터 복원
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
            points_3d[point_id]['xyz'] = xyz.astype(np.float32)
    
    def _compute_residuals(self, params, cameras, points_3d, point_observations):
        """Bundle Adjustment 잔차 계산"""
        # 파라미터 언패킹
        self._unpack_parameters(params, cameras, points_3d)
        
        residuals = []
        
        # 각 관찰에 대한 재투영 오차 계산
        for point_id, observations in point_observations.items():
            if point_id not in points_3d:
                continue
            
            point_3d = points_3d[point_id]['xyz']
            
            for cam_id, observed_pt, conf in observations:
                if cam_id not in cameras:
                    continue
                
                try:
                    cam = cameras[cam_id]
                    K = cam['K']
                    R = cam['R']
                    T = cam['T']
                    
                    # 카메라 좌표계로 변환
                    point_cam = R @ (point_3d - T)
                    
                    if point_cam[2] <= 0:
                        residuals.extend([10.0, 10.0])
                        continue
                    
                    # 재투영
                    point_2d_proj = K @ point_cam
                    point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                    
                    # 잔차 계산
                    residual = point_2d_proj - observed_pt
                    
                    # Huber loss 적용
                    if self.loss_type == 'huber':
                        residual = self._apply_huber_loss(residual, delta=3.0)
                    
                    # 신뢰도 가중치
                    weight = np.clip(conf, 0.1, 1.0)
                    residual = residual * weight
                    
                    residuals.extend(residual)
                    
                except Exception:
                    residuals.extend([2.0, 2.0])
        
        return np.array(residuals)
    
    def _apply_huber_loss(self, residual, delta=3.0):
        """Huber loss 적용"""
        abs_residual = np.abs(residual)
        mask = abs_residual <= delta
        
        result = np.zeros_like(residual)
        result[mask] = residual[mask]
        result[~mask] = delta * np.sign(residual[~mask]) * (2 * np.sqrt(abs_residual[~mask] / delta) - 1)
        
        return result
    
    def _rotation_matrix_to_angle_axis(self, R):
        """회전 행렬을 로드리게스 벡터로 변환"""
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
    
    def monitor_quality(self, metrics):
        """품질 메트릭 모니터링"""
        print(f"    BA Quality: cost={metrics['final_cost']:.6f}, "
              f"iterations={metrics['iterations']}, success={metrics['success']}") 