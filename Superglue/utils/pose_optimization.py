"""
Pose Graph Optimization for camera pose estimation
"""
import numpy as np
import networkx as nx
import cv2 # Added for cv2.Rodrigues


class PoseGraphOptimizer:
    """포즈 그래프 최적화"""
    
    def __init__(self):
        self.pose_graph = nx.Graph()
        self.edge_weights = {}
        self.min_matches = 10
    
    def build_pose_graph(self, matches, cameras):
        """매칭 결과로부터 포즈 그래프 생성"""
        print("  Building pose graph...")
        
        # 노드 추가 (카메라들)
        for cam_id in cameras:
            self.pose_graph.add_node(cam_id)
        
        # 엣지 추가 (매칭이 있는 카메라 쌍)
        for (cam_i, cam_j), match_list in matches.items():
            if len(match_list) >= self.min_matches:
                weight = len(match_list)  # 매칭 수를 가중치로 사용
                self.pose_graph.add_edge(cam_i, cam_j, weight=weight)
                self.edge_weights[(cam_i, cam_j)] = weight
        
        print(f"    Graph has {self.pose_graph.number_of_nodes()} nodes and {self.pose_graph.number_of_edges()} edges")
    
    def remove_outlier_edges(self, min_matches=10):
        """이상치 엣지 제거"""
        print(f"  Removing outlier edges (min_matches={min_matches})...")
        
        edges_to_remove = []
        for (cam_i, cam_j), weight in self.edge_weights.items():
            if weight < min_matches:
                edges_to_remove.append((cam_i, cam_j))
        
        for cam_i, cam_j in edges_to_remove:
            self.pose_graph.remove_edge(cam_i, cam_j)
            del self.edge_weights[(cam_i, cam_j)]
        
        print(f"    Removed {len(edges_to_remove)} outlier edges")
    
    def get_spanning_tree(self):
        """최소 신장 트리 계산"""
        print("  Computing minimum spanning tree...")
        
        try:
            # 최대 가중치 신장 트리 (매칭 수가 많을수록 좋음)
            mst = nx.maximum_spanning_tree(self.pose_graph, weight='weight')
            
            print(f"    MST has {mst.number_of_edges()} edges")
            return mst
            
        except Exception as e:
            print(f"    MST computation failed: {e}")
            # 연결된 컴포넌트 중 가장 큰 것 반환
            largest_cc = max(nx.connected_components(self.pose_graph), key=len)
            subgraph = self.pose_graph.subgraph(largest_cc)
            return subgraph
    
    def optimize_poses(self, cameras, matches):
        """포즈 그래프 최적화 (정교한 구현)"""
        print("  Optimizing pose graph...")
        
        # 1. 초기화
        optimized_cameras = cameras.copy()
        
        # 2. 스패닝 트리 기반 초기 포즈 계산
        mst = self.get_spanning_tree()
        
        # 3. 루프 클로저 검출
        loops = self._detect_loops(mst)
        print(f"    Detected {len(loops)} loop closures")
        
        # 4. 포즈 그래프 최적화 (반복적 개선)
        for iteration in range(5):
            # 4.1 상대 포즈 재계산
            relative_poses = self._compute_relative_poses(optimized_cameras, matches)
            
            # 4.2 포즈 체인 최적화
            optimized_cameras = self._optimize_pose_chain(optimized_cameras, relative_poses, mst)
            
            # 4.3 루프 클로저 제약 적용
            if loops:
                optimized_cameras = self._apply_loop_closure_constraints(optimized_cameras, loops, matches)
            
            # 4.4 수렴 확인
            pose_change = self._compute_pose_change(cameras, optimized_cameras)
            print(f"    Iteration {iteration+1}: pose change = {pose_change:.6f}")
            
            if pose_change < 1e-6:
                break
        
        # 5. 최종 검증
        self._validate_optimized_poses(optimized_cameras)
        
        return optimized_cameras
    
    def _compute_relative_poses(self, cameras, matches):
        """상대 포즈 계산"""
        relative_poses = {}
        
        for (cam_i, cam_j), match_list in matches.items():
            if cam_i in cameras and cam_j in cameras:
                try:
                    # 현재 포즈에서 상대 포즈 계산
                    R_i, T_i = cameras[cam_i]['R'], cameras[cam_i]['T']
                    R_j, T_j = cameras[cam_j]['R'], cameras[cam_j]['T']
                    
                    # 상대 회전과 이동
                    R_rel = R_j @ R_i.T
                    T_rel = T_j - R_rel @ T_i
                    
                    # 매칭 품질 기반 가중치
                    weight = min(len(match_list) / 50.0, 1.0)
                    
                    relative_poses[(cam_i, cam_j)] = {
                        'R': R_rel,
                        'T': T_rel,
                        'weight': weight,
                        'covariance': self._compute_pose_covariance(match_list)
                    }
                    
                except Exception as e:
                    print(f"    Failed to compute relative pose for {cam_i}-{cam_j}: {e}")
                    continue
        
        return relative_poses
    
    def _optimize_pose_chain(self, cameras, relative_poses, mst):
        """포즈 체인 최적화"""
        optimized = cameras.copy()
        
        # MST를 따라 포즈 전파
        root_cam = 0  # 첫 번째 카메라를 루트로 사용
        visited = {root_cam}
        queue = [root_cam]
        
        while queue:
            current_cam = queue.pop(0)
            
            for neighbor in mst.neighbors(current_cam):
                if neighbor in visited:
                    continue
                
                # 상대 포즈 적용
                pair_key = (min(current_cam, neighbor), max(current_cam, neighbor))
                if pair_key in relative_poses:
                    rel_pose = relative_poses[pair_key]
                    
                    # 현재 카메라의 포즈
                    R_current = optimized[current_cam]['R']
                    T_current = optimized[current_cam]['T']
                    
                    # 상대 포즈 적용
                    if current_cam < neighbor:
                        R_new = rel_pose['R'] @ R_current
                        T_new = rel_pose['R'] @ T_current + rel_pose['T']
                    else:
                        R_rel_inv = rel_pose['R'].T
                        T_rel_inv = -R_rel_inv @ rel_pose['T']
                        R_new = R_rel_inv @ R_current
                        T_new = R_rel_inv @ T_current + T_rel_inv
                    
                    optimized[neighbor]['R'] = R_new
                    optimized[neighbor]['T'] = T_new
                    
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return optimized
    
    def _detect_loops(self, mst):
        """루프 클로저 검출"""
        loops = []
        
        # 원래 그래프와 MST의 차이에서 루프 찾기
        for edge in self.pose_graph.edges():
            if not mst.has_edge(*edge):
                loops.append(edge)
        
        return loops
    
    def _apply_loop_closure_constraints(self, cameras, loops, matches):
        """루프 클로저 제약 적용"""
        optimized = cameras.copy()
        
        for cam_i, cam_j in loops:
            pair_key = (min(cam_i, cam_j), max(cam_i, cam_j))
            if pair_key in matches:
                # 루프 클로저 오차 계산
                loop_error = self._compute_loop_error(optimized, cam_i, cam_j, matches[pair_key])
                
                # 오차 분배 (가중치 기반)
                weight_i = len(matches.get((cam_i, cam_j), []))
                weight_j = len(matches.get((cam_j, cam_i), []))
                total_weight = weight_i + weight_j + 1e-6
                
                # 포즈 조정
                adjustment_factor = 0.1  # 부드러운 조정
                
                if loop_error > 0.1:  # 오차 임계값
                    # 회전 조정
                    rot_adjustment = loop_error * adjustment_factor
                    
                    # 카메라 i 조정
                    optimized[cam_i]['R'] = self._adjust_rotation(
                        optimized[cam_i]['R'], rot_adjustment * (weight_j / total_weight)
                    )
                    
                    # 카메라 j 조정
                    optimized[cam_j]['R'] = self._adjust_rotation(
                        optimized[cam_j]['R'], -rot_adjustment * (weight_i / total_weight)
                    )
        
        return optimized
    
    def _compute_loop_error(self, cameras, cam_i, cam_j, matches):
        """루프 클로저 오차 계산"""
        try:
            R_i, T_i = cameras[cam_i]['R'], cameras[cam_i]['T']
            R_j, T_j = cameras[cam_j]['R'], cameras[cam_j]['T']
            
            # 상대 포즈 계산
            R_rel = R_j @ R_i.T
            T_rel = T_j - R_rel @ T_i
            
            # 회전 오차 (각도)
            trace = np.trace(R_rel)
            rot_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            # 이동 오차
            trans_error = np.linalg.norm(T_rel)
            
            # 종합 오차
            total_error = rot_error + trans_error * 0.1
            
            return total_error
            
        except Exception:
            return 0.0
    
    def _adjust_rotation(self, R, adjustment):
        """회전 행렬 조정"""
        try:
            # 로드리게스 벡터로 변환
            angle_axis = self._rotation_to_angle_axis(R)
            
            # 조정 적용
            adjusted_angle_axis = angle_axis + adjustment * angle_axis / (np.linalg.norm(angle_axis) + 1e-6)
            
            # 회전 행렬로 변환
            adjusted_R = self._angle_axis_to_rotation(adjusted_angle_axis)
            
            return adjusted_R
            
        except Exception:
            return R
    
    def _rotation_to_angle_axis(self, R):
        """회전 행렬을 로드리게스 벡터로 변환"""
        # OpenCV 사용 (더 안정적)
        angle_axis, _ = cv2.Rodrigues(R)
        return angle_axis.flatten()
    
    def _angle_axis_to_rotation(self, angle_axis):
        """로드리게스 벡터를 회전 행렬로 변환"""
        # OpenCV 사용 (더 안정적)
        R, _ = cv2.Rodrigues(angle_axis)
        return R
    
    def _compute_pose_covariance(self, matches):
        """포즈 공분산 계산"""
        # 매칭 품질 기반 공분산 추정
        if len(matches) < 10:
            return np.eye(6) * 10.0  # 높은 불확실성
        
        # 매칭 분포 분석
        confidences = [conf for _, _, conf in matches]
        avg_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        # 불확실성 계산
        uncertainty = (1.0 - avg_conf) * (1.0 + std_conf)
        
        return np.eye(6) * uncertainty
    
    def _compute_pose_change(self, cameras_old, cameras_new):
        """포즈 변화량 계산"""
        total_change = 0.0
        count = 0
        
        for cam_id in cameras_old:
            if cam_id in cameras_new:
                R_old, T_old = cameras_old[cam_id]['R'], cameras_old[cam_id]['T']
                R_new, T_new = cameras_new[cam_id]['R'], cameras_new[cam_id]['T']
                
                # 회전 변화
                R_diff = R_new @ R_old.T
                rot_change = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                
                # 이동 변화
                trans_change = np.linalg.norm(T_new - T_old)
                
                total_change += rot_change + trans_change * 0.1
                count += 1
        
        return total_change / (count + 1e-6)
    
    def _validate_optimized_poses(self, cameras):
        """최적화된 포즈 검증"""
        valid_count = 0
        
        for cam_id, cam_data in cameras.items():
            R, T = cam_data['R'], cam_data['T']
            
            # 회전 행렬 유효성 검사
            if self._is_valid_rotation_matrix(R):
                valid_count += 1
            else:
                print(f"    Warning: Invalid rotation matrix for camera {cam_id}")
        
        print(f"    Validated {valid_count}/{len(cameras)} camera poses")
    
    def _is_valid_rotation_matrix(self, R):
        """회전 행렬 유효성 검사"""
        try:
            # 행렬식 검사
            det = np.linalg.det(R)
            if abs(det - 1.0) > 0.1:
                return False
            
            # 직교성 검사
            should_be_identity = R @ R.T
            I = np.eye(3)
            if np.max(np.abs(should_be_identity - I)) > 0.1:
                return False
            
            return True
            
        except Exception:
            return False 