"""
Track Manager for feature point tracking and triangulation
"""
import numpy as np
import cv2
from collections import defaultdict


class TrackManager:
    """특징점 트랙 관리 및 삼각측량"""
    
    def __init__(self):
        self.tracks = {}  # track_id -> {points: [(cam_id, kpt_idx), ...], color: [r,g,b]}
        self.track_id_counter = 0
        self.min_views = 3
        self.min_track_length = 2
    
    def build_tracks(self, matches, image_features):
        """매칭 결과로부터 트랙 생성"""
        print("  Building tracks from matches...")
        
        # 매칭을 그래프로 변환
        track_graph = defaultdict(list)
        
        for (cam_i, cam_j), match_list in matches.items():
            for idx_i, idx_j, conf in match_list:
                # 각 매칭을 노드로 표현
                node_i = (cam_i, idx_i)
                node_j = (cam_j, idx_j)
                
                track_graph[node_i].append((node_j, conf))
                track_graph[node_j].append((node_i, conf))
        
        # 연결된 컴포넌트 찾기 (트랙)
        visited = set()
        tracks = []
        
        for node in track_graph:
            if node in visited:
                continue
            
            # BFS로 연결된 노드들 찾기
            track_nodes = []
            queue = [node]
            visited.add(node)
            
            while queue:
                current = queue.pop(0)
                track_nodes.append(current)
                
                for neighbor, conf in track_graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(track_nodes) >= self.min_track_length:
                tracks.append(track_nodes)
        
        # 트랙을 내부 구조로 변환
        for track_nodes in tracks:
            track_data = {
                'points': track_nodes,
                'color': self._estimate_track_color(track_nodes, image_features),
                'confidence': self._compute_track_confidence(track_nodes, track_graph)
            }
            
            self.tracks[self.track_id_counter] = track_data
            self.track_id_counter += 1
        
        print(f"    Built {len(self.tracks)} tracks")
    
    def filter_tracks(self, min_views=3):
        """트랙 필터링"""
        print(f"  Filtering tracks (min_views={min_views})...")
        
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            # 고유한 카메라 수 계산
            unique_cameras = set(cam_id for cam_id, _ in track_data['points'])
            
            if len(unique_cameras) < min_views:
                tracks_to_remove.append(track_id)
        
        # 필터링된 트랙 제거
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        print(f"    Kept {len(self.tracks)} tracks after filtering")
    
    def triangulate_tracks(self, cameras, image_features):
        """트랙들을 삼각측량하여 3D 포인트 생성"""
        print("  Triangulating tracks...")
        
        triangulated_points = {}
        successful_tracks = 0
        
        for track_id, track_data in self.tracks.items():
            try:
                # 트랙의 각 관찰점 수집
                observations = []
                for cam_id, kpt_idx in track_data['points']:
                    if cam_id in cameras and cam_id in image_features:
                        kpts = image_features[cam_id]['keypoints']
                        if kpt_idx < len(kpts):
                            observations.append((cam_id, kpts[kpt_idx]))
                
                if len(observations) < 2:
                    continue
                
                # 삼각측량 수행
                point_3d = self._triangulate_track(observations, cameras)
                
                if point_3d is not None:
                    triangulated_points[track_id] = {
                        'xyz': point_3d,
                        'color': track_data['color'],
                        'observations': observations
                    }
                    successful_tracks += 1
                    
            except Exception as e:
                print(f"    Track {track_id} triangulation failed: {e}")
                continue
        
        print(f"    Successfully triangulated {successful_tracks} tracks")
        return triangulated_points
    
    def _estimate_track_color(self, track_nodes, image_features):
        """트랙의 색상 추정"""
        # 첫 번째 관찰점의 색상을 사용 (실제로는 이미지에서 샘플링)
        if track_nodes:
            cam_id, kpt_idx = track_nodes[0]
            if cam_id in image_features:
                # 간단한 랜덤 색상 생성
                return np.random.rand(3).astype(np.float32)
        
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    def _compute_track_confidence(self, track_nodes, track_graph):
        """트랙의 신뢰도 계산"""
        if len(track_nodes) < 2:
            return 0.0
        
        # 연결 강도 평균 계산
        total_conf = 0.0
        edge_count = 0
        
        for i, node_i in enumerate(track_nodes):
            for j, node_j in enumerate(track_nodes[i+1:], i+1):
                # 두 노드 간의 연결 찾기
                for neighbor, conf in track_graph[node_i]:
                    if neighbor == node_j:
                        total_conf += conf
                        edge_count += 1
                        break
        
        if edge_count > 0:
            return total_conf / edge_count
        else:
            return 0.0
    
    def _triangulate_track(self, observations, cameras):
        """단일 트랙 삼각측량"""
        if len(observations) < 2:
            return None
        
        # 투영 행렬들 수집
        projection_matrices = []
        points_2d = []
        
        for cam_id, point_2d in observations:
            if cam_id in cameras:
                cam = cameras[cam_id]
                K, R, T = cam['K'], cam['R'], cam['T']
                
                # 투영 행렬 생성
                t = -R @ T
                RT = np.hstack([R, t.reshape(-1, 1)])
                P = K @ RT
                
                projection_matrices.append(P)
                points_2d.append(point_2d)
        
        if len(projection_matrices) < 2:
            return None
        
        try:
            # OpenCV 삼각측량
            points_2d_array = np.array(points_2d).T
            points_4d = cv2.triangulatePoints(
                projection_matrices[0], 
                projection_matrices[1], 
                points_2d_array[:, 0:1], 
                points_2d_array[:, 1:2]
            )
            
            # 4D에서 3D로 변환
            if abs(points_4d[3, 0]) > 1e-10:
                point_3d = (points_4d[:3] / points_4d[3]).flatten()
                
                # 유효성 검사
                if self._is_valid_3d_point(point_3d, observations, cameras):
                    return point_3d.astype(np.float32)
            
        except Exception as e:
            print(f"      Triangulation error: {e}")
        
        return None
    
    def _is_valid_3d_point(self, point_3d, observations, cameras):
        """3D 포인트 유효성 검사"""
        # NaN/Inf 체크
        if np.any(np.isnan(point_3d)) or np.any(np.isinf(point_3d)):
            return False
        
        # 거리 체크
        distance = np.linalg.norm(point_3d)
        if distance > 1000 or distance < 0.001:
            return False
        
        # 재투영 오차 체크
        max_error = 0.0
        for cam_id, point_2d in observations:
            if cam_id in cameras:
                cam = cameras[cam_id]
                K, R, T = cam['K'], cam['R'], cam['T']
                
                # 카메라 좌표계로 변환
                point_cam = R @ (point_3d - T)
                
                if point_cam[2] <= 0:  # 카메라 뒤쪽
                    return False
                
                # 재투영
                point_2d_proj = K @ point_cam
                point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                
                error = np.linalg.norm(point_2d_proj - point_2d)
                max_error = max(max_error, error)
        
        return max_error < 10.0  # 10픽셀 이하 오차 