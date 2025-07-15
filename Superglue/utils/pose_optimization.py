"""
Pose Graph Optimization for camera pose estimation
"""
import numpy as np
import networkx as nx


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
        """포즈 그래프 최적화"""
        print("  Optimizing pose graph...")
        
        # 현재는 간단한 구현 (실제로는 더 복잡한 최적화 필요)
        # 1. 스패닝 트리 기반 초기화
        mst = self.get_spanning_tree()
        
        # 2. 루프 클로저 검출 및 제약 추가
        loops = self._detect_loops(mst)
        
        # 3. 포즈 최적화 (간단한 버전)
        optimized_cameras = self._simple_pose_optimization(cameras, matches, mst)
        
        return optimized_cameras
    
    def _detect_loops(self, mst):
        """루프 클로저 검출"""
        loops = []
        
        # MST에 없는 엣지들을 확인하여 루프 형성
        for edge in self.pose_graph.edges():
            if edge not in mst.edges():
                # 이 엣지를 추가했을 때 형성되는 루프 찾기
                temp_graph = mst.copy()
                temp_graph.add_edge(*edge)
                
                try:
                    cycle = nx.find_cycle(temp_graph)
                    if len(cycle) > 3:  # 3개 이상의 노드로 구성된 루프
                        loops.append(cycle)
                except nx.NetworkXNoCycle:
                    pass
        
        return loops
    
    def _simple_pose_optimization(self, cameras, matches, mst):
        """간단한 포즈 최적화"""
        # 현재는 원본 카메라 반환 (실제 구현에서는 더 복잡한 최적화)
        return cameras.copy() 