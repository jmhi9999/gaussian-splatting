"""
Automatic parameter tuning for SuperGlue pipeline
"""
import numpy as np


class AutoTuner:
    """자동 파라미터 튜닝"""
    
    def __init__(self, config):
        self.config = config
        self.quality_history = []
        self.parameter_history = []
        self.best_parameters = config.copy()
        self.best_quality = 0.0
    
    def evaluate_metrics(self, quality_metrics):
        """품질 메트릭 평가"""
        if not quality_metrics:
            return
        
        # 종합 품질 점수 계산
        quality_score = self._compute_quality_score(quality_metrics)
        
        self.quality_history.append(quality_score)
        
        # 최고 품질 업데이트
        if quality_score > self.best_quality:
            self.best_quality = quality_score
            self.best_parameters = self.config.copy()
            print(f"    New best quality: {quality_score:.4f}")
    
    def adjust_parameters(self):
        """파라미터 자동 조정"""
        if len(self.quality_history) < 2:
            return
        
        # 품질 변화 분석
        recent_quality = self.quality_history[-1]
        previous_quality = self.quality_history[-2]
        
        if recent_quality < previous_quality:
            # 품질이 감소했으면 파라미터를 보수적으로 조정
            self._adjust_parameters_conservative()
        else:
            # 품질이 향상되었으면 더 적극적으로 조정
            self._adjust_parameters_aggressive()
    
    def _compute_quality_score(self, metrics):
        """종합 품질 점수 계산"""
        score = 0.0
        
        # 포즈 추정 성공률
        if 'pose_estimation_success_rate' in metrics:
            score += metrics['pose_estimation_success_rate'] * 0.4
        
        # 평균 매칭 수
        if 'average_matches_per_pair' in metrics:
            avg_matches = metrics['average_matches_per_pair']
            score += min(avg_matches / 50.0, 1.0) * 0.3  # 50개 이상이면 만점
        
        # Bundle Adjustment 품질
        if 'final_cost' in metrics:
            cost = metrics['final_cost']
            score += max(0, 1.0 - cost / 1000.0) * 0.3  # cost가 낮을수록 높은 점수
        
        return score
    
    def _adjust_parameters_conservative(self):
        """보수적 파라미터 조정"""
        print("    Adjusting parameters conservatively...")
        
        # 매칭 임계값 완화
        if 'superglue' in self.config:
            if 'match_threshold' in self.config['superglue']:
                current_threshold = self.config['superglue']['match_threshold']
                self.config['superglue']['match_threshold'] = max(0.01, current_threshold * 0.8)
        
        # 특징점 수 증가
        if 'superpoint' in self.config:
            if 'max_keypoints' in self.config['superpoint']:
                current_max = self.config['superpoint']['max_keypoints']
                self.config['superpoint']['max_keypoints'] = min(16384, current_max * 1.2)
    
    def _adjust_parameters_aggressive(self):
        """적극적 파라미터 조정"""
        print("    Adjusting parameters aggressively...")
        
        # 매칭 임계값 강화
        if 'superglue' in self.config:
            if 'match_threshold' in self.config['superglue']:
                current_threshold = self.config['superglue']['match_threshold']
                self.config['superglue']['match_threshold'] = min(0.2, current_threshold * 1.1)
        
        # 특징점 수 감소 (품질 향상)
        if 'superpoint' in self.config:
            if 'max_keypoints' in self.config['superpoint']:
                current_max = self.config['superpoint']['max_keypoints']
                self.config['superpoint']['max_keypoints'] = max(2048, current_max * 0.9)
    
    def get_best_parameters(self):
        """최고 품질의 파라미터 반환"""
        return self.best_parameters.copy() 