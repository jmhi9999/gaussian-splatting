"""
Parallel processing utilities for SuperGlue pipeline
"""
import concurrent.futures


class ParallelExecutor:
    """병렬 실행 관리"""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.executor = None
    
    def parallel_map(self, func, items):
        """병렬로 함수 실행"""
        if self.max_workers <= 1:
            return [func(item) for item in items]
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(func, items))
            return results
        except Exception as e:
            print(f"    Parallel execution failed: {e}, falling back to sequential")
            return [func(item) for item in items]
    
    def parallel_process_features(self, image_paths, feature_extractor):
        """특징점 추출 병렬 처리"""
        def extract_features(path):
            return feature_extractor.extract(path)
        
        return self.parallel_map(extract_features, image_paths)
    
    def parallel_match_pairs(self, pairs, matcher, image_features):
        """매칭 병렬 처리"""
        def match_pair(pair):
            i, j = pair
            if i in image_features and j in image_features:
                return (pair, matcher.match(image_features[i], image_features[j]))
            return (pair, [])
        
        return self.parallel_map(match_pair, pairs) 