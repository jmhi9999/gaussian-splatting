"""
Feature extraction and matching utilities
"""
import numpy as np
import cv2
import torch


class FeatureExtractor:
    """특징점 추출기"""
    
    def __init__(self, config, device, matching=None):
        self.config = config
        self.device = device
        self.matcher_type = config.get('matcher', 'superglue')
        self.matching = matching

    def extract(self, image):
        """이미지에서 특징점 추출"""
        if self.matcher_type == 'superglue' and self.matching is not None:
            # SuperPoint 특징점 추출
            from models.utils import frame2tensor
            inp = frame2tensor(image, self.device)
            with torch.no_grad():
                pred = self.matching.superpoint({'image': inp})
            return {
                'keypoints': pred['keypoints'][0].cpu().numpy(),
                'descriptors': pred['descriptors'][0].cpu().numpy(),
                'scores': pred['scores'][0].cpu().numpy(),
                'image_size': image.shape[:2]
            }
        else:
            # Fallback: OpenCV SIFT
            try:
                import cv2
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(image, None)
                if keypoints is None or descriptors is None:
                    return None
                kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
                scores = np.array([kp.response for kp in keypoints])
                return {
                    'keypoints': kpts,
                    'descriptors': descriptors.T.astype(np.float32),
                    'scores': scores,
                    'image_size': image.shape[:2]
                }
            except ImportError:
                print("OpenCV not available")
                return None


class Matcher:
    """특징점 매칭기"""
    
    def __init__(self, config, device, matching=None):
        self.config = config
        self.device = device
        self.matcher_type = config.get('matcher', 'superglue')
        self.matching = matching

    def match(self, features1, features2):
        """두 특징점 집합 간의 매칭"""
        if self.matcher_type == 'superglue' and self.matching is not None:
            # SuperGlue 매칭
            data = {
                'image0': torch.zeros((1, 1, 480, 640)).to(self.device),
                'image1': torch.zeros((1, 1, 480, 640)).to(self.device),
                'keypoints0': torch.from_numpy(features1['keypoints']).unsqueeze(0).to(self.device),
                'keypoints1': torch.from_numpy(features2['keypoints']).unsqueeze(0).to(self.device),
                'descriptors0': torch.from_numpy(features1['descriptors']).unsqueeze(0).to(self.device),
                'descriptors1': torch.from_numpy(features2['descriptors']).unsqueeze(0).to(self.device),
                'scores0': torch.from_numpy(features1['scores']).unsqueeze(0).to(self.device),
                'scores1': torch.from_numpy(features2['scores']).unsqueeze(0).to(self.device),
            }
            with torch.no_grad():
                result = self.matching.superglue(data)
            
            indices0 = result['indices0'][0].cpu().numpy()
            indices1 = result['indices1'][0].cpu().numpy()
            mscores0 = result['matching_scores0'][0].cpu().numpy()
            
            matches = []
            for i, j in enumerate(indices0):
                if j >= 0 and mscores0[i] > 0.00001:
                    if j < len(indices1) and indices1[j] == i:
                        matches.append((i, j, mscores0[i]))
            return matches
        else:
            # Fallback: OpenCV BFMatcher
            try:
                import cv2
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(features1['descriptors'].T, features2['descriptors'].T, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append((m.queryIdx, m.trainIdx, 1.0 - m.distance / 1000.0))
                return good_matches
            except ImportError:
                print("OpenCV not available")
                return [] 