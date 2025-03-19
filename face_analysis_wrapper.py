import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face

class EnhancedFaceAnalysis(FaceAnalysis):
    """
    Wrapper for insightface.app.FaceAnalysis that adds the get2 method without needing to patch the library.
    """
    
    def get2(self, img, bboxes, kpss):
        """
        Process multiple detected faces with bounding boxes and keypoints already computed.
        This is more efficient than calling get() which performs detection again.
        
        Args:
            img: Input image
            bboxes: Array of face bounding boxes (N, 5) where each row is [x1, y1, x2, y2, score]
            kpss: Array of face keypoints or None
            
        Returns:
            List of Face objects with all model inferences completed
        """
        if bboxes.shape[0] == 0:
            return []
            
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret
