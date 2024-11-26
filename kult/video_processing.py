from deepface import DeepFace
import numpy as np

from .pose_estimation import PoseEstimator

class VideoProcessor:
    def __init__(self, device='cpu'):
        pass
        #self.pose_estimator = PoseEstimator()

    def face_emotion(self, frame):
        #pose = self.pose_estimator.estimate_pose(frame)

        try:
            # Преобразуем кадр в формат numpy array (BGR), если он не является таковым
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, detector_backend = 'dlib') # detector_backend = 'dlib'

            if isinstance(analysis, list):
                analysis = analysis[0]  # Если анализ возвращает список, берем первый элемент
            dominant_emotion = analysis.get('dominant_emotion', {})
            emotions = analysis.get('emotion', {})
            region = analysis.get('region', {})
            prob = emotions[dominant_emotion] / 100
            return dominant_emotion, prob, region
        except Exception as e:
            return None, None, None