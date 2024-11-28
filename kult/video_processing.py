from deepface import DeepFace
import numpy as np
import cv2

from .pose_estimation import PoseEstimator

detector_backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

DEBUG = False

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

            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, detector_backend = 'opencv')

            if DEBUG:
                # Накладываем боксы и подписи на кадр
                for person in analysis:
                    dominant_emotion = person.get('dominant_emotion', {})
                    emotions = person.get('emotion', {})
                    region = person.get('region', {})
                    prob = emotions[dominant_emotion] / 100
                    if dominant_emotion:
                        x, y, w, h = region.get('x'), region.get('y'), region.get('w'), region.get('h')
                        # Рисуем прямоугольник
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Добавляем текст с доминирующей эмоцией
                        text = f"{dominant_emotion} ({prob:.2f})"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Сохраняем кадр с наложением в виде PNG
                cv2.imwrite('tmp/output_with_boxes.png', frame)

            if isinstance(analysis, list):
                analysis = analysis[0]  # Если анализ возвращает список, берем первый элемент
            dominant_emotion = analysis.get('dominant_emotion', {})
            emotions = analysis.get('emotion', {})
            region = analysis.get('region', {})
            prob = emotions[dominant_emotion] / 100

            return dominant_emotion, prob, region
        except Exception as e:
            return None, None, None