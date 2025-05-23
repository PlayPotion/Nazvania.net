import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

class FaceDetector:
    """ Класс для обнаружения лиц с использованием MediaPipe Face Detection"""

    def __init__(self):
        """ Инициализация детектора лиц"""
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )

    def detect_faces(self, image: np.array) -> List[Tuple[int, int, int, int]]:
        """ Обнаружение лиц на изображении и возвращение bounding boxes"""
        # конвертация цветового пространства для MediaPipe
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = []

        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                h, w, _ = image.shape

                # Конвертация относительных координат в абсолютные
                x, y = int(box.xmin * w), int(box.ymin * h)
                width, height = int(box.width * w), int(box.height * h)

                boxes.append((x, y, width, height))

        return boxes