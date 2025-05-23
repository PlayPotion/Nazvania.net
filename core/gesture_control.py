import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple
import cv2


class GestureRecognizer:
    """ Класс для распознавания жестов с визуализацией kandmarks с использованием MediaPipe Hands """

    def __init__(self):
        """ Инициализация детектора жестов"""
        self.mp_hands = mp.solutions.hands
        self.hands = None # Инициализация по требованию
        self.active = False # Флаг активности распознавания
        self.last_gesture_time = 0
        self.gesture_cooldown = 3  # Задержка между жестами (в секундах)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def enable(self):
        """ Активирует распознавание жестов и инициализирует детектор"""
        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        self.active = True

    def disable(self):
        """ Деактивирует распознавание и освобождает ресурсы"""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        self.active = False


    def recognize(self, image: np.ndarray) -> Optional[str]:
        """ Основной метод распознавания жестов + изображение с landmarks"""
        if not self.active:
            return None, image.copy()

        # Проверка временной задержки между жестами
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return None, image.copy()

        # Проверка входного изображения
        if image is None or image.size == 0:
            return None, image.copy() if image is not None else None

        try:
            # Создаём копию изображения для отрисовки
            annotated_image = image.copy()
            gesture = None

            # Конвертация цветового пространства для MediaPipe
            results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                # Отрисовываем landmarks и соединения
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Берем первую обнаруженную руку
                landmarks = results.multi_hand_landmarks[0].landmark
                gesture = self._classify_gesture(landmarks)

                if gesture:
                    self.last_gesture_time = current_time
                    # Добавляем текст с названием жеста
                    cv2.putText(annotated_image, f"Gesture: {gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return gesture, annotated_image

        except Exception as e:
            print(f"Ошибка распознавания жеста: {e}")
            return None, image.copy()

    def _classify_gesture(self, landmarks) -> Optional[str]:
        """ Классификация жеста по ключевым точкам руки"""
        if len(landmarks) < 21:
            return None

        # Получаем нужные точки руки
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        wrist = landmarks[0]

        # Жест "👍" (большой палец вверх)
        if (self._is_finger_closed(landmarks, [8, 12, 16, 20]) and  # Все пальцы кроме большого согнуты
                thumb_tip.y < wrist.y):  # Большой палец выше запястья (вверх)
            return 'good'

        # Жест "✌️" (знак V)
        if (self._is_finger_open(landmarks, [8, 12]) and  # Указательный и средний разогнуты
                self._is_finger_closed(landmarks, [16, 20])):  # Безымянный и мизинец согнуты
            return 'v'

        # Жест "☝️" (указательный палец вверх)
        if (self._is_finger_open(landmarks, [8]) and
                self._is_finger_closed(landmarks, [4, 12, 16, 20])):
            return 'ai'

        return None

    @staticmethod
    def _distance(p1, p2) -> float:
        """ Вычисление евклидова расстояния между двумя точками"""
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    @staticmethod
    def _is_finger_closed(landmarks, finger_tips) -> bool:
        """ Проверка, согнут ли палец (сравнение с нижележащим суставом)"""
        for tip in finger_tips:
            tip = int(tip)
            if landmarks[tip].y < landmarks[tip - 2].y:
                return False
        return True

    @staticmethod
    def _is_finger_open(landmarks, finger_tips) -> bool:
        """ Проверка, разогнут ли палец"""
        for tip in finger_tips:
            tip = int(tip)
            if landmarks[tip].y > landmarks[tip - 2].y:
                return False
        return True