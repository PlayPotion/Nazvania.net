import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2

class FaceRecognizer:
    """ Класс для генерации эмебддингов лиц с использованием MobileNetV2"""

    def __init__(self):
        """ Инициализация модели MobileNetV2"""
        self.model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """ Подготовка изображения для модели"""
        # Приведение к нужному размеру и формату
        image = cv2.resize(image, (224, 224))
        return preprocess_input(image)

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """ Генерация эмебддинга для лица"""
        try:
            # Проверка входных данных
            if face_image is None or face_image.size == 0:
                raise ValueError("Пустое изображение лица")

            # Предварительная обработка
            processed = self.preprocess_input(face_image)
            if processed is None:
                raise ValueError("Ошибка предвариетльной обработки")

            # Получение эмбеддинга
            embedding = self.model.predict(processed[np.newaxis, ...])[0]
            if embedding is None or len(embedding) == 0:
                raise ValueError("Пустой эмбеддинг")

            return embedding

        except Exception as e:
            print(f"Ошибка в get_embedding: {str(e)}")
            return None

    @staticmethod
    def compare_faces(embedding1: np.ndarray,
                      embedding2: np.ndarray,
                      threshold: float = 0.7) -> bool:
        """ Сравнение двух эмбеддингов с использованием косинусной схожести"""
        # Нормализация векторов
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Выччисление схожести
        similarity = np.dot(embedding1, embedding2)
        return similarity > threshold