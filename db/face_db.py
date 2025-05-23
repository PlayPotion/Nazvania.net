import os
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from bez_mata.config import FACE_DB_PATH
from bez_mata.core.face_recognition import FaceRecognizer


class FaceDB:
    """ Класс для работы с базой данных лиц"""

    def __init__(self, recognizer):
        """ Инициализация базы данных"""
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        self.recognizer = recognizer
        self.embeddings = {}
        self._load_existing_faces()
        self.check_database()

    def check_database(self):
        """ Проверка и вывод стостояния базы данных"""
        print("\nПроверка базы данных:")
        for name, embedding in self.embeddings.items():
            status = "OK" if embedding is not None and len(embedding) > 0 else "CORRUPTED"
            print(f"{name}: {status} (shape={embedding.shape if embedding is not None else 'N/A'})")

    def _load_existing_faces(self):
        """ Загрузка существующих лиц за файлов .npy"""
        for filename in os.listdir(FACE_DB_PATH):
            if filename.endswith('.npy'):
                person_name = filename[:-4]
                self.embeddings[person_name] = np.load(os.path.join(FACE_DB_PATH, filename))

    def add_face(self, face_image: np.ndarray, person_name: str):
        """ Добавление нового лица в базу данных"""
        try:
            # Генерация эмбеддинга для лица
            embedding = self.recognizer.get_embedding(face_image)
            if embedding is None:
                raise ValueError("Ошибка генерации эмбеддинга")

            # Созранение в файл
            np.save(os.path.join(FACE_DB_PATH, f'{person_name}.npy'), embedding)
            self.embeddings[person_name] = embedding

            print(f"[SUCCESS] Лицо '{person_name}' добавлено в базу")
            return True
        except Exception as e:
            print(f'[ERROR] Ошибка добавления лица: {str(e)}')
            return False

    def find_match(self, face_image: np.ndarray):
        """ Поиск соответствия лица в базе данных"""
        try:
            # Проверка вводных данных
            if face_image is None or face_image.size == 0:
                print("Пустое изображение лица")
                return None

            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                print("Изображение лица слишком маленькое")
                return None

            print(f"\nПоиск соответствия среди {len(self.embeddings)} известных лиц")

            # Генерация эмбеддинга
            embedding = self.recognizer.get_embedding(face_image)
            if embedding is None:
                print("Не удалось сгенерировать эмбеддинг")
                return None

            # Нормализация эмбеддинга
            embedding_norm = embedding / np.linalg.norm(embedding)

            best_match = None
            best_similarity = 0.0
            threshold = 0.65

            # Поиск ближайшего соответствия
            for name, db_embed in self.embeddings.items():
                try:
                    if db_embed is None:
                        print(f"Пропускаем пустой эмбеддинг для {name}")
                        continue

                    # Нормализация и сравнение
                    db_embed_norm = db_embed / np.linalg.norm(db_embed)

                    similarity = np.dot(embedding_norm, db_embed_norm)
                    print(f"Cравнение с {name}: схожесть={similarity:.4f}")

                    if similarity > threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_name = name

                except Exception as e:
                    print(f"Ошибка сравнения с {name}: {str(e)}")
                    continue

            print(f"Лучшее соответствие: {best_name} (схожесть={best_similarity:.4f})")
            return best_name if best_similarity > threshold else None

        except Exception as e:
            print(f"Ошибка в  find_match: {str(e)}")
            return None