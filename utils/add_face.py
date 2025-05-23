import cv2
import tkinter as tk
from tkinter import simpledialog
from bez_mata.core.face_detection import FaceDetector
from bez_mata.db.face_db import FaceDB
from bez_mata.core.face_recognition import FaceRecognizer
from bez_mata.config import FACE_DB_PATH

def add_new_face():
    """ Утилита для добавления новых лиц в базу данных"""
    print("\n=== Режим добавления новых лиц ===")

    # Инициализация компонентов
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    face_db = FaceDB(face_recognizer)

    # Инициализация видеопотока
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть камеру")
        return

    while True:
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Проблема с получением кадра")
            continue

        # Детекция лиц
        faces = face_detector.detect_faces(frame)

        # Отрисовка прямоугольников вокруг лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Отображение интерфейса
        cv2.imshow("Add face (SPACE - save, ESC - exit)", frame)

        # Обраотка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(faces) == 1:
                # Вырезаем область с лицом
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w].copy()

                # Запрос имени через GUI
                root = tk.Tk()
                root.withdraw()
                person_name = simpledialog.askstring("Ввод", "Введите имя человека:")

                if person_name:
                    if face_db.add_face(face_img, person_name):
                        break
            else:
                print('[INFO] В кадре должно быть ровно одно лицо')

        elif key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=== Режим добавления завершён ===\n")

if __name__ == "__main__":
    add_new_face()