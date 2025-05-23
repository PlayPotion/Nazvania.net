import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import datetime
import os
import time
from threading import Thread
from bez_mata.config import SCREENS_DIR, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, BLUE, RED, GREEN, GESTURE_CONTROL_ENABLED
from bez_mata.core.face_detection import FaceDetector
from bez_mata.core.face_recognition import FaceRecognizer
from bez_mata.core.gesture_control import GestureRecognizer
from bez_mata.db.face_db import FaceDB


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nazvania.net")
        self.geometry("1200x800")

        # Инициализация компонентов
        try:
            self.face_detector = FaceDetector()
            self.face_recognizer = FaceRecognizer()
            self.gesture_recognizer = GestureRecognizer()
            self.face_db = FaceDB(self.face_recognizer)
        except Exception as e:
            self._show_init_error(e)
            return

        # Вариативные позиции
        self.last_screenshot = None
        self.last_screenshot_faces = []
        self.last_screenshot_embeddings = []
        self.running = True
        self.highlight_mode = False
        self.access_control_mode = False
        self.gesture_control_enabled = GESTURE_CONTROL_ENABLED

        # Создание UI
        self._setup_ui()

        # Запуск видео
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self._show_camera_error()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Запуск видео
        self.video_thread = Thread(target=self._video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

        if GESTURE_CONTROL_ENABLED:
            self.gesture_recognizer.enable()
        else:
            self.gesture_recognizer.disable()

    def _show_camera_error(self):
        messagebox.showerror("Ошибка камеры", "Не получилось открыть видео")
        self.destroy()

    def _show_init_error(self, error):
        messagebox.showerror("Ошибка инициализации", f"Ошибка инициализации компонентов:\n{str(error)}")
        if hasattr(self, 'cap'):
            self.cap.release()
        self.destroy()

    def _setup_ui(self):
        """ Настройка пользовательского интерфейса"""
        # Видеокадры
        self.video_frame = ttk.Label(self)
        self.video_frame.pack(pady=10)

        # Кнопки контроля
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10)

        # Создаем стиль для кнопки управления жестами
        self.style = ttk.Style()
        self.style.configure('Toggle.TButton',
                             foreground='freen' if GESTURE_CONTROL_ENABLED else 'red')

        # Кнопка управления жестами
        self.gesture_btn = ttk.Button(
            control_frame,
            text=f"Жесты ({'ON' if GESTURE_CONTROL_ENABLED else 'OFF'})",
            command=self._toggle_gesture_control,
            style='Toggle.TButton'
        )
        self.gesture_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка управления скриншотами
        self.screenshot_btn = ttk.Button(
            control_frame,
            text="Сохранить скриншот (👍)",
            command=self._take_screenshot
        )
        self.screenshot_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка сравнения с последним скриншотом
        self.highlight_btn = ttk.Button(
            control_frame,
            text="Определение присутствия (✌️)",
            command=self._toggle_highlight
        )
        self.highlight_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка управления доступом
        self.access_btn = ttk.Button(
            control_frame,
            text="Определения доступа",
            command=self._toggle_access_control
        )
        self.access_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка выхода
        self.exit_btn = ttk.Button(
            control_frame,
            text="Выход (☝️)",
            command=self._exit_app
        )
        self.exit_btn.pack(side=tk.LEFT, padx=5)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готово")
        status_bar = ttk.Label(self, textvariable=self.status_var)
        status_bar.pack(fill=tk.X, pady=5)

        # Статус доступа
        self.access_status_var = tk.StringVar()
        self.access_status_var.set("Доступ: -")
        access_status = ttk.Label(self, textvariable=self.access_status_var, font=('Arial', 14))
        access_status.pack(fill=tk.X, pady=5)

    def _video_loop(self):
        """ Основной цикл обработки видео"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                # Создаём копию кадра для обработки
                processed_frame = frame.copy()

                # Обработка лиц
                faces = []
                try:
                    faces = self.face_detector.detect_faces(frame)
                    self.status_var.set(f"Людей в кадре: {len(faces)}")
                except Exception as e:
                    print(f"Ошибка определения лиц: {e}")

                try:
                    if self.access_control_mode:
                        processed_frame = self._process_access_control(frame.copy(), faces)
                    else:
                        processed_frame = self._process_frame(frame.copy())
                except Exception as e:
                    print(f"Ошибка видеопроцесса: {e}")
                    processed_frame = frame

                # Процесс кадров
                if self.access_control_mode:
                    frame = self._process_access_control(frame, self.face_detector.detect_faces(frame))
                processed_frame = self._process_frame(frame.copy())

                # Проверка и определение на жесты
                if self.gesture_control_enabled:
                    try:
                        gesture, gesture_frame = self.gesture_recognizer.recognize(processed_frame)
                        processed_frame = gesture_frame # Используем кадр с отрисованными landmarks

                        if gesture == 'good':
                            self._take_screenshot()
                        elif gesture == 'v':
                            self._toggle_highlight()
                        elif gesture == 'ai':
                            self._exit_app()
                    except Exception as e:
                        print(f"Ошибка распознавания жестов: {e}")

                # Отображение
                img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=imgtk)
                self.video_frame.image = imgtk

            except Exception as e:
                print(f"Ошибка видеопотока: {e}")
                time.sleep(0.5)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """ Обнаружение и распознавание лиц"""
        faces = self.face_detector.detect_faces(frame)
        self.status_var.set(f"Людей в кадре: {len(faces)}")

        # Логика доступа
        if self.access_control_mode and faces:
            self._process_access_control(frame, faces)

        # Логика сравнения
        if self.highlight_mode and self.last_screenshot is not None:
            frame = self._highlight_faces(frame, faces)

        # Отрисовка прямоугольников
        for (x, y, w, h) in faces:
            color = GREEN if not (self.highlight_mode or self.access_control_mode) else BLUE
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        return frame

    def _process_access_control(self, frame, faces):
        """ Сравнение лиц с лицами в базе и вывод"""
        if frame is None:
            self.access_status_var.set("Ошибка: Нет кадра")
            return frame

        if not faces:
            self.access_status_var.set("Доступ: Нет лиц")
            return frame

        access_granted = False
        recognized_name = None

        for (x, y, w, h) in faces:
            try:
                face_img = frame[y:y+h, x:x+w].copy()

                name = self.face_db.find_match(face_img)

                if name:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)
                    cv2.putText(frame, f"Access {name}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                    recognized_name = name
                    access_granted = True
                    break
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), RED, 2)

            except Exception as e:
                print(f"Ошибка обработки лица: {e}")
                continue

        if access_granted:
            self.access_status_var.set(f"Доступ: {recognized_name}")
        else:
            cv2.putText(frame, "В доступе отказано", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
            self.access_status_var.set("Доступ: отклонён")

        return frame

    def _highlight_faces(self, frame, current_faces):
        """ Сравнение лиц с последним скриншотом"""
        for (x, y, w, h) in current_faces:
            face_img = frame[y:y + h, x:x + w]
            current_embedding = self.face_recognizer.get_embedding(face_img)
            is_new = True

            for saved_embedding in self.last_screenshot_embeddings:
                if self.face_recognizer.compare_faces(current_embedding, saved_embedding):
                    is_new = False
                    break

            color = RED if is_new else BLUE
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            label = "New" if is_new else "Known"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

    def _toggle_gesture_control(self):
        """ Переключение режима распознавания жестов с визуальной обратной связью"""
        self.gesture_control_enabled = not self.gesture_control_enabled

        # Обновляем состояние детектора жестов
        if self.gesture_control_enabled:
            self.gesture_recognizer.enable()
            self.style.configure('Toggle.TButton', foreground='green')
        else:
            self.gesture_recognizer.disable()
            self.style.configure('Toggle.TButton', foreground='red')

        # Обновляем текст кнопки
        state = "ON" if self.gesture_control_enabled else "OFF"
        self.gesture_btn.config(text=f"Контроль жестов ({state})")

         # Показываем всплывающее уведомление
        messagebox.showinfo(
            "Контроль жестов",
            f"Режим определения жестов: {state}\n"
            f"👍 - скриншот\n✌️ - сревнение\n☝️ - выход"
        )

    def _take_screenshot(self):
        """ Сохранение действительного кадра как скриншот"""
        ret, frame = self.cap.read()
        if not ret:
            return

        faces = self.face_detector.detect_faces(frame)
        self.last_screenshot_embeddings = []

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            embedding = self.face_recognizer.get_embedding(face_img)
            self.last_screenshot_embeddings.append(embedding)

        os.makedirs(SCREENS_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{len(faces)}.jpg"
        filepath = os.path.join(SCREENS_DIR, filename)
        cv2.imwrite(filepath, frame)

        self.last_screenshot = frame
        messagebox.showinfo("Скриншот сохранён", f"Скриншот сохранён как {filename}")

    def _toggle_highlight(self):
        """ Активация режима сравнения"""
        self.highlight_mode = not self.highlight_mode
        state = "ON" if self.highlight_mode else "OFF"
        messagebox.showinfo("Режим сравнения", f"Режим сравнение: {state}")

    def _toggle_access_control(self):
        """ Активация режима допуска"""
        self.access_control_mode = not self.access_control_mode
        if self.access_control_mode:
            self.highlight_mode = False
        state = "ON" if self.access_control_mode else "OFF"
        messagebox.showinfo("Режим допуска", f"Режим допуска: {state}")

    def _exit_app(self):
        """ Очистка и завершение работы программы"""
        self.running = False
        self.quit()
        self.destroy()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()