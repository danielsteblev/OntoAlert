"""
Модуль для работы с камерой и мониторинга нарушений
"""
import cv2
import asyncio
from typing import Optional
from datetime import datetime
from pathlib import Path
import config
from detector import ViolationDetector
from ontology import ViolationOntology
from telegram_bot import TelegramNotifier


class CameraMonitor:
    """Класс для мониторинга камеры и обработки нарушений"""
    
    def __init__(self):
        # Если в .env задан YOLO_MODEL_PATH (best.pt), мониторинг будет использовать вашу обученную модель
        self.detector = ViolationDetector(model_path=config.YOLO_MODEL_PATH or None)
        self.ontology = ViolationOntology()
        self.notifier = TelegramNotifier()
        self.camera_source = config.CAMERA_SOURCE
        self.detection_interval = config.DETECTION_INTERVAL
        self.is_running = False
        
        # Создание директории для сохранения изображений нарушений
        self.violations_dir = Path("violations_evidence")
        self.violations_dir.mkdir(exist_ok=True)
    
    def start_monitoring(self):
        """Запуск мониторинга камеры"""
        self.is_running = True
        
        # Определение источника видео
        if self.camera_source.isdigit():
            cap = cv2.VideoCapture(int(self.camera_source))
        else:
            cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            print(f"❌ Ошибка: Не удалось открыть камеру {self.camera_source}")
            return
        
        # Установка разрешения
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RESOLUTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RESOLUTION_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FRAME_RATE)
        
        print(f"📹 Камера запущена. Источник: {self.camera_source}")
        print(f"⏱️ Интервал детекции: {self.detection_interval} секунд")
        
        frame_count = 0
        last_detection_time = datetime.now()
        violations = []  # Инициализация переменной
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Не удалось получить кадр")
                    break
                
                frame_count += 1
                
                # Проверка интервала детекции
                current_time = datetime.now()
                time_since_last_detection = (current_time - last_detection_time).total_seconds()
                
                if time_since_last_detection >= self.detection_interval:
                    # Детекция нарушений
                    violations = self.detector.detect_violations(frame)
                    
                    if violations:
                        print(f"🔍 Обнаружено {len(violations)} нарушений на кадре {frame_count}")
                        
                        # Обработка каждого нарушения
                        for violation_data in violations:
                            self._process_violation(violation_data, frame)
                        
                        last_detection_time = current_time
                    else:
                        violations = []  # Сброс если нарушений нет
                
                # Отображение кадра с детекциями и информацией
                display_frame = frame.copy()
                if violations:
                    display_frame = self.detector.draw_detections(display_frame, violations)
                
                # Добавляем информацию о статусе на кадр
                info_text = [
                    f"Frame: {frame_count}",
                    f"Detections: {len(violations)}",
                    f"Press 'q' to quit"
                ]
                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(display_frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Violation Monitor', display_frame)
                
                # Выход по нажатию 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Мониторинг остановлен пользователем")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
            print("✅ Мониторинг завершен")
    
    def _process_violation(self, violation_data: dict, frame):
        """
        Обрабатывает обнаруженное нарушение
        
        Args:
            violation_data: Данные о нарушении от детектора
            frame: Кадр с нарушением
        """
        violation_type = violation_data['type']
        
        # Сохранение изображения нарушения
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{violation_type}_{timestamp}.jpg"
        image_path = self.violations_dir / image_filename
        
        # Вырезание области нарушения
        x1, y1, x2, y2 = violation_data['bbox']
        violation_roi = frame[y1:y2, x1:x2]
        
        # Сохранение изображения
        cv2.imwrite(str(image_path), violation_roi)
        
        # Классификация через онтологию
        violation = self.ontology.classify_violation(
            violation_type=violation_type,
            location="Камера 1",  # В реальной системе должно быть из конфига
            context={'confidence': violation_data['confidence']}
        )
        
        violation.evidence_image_path = str(image_path)
        
        # Отправка уведомления в Telegram
        asyncio.run(self.notifier.send_violation_notification(
            violation=violation,
            image_path=str(image_path)
        ))
        
        print(f"📋 Нарушение обработано: {violation.description}")
        print(f"   Статья: {violation.article}, Штраф: {violation.fine_amount} {violation.fine_currency}")
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.is_running = False
