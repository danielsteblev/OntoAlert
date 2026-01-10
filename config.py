"""
Конфигурационный файл для системы мониторинга нарушений
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram настройки
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Путь к обученной YOLO-модели (best.pt). Если пусто/не найдено — будет использована дефолтная модель.
# Пример: YOLO_MODEL_PATH=runs/violation_detection/train/weights/best.pt
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', '')
YOLO_FALLBACK_MODEL = os.getenv('YOLO_FALLBACK_MODEL', 'yolov8n.pt')

# Переопределение названий классов для вашей модели (если в best.pt они перепутаны).
# Формат: "smoking,littering,graffiti" где позиция = class id (0,1,2,...)
# Пример если 0=graffiti, 1=littering, 2=smoking:
# YOLO_CLASS_NAMES=graffiti,littering,smoking
YOLO_CLASS_NAMES = os.getenv('YOLO_CLASS_NAMES', '').strip()

# Настройки камеры
CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', '0')
FRAME_RATE = int(os.getenv('FRAME_RATE', '30'))
RESOLUTION_WIDTH = int(os.getenv('RESOLUTION_WIDTH', '1920'))
RESOLUTION_HEIGHT = int(os.getenv('RESOLUTION_HEIGHT', '1080'))

# Настройки детекции
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
DETECTION_INTERVAL = int(os.getenv('DETECTION_INTERVAL', '5'))

# Классы нарушений для YOLO
VIOLATION_CLASSES = {
    'smoking': 0,      # Курение
    'littering': 1,    # Выброс мусора
    'graffiti': 2      # Граффити
}
