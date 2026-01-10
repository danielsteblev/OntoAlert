"""
Тестовый скрипт для проверки работы камеры и детекции
"""
import cv2
from detector import ViolationDetector
import config


def test_camera_detection():
    """Тестирует работу камеры и детекции нарушений"""
    print("=" * 60)
    print("🎥 ТЕСТ КАМЕРЫ И ДЕТЕКЦИИ НАРУШЕНИЙ")
    print("=" * 60)
    print()
    
    # Инициализация детектора
    print("📦 Загрузка модели YOLO...")
    try:
        detector = ViolationDetector()
        print("✅ Модель загружена успешно")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    print()
    print("📹 Открытие камеры...")
    
    # Определение источника видео
    camera_source = config.CAMERA_SOURCE
    if camera_source.isdigit():
        cap = cv2.VideoCapture(int(camera_source))
    else:
        cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"❌ Ошибка: Не удалось открыть камеру {camera_source}")
        print("   Проверьте, что камера подключена и доступна")
        return
    
    print(f"✅ Камера открыта: {camera_source}")
    print()
    print("🔍 Запуск детекции...")
    print("   Нажмите 'q' для выхода")
    print()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Не удалось получить кадр")
                break
            
            frame_count += 1
            
            # Детекция каждые 30 кадров (чтобы не перегружать систему и не засорять вывод)
            if frame_count % 30 == 0:
                # Включаем режим отладки для первых проверок
                debug_mode = frame_count <= 60  # Отладка только для первых 2 проверок
                if debug_mode:
                    print(f"\n🔍 Анализ кадра {frame_count}:")
                violations = detector.detect_violations(frame, debug=debug_mode)
                
                if violations:
                    print(f"🔍 Кадр {frame_count}: Обнаружено {len(violations)} нарушений")
                    for v in violations:
                        violation_names = {
                            'smoking': 'Курение',
                            'littering': 'Выброс мусора',
                            'graffiti': 'Граффити'
                        }
                        name = violation_names.get(v['type'], v['type'])
                        print(f"   ⚠️ {name}: уверенность {v['confidence']:.2f}")
                # Убираем вывод при отсутствии нарушений, чтобы не засорять консоль
            
            # Отображение кадра
            display_frame = frame.copy()
            
            # Добавляем информацию на экран
            info_text = [
                f"Frame: {frame_count}",
                f"Press 'q' to quit",
                f"Detection: every 30 frames"
            ]
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', display_frame)
            
            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Тест остановлен пользователем")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Тест завершен")


if __name__ == "__main__":
    test_camera_detection()
