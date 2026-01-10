"""
Скрипт для отладки детекций - показывает все объекты, которые детектирует YOLO
"""
import cv2
from detector import ViolationDetector
import config


def debug_detections():
    """Показывает все детекции на кадре"""
    print("=" * 60)
    print("🔍 ОТЛАДКА ДЕТЕКЦИЙ - ПОКАЗ ВСЕХ ОБЪЕКТОВ")
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
        return
    
    print(f"✅ Камера открыта: {camera_source}")
    print()
    print("🔍 Показываю все детекции объектов...")
    print("   Нажмите 'q' для выхода, 's' для подробного анализа кадра")
    print()
    
    frame_count = 0
    analyze_frame = False
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Не удалось получить кадр")
                break
            
            frame_count += 1
            
            # Анализ каждый 30 кадр или по нажатию 's'
            if frame_count % 30 == 0 or analyze_frame:
                analyze_frame = False
                
                print(f"\n{'='*60}")
                print(f"📸 АНАЛИЗ КАДРА {frame_count}")
                print('='*60)
                
                # Детекция с отладкой
                violations = detector.detect_violations(frame, debug=True)
                
                print()
                if violations:
                    print(f"✅ НАЙДЕНО НАРУШЕНИЙ: {len(violations)}")
                    for v in violations:
                        violation_names = {
                            'smoking': '🚭 Курение',
                            'littering': '🗑️ Выброс мусора',
                            'graffiti': '🎨 Граффити'
                        }
                        name = violation_names.get(v['type'], v['type'])
                        print(f"   {name}: уверенность {v['confidence']:.2f}")
                else:
                    print("❌ Нарушения не обнаружены")
                    print()
                    print("💡 Рекомендации:")
                    print("   • Убедитесь, что в кадре есть люди")
                    print("   • Попробуйте поднести к камере телефон или пульт (может детектироваться как курение)")
                    print("   • Попробуйте положить бутылку/чашку рядом с человеком (может детектироваться как мусор)")
            
            # Отображение кадра
            display_frame = frame.copy()
            
            # Рисуем все детекции для визуализации
            results = detector.model(frame, conf=0.3, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if cls < len(detector.COCO_CLASSES):
                        class_name = detector.COCO_CLASSES[cls]
                        
                        # Рисуем bounding box
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Добавляем подпись
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(display_frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Добавляем информацию
            info_text = [
                f"Frame: {frame_count}",
                f"Press 'q' to quit, 's' to analyze",
                "All detections shown in green"
            ]
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Debug Detections', display_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                analyze_frame = True
    
    except KeyboardInterrupt:
        print("\n⏹️ Отладка остановлена пользователем")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Отладка завершена")


if __name__ == "__main__":
    debug_detections()
