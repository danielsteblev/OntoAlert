"""
Скрипт для диагностики детекции - показывает что именно детектируется на фото
"""
import cv2
import sys
from pathlib import Path
from detector import ViolationDetector


def diagnose_image(image_path: str):
    """Диагностирует детекцию на изображении"""
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА ДЕТЕКЦИИ")
    print("=" * 60)
    print()
    
    # Проверяем файл
    if not Path(image_path).exists():
        print(f"❌ Файл не найден: {image_path}")
        return
    
    print(f"📸 Анализирую изображение: {image_path}")
    print()
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Не удалось загрузить изображение")
        return
    
    print(f"✅ Изображение загружено: {image.shape[1]}x{image.shape[0]} пикселей")
    print()
    
    # Инициализируем детектор
    print("📦 Загрузка модели YOLO...")
    detector = ViolationDetector()
    print("✅ Модель загружена")
    print()
    
    # Запускаем детекцию с debug
    print("🔍 Запуск детекции (режим отладки)...")
    print("-" * 60)
    violations = detector.detect_violations(image, debug=True)
    print("-" * 60)
    print()
    
    # Результаты
    print("=" * 60)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 60)
    print()
    
    if violations:
        print(f"✅ НАЙДЕНО НАРУШЕНИЙ: {len(violations)}")
        print()
        for i, v in enumerate(violations, 1):
            print(f"{i}. Тип: {v['type']}")
            print(f"   Уверенность: {v['confidence']:.2f} ({v['confidence']*100:.1f}%)")
            print(f"   Bounding box: {v['bbox']}")
            print()
    else:
        print("❌ НАРУШЕНИЙ НЕ ОБНАРУЖЕНО")
        print()
        print("💡 Возможные причины:")
        print("   1. Сигарета не детектируется (стандартная YOLO не умеет)")
        print("   2. Нет объектов-индикаторов рядом с человеком")
        print("   3. Объекты не в нужной области (верхняя часть тела)")
        print("   4. Низкая уверенность детекции объектов")
        print()
        print("🔧 Решение:")
        print("   Для надежной детекции нужна обученная модель!")
        print("   См. train_model_guide.md")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python diagnose_detection.py <путь_к_изображению>")
        print()
        print("Пример:")
        print("  python diagnose_detection.py photo.jpg")
        print("  python diagnose_detection.py temp_images/AgACAgIAAxkBAAMUaWFABLb5lO-TE80nlWepEngpncIAArkQaxuqoglL6DoeAkr6WDoBAAMCAAN4AAM4BA.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    diagnose_image(image_path)
