"""
Скрипт для обучения YOLO модели на детекцию курения
"""
from ultralytics import YOLO
from pathlib import Path
import yaml


def create_dataset_config(dataset_path: str, output_file: str = 'dataset.yaml'):
    """
    Создает конфигурационный файл для датасета
    
    Args:
        dataset_path: Путь к папке с датасетом
        output_file: Имя выходного файла
    """
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # Количество классов
        'names': ['smoking']  # Имена классов
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Конфигурационный файл создан: {output_file}")
    return output_file


def train_model(
    dataset_path: str,
    model_size: str = 'n',  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = 'cpu'  # 'cpu' или 'cuda'
):
    """
    Обучает YOLO модель на детекцию курения
    
    Args:
        dataset_path: Путь к датасету
        model_size: Размер модели (n=нано, s=small, m=medium, l=large, x=xlarge)
        epochs: Количество эпох обучения
        imgsz: Размер изображения
        batch: Размер батча
        device: Устройство для обучения ('cpu' или 'cuda')
    """
    print("=" * 60)
    print("🎓 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ДЕТЕКЦИИ КУРЕНИЯ")
    print("=" * 60)
    print()
    
    # Проверяем наличие датасета
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Ошибка: Папка с датасетом не найдена: {dataset_path}")
        print()
        print("💡 Структура датасета должна быть:")
        print("   dataset/")
        print("   ├── images/")
        print("   │   ├── train/")
        print("   │   ├── val/")
        print("   │   └── test/")
        print("   └── labels/")
        print("       ├── train/")
        print("       ├── val/")
        print("       └── test/")
        return None
    
    # Создаем конфигурационный файл
    config_file = create_dataset_config(str(dataset_path))
    
    # Загружаем предобученную модель
    model_name = f'yolov8{model_size}.pt'
    print(f"📦 Загрузка модели: {model_name}")
    model = YOLO(model_name)
    print("✅ Модель загружена")
    print()
    
    # Настройки обучения
    print("⚙️ Настройки обучения:")
    print(f"   Модель: {model_name}")
    print(f"   Эпохи: {epochs}")
    print(f"   Размер изображения: {imgsz}")
    print(f"   Размер батча: {batch}")
    print(f"   Устройство: {device}")
    print()
    
    # Запускаем обучение
    print("🚀 Начало обучения...")
    print("   Это может занять от нескольких часов до дней")
    print("   В зависимости от размера датасета и устройства")
    print()
    
    try:
        results = model.train(
            data=config_file,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='smoking_detection',
            name='train',
            save=True,
            plots=True
        )
        
        print()
        print("=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)
        print()
        print(f"📁 Модель сохранена в: runs/detect/train/weights/best.pt")
        print()
        print("💡 Для использования обученной модели:")
        print("   detector = ViolationDetector(model_path='runs/detect/train/weights/best.pt')")
        print()
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение YOLO модели для детекции курения')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Путь к папке с датасетом')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Размер модели (n=нано, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох обучения')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Размер изображения')
    parser.add_argument('--batch', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Устройство для обучения')
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    # Пример использования без аргументов командной строки
    # Раскомментируйте и укажите путь к вашему датасету
    
    # train_model(
    #     dataset_path='path/to/your/dataset',
    #     model_size='n',  # Начните с 'n' (самая быстрая)
    #     epochs=50,       # Для теста можно меньше
    #     imgsz=640,
    #     batch=8,         # Уменьшите если не хватает памяти
    #     device='cpu'     # Используйте 'cuda' если есть GPU
    # )
    
    # Или используйте командную строку:
    # python train_smoking_model.py --dataset path/to/dataset --epochs 100 --device cuda
    
    main()
