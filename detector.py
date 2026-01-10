"""
Модуль детекции нарушений с использованием компьютерного зрения
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from pathlib import Path
import config
import os

# Отключаем избыточный вывод YOLO
os.environ['YOLO_VERBOSE'] = 'False'


class ViolationDetector:
    """Класс для детекции нарушений на видео"""
    
    # Классы COCO для YOLO (первые 80 классов стандартного датасета)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к обученной YOLO модели. Если None, используется предобученная модель
        """
        # Если путь не передан — пробуем взять из конфига
        resolved_model_path = model_path or (config.YOLO_MODEL_PATH or None)
        self.model_source: str = ""

        self.using_custom_model = False
        self.model_class_names: Dict[int, str] = {}

        if resolved_model_path and Path(resolved_model_path).exists():
            self.model = YOLO(resolved_model_path)
            self.model_source = str(resolved_model_path)
            # У Ultralytics model.names обычно dict[int,str] или list[str]
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                self.model_class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                self.model_class_names = {i: str(n) for i, n in enumerate(names)}
            self.using_custom_model = True
        else:
            # Используем предобученную YOLOv8 модель (COCO)
            self.model = YOLO(config.YOLO_FALLBACK_MODEL)
            self.model_source = str(config.YOLO_FALLBACK_MODEL)
            self.model_class_names = {i: n for i, n in enumerate(self.COCO_CLASSES)}

        # Optional override from env: YOLO_CLASS_NAMES="a,b,c"
        # This fixes cases when label ids in dataset were mapped to different names than expected.
        if config.YOLO_CLASS_NAMES:
            parts = [p.strip() for p in config.YOLO_CLASS_NAMES.split(",") if p.strip()]
            if parts:
                self.model_class_names = {i: name for i, name in enumerate(parts)}
        
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.violation_classes = config.VIOLATION_CLASSES
        # История детекций для анализа поведения
        self.detection_history = []
    
    def detect_violations(self, frame: np.ndarray, debug: bool = False) -> List[Dict]:
        """
        Детектирует нарушения на кадре
        
        Args:
            frame: Кадр изображения (BGR формат)
            debug: Если True, выводит отладочную информацию
        
        Returns:
            Список словарей с информацией о найденных нарушениях
        """
        # Запуск детекции YOLO (verbose=False отключает служебные сообщения)
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        # Собираем все детекции на кадре
        all_detections: List[Dict] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                class_name = self.model_class_names.get(cls, str(cls))
                all_detections.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                })
        
        # Отладочный вывод
        if debug and all_detections:
            print(f"   Детектировано объектов: {len(all_detections)}")
            for det in all_detections:
                print(f"      - {det['class_name']}: {det['confidence']:.2f}")
        
        # Если это ваша обученная модель (smoking/littering/graffiti) — нарушения = классы модели
        if self.using_custom_model:
            violations: List[Dict] = []
            allowed = set(self.violation_classes.keys())
            for det in all_detections:
                if det['class_name'] in allowed:
                    violations.append({
                        'type': det['class_name'],
                        'confidence': det['confidence'],
                        'bbox': det['bbox']
                    })
        else:
            # Иначе — анализ COCO детекций эвристиками
            violations = self._analyze_detections(all_detections, frame, debug)
        
        # Сохраняем историю для анализа поведения
        self.detection_history.append(all_detections)
        if len(self.detection_history) > 10:  # Храним последние 10 кадров
            self.detection_history.pop(0)
        
        return violations
    
    def _analyze_detections(self, detections: List[Dict], frame: np.ndarray, debug: bool = False) -> List[Dict]:
        """
        Анализирует все детекции на кадре и определяет нарушения
        
        Args:
            detections: Список всех детекций на кадре
            frame: Кадр изображения
            debug: Если True, выводит отладочную информацию
        
        Returns:
            Список обнаруженных нарушений
        """
        violations = []
        
        # Поиск людей на кадре (снижаем порог для демонстрации)
        people = [d for d in detections if d['class_name'] == 'person' and d['confidence'] > 0.3]
        
        if debug:
            print(f"   Найдено людей: {len(people)}")
        
        if not people:
            if debug:
                print("   ⚠️ Люди не найдены, нарушения не могут быть определены")
            return violations
        
        # Для каждого человека проверяем возможные нарушения
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            person_center = person['center']
            
            # Поиск объектов рядом с человеком
            nearby_objects = []
            for obj in detections:
                if obj == person:
                    continue
                obj_center = obj['center']
                # Проверяем, находится ли объект в области рядом с человеком
                if self._is_near_person(person['bbox'], obj_center):
                    nearby_objects.append(obj)
            
            # Проверка на курение
            if debug:
                print(f"   Проверка человека для курения. Рядом объектов: {len(nearby_objects)}")
            smoking_violation = self._check_smoking(person, nearby_objects, debug)
            if smoking_violation:
                violations.append(smoking_violation)
                if debug:
                    print(f"   ✅ Курение обнаружено: уверенность {smoking_violation['confidence']:.2f}")
            
            # Проверка на выброс мусора
            littering_violation = self._check_littering(person, nearby_objects, debug)
            if littering_violation:
                violations.append(littering_violation)
                if debug:
                    print(f"   ✅ Выброс мусора обнаружен: уверенность {littering_violation['confidence']:.2f}")
        
        # Проверка на граффити (анализ статических объектов)
        graffiti_violation = self._check_graffiti(detections, frame)
        if graffiti_violation:
            violations.append(graffiti_violation)
        
        return violations
    
    def _is_near_person(self, person_bbox: Tuple[int, int, int, int], obj_center: Tuple[int, int]) -> bool:
        """Проверяет, находится ли объект рядом с человеком"""
        px1, py1, px2, py2 = person_bbox
        # Расширяем область поиска вокруг человека
        margin = 50
        return (px1 - margin <= obj_center[0] <= px2 + margin and 
                py1 - margin <= obj_center[1] <= py2 + margin)
    
    def _check_smoking(self, person: Dict, nearby_objects: List[Dict], debug: bool = False) -> Optional[Dict]:
        """
        Проверяет наличие признаков курения
        
        Эвристика: человек с объектом в области лица/рук (cell phone, remote, bottle, cup)
        """
        # Объекты, которые могут быть ошибочно детектированы вместо сигареты
        # Добавляем больше объектов для повышения чувствительности
        smoking_indicators = ['cell phone', 'remote', 'cigarette', 'lighter', 'cup', 'bottle']
        
        px1, py1, px2, py2 = person['bbox']
        person_height = py2 - py1
        
        if debug:
            print(f"      Рядом объекты: {[obj['class_name'] for obj in nearby_objects]}")
        
        for obj in nearby_objects:
            if obj['class_name'] in smoking_indicators:
                obj_center = obj['center']
                
                # Проверяем, находится ли объект в области лица/верхней части тела
                # Расширяем область проверки для большей чувствительности
                face_region_y = py1 + (py2 - py1) * 0.5  # Верхние 50% тела (расширено с 40%)
                
                # Проверяем, что объект находится в горизонтальной области лица/рук
                if obj_center[1] < face_region_y:
                    # Объект находится в области головы/рук
                    bbox = person['bbox']
                    
                    # Настройка уверенности в зависимости от типа объекта
                    # Для демонстрации делаем более чувствительным
                    if obj['class_name'] == 'cell phone':
                        confidence = min(0.75, obj['confidence'] * 0.9)  # Повышено
                    elif obj['class_name'] == 'remote':
                        confidence = min(0.70, obj['confidence'] * 0.85)  # Повышено
                    elif obj['class_name'] in ['cup', 'bottle']:
                        # Чашка/бутылка в руках тоже может быть признаком курения
                        confidence = min(0.65, obj['confidence'] * 0.75)
                    else:
                        confidence = min(0.75, obj['confidence'] * 0.9)
                    
                    # Снижаем минимальный порог для демонстрации
                    if confidence > 0.45:  # Снижено с 0.5 для большей чувствительности
                        if debug:
                            print(f"      ✅ Объект {obj['class_name']} в области лица/рук - возможное курение")
                        return {
                            'type': 'smoking',
                            'confidence': confidence,
                            'bbox': bbox
                        }
                    elif debug:
                        print(f"      ⚠️ Объект {obj['class_name']} найден, но уверенность низкая: {confidence:.2f}")
        
        if debug and nearby_objects:
            print(f"      ❌ Объекты найдены, но не подходят для детекции курения")
        
        return None
    
    def _check_littering(self, person: Dict, nearby_objects: List[Dict], debug: bool = False) -> Optional[Dict]:
        """
        Проверяет наличие признаков выброса мусора
        
        Эвристика: человек рядом с мусором (bottle, cup, bowl) в области ног
        """
        litter_objects = ['bottle', 'cup', 'bowl']
        
        for obj in nearby_objects:
            if obj['class_name'] in litter_objects:
                px1, py1, px2, py2 = person['bbox']
                obj_center = obj['center']
                
                # Мусор обычно находится в нижней части кадра, рядом с ногами
                feet_region_y = py2 - 50
                
                if obj_center[1] > feet_region_y and obj['confidence'] > 0.5:
                    # Возможен выброс мусора
                    # Объект находится в области ног - потенциально брошенный мусор
                    bbox = person['bbox']
                    # Для демонстрации снижаем порог уверенности
                    confidence = obj['confidence'] * 0.8
                    
                    # Если объект появился недавно (проверяем историю), увеличиваем уверенность
                    if len(self.detection_history) >= 2:
                        was_before = any(
                            any(o['class_name'] == obj['class_name'] and 
                                abs(o['center'][0] - obj['center'][0]) < 100 and
                                abs(o['center'][1] - obj['center'][1]) < 100
                                for o in prev_frame)
                            for prev_frame in self.detection_history[-3:]
                        )
                        if not was_before:
                            confidence = min(0.75, confidence * 1.1)  # Увеличиваем если новый объект
                    
                    if confidence > 0.5:  # Минимальный порог для детекции
                        return {
                            'type': 'littering',
                            'confidence': confidence,
                            'bbox': bbox
                        }
        
        return None
    
    def _check_graffiti(self, detections: List[Dict], frame: np.ndarray) -> Optional[Dict]:
        """
        Проверяет наличие признаков граффити
        
        Эвристика: объекты для рисования (spray can не детектируется стандартной YOLO,
        но можно использовать другие индикаторы)
        """
        # Стандартная YOLO не детектирует баллончик с краской
        # Это требует специальной обученной модели
        # Для демонстрации возвращаем None
        
        # В реальной системе здесь должна быть логика:
        # - Детекция баллончика с краской (spray can)
        # - Детекция маркера (marker)
        # - Анализ поверхности стены на наличие новой краски
        
        return None
    
    def detect_smoking(self, frame: np.ndarray) -> List[Dict]:
        """
        Специализированная детекция курения
        """
        violations = self.detect_violations(frame)
        return [v for v in violations if v['type'] == 'smoking']
    
    def detect_littering(self, frame: np.ndarray) -> List[Dict]:
        """
        Специализированная детекция выброса мусора
        """
        violations = self.detect_violations(frame)
        return [v for v in violations if v['type'] == 'littering']
    
    def detect_graffiti(self, frame: np.ndarray) -> List[Dict]:
        """
        Специализированная детекция граффити
        """
        violations = self.detect_violations(frame)
        return [v for v in violations if v['type'] == 'graffiti']
    
    def draw_detections(self, frame: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """
        Рисует bounding boxes на кадре
        
        Args:
            frame: Исходный кадр
            violations: Список нарушений
        
        Returns:
            Кадр с нарисованными bounding boxes
        """
        result_frame = frame.copy()
        
        for violation in violations:
            x1, y1, x2, y2 = violation['bbox']
            violation_type = violation['type']
            confidence = violation['confidence']
            
            # Цвета для разных типов нарушений
            colors = {
                'smoking': (0, 0, 255),      # Красный
                'littering': (0, 165, 255),  # Оранжевый
                'graffiti': (255, 0, 0)      # Синий
            }
            
            color = colors.get(violation_type, (0, 255, 0))
            
            # Рисование bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Текст с типом нарушения и уверенностью
            violation_names = {
                'smoking': 'Курение',
                'littering': 'Выброс мусора',
                'graffiti': 'Граффити'
            }
            violation_name = violation_names.get(violation_type, violation_type)
            label = f"{violation_name}: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_frame
