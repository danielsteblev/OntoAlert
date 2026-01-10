"""
Онтология для классификации нарушений и определения штрафов
"""
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Violation:
    """Класс для представления нарушения"""
    violation_type: str
    article: str
    description: str
    fine_amount: float
    fine_currency: str = "RUB"
    timestamp: Optional[datetime] = None
    location: Optional[str] = None
    evidence_image_path: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None


class ViolationOntology:

    def __init__(self):
        self.violations_db = self._initialize_violations_db()
    
    def _initialize_violations_db(self) -> Dict[str, Dict]:
        """
        Инициализация базы знаний о нарушениях
        В реальной системе это может быть загружено из RDF/OWL файла
        """
        return {
            'smoking': {
                'article': '6.24 КоАП РФ',
                'description': 'Курение в запрещенных местах',
                'fine_amount': 500.0,
                'fine_currency': 'RUB',
                'category': 'Административное правонарушение',
                'severity': 'low'
            },
            'littering': {
                'article': '8.1 КоАП РФ',
                'description': 'Выброс мусора в неположенном месте',
                'fine_amount': 1000.0,
                'fine_currency': 'RUB',
                'category': 'Административное правонарушение',
                'severity': 'medium'
            },
            'graffiti': {
                'article': '7.17 КоАП РФ',
                'description': 'Порча имущества (рисование граффити)',
                'fine_amount': 5000.0,
                'fine_currency': 'RUB',
                'category': 'Административное правонарушение',
                'severity': 'high'
            }
        }
    
    def classify_violation(self, violation_type: str, 
                          location: Optional[str] = None,
                          context: Optional[Dict] = None) -> Violation:
        """
        Классифицирует нарушение и возвращает полную информацию
        
        Args:
            violation_type: Тип нарушения (smoking, littering, graffiti)
            location: Местоположение нарушения
            context: Дополнительный контекст (может влиять на размер штрафа)
        
        Returns:
            Violation объект с полной информацией
        """
        if violation_type not in self.violations_db:
            raise ValueError(f"Неизвестный тип нарушения: {violation_type}")
        
        violation_data = self.violations_db[violation_type].copy()
        
        # Применение правил онтологии для корректировки штрафа
        fine_amount = violation_data['fine_amount']
        
        # Правила для корректировки штрафа на основе контекста
        if context:
            # Если нарушение повторное
            if context.get('is_repeat', False):
                fine_amount *= 2
            
            # Если нарушение в общественном месте
            if context.get('public_place', False) and violation_type == 'smoking':
                fine_amount *= 1.5
            
            # Если граффити на историческом объекте
            if violation_type == 'graffiti' and context.get('historical_object', False):
                fine_amount *= 3
        
        return Violation(
            violation_type=violation_type,
            article=violation_data['article'],
            description=violation_data['description'],
            fine_amount=fine_amount,
            fine_currency=violation_data['fine_currency'],
            timestamp=datetime.now(),
            location=location,
            category=violation_data.get('category', ''),
            severity=violation_data.get('severity', 'medium')
        )
    
    def get_violation_info(self, violation_type: str) -> Dict:
        """Получить информацию о типе нарушения"""
        return self.violations_db.get(violation_type, {})
