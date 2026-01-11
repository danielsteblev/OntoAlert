
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config

try:
    from rdflib import Graph, Namespace
except Exception:  # pragma: no cover
    Graph = None  # type: ignore
    Namespace = None  # type: ignore


@dataclass
class Violation:
    """Класс для представления нарушения"""
    violation_type: str
    article: str
    description: str
    fine_amount: float
    fine_currency: str = "RUB"
    confidence: Optional[float] = None
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
        Инициализация базы знаний о нарушениях.
        Используется ТОЛЬКО RDF/TTL онтология (без встроенного словаря).
        """
        ttl_path = Path(getattr(config, "ONTOLOGY_TTL_PATH", "violations_ontology.ttl"))
        if Graph is None or Namespace is None:
            raise RuntimeError(
                "rdflib is required to load RDF ontology. Install dependencies and retry."
            )

        if not ttl_path.exists():
            raise FileNotFoundError(
                f"Ontology TTL file not found: {ttl_path}. "
                "Set ONTOLOGY_TTL_PATH in .env or place violations_ontology.ttl in project root."
            )

        return self._load_from_ttl(ttl_path)

    def _load_from_ttl(self, ttl_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Загружает нарушения из Turtle файла.
        Ожидаются индивиды с local-name, совпадающим с violation_type (smoking/littering/graffiti).
        """
        g = Graph()
        g.parse(str(ttl_path), format="turtle")

        VMS = Namespace("http://example.org/vms#")

        def one_str(s, p, default: str = "") -> str:
            v = next(iter(g.objects(s, p)), None)
            return str(v) if v is not None else default

        def one_float(s, p, default: float = 0.0) -> float:
            v = next(iter(g.objects(s, p)), None)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        db: Dict[str, Dict[str, Any]] = {}

        for s in g.subjects(predicate=None, object=None):
            s_str = str(s)
            if not s_str.startswith(str(VMS)):
                continue

            key = s_str.split("#", 1)[-1].strip()
            if not key:
                continue

            article = one_str(s, VMS.hasArticle)
            desc = one_str(s, VMS.hasDescription)
            fine_amount = one_float(s, VMS.hasFineAmount)
            fine_currency = one_str(s, VMS.hasCurrency, default="RUB")
            category = one_str(s, VMS.hasCategory)
            severity = one_str(s, VMS.hasSeverity, default="medium")

            if not article and not desc and fine_amount == 0.0:
                continue

            db[key] = {
                "article": article,
                "description": desc,
                "fine_amount": fine_amount,
                "fine_currency": fine_currency,
                "category": category,
                "severity": severity,
            }

        if not db:
            raise RuntimeError("TTL loaded, but no violation individuals were parsed")

        print(f"✅ RDF онтология загружена: {ttl_path} (types={len(db)})")
        return db
    
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
            confidence=float(context.get('confidence')) if context and context.get('confidence') is not None else None,
            timestamp=datetime.now(),
            location=location,
            category=violation_data.get('category', ''),
            severity=violation_data.get('severity', 'medium')
        )
    
    def get_violation_info(self, violation_type: str) -> Dict:
        """Получить информацию о типе нарушения"""
        return self.violations_db.get(violation_type, {})
