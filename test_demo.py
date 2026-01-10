"""
Демонстрационный скрипт для тестирования системы без камеры
"""
import asyncio
from ontology import ViolationOntology
from telegram_bot import TelegramNotifier


async def demo_system():
    """Демонстрация работы системы"""
    print("=" * 60)
    print("🧪 ДЕМОНСТРАЦИЯ СИСТЕМЫ МОНИТОРИНГА НАРУШЕНИЙ")
    print("=" * 60)
    print()
    
    # Инициализация компонентов
    ontology = ViolationOntology()
    notifier = TelegramNotifier()
    
    # Демонстрация онтологии
    print("📚 Демонстрация онтологии нарушений:")
    print()
    
    violations_to_test = [
        ('smoking', {'public_place': True}),
        ('littering', {}),
        ('graffiti', {'historical_object': True}),
        ('smoking', {'is_repeat': True}),
    ]
    
    for violation_type, context in violations_to_test:
        violation = ontology.classify_violation(
            violation_type=violation_type,
            location="Тестовая локация",
            context=context
        )
        
        print(f"Тип: {violation.violation_type}")
        print(f"Описание: {violation.description}")
        print(f"Статья: {violation.article}")
        print(f"Штраф: {violation.fine_amount} {violation.fine_currency}")
        print(f"Категория: {violation.category}")
        print(f"Серьезность: {violation.severity}")
        print("-" * 60)
    
    print()
    print("📱 Отправка тестового уведомления в Telegram...")
    
    # Создание тестового нарушения
    test_violation = ontology.classify_violation(
        violation_type='smoking',
        location="Демонстрационная камера",
        context={'public_place': True}
    )
    
    # Отправка уведомления
    await notifier.send_violation_notification(test_violation)
    
    print()
    print("✅ Демонстрация завершена!")


if __name__ == "__main__":
    asyncio.run(demo_system())
