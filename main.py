"""
Главный модуль системы мониторинга нарушений
"""
import asyncio
import sys
from camera_monitor import CameraMonitor
from telegram_bot import TelegramNotifier
import config


def main():
    """Главная функция запуска системы"""
    print("=" * 60)
    print("🚨 СИСТЕМА МОНИТОРИНГА НАРУШЕНИЙ")
    print("=" * 60)
    print()
    
    # Проверка конфигурации Telegram
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("⚠️ ВНИМАНИЕ: Telegram бот не настроен!")
        print("   Создайте файл .env и укажите TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")
        print("   Система будет работать, но уведомления не будут отправляться")
        print()
    
    # Тестовая отправка сообщения в Telegram
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        print("📤 Отправка тестового сообщения в Telegram...")
        notifier = TelegramNotifier()
        asyncio.run(notifier.send_test_message())
        print()
    
    # Запуск мониторинга
    print("🎬 Запуск мониторинга камеры...")
    print("   Нажмите 'q' для выхода")
    print()
    
    monitor = CameraMonitor()
    
    try:
        monitor.start_monitoring()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
