"""
Telegram бот для отправки уведомлений о нарушениях
"""
import asyncio
from typing import Optional
from telegram import Bot
from telegram.constants import ParseMode
from pathlib import Path
import config
from ontology import Violation


class TelegramNotifier:
    """Класс для отправки уведомлений в Telegram"""
    
    def __init__(self):
        self.bot = None
        self.chat_id = config.TELEGRAM_CHAT_ID
        # Инициализируем бота только если токен настроен
        if config.TELEGRAM_BOT_TOKEN:
            try:
                self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
            except Exception as e:
                print(f"⚠️ Ошибка инициализации Telegram бота: {e}")
    
    async def send_violation_notification(self, violation: Violation, 
                                         image_path: Optional[str] = None):
        """
        Отправляет уведомление о нарушении в Telegram
        
        Args:
            violation: Объект Violation с информацией о нарушении
            image_path: Путь к изображению с нарушением
        """
        if not self.bot or not self.chat_id:
            print("⚠️ Telegram не настроен. Пропуск отправки уведомления.")
            return
        
        # Формирование сообщения
        message = self._format_violation_message(violation)
        
        try:
            # Отправка текстового сообщения
            if image_path and Path(image_path).exists():
                # Отправка с изображением
                with open(image_path, 'rb') as photo:
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=message,
                        parse_mode=ParseMode.HTML
                    )
            else:
                # Отправка только текста
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=ParseMode.HTML
                )
            
            print(f"✅ Уведомление отправлено в Telegram: {violation.violation_type}")
        
        except Exception as e:
            print(f"❌ Ошибка при отправке уведомления в Telegram: {e}")
    
    def _format_violation_message(self, violation: Violation) -> str:
        """
        Форматирует сообщение о нарушении для Telegram
        
        Args:
            violation: Объект Violation
        
        Returns:
            Отформатированное сообщение в HTML
        """
        timestamp_str = violation.timestamp.strftime("%Y-%m-%d %H:%M:%S") if violation.timestamp else "N/A"

        # По запросу: фиксированное местоположение для демонстрационного режима
        location_str = "Демонстрационная камера"
        probability_line = ""
        if violation.confidence is not None:
            probability_line = f"Вероятность: <b>{violation.confidence * 100:.1f}%</b>\n"

        message = (
            "🚨 Обнаружено нарушение 🚨\n"
            f"Тип нарушения: <b>{violation.description}</b>\n"
            f"{probability_line}"
            f"Статья КоАП: <b>{violation.article}</b>\n"
            f"Штраф: <b>{violation.fine_amount:.0f} {violation.fine_currency}</b>\n\n"
            f"Время: <b>{timestamp_str}</b>\n"
            f"Местоположение: <b>{location_str}</b>"
        )

        return message
    
    async def send_test_message(self):
        """Отправляет тестовое сообщение для проверки работы бота"""
        if not self.bot or not self.chat_id:
            print("⚠️ Telegram не настроен. Невозможно отправить тестовое сообщение.")
            return
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text="🤖 Бот успешно настроен и готов к работе!"
            )
            print("✅ Тестовое сообщение отправлено")
        except Exception as e:
            print(f"❌ Ошибка при отправке тестового сообщения: {e}")
