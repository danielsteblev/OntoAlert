"""
Обработчик команд Telegram бота для анализа фото на нарушения
"""
import asyncio
import cv2
import numpy as np
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError
from telegram.request import HTTPXRequest
import httpx
import config
from detector import ViolationDetector
from ontology import ViolationOntology


class BotHandler:
    """Класс для обработки команд и сообщений бота"""
    
    def __init__(self):
        # Если в .env задан YOLO_MODEL_PATH (best.pt), бот будет использовать вашу обученную модель
        self.detector = ViolationDetector(model_path=config.YOLO_MODEL_PATH or None)
        print("🧠 YOLO model for Telegram bot:")
        print(f"   source: {self.detector.model_source}")
        print(f"   custom: {self.detector.using_custom_model}")
        print(f"   names:  {self.detector.model_class_names}")
        self.ontology = ViolationOntology()
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        try:
            welcome_message = """
🤖 <b>Система мониторинга нарушений</b>

Отправьте фото для анализа на наличие нарушений:
• 🚭 Курение
• 🗑️ Выброс мусора
• 🎨 Граффити

Бот проанализирует изображение и покажет:
• Вероятность нарушения (в процентах)
• Статью КоАП РФ
• Размер штрафа

Просто отправьте фото! 📸
            """
            await update.message.reply_text(welcome_message, parse_mode='HTML', read_timeout=30, write_timeout=30, connect_timeout=30)
        except (TimedOut, NetworkError) as e:
            print(f"Ошибка отправки сообщения /start: {e}")
            # Пытаемся отправить простое сообщение без форматирования
            try:
                await update.message.reply_text("Привет! Отправьте фото для анализа.")
            except:
                pass
        except Exception as e:
            print(f"Неожиданная ошибка в /start: {e}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        try:
            help_message = """
📖 <b>Справка по использованию бота</b>

<b>Команды:</b>
/start - Начать работу с ботом
/help - Показать эту справку

<b>Как использовать:</b>
1. Отправьте фото через Telegram
2. Бот проанализирует изображение
3. Получите результат с вероятностью нарушения

<b>Что анализируется:</b>
• Курение в запрещенных местах
• Выброс мусора
• Граффити на стенах

<b>Примечание:</b>
Система использует эвристические правила на основе стандартной YOLO модели.
Для повышения точности рекомендуется использовать обученную модель.
            """
            await update.message.reply_text(help_message, parse_mode='HTML', read_timeout=30, write_timeout=30, connect_timeout=30)
        except (TimedOut, NetworkError) as e:
            print(f"Ошибка отправки сообщения /help: {e}")
            try:
                await update.message.reply_text("Справка: Отправьте фото для анализа нарушений.")
            except:
                pass
        except Exception as e:
            print(f"Неожиданная ошибка в /help: {e}")
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик получения фото"""
        processing_msg = None
        temp_path = None
        
        try:
            # Отправляем сообщение о начале обработки
            processing_msg = await update.message.reply_text(
                "🔍 Анализирую изображение...",
                read_timeout=30, write_timeout=30, connect_timeout=30
            )
            
            # Получаем фото (используем средний размер для ускорения)
            photo = update.message.photo[-2] if len(update.message.photo) > 1 else update.message.photo[-1]
            
            # Загружаем файл с retry логикой
            max_retries = 3
            file = None
            for attempt in range(max_retries):
                try:
                    file = await asyncio.wait_for(
                        context.bot.get_file(photo.file_id),
                        timeout=30.0
                    )
                    break
                except (TimedOut, httpx.ReadError, httpx.TimeoutException) as e:
                    if attempt < max_retries - 1:
                        await processing_msg.edit_text(
                            f"🔄 Попытка загрузки {attempt + 1}/{max_retries}...",
                            read_timeout=10, write_timeout=10
                        )
                        await asyncio.sleep(2)  # Ждем перед повтором
                    else:
                        raise
            
            # Сохраняем временный файл с retry логикой
            temp_path = self.temp_dir / f"{photo.file_id}.jpg"
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(
                        file.download_to_drive(temp_path),
                        timeout=60.0
                    )
                    break
                except (TimedOut, httpx.ReadError, httpx.TimeoutException) as e:
                    if attempt < max_retries - 1:
                        await processing_msg.edit_text(
                            f"🔄 Попытка скачивания {attempt + 1}/{max_retries}...",
                            read_timeout=10, write_timeout=10
                        )
                        await asyncio.sleep(2)
                    else:
                        raise
            
            # Обновляем сообщение
            await processing_msg.edit_text(
                "📥 Фото загружено, начинаю анализ...",
                read_timeout=30, write_timeout=30
            )
            
            # Загружаем изображение через OpenCV (BGR формат)
            image = cv2.imread(str(temp_path))
            if image is None:
                await processing_msg.edit_text(
                    "❌ Ошибка: Не удалось загрузить изображение",
                    read_timeout=30, write_timeout=30
                )
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                return
            
            # Оптимизируем размер изображения для ускорения обработки
            height, width = image.shape[:2]
            max_size = 1280  # Максимальный размер для обработки
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                await processing_msg.edit_text(
                    "🔄 Оптимизирую размер изображения...",
                    read_timeout=30, write_timeout=30
                )
            
            # Анализируем изображение в отдельной задаче с таймаутом
            await processing_msg.edit_text(
                "🔍 Запускаю детекцию объектов...",
                read_timeout=30, write_timeout=30
            )
            
            # Запускаем детекцию в executor для неблокирующей обработки
            # Включаем debug для диагностики
            loop = asyncio.get_event_loop()
            violations = await asyncio.wait_for(
                loop.run_in_executor(None, self.detector.detect_violations, image, True),  # debug=True для диагностики
                timeout=30.0  # Таймаут 30 секунд на детекцию
            )
            
            # Формируем ответ
            await processing_msg.edit_text(
                "📊 Формирую результат...",
                read_timeout=30, write_timeout=30
            )
            
            if violations:
                # Сводим нарушения по типу: берём максимум уверенности по каждому типу
                best_by_type = {}
                for v in violations:
                    t = v.get('type')
                    c = float(v.get('confidence', 0.0))
                    if t not in best_by_type or c > best_by_type[t]:
                        best_by_type[t] = c

                # Сортируем по уверенности и показываем топ-3, чтобы ответ был коротким
                ordered = sorted(best_by_type.items(), key=lambda x: x[1], reverse=True)[:3]

                lines = ["✅ Обнаружены нарушения:"]
                for vtype, conf in ordered:
                    violation_obj = self.ontology.classify_violation(
                        violation_type=vtype,
                        location="Загружено пользователем",
                        context={'confidence': conf}
                    )
                    probability_percent = conf * 100
                    lines.append(
                        f"- {violation_obj.description}: {probability_percent:.1f}% | {violation_obj.article} | {violation_obj.fine_amount:.0f} {violation_obj.fine_currency}"
                    )

                result_message = "\n".join(lines)

                try:
                    await processing_msg.edit_text(
                        result_message,
                        read_timeout=20, write_timeout=20
                    )
                except (TimedOut, httpx.ReadError, httpx.TimeoutException):
                    try:
                        await update.message.reply_text(
                            result_message,
                            read_timeout=20, write_timeout=20
                        )
                    except:
                        await update.message.reply_text(
                            "✅ Обнаружены нарушения (не удалось отправить полный отчёт).",
                            read_timeout=15, write_timeout=15
                        )
            else:
                # Нарушений не обнаружено - короткое сообщение
                result_message = "✅ Нарушений не обнаружено"
                try:
                    await processing_msg.edit_text(
                        result_message,
                        read_timeout=20, write_timeout=20
                    )
                except (TimedOut, httpx.ReadError, httpx.TimeoutException):
                    try:
                        await update.message.reply_text(
                            result_message,
                            read_timeout=15, write_timeout=15
                        )
                    except:
                        pass
            
            # Удаляем временный файл
            temp_path.unlink()
            
        except (asyncio.TimeoutError, TimedOut, httpx.ReadError, httpx.TimeoutException) as e:
            error_msg = """
⏱️ <b>Превышено время ожидания</b>

Обработка изображения заняла слишком много времени.
Попробуйте:
• Отправить фото меньшего размера
• Убедиться, что интернет-соединение стабильно
• Попробовать позже
            """
            if processing_msg:
                try:
                    await processing_msg.edit_text(
                        error_msg.strip(), 
                        parse_mode='HTML',
                        read_timeout=10, write_timeout=10
                    )
                except:
                    try:
                        await update.message.reply_text(
                            "⏱️ Обработка заняла слишком много времени. Попробуйте позже.",
                            read_timeout=10, write_timeout=10
                        )
                    except:
                        pass
            else:
                try:
                    await update.message.reply_text(
                        "⏱️ Обработка заняла слишком много времени. Попробуйте позже.",
                        read_timeout=10, write_timeout=10
                    )
                except:
                    pass
            print(f"Ошибка обработки фото: Timeout/ReadError - {type(e).__name__}")
            
        except (NetworkError, httpx.ConnectError) as e:
            error_msg = """
⏱️ <b>Проблема с соединением</b>

Не удалось отправить результат из-за проблем с сетью.
Попробуйте:
• Проверить интернет-соединение
• Отправить фото еще раз
• Попробовать позже
            """
            if processing_msg:
                try:
                    await processing_msg.edit_text(error_msg.strip(), parse_mode='HTML', 
                                                   read_timeout=10, write_timeout=10)
                except:
                    try:
                        await update.message.reply_text("⏱️ Проблема с соединением. Попробуйте позже.",
                                                       read_timeout=10, write_timeout=10)
                    except:
                        pass
            else:
                try:
                    await update.message.reply_text("⏱️ Проблема с соединением. Попробуйте позже.",
                                                   read_timeout=10, write_timeout=10)
                except:
                    pass
            print(f"Ошибка сети при обработке фото: {e}")
            
        except Exception as e:
            error_msg = f"❌ Ошибка при обработке изображения: {str(e)}"
            if processing_msg:
                try:
                    await processing_msg.edit_text(error_msg, read_timeout=10, write_timeout=10)
                except:
                    try:
                        await update.message.reply_text("❌ Произошла ошибка при обработке фото",
                                                       read_timeout=10, write_timeout=10)
                    except:
                        pass
            else:
                try:
                    await update.message.reply_text(error_msg, read_timeout=10, write_timeout=10)
                except:
                    pass
            print(f"Ошибка обработки фото: {e}")
            
        finally:
            # Удаляем временный файл
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
    
    def run(self):
        """Запуск бота"""
        if not config.TELEGRAM_BOT_TOKEN:
            print("❌ Ошибка: TELEGRAM_BOT_TOKEN не установлен в .env файле")
            return
        
        print("🤖 Запуск Telegram бота...")
        print("   Отправьте /start боту для начала работы")
        
        # Создаем HTTPXRequest с увеличенными таймаутами
        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30,
            pool_timeout=30
        )
        
        # Создаем приложение с настроенными таймаутами
        application = Application.builder()\
            .token(config.TELEGRAM_BOT_TOKEN)\
            .request(request)\
            .build()
        
        # Регистрируем обработчики
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        
        # Добавляем обработчик ошибок
        application.add_error_handler(self.error_handler)
        
        # Запускаем бота
        print("✅ Бот запущен и готов к работе!")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок бота"""
        error = context.error
        
        if isinstance(error, (TimedOut, httpx.ReadError, httpx.TimeoutException)):
            print(f"⚠️ Таймаут/ошибка чтения при обработке обновления: {error}")
            # Пытаемся отправить сообщение об ошибке, если есть update
            if isinstance(update, Update) and update.effective_message:
                try:
                    await update.effective_message.reply_text(
                        "⏱️ Превышено время ожидания или ошибка сети. Попробуйте позже.",
                        read_timeout=10,
                        write_timeout=10
                    )
                except:
                    pass
        elif isinstance(error, (NetworkError, httpx.ConnectError)):
            print(f"⚠️ Ошибка сети: {error}")
            if isinstance(update, Update) and update.effective_message:
                try:
                    await update.effective_message.reply_text(
                        "🌐 Проблема с интернет-соединением. Проверьте подключение.",
                        read_timeout=10,
                        write_timeout=10
                    )
                except:
                    pass
        else:
            print(f"❌ Неожиданная ошибка: {error}")
            import traceback
            traceback.print_exception(type(error), error, error.__traceback__)


if __name__ == "__main__":
    handler = BotHandler()
    handler.run()
