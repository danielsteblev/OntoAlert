# Настройка Telegram бота

## Шаг 1: Получение токена бота

1. Откройте Telegram
2. Найдите бота **@BotFather** (официальный бот для создания ботов)
3. Отправьте команду `/newbot`
4. Следуйте инструкциям:
   - Придумайте имя бота (например: "Violation Monitor Bot")
   - Придумайте username бота (должен заканчиваться на `bot`, например: `violation_monitor_bot`)
5. BotFather даст вам токен в формате: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
6. **Сохраните этот токен!** Он понадобится для настройки

## Шаг 2: Получение Chat ID

Есть несколько способов:

### Способ 1: Через @userinfobot
1. Найдите бота **@userinfobot** в Telegram
2. Отправьте ему любое сообщение
3. Бот покажет ваш Chat ID (число, например: `123456789`)

### Способ 2: Через API Telegram
1. Сначала напишите любое сообщение вашему боту (если еще не писали)
2. Откройте в браузере: `https://api.telegram.org/bot<ВАШ_ТОКЕН>/getUpdates`
   (замените `<ВАШ_ТОКЕН>` на токен, полученный от BotFather)
3. Найдите в ответе поле `"chat":{"id":123456789}` - это и есть ваш Chat ID

## Шаг 3: Создание файла .env

1. Создайте файл `.env` в корне проекта (в той же папке, где находится `main.py`)
2. Добавьте в него следующее содержимое:

```env
# Telegram Bot настройки
TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather
TELEGRAM_CHAT_ID=ваш_chat_id

# Настройки камеры (опционально)
CAMERA_SOURCE=0
FRAME_RATE=30
RESOLUTION_WIDTH=1920
RESOLUTION_HEIGHT=1080

# Настройки детекции (опционально)
CONFIDENCE_THRESHOLD=0.5
DETECTION_INTERVAL=5
```

**Важно:** Замените `ваш_токен_от_BotFather` и `ваш_chat_id` на реальные значения!

## Пример заполненного .env:

```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
CAMERA_SOURCE=0
FRAME_RATE=30
RESOLUTION_WIDTH=1920
RESOLUTION_HEIGHT=1080
CONFIDENCE_THRESHOLD=0.5
DETECTION_INTERVAL=5
```

## Готово!

После создания файла `.env` с правильными значениями, вы можете запустить тестовую версию:

```bash
python test_demo.py
```

Система автоматически прочитает настройки из файла `.env` и отправит тестовое уведомление в Telegram.
