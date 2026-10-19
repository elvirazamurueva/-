## Диалоговый Telegram-бот с долговременной векторной памятью

## Описание

Учебный проект асинхронного Telegram-бота, который:

- Ведёт **долговременную векторную память** через **Pinecone**.
- Использует **GenAPI** для генерации ответов и создания эмбеддингов.
- Сохраняет только **текст пользователя** (не ответы бота).
- Имеет **изолированную память** для каждого Telegram-пользователя.
- Поддерживает **дедупликацию** записей по порогу сходства.
- Работает через **HTTP-прокси** для Telegram и Pinecone.

## Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌──────────┐
│ Telegram    │────▶│ Bot (PTB)   │────▶│ GenAPI   │
│ API         │◀────│ (bot.py)    │◀────│ (чат)    │
└─────────────┘     │             │     └──────────┘
                    │             │     ┌──────────┐
                    │             │────▶│ GenAPI   │
                    │             │◀────│ (embed)  │
                    │             │     └──────────┘
                    │             │     ┌──────────┐
                    │             │────▶│ Pinecone │
                    │             │◀────│ (memory) │
                    └─────────────┘     └──────────┘
```

**Ключевые файлы:**

| Файл | Назначение |
|------|------------|
| `config.py` | Загрузка и валидация .env |
| `pinecone_manager.py` | Векторная память (Pinecone + GenAPI) |
| `bot.py` | Telegram-бот (python-telegram-bot) |

## Требования

- Python 3.11+
- Аккаунт GenAPI (API-ключ)
- Аккаунт Pinecone (индекс)
- Аккаунт Telegram (токен бота от @BotFather)
- HTTP-прокси для Telegram (обязателен в РФ-сетях)

## Установка

### 1. Клонирование / подготовка

```bash
# Если из репозитория:
git clone <url>
cd <project-dir>

# Или в существующей папке:
```

### 2. Виртуальное окружение

**Bash / Git Bash:**
```bash
python -m venv .venv
source .venv/Scripts/activate   # или .venv/bin/activate на Linux
```

**PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка .env

```bash
# Bash:
cp .env.example .env

# PowerShell:
Copy-Item .env.example .env
```

Заполните `.env`:

```env
GENAPI_API_KEY=ваш_ключ_genapi
PINECONE_API_KEY=ваш_ключ_pinecone
PINECONE_INDEX_NAME=your_index_name
TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather

# Прокси для Telegram (обязателен в РФ)
PROXY_URL=user:password@host:port

# Необязательный прокси для Pinecone
PINECONE_PROXY_URL=

# Таймауты (по умолчанию 30)
TIMEOUT_GENAPI=30
TIMEOUT_PINECONE=30
```

## Создание индекса Pinecone

Индекс создаётся через веб-интерфейс или SDK:

1. Зайдите в [Pinecone Console](https://app.pinecone.io/).
2. Создайте новый Index:
   - Name: `your_index_name` (совпадает с `PINECONE_INDEX_NAME` в .env)
   - Metric: `cosine`
   - Dimensions: **1536** (для модели `text-embedding-3-small`)
   - Environment: выбирайте ближайшую

Или через Python:

```python
import pinecone
pc = pinecone.Pinecone(api_key="your_key")
pc.create_index(
    name="your_index_name",
    dimension=1536,
    metric="cosine",
    spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

## Запуск

```bash
# Bash:
python bot.py

# PowerShell:
python bot.py
```

Бот запустит polling и будет ждать сообщения.

## Тестирование

### Запуск автоматических тестов

```bash
pytest tests/ -v
```

### Диагностика подключения к Pinecone

```bash
python test_pinecone_connection.py
```

Файл **не импортируется** автоматически — его нужно запускать явно, чтобы избежать случайных сетевых проверок при импорте модулей.

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие и инструкция |

## Типы памяти

Бот автоматически определяет тип сообщения и сохраняет его в соответствующее хранилище:

| Тип | Триггеры | Хранение |
|-----|----------|----------|
| `preference` | «люблю», «нравится», «не люблю», «пробовал», «исправил» | Отдельные векторы с фильтром `type=preference` |
| `context` | «меню», «блюда», «позиции» | Отдельные векторы с фильтром `type=context` |
| `memory` | Всё остальное | Обычные векторы с фильтром `type=memory` |

### Исправление предпочтений

Если пользователь говорит «люблю фасоль», а потом «люблю не фасоль, а чечевицу»:
1. Найденное старое предпочтение с тем же `topic` помечается как противоречащее;
2. Старая запись удаляется из Pinecone;
3. Новая запись вставляется.

### Контекст меню

Меню сохраняется как `context` с `context_type=menu`. При ответе бот:
- использует только блюда из меню;
- если меню недоступно, прямо сообщает об этом;
- не придумывает отсутствующие блюда.

### Изоляция пользователей

Все данные изолированы по Telegram `user_id` через Pinecone `namespace` и фильтр `user_id` в метаданных.

## Безопасность

- Токены и ключи хранятся **только** в `.env` (не коммитится).
- Логи **не содержат** токены, ключи, полные URL прокси и полные тексты сообщений.
- Прокси-учётные данные маскируются в логах.

## Работа через прокси (РФ, без VPN)

python-telegram-bot использует **два раздельных HTTP-клиента**:

- `request` — отправка сообщений;
- `get_updates_request` — **polling** (получение обновлений).

Прокси обязан быть настроен для **обоих**, иначе бот «молчит»: отправка работает, а polling не может достучаться до `api.telegram.org`. В `bot.py` это учтено — оба клиента строятся из `PROXY_URL`.

Формат `PROXY_URL` в `.env`: `user:password@host:port` (без схемы — `http://` добавляется автоматически).

Проверка связности (вывод без секретов):

```bash
python check_telegram_proxy.py
```

Ожидаемый результат:

```
Проверяю прокси: ***:***@host:port
[1/2] getMe: OK — бот @your_bot
[2/2] getUpdates: OK — 0 необработанных обновлений
```

## Известные ограничения

1. **GenAPI эмбеддинги** используют собственный контракт:
   - Поле `input` должно быть массивом строк: `"input": ["текст"]`
   - Параметр `is_sync: true` включает синхронный режим (ожидание результата)
   - Ответ приходит в `{"response": {"data": [{"embedding": [...]}]}}`
   - URL: `https://api.gen-api.ru/api/v1/networks/embeddings`
2. **Pinecone Serverless** — индекс должен быть создан вручную.
3. **Нет бэкапа памяти** — данные хранятся только в Pinecone.
4. **Дедупликация** — порог сходства 0.5 по умолчанию; можно настроить в `PineconeManager`.
5. **Противоречия** — определяются эвристически (regex, ключевые слова), неsemantic analysis.
6. **Противоречие при исправлении** — старая запись удаляется из Pinecone автоматически.
7. **Память изолирована** по Telegram user_id — один пользователь не видит память другого.
8. **Не сохраняет ответы бота** — только оригинальные тексты пользователей.
9. **Защита от prompt injection** — сохранённые данные оборачиваются в XML-маркеры `<memories>`, `<preferences>`, `<dialog_context>`.

## Структура проекта

```
VPg05/
├── .env.example
├── .gitignore
├── README.md
├── bot.py
├── check_telegram_proxy.py
├── config.py
├── pinecone_manager.py
├── pytest.ini
├── requirements.txt
├── test_pinecone_connection.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_bot.py
    ├── test_config.py
    ├── test_memory_scenarios.py
    └── test_pinecone_manager.py
```

## Тестирование

```bash
# Все тесты (74 теста):
python -m pytest tests/ -v

# Только сценарии памяти:
python -m pytest tests/test_memory_scenarios.py -v

# Диагностика подключения (требует реальные ключи):
python test_pinecone_connection.py
python check_telegram_proxy.py
```
