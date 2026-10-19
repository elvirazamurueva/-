"""
test_pinecone_connection.py — ручная диагностика подключения к Pinecone.

Запускайте ЯВНО:
    python test_pinecone_connection.py

Этот файл НЕ импортируется при импорте pinecone_manager.py,
чтобы избежать случайных сетевых проверок.
"""

import sys
import os

# Убеждаемся, что текущий каталог в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from pinecone_manager import PineconeManager


def main():
    """Проверка подключения к Pinecone и GenAPI эмбеддингам."""
    print("=" * 50)
    print("Диагностика подключения VPg05")
    print("=" * 50)

    config = load_config()

    # Валидация
    errors = config.validate()
    if errors:
        print("\n[ОШИБКИ КОНФИГУРАЦИИ]")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"\nИндекс: {config.pinecone_index_name}")
    print(f"Модель эмбеддингов: {config.embedding_model}")
    print(f"URL эмбеддингов: {config.genapi_embeddings_url}")
    print(f"URL чата: {config.genapi_url}")

    if config.pinecone_proxy_url:
        print(f"Pinecone прокси: ***настроен***")
    else:
        print("Pinecone прокси: не задан")

    # Проверка Pinecone
    print("\n[1/3] Подключение к Pinecone...")
    try:
        pm = PineconeManager(config)
        print("  OK — Pinecone подключён")
    except Exception as exc:
        print(f"  ОШИБКА: {exc}")
        sys.exit(1)

    # Статистика
    print("\n[2/3] Статистика индекса...")
    try:
        stats = pm.get_index_stats()
        print(f"  OK — размерность={stats.get('dimension', '?')}, "
              f"векторов={stats.get('total_vector_count', '?')}")
    except Exception as exc:
        print(f"  ОШИБКА: {exc}")
        sys.exit(1)

    # Эмбеддинг
    print("\n[3/3] Создание тестового эмбеддинга...")
    try:
        emb = pm.create_embedding("Тестовое предложение для проверки")
        print(f"  OK — размерность={len(emb)}")
    except Exception as exc:
        print(f"  ОШИБКА: {exc}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Все проверки пройдены успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
