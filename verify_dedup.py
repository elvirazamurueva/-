"""
verify_dedup.py — ручная валидация умной фильтрации дубликатов.

Шаги:
1. Удаляет тестовый индекс Pinecone (имя из .env).
2. Пересоздаёт его с той же спецификацией (dim=1536, metric=cosine).
3. Сохраняет две схожие по смыслу фразы через store_document.
4. Проверяет, что вторая дала action=skipped или updated.

Запуск:
    python verify_dedup.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pinecone

from config import load_config
from pinecone_manager import PineconeManager


def main() -> None:
    config = load_config()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"[ОШИБКА] {err}")
        sys.exit(1)

    proxy_url = config.pinecone_proxy_url.strip() or None
    if proxy_url and not proxy_url.startswith(("http://", "https://")):
        proxy_url = "http://" + proxy_url

    kwargs = {"api_key": config.pinecone_api_key}
    if proxy_url:
        kwargs["proxy_url"] = proxy_url
    client = pinecone.Pinecone(**kwargs)

    index_name = config.pinecone_index_name
    print(f"[1/4] Текущая конфигурация индекса: {index_name}")

    # Запоминаем спецификацию до удаления
    old_spec = None
    try:
        old_desc = client.describe_index(name=index_name)
        old_spec = old_desc.spec
        print("      индекс существует, spec получен")
    except Exception as exc:
        print(f"      индекс не найден ({type(exc).__name__}), создадим новый")

    # 1. Удаление индекса
    print(f"[2/4] Удаление индекса '{index_name}'...")
    try:
        client.delete_index(name=index_name)
        # Ждём полного удаления
        for _ in range(60):
            if not client.has_index(name=index_name):
                break
            time.sleep(2)
        print("      индекс удалён")
    except Exception as exc:
        print(f"      не был удалён/не существовал: {type(exc).__name__}")

    # 2. Пересоздание индекса
    print(f"[3/4] Создание индекса '{index_name}' (dim=1536, metric=cosine)...")
    spec = None
    if old_spec is not None:
        try:
            spec = type(old_spec)(
                cloud=old_spec.cloud,
                region=old_spec.region,
            )
        except Exception:
            spec = None
    if spec is None:
        spec = pinecone.ServerlessSpec(cloud="aws", region="us-east-1")

    client.create_index(
        name=index_name,
        dimension=config.embedding_dim,
        metric="cosine",
        spec=spec,
    )
    # Ждём готовности
    for _ in range(90):
        desc = client.describe_index(name=index_name)
        if desc.status and desc.status.ready:
            break
        time.sleep(2)
    print("      индекс создан и готов")

    # 3. Две схожие фразы через менеджер
    print("[4/4] Проверка дедупликации...")
    pm = PineconeManager(config)
    user_id = "verify-user"
    namespace = user_id

    phrase_1 = "Хочу на Марс"
    phrase_2 = "Я полечу на Марс"

    r1 = pm.store_document(text=phrase_1, user_id=user_id, namespace=namespace)
    print(f"      «{phrase_1}» -> action={r1.action}, similarity={r1.similarity_score:.3f}")

    # Ждём, пока вектор станет доступен для query
    time.sleep(3)

    r2 = pm.store_document(text=phrase_2, user_id=user_id, namespace=namespace)
    print(f"      «{phrase_2}» -> action={r2.action}, similarity={r2.similarity_score:.3f}, "
          f"reason={r2.reason}")

    # 4. Итог
    print()
    if r1.action == "inserted" and r2.action in ("updated", "skipped"):
        print(f"УСПЕХ: вторая схожая фраза обработана как дубликат "
              f"(action={r2.action}).")
    else:
        print(f"РЕЗУЛЬТАТ: r1={r1.action}, r2={r2.action} — "
              "проверьте порог SIMILARITY_THRESHOLD и эмбеддинги.")
        sys.exit(2)


if __name__ == "__main__":
    main()
