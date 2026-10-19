"""
config.py — загрузка и проверка конфигурации проекта VPg05.

Загружает переменные окружения из .env, проверяет обязательные параметры,
и предоставляет единый объект Config для всего приложения.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Конфигурация приложения."""

    # GenAPI — чат
    genapi_api_key: str = ""
    genapi_url: str = "https://proxy.gen-api.ru/v1/chat/completions"
    model_id: str = "gpt-4o-mini"

    # GenAPI — эмбеддинги
    genapi_embeddings_url: str = "https://api.gen-api.ru/api/v1/networks/embeddings"
    embedding_model: str = "text-embedding-3-small"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_proxy_url: str = ""

    # Telegram
    telegram_bot_token: str = ""
    proxy_url: str = ""

    # Таймауты
    timeout_genapi: int = 30
    timeout_pinecone: int = 30

    # Порог сходства для дедупликации
    similarity_threshold: float = 0.5

    # Размерность эмбеддингов
    embedding_dim: int = 1536

    def validate(self) -> list[str]:
        """Проверяет обязательные параметры. Возвращает список ошибок."""
        errors: list[str] = []

        if not self.genapi_api_key:
            errors.append("GENAPI_API_KEY не задан")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY не задан")
        if not self.pinecone_index_name:
            errors.append("PINECONE_INDEX_NAME не задан")
        if not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN не задан")

        return errors

    def get_telegram_proxy_url(self) -> Optional[str]:
        """
        Возвращает полный URL прокси для Telegram со схемой http://,
        или None если прокси не настроен.

        Формат PROXY_URL: user:password@host:port (без схемы)
        или user:password@host:port со схемой.
        """
        if not self.proxy_url:
            return None

        raw = self.proxy_url.strip()
        if not raw:
            return None

        # Добавляем схему http:// если отсутствует
        if not raw.startswith(("http://", "https://", "socks5://", "socks5h://")):
            raw = "http://" + raw

        return raw

    def get_pinecone_proxy(self) -> Optional[dict]:
        """
        Возвращает словарь прокси для Pinecone SDK или None.
        """
        if not self.pinecone_proxy_url:
            return None

        raw = self.pinecone_proxy_url.strip()
        if not raw:
            return None

        if not raw.startswith(("http://", "https://")):
            raw = "http://" + raw

        return {"https": raw, "http": raw}

    @staticmethod
    def _safe_log_proxy(raw: str) -> str:
        """Маскирует логин/пароль в URL прокси для логов."""
        if "@" in raw:
            scheme, rest = raw.rsplit("@", 1)
            if "://" in scheme:
                _, creds = scheme.split("://", 1)
                return f"***@{rest}"
            return f"***@{rest}"
        return raw


def load_config(env_path: Optional[str] = None) -> Config:
    """
    Загружает конфигурацию из .env файла и переменных окружения.

    Args:
        env_path: Необязательный путь к .env файлу.

    Returns:
        Объект Config.
    """
    # Определяем путь к .env
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Ищем .env рядом с этим файлом
        default_env = Path(__file__).parent / ".env"
        if default_env.exists():
            load_dotenv(dotenv_path=str(default_env), override=False)
        else:
            load_dotenv(override=False)

    config = Config(
        genapi_api_key=os.getenv("GENAPI_API_KEY", ""),
        genapi_url=os.getenv("GENAPI_URL", "https://proxy.gen-api.ru/v1/chat/completions"),
        model_id=os.getenv("MODEL_ID", "gpt-4o-mini"),
        genapi_embeddings_url=os.getenv(
            "GENAPI_EMBEDDINGS_URL",
            "https://api.gen-api.ru/api/v1/networks/embeddings"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", ""),
        pinecone_proxy_url=os.getenv("PINECONE_PROXY_URL", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        proxy_url=os.getenv("PROXY_URL", ""),
        timeout_genapi=int(os.getenv("TIMEOUT_GENAPI", "30")),
        timeout_pinecone=int(os.getenv("TIMEOUT_PINECONE", "30")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.5")),
    )

    return config
