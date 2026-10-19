"""
conftest.py — общие фикстуры для тестов.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_env():
    """Сбрасывает переменные окружения перед каждым тестом."""
    for key in [
        "GENAPI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "TELEGRAM_BOT_TOKEN",
        "GENAPI_URL",
        "MODEL_ID",
        "GENAPI_EMBEDDINGS_URL",
        "EMBEDDING_MODEL",
        "PINECONE_PROXY_URL",
        "PROXY_URL",
        "TIMEOUT_GENAPI",
        "TIMEOUT_PINECONE",
    ]:
        os.environ.pop(key, None)


@pytest.fixture
def mock_env_file(tmp_path: Path):
    """Создаёт .env файл для тестов и возвращает путь."""
    env_content = """
GENAPI_API_KEY=test_genapi_key
PINECONE_API_KEY=test_pinecone_key
PINECONE_INDEX_NAME=test_index
TELEGRAM_BOT_TOKEN=test_bot_token
GENAPI_URL=https://proxy.gen-api.ru/v1/chat/completions
MODEL_ID=gpt-4o-mini
GENAPI_EMBEDDINGS_URL=https://api.gen-api.ru/api/v1/networks/embeddings
EMBEDDING_MODEL=text-embedding-3-small
PINECONE_PROXY_URL=
PROXY_URL=
TIMEOUT_GENAPI=30
TIMEOUT_PINECONE=30
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    return env_file


@pytest.fixture
def mock_env_with_proxy(tmp_path: Path):
    """Создаёт .env с прокси для тестов."""
    env_content = """
GENAPI_API_KEY=test_genapi_key
PINECONE_API_KEY=test_pinecone_key
PINECONE_INDEX_NAME=test_index
TELEGRAM_BOT_TOKEN=test_bot_token
GENAPI_URL=https://proxy.gen-api.ru/v1/chat/completions
MODEL_ID=gpt-4o-mini
GENAPI_EMBEDDINGS_URL=https://api.gen-api.ru/api/v1/networks/embeddings
EMBEDDING_MODEL=text-embedding-3-small
PINECONE_PROXY_URL=
PROXY_URL=user:pass@proxy.example.com:8080
TIMEOUT_GENAPI=30
TIMEOUT_PINECONE=30
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    return env_file


@pytest.fixture
def mock_genapi_embedding_response():
    """
    Мок-ответ GenAPI для эмбеддингов в синхронном режиме (is_sync=True).

    Реальный контракт GenAPI:
    {"request_id": ..., "response": {"data": [{"embedding": [...]}]}}
    """
    embedding = [0.0] * 1536
    embedding[0] = 0.5
    return {
        "request_id": 55998604,
        "organization_id": 1,
        "model": "embeddings",
        "cost": 0.0003,
        "response": {
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                }
            ],
        },
    }


@pytest.fixture
def mock_genapi_success_response(mock_genapi_embedding_response):
    """Возвращает готовый MagicMock для requests.post (успешный ответ)."""
    return MagicMock(
        status_code=200,
        json=lambda: mock_genapi_embedding_response,
    )


@pytest.fixture
def mock_genapi_chat_response():
    """Возвращает мок-ответ GenAPI для чата."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Это тестовый ответ от GenAPI.",
                },
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture
def mock_pinecone_index():
    """Возвращает мок Pinecone Index."""
    mock = MagicMock()
    mock.query.return_value = {"matches": [], "usage": {"read_units": 1}}
    mock.upsert.return_value = {"upserted_count": 1}
    mock.fetch.return_value = {"vectors": {}}
    mock.delete.return_value = {}
    mock.update.return_value = None
    mock.describe_index_stats.return_value = {
        "dimension": 1536,
        "index_fullness": 0.0,
        "namespaces": {},
        "total_vector_count": 0,
    }
    return mock


@pytest.fixture
def mock_pinecone_client(mock_pinecone_index):
    """Возвращает мок Pinecone Client."""
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_pinecone_index
    return mock_client
