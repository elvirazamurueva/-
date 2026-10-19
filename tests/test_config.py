"""
test_config.py — тесты модуля конфигурации.
"""

import os

import pytest

from config import Config, load_config


class TestConfigLoad:
    """Тесты загрузки конфигурации."""

    def test_load_minimal(self, mock_env_file):
        """Загрузка из .env файла."""
        config = load_config(str(mock_env_file))
        assert config.genapi_api_key == "test_genapi_key"
        assert config.pinecone_api_key == "test_pinecone_key"
        assert config.pinecone_index_name == "test_index"
        assert config.telegram_bot_token == "test_bot_token"

    def test_load_defaults(self, _clear_env, tmp_path):
        """Загрузка без .env — все значения по умолчанию."""
        # Передаём заведомо несуществующий путь, чтобы load_dotenv
        # не нашёл реальный .env файл проекта
        config = load_config(env_path=str(tmp_path / "nonexistent.env"))
        assert config.genapi_api_key == ""
        assert config.genapi_url == "https://proxy.gen-api.ru/v1/chat/completions"
        assert config.model_id == "gpt-4o-mini"
        assert config.timeout_genapi == 30
        assert config.timeout_pinecone == 30
        assert config.similarity_threshold == 0.5

    def test_load_env_vars(self, _clear_env):
        """Загрузка из переменных окружения."""
        os.environ["GENAPI_API_KEY"] = "env_key"
        os.environ["PINECONE_API_KEY"] = "env_pc_key"
        os.environ["PINECONE_INDEX_NAME"] = "env_index"
        os.environ["TELEGRAM_BOT_TOKEN"] = "env_token"
        os.environ["TIMEOUT_GENAPI"] = "60"
        os.environ["SIMILARITY_THRESHOLD"] = "0.85"

        config = load_config()
        assert config.genapi_api_key == "env_key"
        assert config.timeout_genapi == 60
        assert config.similarity_threshold == 0.85


class TestConfigValidate:
    """Тесты валидации конфигурации."""

    def test_all_missing(self):
        """Все обязательные поля отсутствуют."""
        config = Config()
        errors = config.validate()
        assert len(errors) == 4

    def test_all_present(self):
        """Все обязательные поля заданы."""
        config = Config(
            genapi_api_key="key",
            pinecone_api_key="pc_key",
            pinecone_index_name="idx",
            telegram_bot_token="bot_token",
        )
        errors = config.validate()
        assert errors == []

    def test_partial_missing(self):
        """Часть полей отсутствует."""
        config = Config(
            genapi_api_key="key",
            pinecone_api_key="pc_key",
        )
        errors = config.validate()
        assert len(errors) == 2


class TestConfigProxy:
    """Тесты работы с прокси."""

    def test_no_proxy(self):
        """PROXY_URL пустой."""
        config = Config(proxy_url="")
        assert config.get_telegram_proxy_url() is None

    def test_no_scheme_added(self):
        """Добавление http:// если схема отсутствует."""
        config = Config(proxy_url="user:pass@host:8080")
        assert config.get_telegram_proxy_url() == "http://user:pass@host:8080"

    def test_scheme_present(self):
        """Если схема уже есть — не добавляем."""
        config = Config(proxy_url="http://user:pass@host:8080")
        assert config.get_telegram_proxy_url() == "http://user:pass@host:8080"

    def test_socks_scheme(self):
        """Поддержка socks5 схемы."""
        config = Config(proxy_url="socks5://user:pass@host:1080")
        assert config.get_telegram_proxy_url() == "socks5://user:pass@host:1080"

    def test_pinecone_proxy(self):
        """PINECONE_PROXY_URL."""
        config = Config(pinecone_proxy_url="http://proxy:8080")
        assert config.get_pinecone_proxy() == {
            "https": "http://proxy:8080",
            "http": "http://proxy:8080",
        }

    def test_pinecone_proxy_no_scheme(self):
        """PINECONE_PROXY_URL без схемы."""
        config = Config(pinecone_proxy_url="proxy:8080")
        assert config.get_pinecone_proxy() == {
            "https": "http://proxy:8080",
            "http": "http://proxy:8080",
        }
