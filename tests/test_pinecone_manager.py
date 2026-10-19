"""
test_pinecone_manager.py — тесты менеджера векторной памяти.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
import requests

from config import Config
from pinecone_manager import MemoryResult, PineconeManager


@pytest.fixture
def config():
    """Базовая конфигурация для тестов."""
    return Config(
        genapi_api_key="test_key",
        pinecone_api_key="pc_key",
        pinecone_index_name="test_index",
        telegram_bot_token="bot_token",
        timeout_genapi=30,
        timeout_pinecone=30,
        embedding_dim=1536,
    )


@pytest.fixture
def manager(mock_pinecone_client, mock_genapi_embedding_response, config):
    """PineconeManager с моками."""
    with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
        with patch(
            "pinecone_manager.requests.post",
            return_value=MagicMock(
                json=lambda: mock_genapi_embedding_response,
                status_code=200,
            ),
        ):
            pm = PineconeManager(config)
            pm._client = mock_pinecone_client
            pm._index = mock_pinecone_client.Index.return_value
            yield pm


# ------------------------------------------------------------------
# Эмбеддинги
# ------------------------------------------------------------------


class TestCreateEmbedding:
    """Тесты создания эмбеддингов."""

    def test_success(self, manager, mock_genapi_embedding_response):
        """Успешное создание эмбеддинга."""
        with patch(
            "pinecone_manager.requests.post",
            return_value=MagicMock(
                json=lambda: mock_genapi_embedding_response,
                status_code=200,
            ),
        ):
            emb = manager.create_embedding("тест")
            assert len(emb) == 1536
            assert emb[0] == 0.5

    def test_wrong_dimension(self, config, mock_pinecone_client):
        """Эмбеддинг неверной размерности."""
        wrong = {"data": [{"embedding": [0.1] * 100}]}
        with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
            with patch(
                "pinecone_manager.requests.post",
                return_value=MagicMock(
                    json=lambda: wrong,
                    status_code=200,
                ),
            ):
                pm = PineconeManager(config)
                with pytest.raises(RuntimeError, match="Размерность"):
                    pm.create_embedding("тест")

    def test_missing_embedding_field(self, config, mock_pinecone_client):
        """В ответе отсутствует embedding."""
        bad = {"data": [{"index": 0}]}
        with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
            with patch(
                "pinecone_manager.requests.post",
                return_value=MagicMock(
                    json=lambda: bad,
                    status_code=200,
                ),
            ):
                pm = PineconeManager(config)
                with pytest.raises(RuntimeError, match="отсутствует поле"):
                    pm.create_embedding("тест")

    def test_timeout(self, config, mock_pinecone_client):
        """Таймаут при создании эмбеддинга."""
        with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
            with patch(
                "pinecone_manager.requests.post",
                side_effect=requests.exceptions.Timeout("timeout"),
            ):
                pm = PineconeManager(config)
                with pytest.raises(RuntimeError, match="Таймаут"):
                    pm.create_embedding("тест")

    def test_invalid_json(self, config, mock_pinecone_client):
        """Некорректный JSON в ответе."""
        with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
            with patch(
                "pinecone_manager.requests.post",
                return_value=MagicMock(
                    json=MagicMock(side_effect=ValueError("bad json")),
                    status_code=200,
                ),
            ):
                pm = PineconeManager(config)
                with pytest.raises(RuntimeError, match="Некорректный JSON"):
                    pm.create_embedding("тест")


# ------------------------------------------------------------------
# Запись
# ------------------------------------------------------------------


class TestStoreDocument:
    """Тесты сохранения документов."""

    def test_new_document_inserted(self, manager, config, mock_genapi_embedding_response):
        """Сохранение нового документа (action=inserted)."""
        # Поиск не находит похожих
        manager._index.query.return_value = {"matches": []}

        result = manager.store_document(
            text="Мой кот любит рыбу",
            user_id="12345",
            namespace="12345",
        )

        assert result.action == "inserted"
        assert result.similarity_score == 0.0
        assert result.existing_id is None
        assert result.record_id is not None
        assert manager._index.upsert.called

    def test_similar_document_found(self, manager, config, mock_genapi_embedding_response):
        """Похожая запись с тем же объектом и полярностью — skipped (перефраз)."""
        # Поиск находит похожую запись
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "existing-1",
                    "score": 0.85,
                    "metadata": {"text": "Я люблю рыбу", "user_id": "12345"},
                }
            ]
        }

        result = manager.store_document(
            text="Мой кот обожает рыбу",
            user_id="12345",
            namespace="12345",
        )

        assert result.action == "skipped"
        assert result.similarity_score == 0.85
        assert result.existing_id == "existing-1"
        assert not manager._index.update.called

    def test_duplicate_skipped(self, manager, config, mock_genapi_embedding_response):
        """Дубликат — текст идентичен (action=skipped)."""
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "existing-2",
                    "score": 0.95,
                    "metadata": {"text": "Мой кот любит рыбу", "user_id": "12345"},
                }
            ]
        }

        result = manager.store_document(
            text="Мой кот любит рыбу",
            user_id="12345",
            namespace="12345",
        )

        assert result.action == "skipped"
        assert result.existing_id == "existing-2"
        assert not manager._index.update.called

    def test_low_similarity_inserted(self, manager, config, mock_genapi_embedding_response):
        """Низкое сходство — создаём новую запись."""
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "old-1",
                    "score": 0.3,
                    "metadata": {"text": "Я не люблю рыбу", "user_id": "12345"},
                }
            ]
        }

        result = manager.store_document(
            text="Мой кот обожает рыбу",
            user_id="12345",
            namespace="12345",
        )

        # 0.3 < 0.5 — новая запись
        assert result.action == "inserted"

    def test_custom_threshold(self, config, mock_pinecone_client, mock_genapi_embedding_response):
        """Пользовательский порог сходства."""
        with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
            with patch(
                "pinecone_manager.requests.post",
                return_value=MagicMock(
                    json=lambda: mock_genapi_embedding_response,
                    status_code=200,
                ),
            ):
                pm = PineconeManager(config, similarity_threshold=0.9)
                pm._index = mock_pinecone_client.Index.return_value
                pm._index.query.return_value = {
                    "matches": [
                        {
                            "id": "existing-3",
                            "score": 0.7,
                            "metadata": {"text": "Рыба", "user_id": "12345"},
                        }
                    ]
                }

                # 0.7 < 0.9 — новая запись
                result = pm.store_document(
                    text="Мой кот любит рыбу",
                    user_id="12345",
                    namespace="12345",
                )
                assert result.action == "inserted"


# ------------------------------------------------------------------
# Изоляция пользователей
# ------------------------------------------------------------------


class TestUserIsolation:
    """Тесты изоляции памяти разных пользователей."""

    def test_different_users_not_merged(
        self, manager, config, mock_genapi_embedding_response
    ):
        """Воспоминания разных пользователей не объединяются."""
        # Пользователь 11111 — похожая запись, но разные объекты (собак vs котиков)
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "rec-1",
                    "score": 0.9,
                    "metadata": {"text": "Я люблю собак", "user_id": "11111"},
                }
            ]
        }

        result_1 = manager.store_document(
            text="Я люблю котиков",
            user_id="11111",
            namespace="11111",
        )

        # Пользователь 22222 — тот же текст, другой namespace
        manager._index.query.return_value = {"matches": []}

        result_2 = manager.store_document(
            text="Я люблю котиков",
            user_id="22222",
            namespace="22222",
        )

        # 11111: разные объекты → inserted; 22222: нет записей → inserted
        assert result_1.action == "inserted"
        assert result_2.action == "inserted"
        # namespace должен быть уникален
        calls = manager._index.upsert.call_args_list
        assert calls[-1][1]["namespace"] == "22222"

    def test_negative_meaning_preserved(self, config, mock_pinecone_client, mock_genapi_embedding_response):
        """Отрицания не приводят к объединению."""
        with patch("pinecone_manager.pinecone.Pinecone", return_value=mock_pinecone_client):
            with patch(
                "pinecone_manager.requests.post",
                return_value=MagicMock(
                    json=lambda: mock_genapi_embedding_response,
                    status_code=200,
                ),
            ):
                pm = PineconeManager(config)
                pm._index = mock_pinecone_client.Index.return_value

                # Текст с отрицанием — низкое сходство с позитивным
                pm._index.query.return_value = {
                    "matches": [
                        {
                            "id": "rec-pos",
                            "score": 0.45,
                            "metadata": {"text": "Я люблю рыбу", "user_id": "999"},
                        }
                    ]
                }

                result = pm.store_document(
                    text="Я не люблю рыбу",
                    user_id="999",
                    namespace="999",
                )

                # 0.45 < 0.5 — новая запись
                assert result.action == "inserted"


# ------------------------------------------------------------------
# Поиск
# ------------------------------------------------------------------


class TestSearch:
    """Тесты поиска."""

    def test_search_by_text(self, manager, mock_genapi_embedding_response):
        """Поиск по тексту."""
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "rec-1",
                    "score": 0.8,
                    "metadata": {"text": "Рыба", "user_id": "123"},
                }
            ]
        }
        results = manager.search_by_text("рыба", namespace="123")
        assert len(results) == 1
        assert results[0]["id"] == "rec-1"

    def test_search_by_vector(self, manager):
        """Поиск по вектору."""
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "rec-2",
                    "score": 0.7,
                    "metadata": {"text": "Мяу", "user_id": "123"},
                }
            ]
        }
        vector = [0.0] * 1536
        results = manager.search_by_vector(vector, namespace="123")
        assert len(results) == 1

    def test_search_empty(self, manager):
        """Пустой результат поиска."""
        manager._index.query.return_value = {"matches": []}
        results = manager.search_by_text("неизвестное", namespace="123")
        assert results == []

    def test_search_with_filter(self, manager):
        """Поиск с фильтром."""
        manager._index.query.return_value = {"matches": []}
        vector = [0.0] * 1536
        manager.search_by_vector(vector, namespace="123", filter_dict={"user_id": "456"})
        call_kwargs = manager._index.query.call_args[1]
        assert call_kwargs["filter"] == {"user_id": "456"}


# ------------------------------------------------------------------
# Статистика / метаданные / удаление
# ------------------------------------------------------------------


class TestStatsAndDelete:
    """Тесты статистики и удаления."""

    def test_get_stats(self, manager):
        """Получение статистики индекса."""
        stats = manager.get_index_stats()
        assert "dimension" in stats

    def test_update_metadata(self, manager):
        """Обновление метаданных."""
        manager.update_metadata("rec-1", {"new_key": "value"}, namespace="ns")
        manager._index.update.assert_called_once()

    def test_delete_by_id(self, manager):
        """Удаление по ID."""
        manager.delete_by_id(["id1", "id2"], namespace="ns")
        manager._index.delete.assert_called_once()

    def test_delete_by_filter(self, manager):
        """Удаление по фильтру."""
        manager.delete_by_filter({"user_id": "123"}, namespace="ns")
        manager._index.delete.assert_called_once()


# ------------------------------------------------------------------
# Проверка похожих записей
# ------------------------------------------------------------------


class TestCheckSimilar:
    """Тесты проверки похожих записей."""

    def test_similar_found(self, manager, config, mock_genapi_embedding_response):
        """Похожая запись найдена."""
        manager._index.query.return_value = {
            "matches": [
                {
                    "id": "rec-1",
                    "score": 0.75,
                    "metadata": {"text": "Рыба", "user_id": "123"},
                }
            ]
        }
        result = manager.check_similar("Я люблю рыбу", namespace="123")
        assert result["found"] is True
        assert result["score"] == 0.75
        assert result["id"] == "rec-1"

    def test_not_found(self, manager, mock_genapi_embedding_response):
        """Похожая запись не найдена."""
        manager._index.query.return_value = {"matches": []}
        result = manager.check_similar("Новая тема", namespace="123")
        assert result["found"] is False
        assert result["id"] is None
