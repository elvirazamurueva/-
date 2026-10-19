"""
pinecone_manager.py — менеджер векторной памяти на Pinecone.

Отвечает за:
- создание эмбеддингов через GenAPI;
- запись и поиск векторов в Pinecone;
- дедупликацию записей по порогу сходства;
- изоляцию памяти пользователей через namespace;
- отдельное хранение актуальных предпочтений;
- контекст диалога (меню и т.п.).
"""

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pinecone
import requests
from pinecone import PodSpec
from pinecone import ServerlessSpec

from config import Config

logger = logging.getLogger(__name__)

# По умолчанию порог сходства для дедупликации
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Типы записей
TYPE_PREFERENCE = "preference"   # предпочтения пользователя
TYPE_CONTEXT = "context"         # контекст диалога (меню и т.п.)
TYPE_MEMORY = "memory"           # обычные воспоминания

# Темы
TOPIC_FOOD = "еда"
TOPIC_GENERAL = "общее"

#known objects (еда) — используются для детерминированных ключей предпочтений
FOOD_KEYWORDS = [
    "фасоль", "чечевица", "помидор", "помидоры", "сыр", "оливки", "оливка",
    "лобио", "гаспаччо", "карбонара", "салат", "рагу", "суп", "паста",
    "пицца", "рис", "курица", "рыба", "мясо", "овощи", "фрукты", "хлеб",
    "грибы", "яйца", "молоко", "каша", "борщ", "блины", "торт", "шоколад",
    "кофе", "чай", "сок", "компот",
]

# Служебные слова для фолбэка извлечения объекта
_STOP_WORDS = {
    "люблю", "нравится", "нравятся", "очень", "меня", "моему", "теперь",
    "больше", "только", "пробовал", "пробовала", "понравилось", "хочу",
    "буду", "было", "были", "который", "рецепт", "простой", "блюдо",
    "продукт", "любимый", "любимого", "могу", "как", "что", "почему",
}


@dataclass
class MemoryResult:
    """Результат операции записи в память."""
    action: str  # 'inserted', 'updated', 'skipped', 'replaced'
    record_id: str
    similarity_score: float
    existing_id: Optional[str] = None
    reason: str = ""


@dataclass
class PreferenceInfo:
    """Информация о предпочтении."""
    topic: str        # тема: "еда", "напитки", "цвет" и т.д.
    value: str        # значение
    text: str         # исходный текст
    record_id: str    # ID записи
    is_negative: bool = False  # отрицание
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class PineconeManager:
    """
    Менеджер долговременной векторной памяти.

    Использует:
    - GenAPI для создания эмбеддингов;
    - Pinecone для хранения и поиска векторов.
    """

    def __init__(
        self,
        config: Config,
        similarity_threshold: Optional[float] = None,
    ):
        self.config = config
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else config.similarity_threshold
        self._index = None
        self._init_pinecone()

    def _init_pinecone(self) -> None:
        """Инициализация подключения к Pinecone."""
        proxy_url = self.config.pinecone_proxy_url.strip() if self.config.pinecone_proxy_url else None
        if proxy_url and not proxy_url.startswith(("http://", "https://")):
            proxy_url = "http://" + proxy_url

        kwargs: dict[str, Any] = {"api_key": self.config.pinecone_api_key}
        if proxy_url:
            kwargs["proxy_url"] = proxy_url

        self._client = pinecone.Pinecone(**kwargs)
        self._index = self._client.Index(self.config.pinecone_index_name)
        logger.info(
            "Pinecone подключён: индекс=%s", self.config.pinecone_index_name
        )

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _is_negative(text: str) -> bool:
        """Проверяет наличие явных отрицаний."""
        text_lower = text.lower()
        # Прямые маркеры отрицания
        neg_markers = [
            'не люблю', 'не нравится', 'не нравятся', 'не понравилось',
            'не хочу', 'не ем', 'не пью', 'не могу',
            'ненавижу', 'отвратительно', 'не ел', 'не пила', 'не пил',
            'без меня', 'ничего не', 'ни одного',
            'не попробовал', 'не попробовала',
        ]
        for marker in neg_markers:
            if marker in text_lower:
                return True

        # "люблю не X, а Y" — это тоже отрицание X
        if re.search(r'люблю\s+не\s+\w+', text_lower):
            return True
        if re.search(r'нравится\s+не\s+\w+', text_lower):
            return True

        return False

    @staticmethod
    def _extract_topic(text: str) -> str:
        """Извлекает тему из текста."""
        text_lower = text.lower()
        for keyword in FOOD_KEYWORDS:
            if keyword in text_lower:
                return TOPIC_FOOD
        return TOPIC_GENERAL

    @staticmethod
    def _extract_objects(text: str) -> list[str]:
        """
        Извлекает объекты упоминания (что именно любят/не любят).

        Возвращает список объектов в исходной форме (в нижнем регистре).
        """
        text_lower = text.lower()
        found = [kw for kw in FOOD_KEYWORDS if kw in text_lower]
        if found:
            return found

        # Фолбэк: последнее значимое слово как объект
        words = re.findall(r'[а-яёa-z]{3,}', text_lower)
        words = [w for w in words if w not in _STOP_WORDS]
        return [words[-1]] if words else []

    @staticmethod
    def _pref_record_id(topic: str, obj: str) -> str:
        """
        Детерминированный ID записи предпочтения.

        Одна пара (тема, объект) = одна запись. Новое значение перезаписывает
        старое, поэтому актуальное предпочтение всегда единственное.
        """
        digest = hashlib.md5(f"{topic}:{obj}".encode("utf-8")).hexdigest()[:12]
        return f"pref-{digest}"

    @staticmethod
    def _prefs_namespace(user_id: str) -> str:
        """Namespace для актуальных предпочтений пользователя."""
        return f"prefs-{user_id}"

    @staticmethod
    def _ctx_namespace(user_id: str) -> str:
        """Namespace для контекста диалога пользователя."""
        return f"ctx-{user_id}"

    def _list_ids(self, namespace: str, limit: int = 100) -> list[str]:
        """Список ID всех записей в namespace (без векторов)."""
        try:
            return [item.id for item in self._index.list(namespace=namespace, limit=limit)]
        except Exception as exc:
            raise RuntimeError(f"Ошибка списка записей в Pinecone: {exc}") from exc

    def _fetch_records(
        self, namespace: str, ids: list[str]
    ) -> list[dict[str, Any]]:
        """Загрузка записей (метаданные) по ID."""
        if not ids:
            return []
        try:
            result = self._index.fetch(ids=ids, namespace=namespace)
        except Exception as exc:
            raise RuntimeError(f"Ошибка загрузки записей из Pinecone: {exc}") from exc

        vectors = getattr(result, "vectors", None)
        if vectors is None:
            vectors = result.get("vectors", {}) if isinstance(result, dict) else {}

        records = []
        for rid, vec in (vectors or {}).items():
            meta = getattr(vec, "metadata", None)
            if meta is None and isinstance(vec, dict):
                meta = vec.get("metadata", {})
            records.append({"id": rid, "metadata": meta or {}})
        return records

    def _has_contradiction(self, existing_text: str, new_text: str) -> bool:
        """Проверяет противоречие между текстами."""
        existing_lower = existing_text.lower()
        new_lower = new_text.lower()

        # Если оба содержат отрицание одного и того же — не противоречие
        if self._is_negative(existing_text) and self._is_negative(new_text):
            # Проверяем, говорят ли они об одном и том же объекте
            if re.search(r'(?:нрав|люб|ем|ешь|ёшь|ел|пи)', existing_lower) and \
               re.search(r'(?:нрав|люб|ем|ешь|ёшь|ел|пи)', new_lower):
                return False

        # Проверка явного предпочтения vs явного отрицания
        if self._is_negative(new_text) and not self._is_negative(existing_text):
            # Новое — отрицание, старое — утверждение
            if re.search(r'(?:фасоль|чечевица|помидор|еда|блюдо)', existing_lower) and \
               re.search(r'(?:фасоль|чечевица|помидор|еда|блюдо)', new_lower):
                return True

        if self._is_negative(existing_text) and not self._is_negative(new_text):
            # Старое — отрицание, новое — утверждение
            if re.search(r'(?:фасоль|чечевица|помидор|еда|блюдо)', existing_lower) and \
               re.search(r'(?:фасоль|чечевица|помидор|еда|блюдо)', new_lower):
                return True

        # Проверка прямого утверждения vs прямого утверждения разных объектов
        if (not self._is_negative(existing_text) and
            not self._is_negative(new_text) and
            existing_text != new_text):
            # Проверяем, говорят ли о разных объектах одного типа
            old_keywords = set(re.findall(r'фасоль|чечевица|помидор|лобио|гаспаччо|карбонара|салат|рагу|суп|десерт', existing_lower))
            new_keywords = set(re.findall(r'фасоль|чечевица|помидор|лобио|гаспаччо|карбонара|салат|рагу|суп|десерт', new_lower))
            if old_keywords and new_keywords and old_keywords != new_keywords:
                return True

        return False

    def _log_diagnostic(self, action: str, record_id: str, similarity: float,
                        existing_id: Optional[str], reason: str, user_id: str) -> None:
        """Диагностическое логирование без секретов и личных текстов."""
        logger.debug(
            "Память: action=%s user=%s similarity=%.3f existing_id=%s reason=%s",
            action, user_id, similarity, existing_id or "new", reason
        )

    # ------------------------------------------------------------------
    # Сохранение предпочтений
    # ------------------------------------------------------------------

    def store_preference(
        self,
        text: str,
        user_id: str,
        topic: Optional[str] = None,
    ) -> MemoryResult:
        """
        Сохранение актуального предпочтения пользователя.

        Ключ записи = (тема, объект). Новое значение для той же пары
        ПЕРЕЗАПИСЫВАЕТ старое, поэтому противоречия не накапливаются.

        Поддерживает множественные предпочтения с разной полярностью:
        «люблю помидоры и сыр и не люблю оливки» →
          помидоры: нравится, сыр: нравится, оливки: не нравится.
        """
        if not topic:
            topic = self._extract_topic(text)

        namespace = self._prefs_namespace(user_id)

        objects = self._extract_objects(text)
        if not objects:
            objects = [hashlib.md5(text.encode("utf-8")).hexdigest()[:8]]

        # Определяем полярность для каждого объекта отдельно
        # Разбиваем текст на «положительные» и «отрицательные» части
        negated = self._extract_negated_objects(text)
        liked = self._extract_liked_objects(text, objects)

        embedding = self.create_embedding(text)

        last_result: Optional[MemoryResult] = None
        for obj in objects:
            obj_is_negative = obj in negated

            record_id = self._pref_record_id(topic, obj)

            # Существует ли уже такая запись — для action и similarity
            existing_score, action = self._check_pref_exists(namespace, record_id)

            meta = {
                "text": text,
                "user_id": str(user_id),
                "type": TYPE_PREFERENCE,
                "topic": topic,
                "object": obj,
                "is_negative": obj_is_negative,
                "updated_at": time.time(),
            }

            # Детерминированный ID => upsert заменяет старое значение
            self.upsert_vector(embedding, meta, record_id, namespace)

            reason = (
                "Исправление: новое значение заменило старое"
                if existing_score is not None
                else "Новое предпочтение"
            )
            action = "updated" if existing_score is not None else "inserted"

            self._log_diagnostic(
                action, record_id, existing_score or 0.0, None,
                f"{reason} (объект={obj}, negative={obj_is_negative})",
                str(user_id),
            )

            last_result = MemoryResult(
                action=action,
                record_id=record_id,
                similarity_score=existing_score or 0.0,
                existing_id=record_id if existing_score is not None else None,
                reason=reason,
            )

        assert last_result is not None
        return last_result

    def _extract_negated_objects(self, text: str) -> set:
        """Извлекает объекты, которые пользователь НЕ любит."""
        # Ищем паттерны «не люблю X», «не нравится X», «не X»
        negated = set(re.findall(r"не\s+(?:люб|нрав)\s+([а-яё]{4,})", text.lower()))
        # Также «люблю не X, а Y» -> X отрицается
        negated.update(set(re.findall(r"не\s+([а-яё]{4,})", text.lower())))
        return negated

    def _extract_liked_objects(self, text: str, all_objects: list) -> set:
        """Извлекает объекты, которые пользователь любит."""
        liked = set(re.findall(r"(?:люб|нрав)\s+([а-яё]{4,})", text.lower()))
        return liked

    def _check_pref_exists(
        self, namespace: str, record_id: str
    ) -> tuple[Optional[float], str]:
        """Проверяет существование записи предпочтения по детерминированному ID."""
        try:
            existing = self.get_by_id(record_id, namespace=namespace)
        except RuntimeError:
            return None, "inserted"
        if existing:
            return 1.0, "updated"
        return None, "inserted"

    def get_active_preferences(
        self,
        user_id: str,
        topic: Optional[str] = None,
    ) -> list[PreferenceInfo]:
        """
        Получение актуальных предпочтений пользователя.

        Читает namespace предпочтений напрямую (list + fetch), поэтому
        результат не зависит от косинусного сходства.
        """
        namespace = self._prefs_namespace(user_id)

        try:
            ids = self._list_ids(namespace)
            records = self._fetch_records(namespace, ids)
        except RuntimeError as exc:
            logger.warning("Ошибка получения предпочтений: %s", exc)
            return []

        preferences: list[PreferenceInfo] = []
        for rec in records:
            meta = rec.get("metadata", {})
            if meta.get("type") != TYPE_PREFERENCE:
                continue
            rec_topic = meta.get("topic", TOPIC_GENERAL)
            if topic and rec_topic != topic:
                continue
            preferences.append(
                PreferenceInfo(
                    topic=rec_topic,
                    value=meta.get("object", ""),
                    text=meta.get("text", ""),
                    record_id=rec["id"],
                    is_negative=bool(meta.get("is_negative", False)),
                )
            )

        return preferences

    # ------------------------------------------------------------------
    # Сохранение контекста
    # ------------------------------------------------------------------

    def store_context(
        self,
        text: str,
        user_id: str,
        context_type: str = "menu",
    ) -> MemoryResult:
        """
        Сохранение контекста диалога (меню, правила и т.п.).

        Для одной пары (пользователь, context_type) хранится ОДНА актуальная
        запись: новое меню заменяет старое.
        """
        namespace = self._ctx_namespace(user_id)
        record_id = f"ctx-{context_type}-{hashlib.md5(str(user_id).encode()).hexdigest()[:8]}"

        embedding = self.create_embedding(text)

        meta = {
            "text": text,
            "user_id": str(user_id),
            "type": TYPE_CONTEXT,
            "context_type": context_type,
            "updated_at": time.time(),
        }

        existing_score, _ = self._check_pref_exists(namespace, record_id)
        action = "updated" if existing_score is not None else "inserted"

        self.upsert_vector(embedding, meta, record_id, namespace)

        self._log_diagnostic(
            action, record_id, existing_score or 0.0, None,
            f"Контекст: {context_type}", str(user_id)
        )
        return MemoryResult(
            action=action,
            record_id=record_id,
            similarity_score=existing_score or 0.0,
            existing_id=record_id if existing_score is not None else None,
            reason=f"Контекст: {context_type}",
        )

    def get_context(
        self,
        user_id: str,
        context_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Получение контекста пользователя (меню и т.п.)."""
        namespace = self._ctx_namespace(user_id)

        try:
            ids = self._list_ids(namespace)
            records = self._fetch_records(namespace, ids)
        except RuntimeError as exc:
            logger.warning("Ошибка получения контекста: %s", exc)
            return []

        contexts: list[dict[str, Any]] = []
        for rec in records:
            meta = rec.get("metadata", {})
            if meta.get("type") != TYPE_CONTEXT:
                continue
            rec_type = meta.get("context_type", "")
            if context_type and rec_type != context_type:
                continue
            contexts.append({
                "id": rec["id"],
                "context_type": rec_type,
                "text": meta.get("text", ""),
            })

        return contexts

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------

    def create_embedding(self, text: str) -> list[float]:
        """
        Создание вектора эмбеддинга через GenAPI.

        GenAPI использует собственный контракт для эмбеддингов:
        - поле ``input`` должно быть массивом строк;
        - параметр ``is_sync=True`` включает синхронный режим;
        - результат приходит в ``response.data[0].embedding``.

        Args:
            text: Входной текст.

        Returns:
            Список чисел (вектор).

        Raises:
            RuntimeError: при ошибке API или некорректном ответе.
        """
        headers = {
            "Authorization": f"Bearer {self.config.genapi_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.embedding_model,
            "input": [text],
            "is_sync": True,
        }

        try:
            resp = requests.post(
                self.config.genapi_embeddings_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_genapi,
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Таймаут ({self.config.timeout_genapi}с) при создании эмбеддинга"
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ошибка запроса эмбеддинга: {exc}") from exc

        if resp.status_code >= 400:
            raise RuntimeError(
                f"GenAPI embeddings вернул HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError("Некорректный JSON в ответе эмбеддингов") from exc

        # GenAPI (синхронный режим) возвращает:
        # {"request_id": ..., "response": {"data": [{"embedding": [...]}]}}
        response_body = data.get("response")
        if isinstance(response_body, dict):
            embeddings_list = response_body.get("data")
        else:
            embeddings_list = data.get("data")

        if not embeddings_list or not isinstance(embeddings_list, list):
            raise RuntimeError(
                f"Неожиданный формат ответа эмбеддингов: {list(data.keys())}"
            )

        first = embeddings_list[0] if embeddings_list else {}
        embedding = first.get("embedding") if isinstance(first, dict) else None
        if embedding is None:
            raise RuntimeError("В ответе отсутствует поле 'embedding'")

        # Проверка размерности
        if len(embedding) != self.config.embedding_dim:
            raise RuntimeError(
                f"Размерность эмбеддинга {len(embedding)} "
                f"не совпадает с ожидаемой {self.config.embedding_dim}"
            )

        return embedding

    # ------------------------------------------------------------------
    # Запись векторов
    # ------------------------------------------------------------------

    def upsert_vector(
        self,
        vector: list[float],
        metadata: dict[str, Any],
        record_id: str,
        namespace: str = "",
    ) -> None:
        """Запись одного вектора."""
        self._index.upsert(
            vectors=[{"id": record_id, "values": vector, "metadata": metadata}],
            namespace=namespace,
        )
        logger.debug("Вектор записан: id=%s namespace=%s", record_id, namespace)

    def upsert_vectors(
        self,
        vectors: list[dict[str, Any]],
        namespace: str = "",
    ) -> None:
        """
        Массовая запись векторов.

        Args:
            vectors: Список словарей {id, values, metadata}.
            namespace: Пространство имён.
        """
        self._index.upsert(vectors=vectors, namespace=namespace)
        logger.debug("Записано %d векторов в namespace=%s", len(vectors), namespace)

    # ------------------------------------------------------------------
    # Умное сохранение с типом
    # ------------------------------------------------------------------

    def store_document(
        self,
        text: str,
        user_id: str,
        metadata: Optional[dict[str, Any]] = None,
        namespace: str = "",
        doc_type: str = TYPE_MEMORY,
    ) -> MemoryResult:
        """
        Сохранение текста с автоматическим созданием эмбеддинга.

        Args:
            text: Текст для сохранения.
            user_id: Telegram ID пользователя.
            metadata: Дополнительные метаданные.
            namespace: Пространство имён (по умолчанию = user_id).
            doc_type: Тип записи ('preference', 'context', 'memory').

        Returns:
            MemoryResult с действием, скором сходства и ID.
        """
        # Предпочтения и контекст имеют собственную логику замены
        if doc_type == TYPE_PREFERENCE:
            return self.store_preference(text=text, user_id=user_id)
        if doc_type == TYPE_CONTEXT:
            return self.store_context(text=text, user_id=user_id)

        if not namespace:
            namespace = str(user_id)

        embedding = self.create_embedding(text)

        meta = {
            "text": text,
            "user_id": str(user_id),
            "type": TYPE_MEMORY,
        }
        if metadata:
            meta.update(metadata)

        # Поиск похожей записи
        existing = self._find_similar(embedding, namespace)

        if existing and existing["score"] >= self.similarity_threshold:
            existing_id = existing["id"]
            existing_meta = existing.get("metadata", {})
            existing_text = existing_meta.get("text", "")

            # 1. Текст идентичен — пропускаем
            if existing_text.strip().lower() == text.strip().lower():
                self._log_diagnostic(
                    "skipped", existing_id, existing["score"], existing_id,
                    "Полное совпадение текста", str(user_id)
                )
                return MemoryResult(
                    action="skipped",
                    record_id=existing_id,
                    similarity_score=existing["score"],
                    existing_id=existing_id,
                    reason="Полное совпадение текста",
                )

            # 2. Разные темы — НЕ объединяем, сохраняем отдельно
            existing_topic = existing_meta.get("topic") or self._extract_topic(existing_text)
            new_topic = self._extract_topic(text)
            if existing_topic != new_topic:
                self._log_diagnostic(
                    "inserted", "", existing["score"], existing_id,
                    f"Разные темы: {existing_topic} != {new_topic}", str(user_id)
                )
                return self._insert_new(embedding, meta, namespace, existing["score"],
                                        existing_id, f"Разные темы: {existing_topic} != {new_topic}")

            # 3. Разные объекты — НЕ объединяем, сохраняем отдельно
            existing_objects = set(existing_meta.get("objects") or self._extract_objects(existing_text))
            new_objects = set(self._extract_objects(text))
            if existing_objects and new_objects and existing_objects != new_objects:
                self._log_diagnostic(
                    "inserted", "", existing["score"], existing_id,
                    f"Разные объекты: {sorted(existing_objects)} != {sorted(new_objects)}",
                    str(user_id),
                )
                return self._insert_new(
                    embedding, meta, namespace, existing["score"], existing_id,
                    f"Разные объекты: {sorted(existing_objects)} != {sorted(new_objects)}",
                )

            # 4. Противоречие (в т.ч. отрицание) — новое заменяет старое
            if self._has_contradiction(existing_text, text):
                self._index.update(
                    id=existing_id,
                    values=embedding,
                    set_metadata=meta,
                    namespace=namespace,
                )
                self._log_diagnostic(
                    "updated", existing_id, existing["score"], existing_id,
                    "Противоречие: новое значение заменяет старое", str(user_id)
                )
                return MemoryResult(
                    action="updated",
                    record_id=existing_id,
                    similarity_score=existing["score"],
                    existing_id=existing_id,
                    reason="Противоречие: новое значение заменяет старое",
                )

            # 5. Тот же объект и та же полярность, но перефразировано — пропускаем
            if self._is_negative(existing_text) == self._is_negative(text):
                self._log_diagnostic(
                    "skipped", existing_id, existing["score"], existing_id,
                    "Тот же факт в другой формулировке", str(user_id)
                )
                return MemoryResult(
                    action="skipped",
                    record_id=existing_id,
                    similarity_score=existing["score"],
                    existing_id=existing_id,
                    reason="Тот же факт в другой формулировке",
                )

            # Иначе — обновляем
            self._index.update(
                id=existing_id,
                values=embedding,
                set_metadata=meta,
                namespace=namespace,
            )
            self._log_diagnostic(
                "updated", existing_id, existing["score"], existing_id,
                "Уточнение того же факта", str(user_id)
            )
            return MemoryResult(
                action="updated",
                record_id=existing_id,
                similarity_score=existing["score"],
                existing_id=existing_id,
                reason="Уточнение того же факта",
            )

        # Новая запись
        return self._insert_new(embedding, meta, namespace, 0.0, None,
                                "Новая запись типа=memory")

    def _insert_new(
        self,
        embedding: list[float],
        meta: dict[str, Any],
        namespace: str,
        similarity: float,
        existing_id: Optional[str],
        reason: str,
    ) -> MemoryResult:
        """Вставка новой записи с диагностикой."""
        new_id = str(uuid.uuid4())
        self.upsert_vector(embedding, meta, new_id, namespace)
        self._log_diagnostic(
            "inserted", new_id, similarity, existing_id, reason,
            str(meta.get("user_id", "")),
        )
        return MemoryResult(
            action="inserted",
            record_id=new_id,
            similarity_score=similarity,
            existing_id=existing_id,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search_by_text(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Поиск по тексту (создаёт эмбеддинг запроса)."""
        embedding = self.create_embedding(query)
        return self.search_by_vector(embedding, namespace=namespace, top_k=top_k)

    def search_by_vector(
        self,
        vector: list[float],
        namespace: str = "",
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Поиск по вектору."""
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True,
        }
        if filter_dict:
            kwargs["filter"] = filter_dict

        try:
            result = self._index.query(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Ошибка поиска в Pinecone: {exc}") from exc

        matches = []
        for m in result.get("matches", []):
            matches.append({
                "id": m["id"],
                "score": m["score"],
                "metadata": m.get("metadata", {}),
            })
        return matches

    def _find_similar(
        self, vector: list[float], namespace: str
    ) -> Optional[dict[str, Any]]:
        """Поиск наиболее похожей записи (top-1)."""
        results = self.search_by_vector(vector, namespace=namespace, top_k=1)
        if results:
            return results[0]
        return None

    # ------------------------------------------------------------------
    # Получение / удаление
    # ------------------------------------------------------------------

    def get_by_id(
        self, record_id: str, namespace: str = ""
    ) -> Optional[dict[str, Any]]:
        """Получение записи по ID."""
        try:
            result = self._index.fetch(ids=[record_id], namespace=namespace)
        except Exception as exc:
            raise RuntimeError(f"Ошибка получения записей: {exc}") from exc

        vectors = result.get("vectors", {})
        if record_id in vectors:
            v = vectors[record_id]
            return {
                "id": v["id"],
                "values": v.get("values", []),
                "metadata": v.get("metadata", {}),
            }
        return None

    def delete_by_id(self, record_ids: list[str], namespace: str = "") -> None:
        """Удаление записей по списку ID."""
        self._index.delete(ids=record_ids, namespace=namespace)
        logger.debug("Удалены записи: %s", record_ids)

    def delete_by_filter(
        self, filter_dict: dict[str, Any], namespace: str = ""
    ) -> None:
        """Удаление записей по фильтру метаданных."""
        self._index.delete(filter=filter_dict, namespace=namespace)
        logger.debug("Удалены записи по фильтру: %s", filter_dict)

    # ------------------------------------------------------------------
    # Статистика и обновление метаданных
    # ------------------------------------------------------------------

    def get_index_stats(self) -> dict[str, Any]:
        """Получение статистики индекса."""
        try:
            stats = self._index.describe_index_stats()
            # Pinecone SDK 6.x возвращает объект
            if hasattr(stats, "to_dict"):
                return stats.to_dict()
            return dict(stats)
        except Exception as exc:
            raise RuntimeError(f"Ошибка получения статистики: {exc}") from exc

    def update_metadata(
        self,
        record_id: str,
        new_metadata: dict[str, Any],
        namespace: str = "",
    ) -> None:
        """Обновление метаданных существующей записи."""
        try:
            self._index.update(
                id=record_id,
                set_metadata=new_metadata,
                namespace=namespace,
            )
        except Exception as exc:
            raise RuntimeError(f"Ошибка обновления метаданных: {exc}") from exc
        logger.debug("Метаданные обновлены: id=%s", record_id)

    # ------------------------------------------------------------------
    # Проверка похожих записей
    # ------------------------------------------------------------------

    def check_similar(
        self,
        text: str,
        namespace: str = "",
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Проверка наличия похожих записей перед сохранением.

        Args:
            text: Текст для проверки.
            namespace: Пространство имён.
            threshold: Порог сходства (по умолчанию self.similarity_threshold).

        Returns:
            Словарь с keys: found (bool), id, score, text.
        """
        if threshold is None:
            threshold = self.similarity_threshold

        embedding = self.create_embedding(text)
        existing = self._find_similar(embedding, namespace)

        if existing and existing["score"] >= threshold:
            return {
                "found": True,
                "id": existing["id"],
                "score": existing["score"],
                "text": existing.get("metadata", {}).get("text", ""),
            }

        return {"found": False, "id": None, "score": 0.0, "text": ""}
