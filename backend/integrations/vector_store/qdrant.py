import math
from collections.abc import Callable, Sequence
from threading import RLock
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from backend.core.config import Settings
from backend.core.errors import ProviderError


class VectorRecord(BaseModel):
    id: UUID
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    def __init__(
        self,
        settings: Settings,
        dimension: Callable[[], int],
        client_factory: Callable[..., Any] = QdrantClient,
    ):
        self.settings = settings
        self._dimension = dimension
        self._client_factory = client_factory
        self._client = None
        self._collection_ready = False
        self._lock = RLock()

    @property
    def initialized(self) -> bool:
        return self._client is not None

    @property
    def collection_ready(self) -> bool:
        return self._collection_ready

    def _get_client(self):
        if self._client is None:
            if self.settings.qdrant_url:
                self._client = self._client_factory(
                    url=self.settings.qdrant_url,
                    api_key=(
                        self.settings.qdrant_api_key.get_secret_value()
                        if self.settings.qdrant_api_key
                        else None
                    ),
                    timeout=self.settings.qdrant_timeout_seconds,
                )
            else:
                self.settings.qdrant_path.mkdir(parents=True, exist_ok=True)
                self._client = self._client_factory(path=str(self.settings.qdrant_path))
        return self._client

    def ensure_collection(self) -> None:
        with self._lock:
            if self._collection_ready:
                return
            dimension = self._dimension()
            if dimension <= 0:
                raise ValueError("Vector dimension must be positive.")
            try:
                client = self._get_client()
                name = self.settings.qdrant_collection
                if not client.collection_exists(name):
                    client.create_collection(
                        collection_name=name,
                        vectors_config=models.VectorParams(
                            size=dimension, distance=models.Distance.COSINE
                        ),
                    )
                config = client.get_collection(name).config.params.vectors
                if (
                    not isinstance(config, models.VectorParams)
                    or config.size != dimension
                    or config.distance != models.Distance.COSINE
                ):
                    raise ProviderError(
                        "vector_collection_mismatch",
                        "Existing Qdrant collection does not match the embedding dimension "
                        "or cosine distance. Use a new collection and reindex.",
                        409,
                    )
                self._collection_ready = True
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    "vector_store_unavailable",
                    "Cannot initialize Qdrant. Check its URL or local storage path; "
                    "local mode supports one backend process per path.",
                    503,
                ) from exc

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension():
            raise ValueError("Vector dimension does not match the embedding model.")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Vector values must be finite.")
        if not any(value != 0 for value in vector):
            raise ValueError("Cosine similarity requires a nonzero vector.")

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        with self._lock:
            for record in records:
                self._validate_vector(record.vector)
            self.ensure_collection()
            try:
                self._client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=[
                        models.PointStruct(id=str(r.id), vector=r.vector, payload=r.payload)
                        for r in records
                    ],
                    wait=True,
                )
            except Exception as exc:
                raise ProviderError(
                    "vector_upsert_failed", "Qdrant could not store vectors."
                ) from exc

    def query(
        self,
        vector: Sequence[float],
        limit: int = 5,
        query_filter: models.Filter | None = None,
    ) -> list[SearchHit]:
        if not 1 <= limit <= 100:
            raise ValueError("Query limit must be between 1 and 100.")
        with self._lock:
            self._validate_vector(vector)
            self.ensure_collection()
            try:
                points = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=list(vector),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                ).points
            except Exception as exc:
                raise ProviderError("vector_query_failed", "Qdrant retrieval failed.") from exc
            return [SearchHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in points]

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
            self._collection_ready = False
