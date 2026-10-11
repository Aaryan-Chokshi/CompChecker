from collections.abc import Callable, Sequence
from threading import RLock
from typing import Any

from backend.core.config import Settings
from backend.core.errors import ProviderError


class HuggingFaceEmbeddings:
    def __init__(self, settings: Settings, model_factory: Callable[..., Any] | None = None):
        self.settings = settings
        self._model_factory = model_factory
        self._model = None
        self._lock = RLock()

    @property
    def initialized(self) -> bool:
        return self._model is not None

    def _get_model(self):
        with self._lock:
            if self._model is None:
                try:
                    factory = self._model_factory
                    if factory is None:
                        # Keep torch/transformers imports and downloads out of startup.
                        from sentence_transformers import SentenceTransformer

                        factory = SentenceTransformer
                    self._model = factory(
                        self.settings.embedding_model_id,
                        device=self.settings.embedding_device,
                        cache_folder=str(self.settings.embedding_cache_dir),
                        revision=self.settings.embedding_revision,
                        local_files_only=self.settings.embedding_local_files_only,
                        trust_remote_code=True,  # Required by the selected GTE model card.
                    )
                except Exception as exc:
                    raise ProviderError(
                        "embedding_initialization_failed",
                        "Cannot load the local embedding model. Check the model cache, "
                        "device, Hugging Face connectivity, and offline setting.",
                        503,
                    ) from exc
            return self._model

    @property
    def dimension(self) -> int:
        with self._lock:
            dimension = self._get_model().get_sentence_embedding_dimension()
            if not isinstance(dimension, int) or dimension <= 0:
                raise ProviderError("embedding_dimension_invalid", "Invalid embedding dimension.")
            return dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty strings.")
        with self._lock:
            model = self._get_model()
            try:
                vectors = model.encode(
                    list(texts),
                    batch_size=self.settings.embedding_batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).tolist()
            except Exception as exc:
                raise ProviderError(
                    "embedding_failed", "Local embedding generation failed."
                ) from exc
            if len(vectors) != len(texts) or any(len(v) != self.dimension for v in vectors):
                raise ProviderError(
                    "embedding_dimension_invalid",
                    "Embedding output has an unexpected shape.",
                )
            return vectors

    def close(self) -> None:
        with self._lock:
            self._model = None
