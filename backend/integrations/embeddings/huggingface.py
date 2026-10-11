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
                    self.settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
                    options = dict(
                        device=self.settings.embedding_device,
                        cache_folder=str(self.settings.embedding_cache_dir),
                        revision=self.settings.embedding_revision,
                        trust_remote_code=True,  # Required by the selected GTE model card.
                    )
                    try:
                        # Check for a complete, locally loadable checkpoint first.
                        self._model = factory(
                            self.settings.embedding_model_id, local_files_only=True, **options
                        )
                    except (OSError, ValueError):
                        if self.settings.embedding_local_files_only:
                            raise
                        self._model = factory(
                            self.settings.embedding_model_id, local_files_only=False, **options
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
            max_length = getattr(model, "max_seq_length", None)
            tokenizer = getattr(model, "tokenizer", None)
            if isinstance(max_length, int) and tokenizer is not None:
                for text in texts:
                    if len(tokenizer.encode(text, truncation=False)) > max_length:
                        raise ProviderError(
                            "embedding_input_too_long",
                            "A chunk exceeds the embedding model token limit. Shorten the row "
                            "or reduce CHUNK_MAX_CHARS; content will not be silently truncated.",
                            422,
                        )
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
