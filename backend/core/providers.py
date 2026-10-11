import logging

from backend.core.config import Settings
from backend.integrations.embeddings.huggingface import HuggingFaceEmbeddings
from backend.integrations.llm.groq import GroqLLM
from backend.integrations.vector_store.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


class Providers:
    """One lazy provider bundle per FastAPI application lifespan."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = HuggingFaceEmbeddings(settings)
        self.vector_store = QdrantVectorStore(settings, lambda: self.embeddings.dimension)
        self.llm = GroqLLM(settings)

    def status(self) -> dict:
        # Inspect state only: no downloads, network probes, or billable requests.
        return {
            "embeddings": {
                "configured": True,
                "initialized": self.embeddings.initialized,
                "model": self.settings.embedding_model_id,
                "device": self.settings.embedding_device,
            },
            "vector_store": {
                "configured": True,
                "initialized": self.vector_store.initialized,
                "collection_ready": self.vector_store.collection_ready,
                "mode": "server" if self.settings.qdrant_url else "local",
                "collection": self.settings.qdrant_collection,
            },
            "llm": {
                "configured": self.settings.groq_api_key is not None,
                "initialized": self.llm.initialized,
                "model": self.settings.groq_model,
            },
        }

    def close(self) -> None:
        for provider in (self.llm, self.vector_store, self.embeddings):
            try:
                provider.close()
            except Exception as exc:  # noqa: BLE001 - release every provider on shutdown
                # Continue closing other resources; do not log upstream secrets.
                logger.error(
                    "Provider shutdown failed: %s (%s)",
                    type(provider).__name__,
                    type(exc).__name__,
                )
