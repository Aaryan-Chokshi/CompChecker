import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from backend.core.errors import ApplicationError, RecordNotFound
from backend.features.documents.chunking import AgenticChunker
from backend.features.documents.models import Document, DocumentChunk
from backend.features.documents.parsing import SUPPORTED_EXTENSIONS, parse_document
from backend.features.documents.repository import DocumentRepository
from backend.features.documents.schemas import DocumentRole, DocumentStatus
from backend.integrations.vector_store.qdrant import VectorRecord

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, repository: DocumentRepository, providers=None):
        self.repository = repository
        self.providers = providers

    def list(self, role: DocumentRole | None, limit: int, offset: int):
        return self.repository.list(role, limit, offset)

    def get(self, document_id: UUID):
        document = self.repository.get(document_id)
        if document is None:
            raise RecordNotFound("Document")
        return document

    def upload(self, file: UploadFile, role: DocumentRole):
        settings = self.providers.settings
        filename = (file.filename or "").replace("\\", "/").split("/")[-1]
        if (
            not filename
            or len(filename) > 255
            or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            raise ApplicationError(
                "unsupported_document",
                "Use PDF, DOCX, TXT, Markdown, CSV or TSV with a valid filename.",
                415,
            )
        content = file.file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise ApplicationError("document_too_large", "Upload exceeds MAX_UPLOAD_BYTES.", 413)
        if not content:
            raise ApplicationError("document_empty", "Upload contains no content.", 422)
        # Validate parsing before creating a processing record or contacting providers.
        units = parse_document(filename, content, settings.max_document_chunks)
        document = Document(
            id=str(uuid4()), filename=filename, role=role, status=DocumentStatus.PROCESSING
        )
        session = self.repository.session
        session.add(document)
        session.commit()
        vector_write_started = False
        try:
            settings.upload_dir.mkdir(parents=True, exist_ok=True)
            (settings.upload_dir / f"{document.id}{Path(filename).suffix.lower()}").write_bytes(
                content
            )
            chunks = AgenticChunker(self.providers.llm, settings).chunk(units)
            records = []
            for ordinal, chunk in enumerate(chunks):
                chunk_id = str(uuid4())
                metadata = {
                    **chunk.provenance,
                    "embedding_model": settings.embedding_model_id,
                    "embedding_revision": settings.embedding_revision,
                }
                session.add(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        ordinal=ordinal,
                        text=chunk.text,
                        provenance=metadata,
                    )
                )
                records.append((chunk_id, chunk.text, metadata, ordinal))
            # Bounded batches limit model memory and Qdrant request size.
            for start in range(0, len(records), settings.embedding_batch_size):
                batch = records[start : start + settings.embedding_batch_size]
                vectors = self.providers.embeddings.embed([record[1] for record in batch])
                vector_write_started = True
                self.providers.vector_store.upsert(
                    [
                        VectorRecord(
                            id=chunk_id,
                            vector=vector,
                            payload={
                                "document_id": document.id,
                                "role": role.value,
                                "filename": filename,
                                "ordinal": ordinal,
                                "text": text,
                                "provenance": metadata,
                            },
                        )
                        for (chunk_id, text, metadata, ordinal), vector in zip(
                            batch, vectors, strict=True
                        )
                    ]
                )
            document.status = DocumentStatus.READY
            session.commit()
        except Exception as exc:  # noqa: BLE001 - record failure and compensate external writes
            session.rollback()
            if vector_write_started:
                try:
                    self.providers.vector_store.delete_document(document.id)
                except ApplicationError:
                    logger.error("Could not clean up vectors for failed document %s", document.id)
            document.status = DocumentStatus.FAILED
            session.commit()
            if isinstance(exc, ApplicationError):
                raise
            raise ApplicationError(
                "ingestion_failed", "Document ingestion failed. Check backend storage and retry."
            ) from exc
        return document

    def chunks(self, document_id: UUID, limit: int, offset: int):
        self.get(document_id)
        return self.repository.chunks(document_id, limit, offset)
