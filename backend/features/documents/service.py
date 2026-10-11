from uuid import UUID

from fastapi import UploadFile

from backend.core.errors import RecordNotFound, WorkflowNotImplemented
from backend.features.documents.repository import DocumentRepository
from backend.features.documents.schemas import DocumentRole


class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository

    def list(self, role: DocumentRole | None, limit: int, offset: int):
        return self.repository.list(role, limit, offset)

    def get(self, document_id: UUID):
        document = self.repository.get(document_id)
        if document is None:
            raise RecordNotFound("Document")
        return document

    def upload(self, file: UploadFile, role: DocumentRole):
        # Future: validate -> persist -> parse/chunk -> embed/index -> update status.
        raise WorkflowNotImplemented("Document ingestion")
