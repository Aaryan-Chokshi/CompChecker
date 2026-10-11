from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.features.documents.models import Document
from backend.features.documents.schemas import DocumentRole


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, role: DocumentRole | None, limit: int, offset: int) -> list[Document]:
        query = select(Document)
        if role is not None:
            query = query.where(Document.role == role)
        query = query.order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset)
        return list(self.session.scalars(query))

    def get(self, document_id: UUID) -> Document | None:
        return self.session.get(Document, str(document_id))
