from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, utc_now
from backend.features.documents.schemas import DocumentRole, DocumentStatus


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    role: Mapped[DocumentRole] = mapped_column(Enum(DocumentRole, native_enum=False))
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False), default=DocumentStatus.UPLOADED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
