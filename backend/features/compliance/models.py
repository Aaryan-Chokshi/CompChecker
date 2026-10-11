from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, utc_now
from backend.features.compliance.schemas import RunStatus


class ComplianceRun(Base):
    __tablename__ = "compliance_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    policy_ids: Mapped[list[str]] = mapped_column(JSON)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False), default=RunStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
