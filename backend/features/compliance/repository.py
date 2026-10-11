from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.features.compliance.models import ComplianceRun


class ComplianceRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, limit: int, offset: int) -> list[ComplianceRun]:
        query = (
            select(ComplianceRun)
            .order_by(ComplianceRun.created_at.desc(), ComplianceRun.id)
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(query))

    def get(self, check_id: UUID) -> ComplianceRun | None:
        return self.session.get(ComplianceRun, str(check_id))
