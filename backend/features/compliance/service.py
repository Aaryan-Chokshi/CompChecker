from uuid import UUID

from backend.core.errors import RecordNotFound, WorkflowNotImplemented
from backend.features.compliance.repository import ComplianceRepository
from backend.features.compliance.schemas import CheckCreate


class ComplianceService:
    def __init__(self, repository: ComplianceRepository):
        self.repository = repository

    def list(self, limit: int, offset: int):
        return self.repository.list(limit, offset)

    def get(self, check_id: UUID):
        run = self.repository.get(check_id)
        if run is None:
            raise RecordNotFound("Compliance check")
        return run

    def create(self, request: CheckCreate):
        # Future: validate ready documents -> retrieve policies -> evaluate -> persist.
        raise WorkflowNotImplemented("Compliance evaluation")
