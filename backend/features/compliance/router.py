from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from backend.core.dependencies import SessionDependency
from backend.features.compliance.repository import ComplianceRepository
from backend.features.compliance.schemas import CheckCreate, CheckRead
from backend.features.compliance.service import ComplianceService

router = APIRouter(prefix="/checks", tags=["compliance"])


@router.get("", response_model=list[CheckRead])
def list_checks(
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return ComplianceService(ComplianceRepository(session)).list(limit, offset)


@router.get("/{check_id}", response_model=CheckRead)
def get_check(check_id: UUID, session: SessionDependency):
    return ComplianceService(ComplianceRepository(session)).get(check_id)


@router.post(
    "",
    response_model=CheckRead,
    responses={501: {"description": "Compliance evaluation is not implemented."}},
)
def create_check(request: CheckCreate, session: SessionDependency):
    return ComplianceService(ComplianceRepository(session)).create(request)
