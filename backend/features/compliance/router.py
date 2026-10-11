from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import Response

from backend.core.dependencies import ProviderDependency, SessionDependency
from backend.core.errors import ApplicationError
from backend.features.compliance.clarifications import export_sheet, sheet_lines
from backend.features.compliance.repository import ComplianceRepository
from backend.features.compliance.schemas import CheckCreate, CheckRead, ComplianceResult, RunStatus
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
    status_code=201,
)
def create_check(request: CheckCreate, session: SessionDependency, providers: ProviderDependency):
    return ComplianceService(ComplianceRepository(session), providers).create(request)


@router.get("/{check_id}/clarifications")
def download_clarifications(
    check_id: UUID, session: SessionDependency, format: Literal["md", "docx", "pdf"] = "md"
):
    run = ComplianceService(ComplianceRepository(session)).get(check_id)
    if run.status != RunStatus.COMPLETED or run.result is None:
        raise ApplicationError(
            "check_not_completed", "Clarifications are available after a completed check.", 409
        )
    from backend.features.documents.models import Document

    document = session.get(Document, run.document_id)
    result = ComplianceResult.model_validate(run.result)
    content, media_type = export_sheet(
        sheet_lines(run.id, document.filename, result.clarification_questions), format
    )
    return Response(
        content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="clarifications-{check_id}.{format}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
