from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, UploadFile

from backend.core.dependencies import ProviderDependency, SessionDependency
from backend.features.documents.repository import DocumentRepository
from backend.features.documents.schemas import ChunkRead, DocumentRead, DocumentRole
from backend.features.documents.service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentRead])
def list_documents(
    session: SessionDependency,
    role: DocumentRole | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return DocumentService(DocumentRepository(session)).list(role, limit, offset)


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: UUID, session: SessionDependency):
    return DocumentService(DocumentRepository(session)).get(document_id)


@router.post(
    "",
    response_model=DocumentRead,
    status_code=201,
)
def upload_document(
    session: SessionDependency,
    providers: ProviderDependency,
    file: Annotated[UploadFile, File()],
    role: Annotated[DocumentRole, Form()],
):
    return DocumentService(DocumentRepository(session), providers).upload(file, role)


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
def list_chunks(
    document_id: UUID,
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return DocumentService(DocumentRepository(session)).chunks(document_id, limit, offset)
