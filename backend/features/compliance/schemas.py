from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(StrEnum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    criterion: str
    verdict: Verdict
    explanation: str
    evidence_excerpt: str | None


class ComplianceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    verdict: Verdict
    findings: list[Finding]


class CheckCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    policy_ids: list[UUID] = Field(min_length=1)


class CheckRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    policy_ids: list[UUID]
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None
    result: ComplianceResult | None

    @field_validator("created_at", "completed_at")
    @classmethod
    def normalize_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
