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


class Satisfaction(StrEnum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    NOT_SATISFIED = "not_satisfied"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    excerpt: str = Field(min_length=1)


class Evaluation(BaseModel):
    """Strict, required fields for structured model output."""

    model_config = ConfigDict(extra="forbid")
    requirement: str
    satisfaction: Satisfaction
    extent_percent: float | None = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    explanation: str
    policy_evidence: list[Citation]
    supplier_evidence: list[Citation]
    clarification_questions: list[str]


class NumberedEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: int
    evaluation: Evaluation


class EvaluationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluations: list[NumberedEvaluation]


class EvidenceReference(Citation):
    document_id: str
    filename: str
    provenance: dict


class ComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    basis: str
    specification: str
    specification_chunk_id: str | None
    policy_id: str
    requirement: str
    satisfaction: Satisfaction
    extent_percent: float | None
    confidence: float
    explanation: str
    policy_evidence: list[EvidenceReference]
    supplier_evidence: list[EvidenceReference]
    # Search candidates are source context, independently of the LLM's citations.
    retrieved_policy_chunks: list[EvidenceReference] = Field(default_factory=list)
    retrieved_supplier_chunks: list[EvidenceReference] = Field(default_factory=list)


class ClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comparison_id: str
    question: str


class ComplianceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    verdict: Verdict
    findings: list[Finding]
    comparison_matrix: list[ComparisonRow] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    clarification_markdown: str = ""


class CheckCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    policy_ids: list[UUID] = Field(min_length=1, max_length=20)
    specifications: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("policy_ids")
    @classmethod
    def unique_policies(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Policy IDs must be unique.")
        return value

    @field_validator("specifications")
    @classmethod
    def valid_specifications(cls, value):
        if any(not text.strip() or len(text) > 3000 for text in value):
            raise ValueError("Specifications must contain 1 to 3000 characters.")
        return [text.strip() for text in value]


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
