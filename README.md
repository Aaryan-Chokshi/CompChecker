# Compliance Checker

FastAPI parses policy/regulation documents and supplier specifications, creates row-preserving chunks, generates local Hugging Face embeddings, and stores each chunk as one Qdrant point. Groq selects semantic boundaries for prose and compares retrieved evidence. Dash supports uploads, reviews, comparison matrices, and supplier clarification downloads.

## Run locally

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GROQ_API_KEY='your-key'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal, activate the environment and run `python -m ui.app`. Open [the UI](http://127.0.0.1:8050) or [the API documentation](http://127.0.0.1:8000/docs).

Upload policies in **Policy Library**, upload a supplier document in **Compliance Check**, select ready documents, optionally enter additional specifications, and run the review. Results contain a comparison matrix and supplier clarification sheet with Markdown, DOCX and PDF downloads. Historical results and downloads are available in **History**.

The backend reads process environment variables; it does not load `.env`. `.env.example` lists supported settings. The UI loads only `UI_*` settings from environment or `.env`. Groq credentials remain in the backend.

## Shared ingestion and row boundaries

Both `policy` and `subject` documents use `DocumentService` → `parse_document` → `AgenticChunker` → local embeddings → Qdrant. Supported inputs are UTF-8 TXT, Markdown, CSV, TSV, DOCX, and text-based PDF.

- CSV/TSV files use the first record as headers. Quoted multiline cells remain in their row. Blank data rows are skipped; duplicate header names remain separate columns.
- Markdown tables, tab-separated text blocks, DOCX tables, and detected PDF tables become structural row units. Each data row produces exactly one chunk and one vector point, with only its values and column labels. There is no row overlap or indexed table-wide summary.
- DOCX header flags and conservative recognized header names determine whether the first table row supplies column labels. PDF uses the same conservative recognition. Unknown first rows remain data to avoid discarding requirements.
- For prose, Groq groups contiguous sentence units around independently assessable specifications. The backend verifies complete, ordered coverage without duplicated IDs and reconstructs the original text. Invalid or oversized groups fail explicitly. Rows bypass semantic grouping because their boundaries are fixed.
- SQLite and Qdrant share chunk IDs. Payloads include document ID, role, filename, ordinal, source text, page/table/row provenance, structured columns, and embedding model/revision.

Valid uploads transition through `processing` to `ready`; processing failures persist as `failed`. Original files are saved with generated UUID names under `UPLOAD_DIR`. SQLite chunks are committed after indexing succeeds. Failed writes trigger deletion of that document's Qdrant points; cleanup failures are logged and failed documents remain ineligible for review.

Scanned PDFs and image-only pages require OCR before upload. Table detection works best with ruled, machine-generated PDFs; unusual or borderless layouts may need CSV/DOCX conversion. Nested DOCX tables and vertically merged cells must be flattened/unmerged. Headers, footers, text boxes and embedded images are not extracted. Oversized rows are rejected rather than split or silently truncated. DOCX expanded-size and PDF page-count guards supplement upload/chunk limits.

## Local embeddings and storage

On first use, the embedding adapter checks for a complete local checkpoint with `local_files_only=True`. If unavailable and offline mode is disabled, Sentence Transformers downloads and caches it, then runs inference on `EMBEDDING_DEVICE`. Later calls reuse the loaded model. Startup and health checks do not download models or call Groq.

The default English model, `Alibaba-NLP/gte-large-en-v1.5`, requires `trust_remote_code=True` per its [model card](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5). Set `EMBEDDING_REVISION` to a commit for reproducibility. Expect substantial model disk/RAM usage. Embeddings are normalized; dimensions are derived from the model. Actual tokenizer length is checked before encoding to prevent truncation. `EMBEDDING_LOCAL_FILES_ONLY=true` forbids downloads and fails explicitly if the checkpoint is missing.

Local Qdrant storage requires one backend worker per path. Set `QDRANT_URL` for server mode. Dimension/distance mismatches return errors without deleting collections. When changing embedding models or revisions, use a new collection and upload documents again. Reviews reject documents indexed with a different model/revision, even when dimensions match.

## Comparison and clarifications

Every supplier chunk and optional input specification is compared against retrieved evidence from each selected policy. Reverse retrieval also checks every indexed policy chunk against supplier evidence, so obligations omitted from the supplier document remain visible. Queries filter by selected document ID and role.

Groq's Chat Completions API returns [strict JSON schema outputs](https://console.groq.com/docs/structured-outputs), which the backend validates with Pydantic. The default model is `openai/gpt-oss-120b` hosted by Groq. Review instructions cover units, thresholds, test methods, scope, exceptions, missing evidence and prompt injection. Each matrix row includes:

| Field | Meaning |
| --- | --- |
| `basis` | `specification` or reverse `policy_requirement` coverage |
| `specification`, `requirement` | Submitted text and assessed regulatory obligation |
| `satisfaction` | `satisfied`, `partial`, `not_satisfied`, `insufficient_evidence`, `not_applicable` |
| `extent_percent` | 100 for satisfied; 0 for not satisfied; between 0 and 100 for partial; null for unknown/inapplicable |
| `confidence` | LLM self-assessed decision confidence, 0–1; **not a calibrated probability** |
| `explanation` | Decision rationale |
| `policy_evidence`, `supplier_evidence` | Validated chunk IDs, exact excerpts and source provenance |

The backend rejects invented citations, excerpts outside retrieved chunks, inconsistent satisfaction/extent, and decisions lacking required evidence. Missing evidence prompts review rather than establishing noncompliance. Any partial/not-satisfied decision gives `non_compliant`; otherwise insufficient evidence or confidence below 0.7 gives `needs_review`. Remaining decisions give `compliant` within the extracted/retrieved evidence scope. Multiple obligations in one chunk use the least satisfied obligation.

Supplier questions are tied to matrix row IDs. Gap/low-confidence decisions without LLM questions receive a fallback request for measurements, methods and certificates. Persisted Markdown and downloadable DOCX/PDF share the same questions and include spaces for answers and supporting evidence. Nothing is sent to the supplier.

## API

Feature endpoints use `/api/v1`; IDs are UUIDs. Writes return `201`. Lists accept `limit` (1–100, default 50) and `offset` (default 0).

| Method | Endpoint | Behavior |
| --- | --- | --- |
| GET | `/health/live` | Process liveness |
| GET | `/health/dependencies` | Initialization/configuration without provider calls |
| POST | `/documents` | Multipart `file`, `role=policy` or `subject`; parse, chunk and index |
| GET | `/documents` | Document metadata; optional role filter |
| GET | `/documents/{id}` | Document metadata |
| GET | `/documents/{id}/chunks` | Paginated chunks and provenance |
| POST | `/checks` | JSON `document_id`, unique nonempty `policy_ids`, optional `specifications` |
| GET | `/checks` | History including results |
| GET | `/checks/{id}` | Review and comparison matrix |
| GET | `/checks/{id}/clarifications?format=md` | Download; also `docx` or `pdf` |

Example:

```json
{
  "document_id": "00000000-0000-0000-0000-000000000001",
  "policy_ids": ["00000000-0000-0000-0000-000000000002"],
  "specifications": ["Tensile strength must be at least 100 MPa."]
}
```

Application errors use `{"error":{"code":"...","message":"..."}}`; invalid requests use FastAPI's `422`. Unsupported/malformed/empty inputs, oversized uploads, wrong roles, non-ready documents and provider failures return explicit errors. Failed checks remain in history without results. Downloads require a completed review.

## Configuration

| Setting | Default |
| --- | --- |
| `DATABASE_PATH` | `data/compliance.sqlite3` |
| `GROQ_API_KEY` | Required for prose chunking and review; process environment only |
| `GROQ_MODEL` / `GROQ_TIMEOUT_SECONDS` | `openai/gpt-oss-120b` / `60` per call; SDK retries disabled |
| `EMBEDDING_MODEL_ID` / `EMBEDDING_DEVICE` | `Alibaba-NLP/gte-large-en-v1.5` / `cpu` |
| `EMBEDDING_CACHE_DIR` / `EMBEDDING_LOCAL_FILES_ONLY` | `data/models` / `false` |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_REVISION` | `8` / unset |
| `UPLOAD_DIR` / `MAX_UPLOAD_BYTES` | `data/uploads` / `20971520` (20 MB) |
| `MAX_DOCUMENT_CHUNKS` / `CHUNK_MAX_CHARS` | `1000` / `3000` |
| `CHUNK_AGENT_BATCH_CHARS` / `RETRIEVAL_TOP_K` | `18000` / `5` per document |
| `QDRANT_PATH` / `QDRANT_COLLECTION` | `data/qdrant` / `policy_chunks` (both roles) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Unset; URL switches to server mode |
| `QDRANT_TIMEOUT_SECONDS` | `10` in server mode |
| `UI_API_BASE_URL` | `http://127.0.0.1:8000/api/v1` |
| `UI_API_TIMEOUT_SECONDS` / `UI_WORKFLOW_TIMEOUT_SECONDS` | `5` for reads / `600` for writes |
| `UI_HOST` / `UI_PORT` / `UI_DEBUG` | `127.0.0.1` / `8050` / `false` |

Relative paths resolve from the project root. Runtime data is Git-ignored. Schema initialization adds `document_chunks` without altering existing tables. Old ready documents without chunks require re-upload; historical result JSON remains readable with defaults for new fields.

Processing is synchronous. Reviews require supplier chunk count × selected policy count plus every policy chunk in LLM calls; large reviews can exceed the UI timeout. Refresh history before retrying. Background jobs, recovery after process interruption, authentication, OCR, calibrated confidence and production deployment are outside this implementation.

## Verification

```bash
python -m pytest -q
python -m ruff check backend ui tests
python -m ruff format --check backend ui tests
```

Tests exercise real temporary SQLite and local Qdrant, mocked embeddings, and Groq SDK HTTP responses. They cover shared ingestion, row isolation, formats, agent plan coverage, cache/download behavior, token limits, filtering, citations, bidirectional coverage, matrix scores, failure cleanup/persistence, downloads and Dash callbacks. They do not download weights or make paid Groq calls; live inference and evaluation quality need separate verification with configured services.
