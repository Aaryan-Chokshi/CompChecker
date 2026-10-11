# Compliance Checker

A runnable application scaffold for reviewing documents against a policy library. Dash provides the UI, FastAPI owns the API and business logic, SQLite stores application records, and Qdrant stores policy vectors. Embeddings run locally through Hugging Face; LLM requests use OpenAI's API.

**This is the first-level scaffold, not an end-to-end compliance checker.** Document ingestion and compliance evaluation are explicit extension points. Their API endpoints return `501 Not Implemented`, and the corresponding UI controls are disabled. No sample findings or successful checks are fabricated.

## Quick start

Use Python **3.11**. From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, create the environment with `py -3.11 -m venv .venv`, activate using `.venv\Scripts\Activate.ps1`, and copy the configuration with `Copy-Item .env.example .env`.

Start the backend in one terminal:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Activate the same environment in a second terminal and start Dash:

```bash
python -m ui.app
```

Open [the UI](http://127.0.0.1:8050), [interactive API documentation](http://127.0.0.1:8000/docs), or [the OpenAPI schema](http://127.0.0.1:8000/openapi.json).

Startup requires neither an OpenAI API key nor an embedding model download. SQLite initializes during backend startup. The UI loads real API records, initially empty, and refreshes every 30 seconds or when Refresh is clicked. It remains accessible if the API is down and displays an actionable error.

## Project structure

```text
backend/
  main.py                         Application factory, routes, lifespan
  core/                           Settings, logging, errors, dependency injection
  db/                             SQLite engine, request sessions, table setup
  integrations/
    embeddings/                   Lazy Hugging Face/Sentence Transformers adapter
    vector_store/                 Local/server Qdrant adapter
    llm/                          OpenAI structured-output adapter
  features/
    documents/                    Policy and subject metadata
    compliance/                   Check contracts and historical results
ui/
  app.py                          Dash factory, navigation and view callback
  config.py                       UI-only environment settings
  api/                            HTTP client and API error handling
  components/                     Shared navigation, tables, badges, empty states
  features/                       Overview, policy library, checks and history
  assets/                         Responsive CSS
tests/                            API, persistence, provider and Dash checks
data/                             Ignored runtime storage
requirements.txt                  Shared runtime and verification dependencies
.env.example                      Configuration template without credentials
```

Feature routers delegate to services; services use repositories for persistence and provider adapters for future orchestration. Each request receives its own SQLAlchemy session. SQLite foreign keys are enabled. The UI only calls HTTP endpoints and never imports backend modules, accesses SQLite/Qdrant, or reads the OpenAI key into its settings.

## Configuration and storage

Backend settings are loaded through Pydantic Settings from process environment variables. The OpenAI key is read only from `OPENAI_API_KEY`; it is never read from `.env` or `.env.example`. The UI only consumes `UI_*` settings. Relative storage paths resolve against the project root. `.env`, databases, model files, and vector storage are ignored by Git.

| Setting | Default / purpose |
| --- | --- |
| `DATABASE_PATH` | `data/compliance.sqlite3` |
| `OPENAI_API_KEY` | Process environment variable only; optional at startup and required when invoking the LLM adapter |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_TIMEOUT_SECONDS` | `60`; automatic SDK retries disabled |
| `EMBEDDING_MODEL_ID` | `Alibaba-NLP/gte-large-en-v1.5` |
| `EMBEDDING_DEVICE` | `cpu`; set a supported Torch device explicitly for acceleration |
| `EMBEDDING_CACHE_DIR` | `data/models` |
| `EMBEDDING_LOCAL_FILES_ONLY` | `false`; set `true` after caching for offline loading |
| `EMBEDDING_BATCH_SIZE` | `8` |
| `EMBEDDING_REVISION` | Optional Hugging Face commit ID |
| `QDRANT_PATH` | `data/qdrant` in persistent local mode |
| `QDRANT_COLLECTION` | `policy_chunks` |
| `QDRANT_URL` | Unset; when set, server mode replaces local mode |
| `QDRANT_API_KEY` | Optional server credential |
| `QDRANT_TIMEOUT_SECONDS` | `10` in server mode |
| `UI_API_BASE_URL` | `http://127.0.0.1:8000/api/v1` |
| `UI_API_TIMEOUT_SECONDS` | `5` per HTTP request |
| `UI_HOST`, `UI_PORT`, `UI_DEBUG` | `127.0.0.1`, `8050`, `false` |

Use one backend worker in local Qdrant mode: only one process can hold a storage path. `--reload` is for development; omit it for a stable local session. To use a separate Qdrant instance, set `QDRANT_URL=http://127.0.0.1:6333` and provide its API key if required. No Docker service is needed for the default configuration.

### Embeddings

The chosen English GTE checkpoint is loaded with `SentenceTransformer(..., trust_remote_code=True)`, following its [Hugging Face model card](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5). It returns normalized vectors and runs locally, not through a hosted embedding API. The checkpoint has 1024-dimensional embeddings; the adapter derives the dimension from the loaded model rather than hardcoding it.

The first explicit embedding or vector-store operation loads the model and may download model weights and model code. Allow disk space and memory for a large model. Subsequent loads reuse cached artifacts; Hugging Face may additionally use its standard cache for custom model code. `EMBEDDING_LOCAL_FILES_ONLY=true` requires those artifacts to be cached already. A fixed `EMBEDDING_REVISION` can pin weights and custom code to a particular commit.

Qdrant initializes a cosine-distance collection lazily and verifies its vector dimension. A mismatched existing collection raises an explicit error; it is never deleted or recreated automatically. After changing models, use a new collection and reindex even if the new model uses the same dimensions.

### OpenAI

`OpenAILLM.generate_structured(instructions, user_input, response_model)` reads the existing `OPENAI_API_KEY` process variable and calls the Responses API with `gpt-4o-mini`, a Pydantic output schema, and `store=False`, following [OpenAI's structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

The adapter distinguishes missing configuration, API failures, timeouts, refusals, incomplete responses, and invalid structured output. Only explicit adapter invocation contacts OpenAI and can incur API charges. It supplies a provider primitive; the future compliance service must assemble retrieved evidence and evaluation instructions.

All provider adapters initialize lazily. Application shutdown closes initialized clients and releases the model reference. Health endpoints never download models or make upstream requests.

## API contracts

| Method | Endpoint | Scaffold behavior |
| --- | --- | --- |
| GET | `/api/v1/health/live` | Process liveness: `{"status":"ok"}` |
| GET | `/api/v1/health/dependencies` | Configuration and initialization state only; no availability probes |
| GET | `/api/v1/documents` | SQLite records; optional `role=policy` or `role=subject` |
| GET | `/api/v1/documents/{document_id}` | Metadata, or `404` for an unknown UUID |
| POST | `/api/v1/documents` | Multipart `file` + `role`; valid requests return `501` |
| GET | `/api/v1/checks` | Historical run records from SQLite |
| GET | `/api/v1/checks/{check_id}` | Run with structured result, or `404` |
| POST | `/api/v1/checks` | JSON `document_id` + nonempty `policy_ids`; valid requests return `501` |

List endpoints return arrays with `limit` (default 50, maximum 100) and `offset` (default 0), ordered newest first. Empty lists return `[]`. The UI displays at most the latest 100 records per request; Overview counts are counts of loaded records, not database totals.

Identifiers are UUIDs. Document responses include `filename`, `role`, processing `status`, and UTC `created_at`. Run responses include `document_id`, `policy_ids`, run `status`, timestamps, and an optional typed result. Results contain a summary, verdict, and findings with policy references and evidence excerpts. No workflow produces these results in this scaffold.

Application errors use `{"error":{"code":"...","message":"..."}}`. Invalid requests use FastAPI's standard `422` validation response. `501` responses do not write records; multipart parsing may use temporary upload storage managed by FastAPI.

## Extending the scaffold

1. Add a module under `backend/features/<feature>/` with its router, Pydantic schemas, service, and repository/models as needed. Register the router under `/api/v1` in `backend/main.py`; register new models in `backend/db/initialize.py`.
2. Use `SessionDependency` for request-scoped persistence and `ProviderDependency` for the lazy provider bundle. Services own orchestration; repositories own SQL queries. Future write services should commit deliberately and roll back on failure.
3. Add a matching UI feature and navigation entry. Extend `ui/api/client.py` with HTTP operations and render through the Dash view callback. Enable existing workflow controls only after the underlying endpoints work.
4. Add behavior tests. The next ingestion implementation should validate/persist documents, parse and chunk text, embed policies, index vectors, and update statuses. The next compliance implementation should validate ready document references, retrieve policy evidence, generate structured findings, and persist the run/result.

SQLite initialization uses `create_all` for initial tables, not schema migration. Introduce migrations before changing a populated database. Policy references are a JSON snapshot in this initial scaffold; future write workflows must validate policy existence and readiness. Authentication, background job infrastructure, parsing/chunking, indexing orchestration, evaluation, result-detail UI, and production deployment are not implemented.

## Verification

```bash
python -m pytest -q
python -m ruff check backend ui tests
python -m ruff format --check backend ui tests
```

Tests use temporary SQLite and persistent local Qdrant storage, a mocked embedding model, and the real OpenAI SDK over an HTTP mock transport. They cover lazy startup, persistence across restart, API contracts, provider failure handling and shutdown, Dash navigation, callback execution, disabled controls, and API-unavailable feedback. They do not download a model, use credentials, or make paid calls. Real Hugging Face inference and live OpenAI evaluation require separate opt-in verification once those workflows are implemented.
