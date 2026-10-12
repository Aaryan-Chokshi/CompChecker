# Document comparison investigation — 12 October 2026

The stored vector index is intact. The current evidence loss occurs after retrieval, or when a review fails before its result is committed. Past failures cannot be attributed individually because the original code did not persist failure reasons and no saved backend log was found.

## Measurements

The configured local collection `policy_chunks` has 407 points, matching all 407 SQLite chunks. There are no missing IDs, orphan points, or mismatched text/document/role payloads.

For `02_Technical_Supplier_response.docx` (135 chunks) against `03_Manufacturing_Servo_Conveyor_Module.pdf` (239 chunks), all 374 bidirectional queries returned five candidates under the application's document ID and role filters. The audit reused the stored query vectors on a temporary snapshot; it did not regenerate embeddings.

Examples of relevant retrieval:

- Supplier speed range `0.05–1.00 m/s` retrieved the policy's `0.05 to 1.0 m per second` requirement first, cosine similarity **0.808**. The policy additionally restricts reverse motion to controlled recovery mode, so the range match alone does not establish full compliance.
- Supplier duty profile `16 h/day, 300 days/year, five-year design mission` retrieved the equivalent policy duty requirement first, similarity **0.895**.

A broader system architecture query returned repeated document title/page-header chunks among its top five results. This is a separate retrieval-quality issue: redundant PDF boilerplate can displace relevant requirements. The investigation did not alter parsing or reindex the existing documents.

At the initial history snapshot, there were **9 failed runs, 3 running runs, and no completed runs**. None had saved results. Current code saves the complete comparison result only after all Ollama batches finish. Any provider or citation-validation failure loses that run's accumulated rows; interruption can leave it marked running. A running status alone does not establish that processing is still active.

## Code findings and changes

Previously the UI displayed only the citations chosen by Ollama, not the chunks retrieved and passed to it. An `insufficient_evidence` response was allowed to contain empty citation lists. Thus an empty evidence cell could not distinguish unsuccessful search from a reviewer omitting citations.

Completed rows now retain their source candidates separately from validated citations, and the UI labels them as source context rather than confirmed supporting evidence. Missing Qdrant points and IDs absent from SQLite now cause explicit errors rather than silently becoming empty context. Reverse policy reviews must cite their original policy requirement even if the supplier information is incomplete. Invalid single-task responses receive one bounded retry with validation feedback. Run failures, transport failures, and invalid JSON field locations now have diagnostic logs.

These changes preserve strict citation validation. They do not convert nearest-neighbor candidates into proof of compliance.

## Live Ollama checks

The configured model is `gpt-oss:20b-cloud`, called at `https://ollama.com`. A single real speed-range comparison through the original adapter failed with `llm_api_error` before returning structured output. A subsequent small structured `2 + 2` request, limited to 20 seconds, failed with a confirmed `httpx.ReadTimeout`. Authenticated `GET /api/tags` returned HTTP 200, and the local backend health endpoint returned HTTP 200. Thus reachability alone did not establish working chat inference.

Ollama's official documentation states that its Cloud service currently does not support structured outputs: https://docs.ollama.com/capabilities/structured-outputs . Passing a JSON schema therefore does not guarantee a schema-valid response. The adapter also includes the schema in its prompt and validates the returned JSON in Python. A final tiny request without a schema, using low reasoning effort, also hit a 20-second read timeout. The observed connection/inference failure was therefore not specific to the comparison input or JSON schema.

The configured per-call timeout is 500 seconds. Hundreds of tasks, batch retries, and all-or-nothing result persistence explain why long or failed reviews leave no visible evidence. No successful full live comparison was obtained during this investigation, so model response quality and the historical failure reasons remain unverified.

## Reproduction and validation

Run `python -m scripts.diagnose_retrieval --samples 3` to repeat the read-only vector audit. It operates on a temporary local snapshot, does not lock the active Qdrant directory, and does not call Ollama or load embedding weights.

The updated test suite passes **76 tests**. Ruff checks, formatting checks, and `git diff --check` pass. Regression cases cover missing vectors, unmappable point IDs, retained source context when citations are omitted, retrying missing policy citations, displaying candidates separately in the UI, and auditing a snapshot while the backend holds the live Qdrant lock.

Restart the backend and UI to load these source changes. Existing failed/running history entries and stored documents were not rewritten.
