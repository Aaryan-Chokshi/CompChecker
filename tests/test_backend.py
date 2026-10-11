import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import inspect

from backend.core.config import Settings
from backend.db.base import utc_now
from backend.features.compliance.models import ComplianceRun
from backend.features.compliance.schemas import RunStatus
from backend.features.documents.models import Document
from backend.features.documents.schemas import DocumentRole, DocumentStatus
from backend.main import create_app


def test_groq_key_is_read_from_process_environment_only(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "obsolete-secret")
    monkeypatch.setenv("GROQ_API_KEY", "process-secret")
    settings = Settings(_env_file=None, database_path=tmp_path / "db.sqlite3")
    assert settings.groq_api_key.get_secret_value() == "process-secret"
    assert "openai_api_key" not in Settings.model_fields


def test_imports_do_not_load_embedding_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import backend.main; import ui.app; assert 'sentence_transformers' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_startup_initializes_only_database(api, settings):
    assert settings.database_path.is_file()
    assert set(inspect(api.app.state.database.engine).get_table_names()) == {
        "documents",
        "compliance_runs",
        "document_chunks",
    }
    assert api.get("/api/v1/health/live").json() == {"status": "ok"}
    providers = api.app.state.providers
    assert not providers.embeddings.initialized
    assert not providers.vector_store.initialized
    assert not providers.llm.initialized
    assert not settings.qdrant_path.exists()
    assert not settings.embedding_cache_dir.exists()


def test_dependency_status_does_not_initialize_or_expose_credentials(settings, monkeypatch):
    settings.groq_api_key = SecretStr("test-secret-not-for-ui")
    app = create_app(settings)
    with TestClient(app) as client:

        def forbidden(*args, **kwargs):
            pytest.fail("Health endpoint tried to initialize a provider")

        providers = app.state.providers
        monkeypatch.setattr(providers.embeddings, "_get_model", forbidden)
        monkeypatch.setattr(providers.vector_store, "_get_client", forbidden)
        monkeypatch.setattr(providers.llm, "_get_client", forbidden)
        response = client.get("/api/v1/health/dependencies")
        assert response.status_code == 200
        assert response.json()["llm"] == {
            "configured": True,
            "initialized": False,
            "model": "openai/gpt-oss-120b",
        }
        assert "test-secret-not-for-ui" not in response.text


@pytest.mark.parametrize("resource", ["documents", "checks"])
def test_empty_lists_and_missing_records(api, resource):
    assert api.get(f"/api/v1/{resource}").json() == []
    response = api.get(f"/api/v1/{resource}/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert api.get(f"/api/v1/{resource}/invalid").status_code == 422
    assert api.get(f"/api/v1/{resource}?limit=101").status_code == 422
    assert api.get(f"/api/v1/{resource}?offset=-1").status_code == 422


def test_missing_configuration_persists_failed_ingestion_and_missing_check_is_rejected(api):
    upload = api.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", b"Policy text", "text/plain")},
        data={"role": "policy"},
    )
    check = api.post(
        "/api/v1/checks",
        json={"document_id": str(uuid4()), "policy_ids": [str(uuid4())]},
    )
    assert upload.status_code == 503
    assert upload.json()["error"]["code"] == "llm_not_configured"
    assert check.status_code == 404
    assert api.get("/api/v1/documents").json()[0]["status"] == "failed"
    assert api.get("/api/v1/checks").json() == []


def test_request_contracts(api):
    assert api.post("/api/v1/documents", data={"role": "policy"}).status_code == 422
    assert (
        api.post("/api/v1/checks", json={"document_id": str(uuid4()), "policy_ids": []}).status_code
        == 422
    )
    assert api.get("/api/v1/documents?role=invalid").status_code == 422
    schema = api.get("/openapi.json").json()
    assert (
        "multipart/form-data"
        in schema["paths"]["/api/v1/documents"]["post"]["requestBody"]["content"]
    )


def test_sqlite_round_trip_filtering_and_restart(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        with app.state.database.sessions() as session:
            subject = Document(
                filename="contract.txt",
                role=DocumentRole.SUBJECT,
                status=DocumentStatus.READY,
            )
            policy = Document(
                filename="policy.txt",
                role=DocumentRole.POLICY,
                status=DocumentStatus.READY,
            )
            session.add_all([subject, policy])
            session.flush()
            result = {
                "summary": "Manual fixture",
                "verdict": "needs_review",
                "findings": [],
            }
            run = ComplianceRun(
                document_id=subject.id,
                policy_ids=[policy.id],
                status=RunStatus.COMPLETED,
                completed_at=utc_now(),
                result=result,
            )
            session.add(run)
            session.commit()
            subject_id, policy_id, check_id = subject.id, policy.id, run.id

        assert [row["id"] for row in client.get("/api/v1/documents?role=policy").json()] == [
            policy_id
        ]
        assert len(client.get("/api/v1/documents?limit=1").json()) == 1
        assert client.get(f"/api/v1/documents/{subject_id}").json()["created_at"].endswith("Z")
        check = client.get(f"/api/v1/checks/{check_id}").json()
        assert all(check["result"][key] == value for key, value in result.items())
        assert check["policy_ids"] == [policy_id]
        assert check["completed_at"].endswith("Z")

    with TestClient(create_app(settings)) as client:
        assert len(client.get("/api/v1/documents").json()) == 2
        assert client.get(f"/api/v1/checks/{check_id}").json()["document_id"] == subject_id


def test_lifespan_closes_initialized_clients(settings):
    app = create_app(settings)
    with TestClient(app):
        providers = app.state.providers
        providers.vector_store._dimension = lambda: 3
        providers.vector_store.ensure_collection()
        assert providers.vector_store.initialized
    assert not providers.vector_store.initialized
    assert not providers.embeddings.initialized
    assert not providers.llm.initialized
