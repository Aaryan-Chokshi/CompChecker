import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app


@pytest.fixture
def settings(tmp_path, monkeypatch):
    # Ignore a developer's credentials and external service configuration.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return Settings(
        _env_file=None,
        database_path=tmp_path / "compliance.sqlite3",
        embedding_cache_dir=tmp_path / "models",
        qdrant_path=tmp_path / "qdrant",
        embedding_local_files_only=True,
    )


@pytest.fixture
def api(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        yield client
