from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.core.config import Settings
from backend.core.errors import register_error_handlers
from backend.core.health import router as health_router
from backend.core.logging import configure_logging
from backend.core.providers import Providers
from backend.db.initialize import initialize_database
from backend.db.session import Database
from backend.features.compliance.router import router as compliance_router
from backend.features.documents.router import router as documents_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        database = Database(settings.database_path)
        providers = Providers(settings)
        try:
            initialize_database(database)
            app.state.database = database
            app.state.providers = providers
            yield
        finally:
            providers.close()
            database.close()

    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description="Local document embeddings, row-preserving ingestion and evidence-based compliance comparison.",
        lifespan=lifespan,
    )
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(compliance_router, prefix="/api/v1")
    register_error_handlers(app)
    return app


app = create_app()
