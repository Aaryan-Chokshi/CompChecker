from fastapi import APIRouter

from backend.core.dependencies import ProviderDependency

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def liveness():
    return {"status": "ok"}


@router.get("/dependencies")
def dependencies(providers: ProviderDependency):
    return {
        "database": {"configured": True, "initialized": True, "engine": "sqlite"},
        **providers.status(),
        "note": "Configuration and initialization state only; availability is not probed.",
    }
