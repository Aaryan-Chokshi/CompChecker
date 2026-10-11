from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApplicationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class WorkflowNotImplemented(ApplicationError):
    def __init__(self, workflow: str):
        super().__init__(
            "workflow_not_implemented",
            f"{workflow} is an extension point in this scaffold and is not implemented yet.",
            501,
        )


class RecordNotFound(ApplicationError):
    def __init__(self, resource: str):
        super().__init__("not_found", f"{resource} was not found.", 404)


class ProviderError(ApplicationError):
    """Safe, actionable messages without raw upstream errors or credentials."""

    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(code, message, status_code)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )
