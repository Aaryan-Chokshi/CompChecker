from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from backend.core.providers import Providers


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.database.sessions() as session:
        yield session


def get_providers(request: Request) -> Providers:
    return request.app.state.providers


SessionDependency = Annotated[Session, Depends(get_session)]
ProviderDependency = Annotated[Providers, Depends(get_providers)]
