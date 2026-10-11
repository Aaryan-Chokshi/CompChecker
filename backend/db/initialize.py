from backend.db.base import Base
from backend.db.session import Database
from backend.features.compliance import models as compliance_models  # noqa: F401
from backend.features.documents import models as document_models  # noqa: F401


def initialize_database(database: Database) -> None:
    # New chunk storage is additive; existing document/run columns are unchanged.
    Base.metadata.create_all(database.engine)
