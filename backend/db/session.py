from pathlib import Path

from sqlalchemy import URL, create_engine, event
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            URL.create("sqlite", database=str(path)),
            connect_args={"check_same_thread": False, "timeout": 10},
        )

        @event.listens_for(self.engine, "connect")
        def configure_sqlite(connection, record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def close(self) -> None:
        self.engine.dispose()
