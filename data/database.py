"""Database connection helpers (SQLite through SQLAlchemy)."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from data.models import Base

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "sqlite:///recsys.db"
_IN_MEMORY_URLS = ("sqlite://", "sqlite:///:memory:")
SQLITE_BUSY_TIMEOUT_MS = 5000


def make_engine(url: str | None = None) -> Engine:
    """Create an engine.

    Falls back to the ``DATABASE_URL`` environment variable and then to a
    local ``recsys.db`` file.  In-memory SQLite databases use a static
    pool so every thread sees the same database.  File-based SQLite uses
    WAL journaling and a busy timeout so concurrent threads wait for each
    other instead of failing with "database is locked".  Other databases
    get ``pool_pre_ping`` to survive dropped connections.
    """
    url = url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    kwargs: dict[str, Any] = {}
    is_sqlite = url.startswith("sqlite")
    in_memory = url in _IN_MEMORY_URLS
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
        if in_memory:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if not in_memory:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(engine)


def drop_db(engine: Engine) -> None:
    """Drop every table (used by the seed script and tests)."""
    Base.metadata.drop_all(engine)


@contextmanager
def session_scope(
    session_factory: sessionmaker[Session],
) -> Iterator[Session]:
    """Provide a transactional scope: commit on success, else roll back.

    Exceptions are re-raised unchanged after the rollback.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.debug(
            "transaction rolled back", extra={"error": type(exc).__name__}
        )
        raise
    finally:
        session.close()
