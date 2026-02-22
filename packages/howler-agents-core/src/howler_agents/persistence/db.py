"""Async SQLite connection manager for howler-agents persistence."""

from __future__ import annotations

import structlog
from pathlib import Path
from typing import Any

import aiosqlite

log = structlog.get_logger(__name__)

_PRAGMA_WAL = "PRAGMA journal_mode = WAL"
_PRAGMA_FK = "PRAGMA foreign_keys = ON"


class DatabaseManager:
    """Manages an aiosqlite connection with WAL mode and foreign keys enabled.

    Reusable by the experience store, hivemind, and mcp_server modules.

    Usage::

        db = DatabaseManager()           # auto-detects repo root
        await db.initialize()
        rows = await db.execute("SELECT * FROM runs")
        await db.close()
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            from howler_agents.persistence.repo import get_db_path
            db_path = get_db_path()

        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        log.debug("db_manager_created", path=str(self._db_path))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the connection, enable pragmas, and run migrations."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        from howler_agents.persistence.repo import ensure_gitignore
        ensure_gitignore(self._db_path.parent)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute(_PRAGMA_WAL)
        await self._conn.execute(_PRAGMA_FK)
        await self._conn.commit()

        from howler_agents.persistence.migrations import run_migrations
        await run_migrations(self)

        log.info("db_initialized", path=str(self._db_path))

    async def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            log.debug("db_closed", path=str(self._db_path))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a SELECT statement and return rows as plain dicts."""
        self._require_connection()
        async with self._conn.execute(sql, params) as cursor:  # type: ignore[union-attr]
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def execute_write(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """Execute an INSERT / UPDATE / DELETE / DDL statement.

        Returns the number of rows affected (0 for DDL).
        """
        self._require_connection()
        async with self._conn.execute(sql, params) as cursor:  # type: ignore[union-attr]
            await self._conn.commit()  # type: ignore[union-attr]
            return cursor.rowcount if cursor.rowcount >= 0 else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connection(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "DatabaseManager is not initialized. Call await db.initialize() first."
            )
