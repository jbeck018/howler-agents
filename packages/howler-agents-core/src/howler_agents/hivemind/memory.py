"""Collective memory for the hive-mind: CRUD + text search over the memory table."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from howler_agents.persistence.db import DatabaseManager

log = structlog.get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CollectiveMemory:
    """CRUD + search for the hive-mind's collective memory.

    Memory persists in the SQLite ``memory`` table across sessions.
    Supports namespace-based organisation and text search.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def store(
        self,
        namespace: str,
        key: str,
        value: str,
        tags: list[str] | None = None,
        score: float = 0.0,
    ) -> str:
        """Store a memory entry and return its ID.

        Uses UPSERT: if (namespace, key) already exists the value, score,
        tags, and updated_at timestamp are refreshed.
        """
        entry_id = str(uuid.uuid4())
        tags_json = json.dumps(tags or [])
        now = _now_iso()

        await self._db.execute_write(
            """
            INSERT INTO memory (id, namespace, key, value, tags, score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value      = excluded.value,
                tags       = excluded.tags,
                score      = excluded.score,
                updated_at = excluded.updated_at
            """,
            (entry_id, namespace, key, value, tags_json, score, now, now),
        )

        log.debug("memory_stored", namespace=namespace, key=key, score=score)
        return entry_id

    async def retrieve(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Return a single memory entry and increment its access counter."""
        rows = await self._db.execute(
            "SELECT * FROM memory WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        if not rows:
            return None

        entry = rows[0]
        await self._db.execute_write(
            "UPDATE memory SET access_count = access_count + 1, updated_at = ? WHERE namespace = ? AND key = ?",
            (_now_iso(), namespace, key),
        )
        log.debug("memory_retrieved", namespace=namespace, key=key)
        return entry

    async def search(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memory by text matching (LIKE on key and value columns).

        Falls back to listing by score when *query* is empty.
        """
        if not query:
            return await self.list(namespace=namespace, limit=limit)

        pattern = f"%{query}%"

        if namespace is not None:
            rows = await self._db.execute(
                """
                SELECT * FROM memory
                WHERE namespace = ?
                  AND (key LIKE ? OR value LIKE ?)
                ORDER BY score DESC, access_count DESC
                LIMIT ?
                """,
                (namespace, pattern, pattern, limit),
            )
        else:
            rows = await self._db.execute(
                """
                SELECT * FROM memory
                WHERE key LIKE ? OR value LIKE ?
                ORDER BY score DESC, access_count DESC
                LIMIT ?
                """,
                (pattern, pattern, limit),
            )

        log.debug("memory_search", query=query, namespace=namespace, hits=len(rows))
        return rows

    async def list(
        self,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List memory entries ordered by recency, optionally filtered by namespace."""
        if namespace is not None:
            return await self._db.execute(
                "SELECT * FROM memory WHERE namespace = ? ORDER BY updated_at DESC LIMIT ?",
                (namespace, limit),
            )
        return await self._db.execute(
            "SELECT * FROM memory ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a memory entry. Returns True when a row was removed."""
        affected = await self._db.execute_write(
            "DELETE FROM memory WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        log.debug("memory_deleted", namespace=namespace, key=key, found=affected > 0)
        return affected > 0

    async def stats(self) -> dict[str, Any]:
        """Return aggregate statistics: totals and per-namespace breakdowns."""
        total_rows = await self._db.execute("SELECT COUNT(*) AS n FROM memory")
        total = total_rows[0]["n"] if total_rows else 0

        ns_rows = await self._db.execute(
            """
            SELECT namespace, COUNT(*) AS count, AVG(score) AS avg_score
            FROM memory
            GROUP BY namespace
            ORDER BY count DESC
            """
        )

        return {
            "total_entries": total,
            "namespaces": [dict(r) for r in ns_rows],
        }
