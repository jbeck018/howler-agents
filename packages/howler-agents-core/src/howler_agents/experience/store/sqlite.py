"""SQLite-backed experience store for durable per-repo persistence."""

from __future__ import annotations

import json
import structlog
from datetime import datetime, timezone

from howler_agents.experience.trace import EvolutionaryTrace
from howler_agents.persistence.db import DatabaseManager

log = structlog.get_logger(__name__)

_INSERT_SQL = """
    INSERT INTO traces (
        id, agent_id, run_id, generation,
        task_description, outcome, score,
        key_decisions, lessons_learned, patches_applied, parent_ids,
        recorded_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO NOTHING
"""

_SELECT_BY_AGENT = """
    SELECT * FROM traces WHERE agent_id = ? ORDER BY recorded_at DESC
"""

_SELECT_BY_RUN = """
    SELECT * FROM traces WHERE run_id = ? ORDER BY recorded_at DESC LIMIT ?
"""

_SELECT_BY_GENERATION = """
    SELECT * FROM traces WHERE run_id = ? AND generation = ? ORDER BY recorded_at DESC
"""

_DELETE_BY_RUN = "DELETE FROM traces WHERE run_id = ?"


def _serialize(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _deserialize(raw: str) -> list[str]:
    return json.loads(raw)


def _parse_dt(value: str) -> datetime:
    """Parse an ISO datetime string, always returning a UTC-aware datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_trace(row: dict) -> EvolutionaryTrace:
    return EvolutionaryTrace(
        id=row["id"],
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        generation=row["generation"],
        task_description=row["task_description"],
        outcome=row["outcome"],
        score=row["score"],
        key_decisions=_deserialize(row["key_decisions"]),
        lessons_learned=_deserialize(row["lessons_learned"]),
        patches_applied=_deserialize(row["patches_applied"]),
        parent_ids=_deserialize(row["parent_ids"]),
        recorded_at=_parse_dt(row["recorded_at"]),
    )


class SQLiteStore:
    """Durable SQLite-backed experience store for per-repo persistence."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, trace: EvolutionaryTrace) -> None:
        """Persist an EvolutionaryTrace, silently ignoring duplicate IDs."""
        params = (
            trace.id,
            trace.agent_id,
            trace.run_id,
            trace.generation,
            trace.task_description,
            trace.outcome,
            trace.score,
            _serialize(trace.key_decisions),
            _serialize(trace.lessons_learned),
            _serialize(trace.patches_applied),
            _serialize(trace.parent_ids),
            trace.recorded_at.isoformat(),
        )
        await self._db.execute_write(_INSERT_SQL, params)
        log.debug("trace_saved", trace_id=trace.id, agent_id=trace.agent_id)

    async def get_by_agent(self, agent_id: str) -> list[EvolutionaryTrace]:
        """Return all traces for a given agent, newest first."""
        rows = await self._db.execute(_SELECT_BY_AGENT, (agent_id,))
        return [_row_to_trace(r) for r in rows]

    async def get_by_run(self, run_id: str, limit: int = 100) -> list[EvolutionaryTrace]:
        """Return up to `limit` traces for a run, newest first."""
        rows = await self._db.execute(_SELECT_BY_RUN, (run_id, limit))
        return [_row_to_trace(r) for r in rows]

    async def get_by_generation(
        self, run_id: str, generation: int
    ) -> list[EvolutionaryTrace]:
        """Return all traces for a specific generation within a run."""
        rows = await self._db.execute(_SELECT_BY_GENERATION, (run_id, generation))
        return [_row_to_trace(r) for r in rows]

    async def delete_by_run(self, run_id: str) -> int:
        """Delete all traces for a run. Returns the count of deleted rows."""
        count = await self._db.execute_write(_DELETE_BY_RUN, (run_id,))
        log.info("traces_deleted", run_id=run_id, count=count)
        return count
