"""PostgreSQL + pgvector experience store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from howler_agents.experience.trace import EvolutionaryTrace

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class PostgresStore:
    """Durable experience store backed by PostgreSQL."""

    def __init__(self, session_factory: object) -> None:
        self._session_factory = session_factory

    async def _get_session(self) -> AsyncSession:
        from sqlalchemy.ext.asyncio import AsyncSession

        factory = self._session_factory
        if callable(factory):
            session = factory()
            if isinstance(session, AsyncSession):
                return session
        raise TypeError("session_factory must return an AsyncSession")

    async def save(self, trace: EvolutionaryTrace) -> None:
        from sqlalchemy import text

        async with await self._get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO evolutionary_traces
                        (id, agent_id, run_id, generation, task_description, outcome,
                         score, key_decisions, lessons_learned, recorded_at)
                    VALUES (:id, :agent_id, :run_id, :generation, :task_description,
                            :outcome, :score, :key_decisions, :lessons_learned, :recorded_at)
                """),
                {
                    "id": trace.id,
                    "agent_id": trace.agent_id,
                    "run_id": trace.run_id,
                    "generation": trace.generation,
                    "task_description": trace.task_description,
                    "outcome": trace.outcome,
                    "score": trace.score,
                    "key_decisions": trace.key_decisions,
                    "lessons_learned": trace.lessons_learned,
                    "recorded_at": trace.recorded_at,
                },
            )
            await session.commit()

    async def get_by_agent(self, agent_id: str) -> list[EvolutionaryTrace]:
        from sqlalchemy import text

        async with await self._get_session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM evolutionary_traces WHERE agent_id = :agent_id ORDER BY recorded_at"
                ),
                {"agent_id": agent_id},
            )
            return [self._row_to_trace(row) for row in result.mappings()]

    async def get_by_run(self, run_id: str, limit: int = 100) -> list[EvolutionaryTrace]:
        from sqlalchemy import text

        async with await self._get_session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM evolutionary_traces WHERE run_id = :run_id ORDER BY recorded_at DESC LIMIT :limit"
                ),
                {"run_id": run_id, "limit": limit},
            )
            return [self._row_to_trace(row) for row in result.mappings()]

    async def get_by_generation(self, run_id: str, generation: int) -> list[EvolutionaryTrace]:
        from sqlalchemy import text

        async with await self._get_session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM evolutionary_traces WHERE run_id = :run_id AND generation = :gen"
                ),
                {"run_id": run_id, "gen": generation},
            )
            return [self._row_to_trace(row) for row in result.mappings()]

    async def delete_by_run(self, run_id: str) -> int:
        from sqlalchemy import text

        async with await self._get_session() as session:
            result = await session.execute(
                text("DELETE FROM evolutionary_traces WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

    @staticmethod
    def _row_to_trace(row: object) -> EvolutionaryTrace:
        r = row  # type: ignore[assignment]
        return EvolutionaryTrace(
            id=str(r["id"]),
            agent_id=str(r["agent_id"]),
            run_id=str(r["run_id"]),
            generation=r["generation"],
            task_description=r["task_description"],
            outcome=r["outcome"],
            score=r["score"],
            key_decisions=list(r["key_decisions"]) if r["key_decisions"] else [],
            lessons_learned=list(r["lessons_learned"]) if r["lessons_learned"] else [],
            recorded_at=r["recorded_at"],
        )
