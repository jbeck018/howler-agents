"""Repository for evolutionary traces."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from howler_agents_service.db.models import EvolutionaryTraceModel


class TracesRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **kwargs) -> EvolutionaryTraceModel:
        trace = EvolutionaryTraceModel(**kwargs)
        self._session.add(trace)
        await self._session.commit()
        await self._session.refresh(trace)
        return trace

    async def list_by_run(self, run_id: UUID, limit: int = 100) -> list[EvolutionaryTraceModel]:
        result = await self._session.execute(
            select(EvolutionaryTraceModel)
            .where(EvolutionaryTraceModel.run_id == run_id)
            .order_by(EvolutionaryTraceModel.recorded_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_agent(self, agent_id: UUID) -> list[EvolutionaryTraceModel]:
        result = await self._session.execute(
            select(EvolutionaryTraceModel)
            .where(EvolutionaryTraceModel.agent_id == agent_id)
            .order_by(EvolutionaryTraceModel.recorded_at)
        )
        return list(result.scalars().all())
