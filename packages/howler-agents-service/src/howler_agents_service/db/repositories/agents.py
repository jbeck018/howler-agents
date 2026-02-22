"""Repository for agents."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from howler_agents_service.db.models import AgentModel


class AgentsRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **kwargs) -> AgentModel:
        agent = AgentModel(**kwargs)
        self._session.add(agent)
        await self._session.commit()
        await self._session.refresh(agent)
        return agent

    async def get(self, agent_id: UUID) -> AgentModel | None:
        return await self._session.get(AgentModel, agent_id)

    async def list_by_run(self, run_id: UUID) -> list[AgentModel]:
        result = await self._session.execute(
            select(AgentModel).where(AgentModel.run_id == run_id).order_by(AgentModel.combined_score.desc())
        )
        return list(result.scalars().all())

    async def get_best(self, run_id: UUID, top_k: int = 5) -> list[AgentModel]:
        result = await self._session.execute(
            select(AgentModel)
            .where(AgentModel.run_id == run_id)
            .order_by(AgentModel.combined_score.desc())
            .limit(top_k)
        )
        return list(result.scalars().all())
