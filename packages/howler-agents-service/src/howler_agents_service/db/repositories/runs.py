"""Repository for evolution runs."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from howler_agents_service.db.models import EvolutionRunModel


class RunsRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        config: dict[str, Any],
        total_generations: int,
        org_id: Any = None,
    ) -> EvolutionRunModel:
        run = EvolutionRunModel(
            config=config,
            total_generations=total_generations,
            org_id=org_id,
        )
        self._session.add(run)
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def get(self, run_id: UUID) -> EvolutionRunModel | None:
        return await self._session.get(EvolutionRunModel, run_id)

    async def list(
        self, limit: int = 20, offset: int = 0, status: str | None = None
    ) -> tuple[list[EvolutionRunModel], int]:
        query = select(EvolutionRunModel)
        count_query = select(func.count()).select_from(EvolutionRunModel)
        if status:
            query = query.where(EvolutionRunModel.status == status)
            count_query = count_query.where(EvolutionRunModel.status == status)
        query = query.order_by(EvolutionRunModel.created_at.desc()).offset(offset).limit(limit)
        result = await self._session.execute(query)
        total_result = await self._session.execute(count_query)
        return list(result.scalars().all()), total_result.scalar_one()

    async def update_status(
        self, run_id: UUID, status: str, **kwargs: Any
    ) -> EvolutionRunModel | None:
        run = await self.get(run_id)
        if run:
            run.status = status
            for key, value in kwargs.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            await self._session.commit()
            await self._session.refresh(run)
        return run
