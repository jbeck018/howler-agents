"""FastAPI dependency injection for database sessions and repositories."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from howler_agents_service.db.engine import get_session_factory
from howler_agents_service.db.repositories.agents import AgentsRepo
from howler_agents_service.db.repositories.auth import AuthRepo
from howler_agents_service.db.repositories.runs import RunsRepo
from howler_agents_service.db.repositories.traces import TracesRepo


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session, auto-closing on exit."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_runs_repo(session: SessionDep) -> RunsRepo:
    return RunsRepo(session)


def get_agents_repo(session: SessionDep) -> AgentsRepo:
    return AgentsRepo(session)


def get_traces_repo(session: SessionDep) -> TracesRepo:
    return TracesRepo(session)


def get_auth_repo(session: SessionDep) -> AuthRepo:
    return AuthRepo(session)


RunsRepoDep = Annotated[RunsRepo, Depends(get_runs_repo)]
AgentsRepoDep = Annotated[AgentsRepo, Depends(get_agents_repo)]
TracesRepoDep = Annotated[TracesRepo, Depends(get_traces_repo)]
AuthRepoDep = Annotated[AuthRepo, Depends(get_auth_repo)]
