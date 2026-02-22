"""Service test fixtures with in-memory mock repos."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from howler_agents_service.auth.deps import get_current_user
from howler_agents_service.auth.models import CurrentUser
from howler_agents_service.db.deps import get_agents_repo, get_runs_repo, get_traces_repo
from howler_agents_service.rest.routes.agents import router as agents_router
from howler_agents_service.rest.routes.experience import router as experience_router
from howler_agents_service.rest.routes.health import router as health_router
from howler_agents_service.rest.routes.runs import router as runs_router


class InMemoryRunsRepo:
    """In-memory runs repository for testing."""

    def __init__(self):
        self._runs: dict[str, Any] = {}

    async def create(self, config: dict, total_generations: int, org_id=None):
        run_id = uuid.uuid4()
        now = datetime.now(UTC)
        run = MagicMock()
        run.id = run_id
        run.config = config
        run.status = "pending"
        run.current_generation = 0
        run.total_generations = total_generations
        run.best_score = 0
        run.best_agent_id = None
        run.error_message = None
        run.org_id = org_id
        run.created_at = now
        run.updated_at = now
        self._runs[str(run_id)] = run
        return run

    async def get(self, run_id):
        return self._runs.get(str(run_id))

    async def list(self, limit=20, offset=0, status=None):
        runs = list(self._runs.values())
        if status:
            runs = [r for r in runs if r.status == status]
        total = len(runs)
        return runs[offset : offset + limit], total

    async def update_status(self, run_id, status, **kwargs):
        run = self._runs.get(str(run_id))
        if run:
            run.status = status
            for k, v in kwargs.items():
                setattr(run, k, v)
            run.updated_at = datetime.now(UTC)
        return run


class InMemoryAgentsRepo:
    """In-memory agents repository for testing."""

    def __init__(self):
        self._agents: list[Any] = []

    async def create(self, **kwargs):
        agent = MagicMock()
        agent.id = kwargs.get("id", uuid.uuid4())
        for k, v in kwargs.items():
            setattr(agent, k, v)
        self._agents.append(agent)
        return agent

    async def list_by_run(self, run_id):
        return [a for a in self._agents if str(a.run_id) == str(run_id)]

    async def get_best(self, run_id, top_k=5):
        agents = await self.list_by_run(run_id)
        return sorted(agents, key=lambda a: getattr(a, "combined_score", 0), reverse=True)[:top_k]


class InMemoryTracesRepo:
    """In-memory traces repository for testing."""

    def __init__(self):
        self._traces: list[Any] = []

    async def create(self, **kwargs):
        trace = MagicMock()
        trace.id = uuid.uuid4()
        for k, v in kwargs.items():
            setattr(trace, k, v)
        self._traces.append(trace)
        return trace

    async def list_by_run(self, run_id, limit=100):
        return [t for t in self._traces if str(t.run_id) == str(run_id)][:limit]


@pytest.fixture
def client():
    """Create a test client with in-memory repos (no database needed)."""
    app = FastAPI(title="Howler Agents API (test)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(runs_router, prefix="/api/v1", tags=["runs"])
    app.include_router(agents_router, prefix="/api/v1", tags=["agents"])
    app.include_router(experience_router, prefix="/api/v1", tags=["experience"])

    # Override DB dependencies with in-memory versions
    runs_repo = InMemoryRunsRepo()
    agents_repo = InMemoryAgentsRepo()
    traces_repo = InMemoryTracesRepo()

    # Provide a default authenticated user so existing tests don't need auth headers
    _test_user = CurrentUser(
        user_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        email="test@example.com",
        role="owner",
    )

    app.dependency_overrides[get_runs_repo] = lambda: runs_repo
    app.dependency_overrides[get_agents_repo] = lambda: agents_repo
    app.dependency_overrides[get_traces_repo] = lambda: traces_repo
    app.dependency_overrides[get_current_user] = lambda: _test_user

    return TestClient(app)
