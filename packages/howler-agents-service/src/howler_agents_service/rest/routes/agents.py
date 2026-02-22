"""Agent query endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from howler_agents_service.auth.deps import CurrentUserDep
from howler_agents_service.db.deps import AgentsRepoDep
from howler_agents_service.rest.schemas import AgentSchema

router = APIRouter()


def _agent_to_schema(agent) -> AgentSchema:
    """Convert an ORM AgentModel to the REST AgentSchema."""
    return AgentSchema(
        id=str(agent.id),
        run_id=str(agent.run_id),
        generation=agent.generation,
        parent_id=str(agent.parent_id) if agent.parent_id else None,
        group_id=str(agent.group_id) if agent.group_id else None,
        performance_score=agent.performance_score or 0,
        novelty_score=agent.novelty_score or 0,
        combined_score=agent.combined_score or 0,
        capability_vector=agent.capability_vector or [],
        created_at=agent.created_at,
    )


@router.get("/runs/{run_id}/agents", response_model=list[AgentSchema])
async def list_agents(
    run_id: str,
    repo: AgentsRepoDep,
    current_user: CurrentUserDep,
) -> list[AgentSchema]:
    agents = await repo.list_by_run(UUID(run_id))
    return [_agent_to_schema(a) for a in agents]


@router.get("/runs/{run_id}/agents/best", response_model=list[AgentSchema])
async def get_best_agents(
    run_id: str,
    repo: AgentsRepoDep,
    current_user: CurrentUserDep,
    top_k: int = 5,
) -> list[AgentSchema]:
    agents = await repo.get_best(UUID(run_id), top_k=top_k)
    return [_agent_to_schema(a) for a in agents]
