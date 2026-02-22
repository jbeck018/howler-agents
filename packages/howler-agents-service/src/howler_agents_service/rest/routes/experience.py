"""Experience submission endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from howler_agents_service.auth.deps import CurrentUserDep
from howler_agents_service.db.deps import TracesRepoDep
from howler_agents_service.rest.schemas import (
    ProbeResultsRequest,
    ProbeResultsResponse,
    TraceSchema,
    TraceSubmitRequest,
)

router = APIRouter()


@router.get("/runs/{run_id}/traces", response_model=list[TraceSchema])
async def list_traces(
    run_id: str,
    repo: TracesRepoDep,
    current_user: CurrentUserDep,
    limit: int = 100,
) -> list[TraceSchema]:
    traces = await repo.list_by_run(UUID(run_id), limit=limit)
    return [
        TraceSchema(
            id=str(t.id),
            agent_id=str(t.agent_id),
            run_id=str(t.run_id),
            generation=t.generation,
            task_description=t.task_description,
            outcome=t.outcome,
            score=t.score,
            key_decisions=list(t.key_decisions or []),
            lessons_learned=list(t.lessons_learned or []),
            recorded_at=t.recorded_at,
        )
        for t in traces
    ]


@router.post("/runs/{run_id}/experience")
async def submit_experience(
    run_id: str,
    request: TraceSubmitRequest,
    repo: TracesRepoDep,
    current_user: CurrentUserDep,
) -> dict[str, bool]:
    await repo.create(
        agent_id=UUID(request.agent_id),
        run_id=UUID(run_id),
        generation=0,
        task_description=request.task_description,
        outcome=request.outcome,
        score=request.score,
        key_decisions=request.key_decisions,
        lessons_learned=request.lessons_learned,
        org_id=current_user.org_id,
    )
    return {"accepted": True}


@router.post("/runs/{run_id}/probes", response_model=ProbeResultsResponse)
async def submit_probe_results(
    run_id: str,
    request: ProbeResultsRequest,
    current_user: CurrentUserDep,
) -> ProbeResultsResponse:
    vector = [1.0 if r else 0.0 for r in request.results]
    return ProbeResultsResponse(capability_vector=vector)
