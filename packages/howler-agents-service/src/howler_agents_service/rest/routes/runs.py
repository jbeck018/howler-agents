"""Evolution run management endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from howler_agents_service.auth.deps import CurrentUserDep
from howler_agents_service.db.deps import RunsRepoDep
from howler_agents_service.rest.schemas import (
    CreateRunRequest,
    RunConfigSchema,
    RunListResponse,
    RunSchema,
)

router = APIRouter()


def _run_to_schema(run) -> RunSchema:
    """Convert an ORM EvolutionRunModel to the REST RunSchema."""
    config_data = run.config if isinstance(run.config, dict) else {}
    return RunSchema(
        id=str(run.id),
        config=RunConfigSchema(**config_data),
        status=run.status,
        current_generation=run.current_generation,
        total_generations=run.total_generations,
        best_score=run.best_score or 0,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.post("/runs", response_model=RunSchema, status_code=201)
async def create_run(
    request: CreateRunRequest,
    repo: RunsRepoDep,
    current_user: CurrentUserDep,
) -> RunSchema:
    run = await repo.create(
        config=request.config.model_dump(),
        total_generations=request.config.num_iterations,
        org_id=current_user.org_id,
    )
    return _run_to_schema(run)


@router.get("/runs/{run_id}", response_model=RunSchema)
async def get_run(
    run_id: str,
    repo: RunsRepoDep,
    current_user: CurrentUserDep,
) -> RunSchema:
    run = await repo.get(UUID(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_schema(run)


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    repo: RunsRepoDep,
    current_user: CurrentUserDep,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
) -> RunListResponse:
    runs, total = await repo.list(limit=limit, offset=offset, status=status)
    return RunListResponse(
        runs=[_run_to_schema(r) for r in runs],
        total=total,
    )


@router.post("/runs/{run_id}/step", response_model=RunSchema)
async def step_evolution(
    run_id: str,
    repo: RunsRepoDep,
    current_user: CurrentUserDep,
) -> RunSchema:
    uid = UUID(run_id)
    run = await repo.get(uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status == "completed":
        raise HTTPException(status_code=400, detail="Run already completed")

    new_gen = min(run.current_generation + 1, run.total_generations)
    new_status = "completed" if new_gen >= run.total_generations else "running"

    updated = await repo.update_status(
        uid,
        status=new_status,
        current_generation=new_gen,
        best_score=run.best_score,
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update run")
    return _run_to_schema(updated)
