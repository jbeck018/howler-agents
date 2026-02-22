"""Pydantic request/response models for REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunConfigSchema(BaseModel):
    population_size: int = 10
    group_size: int = 3
    num_iterations: int = 5
    alpha: float = 0.5
    num_probes: int = 20
    llm_config: dict[str, str] = Field(default_factory=dict)
    task_domain: str = "general"
    task_config: dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    config: RunConfigSchema


class AgentSchema(BaseModel):
    id: str
    run_id: str
    generation: int
    parent_id: str | None = None
    group_id: str | None = None
    performance_score: float = 0
    novelty_score: float = 0
    combined_score: float = 0
    capability_vector: list[float] = Field(default_factory=list)
    created_at: datetime | None = None


class RunSchema(BaseModel):
    id: str
    config: RunConfigSchema
    status: str
    current_generation: int
    total_generations: int
    best_score: float = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RunListResponse(BaseModel):
    runs: list[RunSchema]
    total: int


class TraceSubmitRequest(BaseModel):
    agent_id: str
    task_description: str
    outcome: str
    score: float
    key_decisions: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)


class ProbeResultsRequest(BaseModel):
    agent_id: str
    results: list[bool]


class TraceSchema(BaseModel):
    id: str
    agent_id: str
    run_id: str
    generation: int
    task_description: str
    outcome: str
    score: float
    key_decisions: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)
    recorded_at: datetime | None = None


class ProbeResultsResponse(BaseModel):
    capability_vector: list[float]
