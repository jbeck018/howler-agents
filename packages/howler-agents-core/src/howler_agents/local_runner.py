"""Local evolution runner - wraps EvolutionLoop for CLI/MCP use.

Provides a zero-config entry point for running evolution loops locally.
State is held in memory during a run and optionally persisted to SQLite
when a DatabaseManager is supplied.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.agents.pool import AgentPool
from howler_agents.config import HowlerConfig, LLMRole, RoleModelConfig
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.experience.trace import EvolutionaryTrace
from howler_agents.llm.router import LLMRouter
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector

logger = structlog.get_logger()


class LLMBackedAgent(Agent):
    """A concrete agent implementation that delegates tasks to an LLM.

    This agent uses the ACTING role of the LLMRouter to execute tasks
    and records its decisions and lessons for the experience pool.
    """

    def __init__(self, config: AgentConfig, llm: LLMRouter, task_domain: str) -> None:
        super().__init__(config)
        self._llm = llm
        self._task_domain = task_domain

    async def run_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute a task by prompting the ACTING LLM."""
        description = task.get("description", str(task))
        expected = task.get("expected", "")

        system_prompt = (
            f"You are an AI agent specializing in {self._task_domain} tasks.\n"
            f"Framework config: {self.config.framework_config}\n"
            "Solve the task concisely. After your answer, on a new line write:\n"
            "DECISION: <your key decision>\n"
            "LESSON: <what you learned>\n"
            "SCORE: <0.0-1.0 self-assessment>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ]

        try:
            response = await self._llm.complete(role=LLMRole.ACTING, messages=messages)
            lines = response.strip().splitlines()

            decision = ""
            lesson = ""
            score = 0.5
            answer_lines = []

            for line in lines:
                upper = line.upper()
                if upper.startswith("DECISION:"):
                    decision = line.split(":", 1)[1].strip()
                elif upper.startswith("LESSON:"):
                    lesson = line.split(":", 1)[1].strip()
                elif upper.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        score = 0.5
                else:
                    answer_lines.append(line)

            answer = "\n".join(answer_lines).strip()

            # Determine success by comparing to expected if provided
            success = True
            if expected:
                success = expected.lower() in answer.lower()

            return TaskResult(
                success=success,
                score=score,
                output=answer,
                key_decisions=[decision] if decision else [],
                lessons_learned=[lesson] if lesson else [],
            )

        except Exception as exc:
            logger.warning("agent_task_error", agent_id=self.id, error=str(exc))
            return TaskResult(
                success=False,
                score=0.0,
                output=f"Error: {exc}",
                key_decisions=[],
                lessons_learned=["Encountered an error; improve error handling"],
            )

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        """Apply a mutation patch to this agent's framework config."""
        self.patches.append(patch)
        # Patches are advisory descriptions; store them in framework_config
        if patch.intent:
            patches_list: list[str] = self.config.framework_config.get("applied_patches", [])
            patches_list.append(patch.intent)
            self.config.framework_config["applied_patches"] = patches_list
        logger.debug("patch_applied", agent_id=self.id, intent=patch.intent)


@dataclass
class RunRecord:
    """Metadata and live state for a single local evolution run."""

    run_id: str
    config: HowlerConfig
    status: str = "pending"  # pending | running | completed | failed
    current_generation: int = 0
    best_score: float = 0.0
    best_agent_id: str | None = None
    generation_summaries: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None

    # Live objects kept alive for the duration of the run
    pool: AgentPool = field(default_factory=AgentPool)
    experience: SharedExperiencePool | None = None


class LocalRunner:
    """Manages local evolution runs with optional SQLite persistence.

    A single LocalRunner instance can manage multiple concurrent runs.
    Each run gets its own AgentPool, ExperiencePool, and store.

    When a DatabaseManager is supplied, runs are persisted to SQLite and
    hydrated back on startup so they survive process restarts. Without a
    DatabaseManager the runner falls back to purely in-memory state,
    preserving full backward compatibility.
    """

    def __init__(self, db: Any | None = None) -> None:
        self._runs: dict[str, RunRecord] = {}
        # Session-level model config overrides
        self._session_config: dict[str, Any] = {}
        self._db: Any | None = db  # DatabaseManager | None
        self._store: Any | None = None  # SQLiteStore | InMemoryStore | None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialize persistence. Must be called before operations when db is set."""
        if self._db is not None:
            await self._db.initialize()
            from howler_agents.experience.store.sqlite import SQLiteStore
            self._store = SQLiteStore(self._db)
            await self._hydrate_from_db()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()

    async def _hydrate_from_db(self) -> None:
        """Load completed/failed runs from SQLite so they appear in list_runs()."""
        if self._db is None:
            return
        rows = await self._db.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 50"
        )
        for row in rows:
            if row["run_id"] not in self._runs:
                config = self._build_config(
                    population_size=row["population_size"],
                    group_size=row["group_size"],
                    num_iterations=row["num_iterations"],
                    alpha=row["alpha"],
                    task_domain=row["task_domain"],
                    model=row["model"],
                )
                record = RunRecord(
                    run_id=row["run_id"],
                    config=config,
                    status=row["status"],
                    current_generation=row["current_generation"],
                    best_score=row["best_score"],
                    best_agent_id=row.get("best_agent_id"),
                    generation_summaries=json.loads(row["generation_summaries"]),
                    started_at=datetime.fromisoformat(row["started_at"]),
                    finished_at=(
                        datetime.fromisoformat(row["finished_at"])
                        if row["finished_at"]
                        else None
                    ),
                    error=row.get("error"),
                )
                # Mark interrupted runs as failed
                if record.status == "running":
                    record.status = "failed"
                    record.error = "Interrupted (process restarted)"
                    if self._db is not None:
                        await self._db.execute_write(
                            "UPDATE runs SET status = 'failed', error = 'Interrupted (process restarted)' WHERE run_id = ?",
                            (row["run_id"],),
                        )
                self._runs[row["run_id"]] = record

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def configure(
        self,
        acting_model: str | None = None,
        evolving_model: str | None = None,
        reflecting_model: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Persist model configuration for the session."""
        if acting_model:
            self._session_config["acting_model"] = acting_model
        if evolving_model:
            self._session_config["evolving_model"] = evolving_model
        if reflecting_model:
            self._session_config["reflecting_model"] = reflecting_model
        if api_key is not None:
            self._session_config["api_key"] = api_key
        return dict(self._session_config)

    def get_session_config(self) -> dict[str, Any]:
        return dict(self._session_config)

    # ------------------------------------------------------------------ #
    # Run lifecycle                                                        #
    # ------------------------------------------------------------------ #

    def _build_config(
        self,
        population_size: int,
        group_size: int,
        num_iterations: int,
        alpha: float,
        task_domain: str,
        model: str,
    ) -> HowlerConfig:
        """Build a HowlerConfig merging args with session-level overrides."""
        acting_model = self._session_config.get("acting_model", model)
        evolving_model = self._session_config.get("evolving_model", model)
        reflecting_model = self._session_config.get("reflecting_model", model)
        api_key: str | None = self._session_config.get("api_key")

        return HowlerConfig(
            population_size=population_size,
            group_size=group_size,
            num_iterations=num_iterations,
            alpha=alpha,
            task_domain=task_domain,
            role_models={
                LLMRole.ACTING: RoleModelConfig(model=acting_model, api_key=api_key),
                LLMRole.EVOLVING: RoleModelConfig(model=evolving_model, api_key=api_key),
                LLMRole.REFLECTING: RoleModelConfig(model=reflecting_model, api_key=api_key),
            },
        )

    def _build_domain_tasks(self, config: HowlerConfig) -> list[dict[str, Any]]:
        """Return a list of tasks appropriate for the configured domain."""
        domain = config.task_domain.lower()
        domain_tasks: dict[str, list[dict[str, Any]]] = {
            "coding": [
                {"description": "Write a Python function to compute the factorial of n.", "type": "code_gen"},
                {"description": "Debug this code: for i in range(10): print(i", "type": "debugging"},
                {"description": "Refactor: replace nested if/else with a dict dispatch table.", "type": "refactoring"},
            ],
            "math": [
                {"description": "Compute the derivative of f(x) = x^3 + 2x^2 - 5.", "type": "calculus"},
                {"description": "Solve: 3x + 7 = 22", "type": "algebra"},
                {"description": "Find the area of a circle with radius 5.", "type": "geometry"},
            ],
            "writing": [
                {"description": "Write a persuasive paragraph about renewable energy.", "type": "persuasion"},
                {"description": "Summarize the concept of machine learning in two sentences.", "type": "summarization"},
                {"description": "Edit this sentence for clarity: 'The thing that it does is bad.'", "type": "editing"},
            ],
        }
        return domain_tasks.get(domain, [
            {"description": "Explain what you are capable of.", "type": "general"},
            {"description": "List three strategies for solving complex problems.", "type": "general"},
            {"description": "Describe a best practice for collaborative AI systems.", "type": "general"},
        ])

    def start_run(
        self,
        population_size: int = 10,
        group_size: int = 3,
        num_iterations: int = 5,
        alpha: float = 0.5,
        task_domain: str = "general",
        model: str = "claude-sonnet-4-20250514",
    ) -> str:
        """Register a new run and return its run_id.

        The run does NOT start immediately; call ``run_async`` to execute it.
        """
        run_id = str(uuid.uuid4())
        config = self._build_config(
            population_size=population_size,
            group_size=group_size,
            num_iterations=num_iterations,
            alpha=alpha,
            task_domain=task_domain,
            model=model,
        )
        record = RunRecord(run_id=run_id, config=config)
        self._runs[run_id] = record
        logger.info("run_registered", run_id=run_id, domain=task_domain, model=model)
        return run_id

    async def run_async(self, run_id: str) -> dict[str, Any]:
        """Execute the evolution loop for a registered run.

        Returns the final run summary dict.
        """
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if record.status == "running":
            raise RuntimeError(f"Run {run_id} is already executing.")

        record.status = "running"
        config = record.config

        # Persist the new run to DB before starting execution
        if self._db is not None:
            await self._db.execute_write(
                """INSERT OR REPLACE INTO runs
                   (run_id, status, task_domain, population_size, group_size,
                    num_iterations, alpha, model, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    "running",
                    config.task_domain,
                    config.population_size,
                    config.group_size,
                    config.num_iterations,
                    config.alpha,
                    config.role_models[LLMRole.ACTING].model,
                    record.started_at.isoformat(),
                ),
            )

        try:
            # Build store: use SQLiteStore if available, else fall back to in-memory
            if self._store is not None:
                store = self._store
            else:
                store = InMemoryStore()
            experience = SharedExperiencePool(store)
            record.experience = experience

            llm = LLMRouter(config)
            probe_registry = ProbeRegistry()
            probe_registry.register_default_probes(config.num_probes)

            # Seed the agent pool
            for i in range(config.population_size):
                group_id = f"group_{i % max(1, config.population_size // config.group_size)}"
                agent_config = AgentConfig(
                    generation=0,
                    group_id=group_id,
                    framework_config={"domain": config.task_domain, "index": i},
                )
                agent = LLMBackedAgent(config=agent_config, llm=llm, task_domain=config.task_domain)
                record.pool.add(agent)

            selector = PerformanceNoveltySelector(alpha=config.alpha)
            reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)
            probe_evaluator = ProbeEvaluator(registry=probe_registry)

            loop = EvolutionLoop(
                config=config,
                pool=record.pool,
                selector=selector,
                reproducer=reproducer,
                experience=experience,
                probe_evaluator=probe_evaluator,
            )

            tasks = self._build_domain_tasks(config)
            result = await loop.run(run_id=run_id, tasks=tasks)

            record.status = "completed"
            record.generation_summaries = result.get("generations", [])
            record.best_score = result.get("best_score", 0.0)
            if record.generation_summaries:
                last = record.generation_summaries[-1]
                record.best_agent_id = last.get("best_agent_id")
            record.current_generation = config.num_iterations
            record.finished_at = datetime.now(timezone.utc)

            agents = record.pool.agents
            mean_score = (
                sum(a.combined_score for a in agents) / len(agents) if agents else 0.0
            )

            if self._db is not None:
                await self._db.execute_write(
                    """UPDATE runs SET status = ?, best_score = ?, best_agent_id = ?,
                       current_generation = ?, generation_summaries = ?,
                       finished_at = ?, mean_score = ?
                       WHERE run_id = ?""",
                    (
                        "completed",
                        record.best_score,
                        record.best_agent_id,
                        record.current_generation,
                        json.dumps(record.generation_summaries),
                        record.finished_at.isoformat(),
                        mean_score,
                        run_id,
                    ),
                )
                await self._seed_hivemind(run_id)

            logger.info("run_completed", run_id=run_id, best_score=record.best_score)
            return result

        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.finished_at = datetime.now(timezone.utc)

            if self._db is not None:
                await self._db.execute_write(
                    "UPDATE runs SET status = 'failed', error = ?, finished_at = ? WHERE run_id = ?",
                    (
                        str(exc),
                        record.finished_at.isoformat(),
                        run_id,
                    ),
                )

            logger.error("run_failed", run_id=run_id, error=str(exc))
            raise

    async def _seed_hivemind(self, run_id: str) -> None:
        """Seed hive-mind memory from a completed run's traces."""
        if self._db is None:
            return
        try:
            from howler_agents.hivemind.coordinator import HiveMindCoordinator
            coordinator = HiveMindCoordinator(self._db)
            result = await coordinator.seed_from_run(run_id)
            logger.info("hivemind_seeded", run_id=run_id, **result)
        except Exception as exc:
            logger.warning("hivemind_seed_failed", run_id=run_id, error=str(exc))

    def get_status(self, run_id: str) -> dict[str, Any]:
        """Return current status of a run."""
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id: {run_id}")

        agents = record.pool.agents
        mean_score = (
            sum(a.combined_score for a in agents) / len(agents) if agents else 0.0
        )

        return {
            "run_id": run_id,
            "status": record.status,
            "current_generation": record.current_generation,
            "total_generations": record.config.num_iterations,
            "population_size": record.pool.size,
            "best_score": record.best_score,
            "best_agent_id": record.best_agent_id,
            "mean_score": round(mean_score, 4),
            "task_domain": record.config.task_domain,
            "started_at": record.started_at.isoformat(),
            "finished_at": record.finished_at.isoformat() if record.finished_at else None,
            "error": record.error,
        }

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary list of all known runs."""
        return [self.get_status(run_id) for run_id in self._runs]

    def list_agents(self, run_id: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """List agents in the pool for a run."""
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id: {run_id}")

        agents = record.pool.agents
        if top_k is not None:
            agents = record.pool.top_k(top_k)

        return [
            {
                "agent_id": a.id,
                "generation": a.config.generation,
                "group_id": a.config.group_id,
                "performance_score": round(a.performance_score, 4),
                "novelty_score": round(a.novelty_score, 4),
                "combined_score": round(a.combined_score, 4),
                "capability_vector": a.capability_vector,
                "patches_applied": len(a.patches),
            }
            for a in agents
        ]

    async def submit_experience(
        self,
        run_id: str,
        agent_id: str,
        task_description: str,
        outcome: str,
        score: float,
        key_decisions: list[str],
        lessons_learned: list[str],
    ) -> str:
        """Manually submit a task experience trace to the pool."""
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if record.experience is None:
            raise RuntimeError(f"Run {run_id} has not been initialized yet.")

        trace = EvolutionaryTrace(
            agent_id=agent_id,
            run_id=run_id,
            generation=record.current_generation,
            task_description=task_description,
            outcome=outcome,
            score=score,
            key_decisions=key_decisions,
            lessons_learned=lessons_learned,
        )
        await record.experience.submit(trace)
        return trace.id

    async def get_experience_context(
        self, run_id: str, group_id: str | None = None
    ) -> str:
        """Retrieve formatted experience context for a run/group."""
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if record.experience is None:
            return "No experience available — run has not started."

        gid = group_id or "default"
        return await record.experience.get_group_context(
            run_id=run_id,
            group_id=gid,
            generation=record.current_generation,
        )
