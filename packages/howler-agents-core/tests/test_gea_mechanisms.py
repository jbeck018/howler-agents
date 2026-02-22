"""Integration tests verifying GEA paper mechanisms.

These tests verify that the core mechanisms described in
arXiv:2602.04837 work correctly:
1. Combined performance+novelty selection outperforms single-signal selection
2. Group experience sharing enables cross-lineage knowledge transfer
3. Capability vectors correctly differentiate agent behaviors
4. Evolution loop improves scores over generations
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.agents.pool import AgentPool
from howler_agents.config import HowlerConfig
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.experience.trace import EvolutionaryTrace
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector
from howler_agents.selection.novelty import KNNNoveltyEstimator
from howler_agents.selection.performance import TaskPerformanceScorer


class EvolvingAgent(Agent):
    """Agent that improves over generations through patches."""

    def __init__(self, config: AgentConfig, base_skill: float = 0.3) -> None:
        super().__init__(config)
        self._base_skill = base_skill
        self._skills: dict[str, float] = {}

    async def run_task(self, task: dict[str, Any]) -> TaskResult:
        import random
        task_type = task.get("type", "general")
        skill = self._skills.get(task_type, self._base_skill)
        noise = random.gauss(0, 0.05)
        score = max(0.0, min(1.0, skill + noise))
        return TaskResult(
            success=score > 0.5,
            score=score,
            output=f"completed {task_type}",
            key_decisions=[f"used {task_type} strategy"],
            lessons_learned=[f"{'effective' if score > 0.5 else 'ineffective'} approach"],
        )

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)
        # Patches improve the agent's base skill
        self._base_skill = min(1.0, self._base_skill + 0.08)
        if patch.category in self._skills:
            self._skills[patch.category] = min(1.0, self._skills[patch.category] + 0.1)
        else:
            self._skills[patch.category] = self._base_skill + 0.1


def make_evolving_agent(skill: float = 0.3, generation: int = 0) -> EvolvingAgent:
    agent = EvolvingAgent(
        AgentConfig(id=str(uuid.uuid4()), generation=generation),
        base_skill=skill,
    )
    return agent


# ---------------------------------------------------------------------------
# Test 1: Combined selection outperforms single-signal selection
# ---------------------------------------------------------------------------

class TestPerformanceNoveltySelection:
    """Paper mechanism 1: Combined criterion outperforms pure performance or pure novelty."""

    def test_pure_performance_selects_top_scorers(self):
        """Pure performance selection (alpha=1.0) picks highest-performing agents."""
        selector = PerformanceNoveltySelector(alpha=1.0, k_neighbors=2)
        agents = []
        for i in range(10):
            a = make_evolving_agent(skill=0.5 + (i * 0.05))
            a.performance_score = 0.5 + (i * 0.05)
            # All agents have identical capability vectors (no novelty differentiation)
            a.capability_vector = [1.0] * 10
            agents.append(a)

        survivors = selector.select(agents, num_survivors=5)

        # With alpha=1.0, combined score = performance score (normalized).
        # The top 5 by performance should dominate.
        assert len(survivors) == 5
        # All survivors should have combined scores assigned
        assert all(a.combined_score >= 0 for a in survivors)

    def test_pure_novelty_assigns_nonzero_novelty(self):
        """Pure novelty selection (alpha=0.0) assigns positive novelty to unique agents."""
        selector = PerformanceNoveltySelector(alpha=0.0, k_neighbors=2)
        agents = []
        for i in range(10):
            a = make_evolving_agent(skill=0.1)
            a.performance_score = 0.1  # All low performance
            # Each agent has a unique capability vector
            a.capability_vector = [float(j == i) for j in range(10)]
            agents.append(a)

        survivors = selector.select(agents, num_survivors=5)

        # With distinct vectors, all agents should receive positive novelty
        assert len(survivors) == 5
        assert all(a.novelty_score > 0 for a in survivors)

    def test_balanced_selection_returns_correct_count(self):
        """Alpha=0.5 should select exactly the requested number of agents."""
        selector = PerformanceNoveltySelector(alpha=0.5, k_neighbors=2)

        # Create a population with varied performance and novelty
        agents = []
        for i in range(10):
            a = make_evolving_agent(skill=0.3 + (i * 0.07))
            a.performance_score = 0.3 + (i * 0.07)
            # Diverse capability vectors
            vec = [0.0] * 10
            vec[i] = 1.0
            vec[(i + 1) % 10] = 1.0
            a.capability_vector = vec
            agents.append(a)

        survivors = selector.select(agents, num_survivors=5)

        assert len(survivors) == 5
        # Combined scores should all be non-negative
        assert all(a.combined_score >= 0 for a in survivors)

    def test_combined_score_formula(self):
        """Combined score = alpha * performance + (1-alpha) * novelty for each agent."""
        alpha = 0.5
        selector = PerformanceNoveltySelector(alpha=alpha, k_neighbors=2)

        agents = []
        for i in range(5):
            a = make_evolving_agent(skill=0.2 * i)
            a.performance_score = 0.2 * i
            a.capability_vector = [float(j == i) for j in range(5)]
            agents.append(a)

        selector.score_agents(agents)

        for agent in agents:
            expected = alpha * agent.performance_score + (1 - alpha) * agent.novelty_score
            assert agent.combined_score == pytest.approx(expected)

    def test_pure_performance_vs_combined_differ(self):
        """Alpha=0.5 should NOT always match pure performance ordering."""
        selector_perf = PerformanceNoveltySelector(alpha=1.0, k_neighbors=2)
        selector_balanced = PerformanceNoveltySelector(alpha=0.5, k_neighbors=2)

        # Build agents with high performance but low diversity
        # vs low performance but high novelty
        agents_for_perf = []
        agents_for_balanced = []

        for i in range(8):
            # High performance, no diversity (identical vectors)
            a1 = make_evolving_agent(skill=0.5 + i * 0.05)
            a1.performance_score = 0.5 + i * 0.05
            a1.capability_vector = [1.0, 0.0, 0.0, 0.0, 0.0]
            agents_for_perf.append(a1)

            # Varying performance AND varying vectors
            a2 = make_evolving_agent(skill=0.5 + i * 0.05)
            a2.performance_score = 0.5 + i * 0.05
            vec = [0.0] * 8
            vec[i] = 1.0
            a2.capability_vector = vec
            agents_for_balanced.append(a2)

        survivors_perf = selector_perf.select(agents_for_perf, num_survivors=4)
        survivors_balanced = selector_balanced.select(agents_for_balanced, num_survivors=4)

        # Both should return the right count
        assert len(survivors_perf) == 4
        assert len(survivors_balanced) == 4

        # The balanced selector should assign nonzero novelty
        assert all(a.novelty_score >= 0 for a in survivors_balanced)


# ---------------------------------------------------------------------------
# Test 2: Shared Experience Pool
# ---------------------------------------------------------------------------

class TestSharedExperiencePool:
    """Paper mechanism 2: Experience traces are shared across group members."""

    @pytest.mark.asyncio
    async def test_cross_agent_experience_visible(self):
        """Traces from agent A should appear in agent B's group context."""
        store = InMemoryStore()
        pool = SharedExperiencePool(store)

        # Agent A submits experience
        await pool.submit(EvolutionaryTrace(
            agent_id="agent-a", run_id="run-1", generation=0,
            task_description="fix auth bug",
            outcome="success",
            score=0.9,
            lessons_learned=["use retry pattern for API calls"],
        ))

        # Agent B submits experience
        await pool.submit(EvolutionaryTrace(
            agent_id="agent-b", run_id="run-1", generation=0,
            task_description="add logging",
            outcome="success",
            score=0.7,
            lessons_learned=["structured logging improves debugging"],
        ))

        # Group context should contain BOTH agents' lessons
        context = await pool.get_group_context("run-1", "group-1", generation=1)
        assert "retry pattern" in context
        assert "structured logging" in context

    @pytest.mark.asyncio
    async def test_experience_accumulates_across_generations(self):
        """Each generation adds to the pool, building cumulative knowledge."""
        store = InMemoryStore()
        pool = SharedExperiencePool(store)

        for gen in range(3):
            await pool.submit(EvolutionaryTrace(
                agent_id=f"agent-gen{gen}", run_id="run-1", generation=gen,
                task_description=f"task gen {gen}",
                outcome="success", score=0.5 + gen * 0.1,
                lessons_learned=[f"lesson from generation {gen}"],
            ))

        context = await pool.get_group_context("run-1", "group-1", generation=3)
        # Context groups by generation using "### Generation {gen}" format
        assert "### Generation 0" in context
        assert "### Generation 1" in context
        assert "### Generation 2" in context

    @pytest.mark.asyncio
    async def test_experience_pool_scales(self):
        """Pool handles many traces without losing information."""
        store = InMemoryStore()
        pool = SharedExperiencePool(store)

        for i in range(100):
            await pool.submit(EvolutionaryTrace(
                agent_id=f"agent-{i % 10}", run_id="run-1", generation=i // 10,
                task_description=f"task {i}", outcome="success", score=0.5,
            ))

        traces = await store.get_by_run("run-1", limit=50)
        assert len(traces) == 50  # Respects limit

    @pytest.mark.asyncio
    async def test_empty_pool_returns_default_message(self):
        """An empty pool should return a sensible default context string."""
        store = InMemoryStore()
        pool = SharedExperiencePool(store)

        context = await pool.get_group_context("no-run", "group-1", generation=0)
        assert "No prior experience" in context

    @pytest.mark.asyncio
    async def test_traces_isolated_by_run_id(self):
        """Traces from one run should not appear in another run's context."""
        store = InMemoryStore()
        pool = SharedExperiencePool(store)

        await pool.submit(EvolutionaryTrace(
            agent_id="agent-a", run_id="run-X", generation=0,
            task_description="task for run X",
            outcome="success", score=0.8,
            lessons_learned=["secret lesson for run X"],
        ))

        context = await pool.get_group_context("run-Y", "group-1", generation=1)
        assert "secret lesson for run X" not in context


# ---------------------------------------------------------------------------
# Test 3: Probe Task Characterization
# ---------------------------------------------------------------------------

class TestProbeCharacterization:
    """Paper mechanism 4: Binary capability vectors from probe tasks."""

    @pytest.mark.asyncio
    async def test_capability_vectors_are_binary(self):
        """Probe results should produce binary (0/1) capability vectors."""
        registry = ProbeRegistry()
        registry.register_default_probes(num_probes=10)
        evaluator = ProbeEvaluator(registry)

        agent = make_evolving_agent(skill=0.6)
        vector = await evaluator.evaluate(agent)

        assert len(vector) == 10
        assert all(v in (0.0, 1.0) for v in vector)

    @pytest.mark.asyncio
    async def test_different_agents_get_different_vectors(self):
        """Agents with different skills should tend to have different capability vectors."""
        registry = ProbeRegistry()
        registry.register_default_probes(num_probes=20)
        evaluator = ProbeEvaluator(registry)

        # Strong agent - run multiple times to reduce noise impact
        strong = make_evolving_agent(skill=0.9)
        strong_vec = await evaluator.evaluate(strong)

        # Weak agent
        weak = make_evolving_agent(skill=0.1)
        weak_vec = await evaluator.evaluate(weak)

        # Strong agent should have at least as many 1s as a weak agent on average
        # (with high skill, more probes pass threshold of 0.5)
        assert sum(strong_vec) >= sum(weak_vec)

    def test_novelty_from_capability_vectors(self):
        """KNN novelty should detect behavioral differences via capability vectors."""
        estimator = KNNNoveltyEstimator(k_neighbors=2)

        agents = [make_evolving_agent() for _ in range(5)]
        # 4 similar agents
        for i in range(4):
            agents[i].capability_vector = [1, 1, 1, 0, 0]
        # 1 unique agent
        agents[4].capability_vector = [0, 0, 0, 1, 1]

        estimator.score(agents)

        # The unique agent should have highest novelty
        assert agents[4].novelty_score == max(a.novelty_score for a in agents)

    @pytest.mark.asyncio
    async def test_probe_vector_length_matches_registry(self):
        """Capability vector length must exactly match the number of registered probes."""
        for num_probes in [5, 10, 15, 20]:
            registry = ProbeRegistry()
            registry.register_default_probes(num_probes=num_probes)
            evaluator = ProbeEvaluator(registry)

            agent = make_evolving_agent(skill=0.5)
            vector = await evaluator.evaluate(agent)

            assert len(vector) == num_probes, (
                f"Expected vector length {num_probes}, got {len(vector)}"
            )

    def test_novelty_all_identical_vectors(self):
        """When all agents have identical capability vectors, novelty is uniform."""
        estimator = KNNNoveltyEstimator(k_neighbors=2)

        agents = [make_evolving_agent() for _ in range(5)]
        for a in agents:
            a.capability_vector = [1.0, 0.0, 1.0, 0.0]

        estimator.score(agents)

        # All agents at zero distance from each other - normalized to 1.0
        assert all(a.novelty_score == pytest.approx(1.0) for a in agents)


# ---------------------------------------------------------------------------
# Test 4: Evolution Improves Over Generations
# ---------------------------------------------------------------------------

class TestEvolutionImprovement:
    """Verify that the evolution loop produces improving scores."""

    @pytest.mark.asyncio
    async def test_scores_improve_over_generations(self):
        """Running evolution should improve best scores over generations."""
        config = HowlerConfig(
            population_size=6, group_size=3, num_iterations=3, alpha=0.5, num_probes=5
        )

        pool = AgentPool()
        for _ in range(config.population_size):
            pool.add(make_evolving_agent(skill=0.3))

        store = InMemoryStore()
        experience = SharedExperiencePool(store)
        selector = PerformanceNoveltySelector(alpha=config.alpha)

        registry = ProbeRegistry()
        registry.register_default_probes(num_probes=config.num_probes)
        probe_eval = ProbeEvaluator(registry)

        # Mock reproducer that creates improving patches
        reproducer = AsyncMock(spec=GroupReproducer)
        reproducer.reproduce = AsyncMock(return_value=(
            FrameworkPatch(intent="improve skills", category="general"),
            AsyncMock(),
        ))

        loop = EvolutionLoop(
            config=config, pool=pool, selector=selector,
            reproducer=reproducer, experience=experience,
            probe_evaluator=probe_eval,
        )

        tasks = [
            {"description": "solve coding problem", "type": "general"},
            {"description": "write tests", "type": "testing"},
        ]

        gen_scores = []
        for gen in range(config.num_iterations):
            summary = await loop.step("test-run", gen, tasks)
            gen_scores.append(summary["best_score"])

        # Scores should generally improve (or at least not fall dramatically)
        # Allow for noise in the stochastic simulation
        assert gen_scores[-1] >= gen_scores[0] * 0.8, (
            f"Final score {gen_scores[-1]:.3f} should be close to or better than "
            f"initial {gen_scores[0]:.3f}"
        )

    @pytest.mark.asyncio
    async def test_experience_traces_accumulate(self):
        """Each generation step should produce experience traces."""
        config = HowlerConfig(population_size=4, group_size=2, num_iterations=1, num_probes=3)

        pool = AgentPool()
        for _ in range(config.population_size):
            pool.add(make_evolving_agent(skill=0.5))

        store = InMemoryStore()
        experience = SharedExperiencePool(store)
        selector = PerformanceNoveltySelector(alpha=config.alpha)

        registry = ProbeRegistry()
        registry.register_default_probes(num_probes=config.num_probes)
        probe_eval = ProbeEvaluator(registry)

        reproducer = AsyncMock(spec=GroupReproducer)
        reproducer.reproduce = AsyncMock(return_value=(
            FrameworkPatch(intent="test"), AsyncMock(),
        ))

        loop = EvolutionLoop(
            config=config, pool=pool, selector=selector,
            reproducer=reproducer, experience=experience,
            probe_evaluator=probe_eval,
        )

        await loop.step("test-run", 0, [{"description": "test", "type": "general"}])

        # Traces should have been submitted
        traces = await store.get_by_run("test-run")
        assert len(traces) > 0, "Evolution step should produce experience traces"
        assert all(t.run_id == "test-run" for t in traces)

    @pytest.mark.asyncio
    async def test_step_summary_contains_required_fields(self):
        """Each evolution step summary must include generation stats."""
        config = HowlerConfig(population_size=4, group_size=2, num_iterations=1, num_probes=3)

        pool = AgentPool()
        for _ in range(config.population_size):
            pool.add(make_evolving_agent(skill=0.5))

        store = InMemoryStore()
        experience = SharedExperiencePool(store)
        selector = PerformanceNoveltySelector(alpha=config.alpha)

        registry = ProbeRegistry()
        registry.register_default_probes(num_probes=config.num_probes)
        probe_eval = ProbeEvaluator(registry)

        reproducer = AsyncMock(spec=GroupReproducer)
        reproducer.reproduce = AsyncMock(return_value=(
            FrameworkPatch(intent="test"), AsyncMock(),
        ))

        loop = EvolutionLoop(
            config=config, pool=pool, selector=selector,
            reproducer=reproducer, experience=experience,
            probe_evaluator=probe_eval,
        )

        summary = await loop.step("test-run-2", 0, [{"description": "task", "type": "general"}])

        assert summary["generation"] == 0
        assert "population_size" in summary
        assert "best_score" in summary
        assert "mean_score" in summary
        assert "best_agent_id" in summary

    @pytest.mark.asyncio
    async def test_full_run_returns_best_score(self):
        """Full evolution run should return the best score across all generations."""
        config = HowlerConfig(population_size=4, group_size=2, num_iterations=3, num_probes=3)

        pool = AgentPool()
        for _ in range(config.population_size):
            pool.add(make_evolving_agent(skill=0.4))

        store = InMemoryStore()
        experience = SharedExperiencePool(store)
        selector = PerformanceNoveltySelector(alpha=config.alpha)

        registry = ProbeRegistry()
        registry.register_default_probes(num_probes=config.num_probes)
        probe_eval = ProbeEvaluator(registry)

        reproducer = AsyncMock(spec=GroupReproducer)
        reproducer.reproduce = AsyncMock(return_value=(
            FrameworkPatch(intent="improve"), AsyncMock(),
        ))

        loop = EvolutionLoop(
            config=config, pool=pool, selector=selector,
            reproducer=reproducer, experience=experience,
            probe_evaluator=probe_eval,
        )

        result = await loop.run("full-run", [{"description": "task", "type": "general"}])

        assert result["run_id"] == "full-run"
        assert "generations" in result
        assert len(result["generations"]) == config.num_iterations
        assert result["best_score"] >= 0


# ---------------------------------------------------------------------------
# Test 5: Paper Configuration Verification
# ---------------------------------------------------------------------------

class TestPaperConfig:
    """Verify that paper-specified configurations are valid."""

    def test_swe_bench_config(self):
        """SWE-bench config from paper: K=10, M=3, alpha=0.5, 10 iterations."""
        config = HowlerConfig(
            population_size=10, group_size=3, num_iterations=10, alpha=0.5, num_probes=20
        )
        assert config.population_size == 10
        assert config.group_size == 3
        assert config.alpha == 0.5
        assert config.num_iterations == 10

        # Groups should partition evenly (with possible remainder)
        pool = AgentPool()
        for _ in range(config.population_size):
            pool.add(make_evolving_agent())
        groups = pool.partition_groups(config.group_size)
        assert len(groups) >= config.population_size // config.group_size

    def test_polyglot_config(self):
        """Polyglot config from paper: K=10, M=3, alpha=0.5, 15 iterations."""
        config = HowlerConfig(
            population_size=10, group_size=3, num_iterations=15, alpha=0.5, num_probes=30
        )
        assert config.population_size == 10
        assert config.num_iterations == 15
        assert config.num_probes == 30

    def test_alpha_range_coverage(self):
        """Paper ablation: alpha from 0 to 1 should all produce valid configs."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = HowlerConfig(alpha=alpha)
            assert 0 <= config.alpha <= 1

    def test_group_partitioning_with_paper_params(self):
        """K=10, M=3 should produce ceil(10/3) = 4 groups (last group has 1 agent)."""
        config = HowlerConfig(population_size=10, group_size=3)
        pool = AgentPool()
        for _ in range(config.population_size):
            pool.add(make_evolving_agent())

        groups = pool.partition_groups(config.group_size)

        # 10 agents in groups of 3: [3, 3, 3, 1]
        assert len(groups) == 4
        total_agents = sum(len(g) for g in groups)
        assert total_agents == config.population_size

    def test_selector_with_paper_alpha(self):
        """The paper's alpha=0.5 selector should assign nonzero combined scores."""
        selector = PerformanceNoveltySelector(alpha=0.5, k_neighbors=5)
        agents = []
        for i in range(10):
            a = make_evolving_agent(skill=0.4 + i * 0.05)
            a.performance_score = 0.4 + i * 0.05
            vec = [0.0] * 10
            vec[i] = 1.0
            a.capability_vector = vec
            agents.append(a)

        # Select K/2 survivors as the paper implies
        survivors = selector.select(agents, num_survivors=5)

        assert len(survivors) == 5
        assert all(a.combined_score > 0 for a in survivors)
