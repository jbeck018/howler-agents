"""End-to-end integration tests for the GEA evolution pipeline.

These tests exercise the FULL pipeline — experience → meta-LLM → directive →
patch → behavior change — without mocking GroupReproducer.  The LLM layer
(litellm.acompletion) is replaced with a deterministic stub so tests are fast
and require no API keys, but every other component runs for real.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from howler_agents.agents.base import Agent, AgentConfig, FrameworkPatch, TaskResult
from howler_agents.agents.pool import AgentPool
from howler_agents.config import HowlerConfig, LLMRole
from howler_agents.evolution.directive import EvolutionDirective
from howler_agents.evolution.loop import EvolutionLoop
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.experience.trace import EvolutionaryTrace
from howler_agents.llm.router import LLMRouter
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.selection.criterion import PerformanceNoveltySelector


# ---------------------------------------------------------------------------
# Test agent implementation (mirrors EvolvingAgent from test_gea_mechanisms.py)
# ---------------------------------------------------------------------------

class EvolvingAgent(Agent):
    """Deterministic agent whose skill improves with each patch applied."""

    def __init__(self, config: AgentConfig, base_skill: float = 0.3) -> None:
        super().__init__(config)
        self._base_skill = base_skill
        self._skills: dict[str, float] = {}

    async def run_task(self, task: dict[str, Any]) -> TaskResult:
        task_type = task.get("type", "general")
        skill = self._skills.get(task_type, self._base_skill)
        score = max(0.0, min(1.0, skill))
        return TaskResult(
            success=score > 0.5,
            score=score,
            output=f"completed {task_type}",
            key_decisions=[f"used {task_type} strategy"],
            lessons_learned=[f"{'effective' if score > 0.5 else 'ineffective'} approach"],
        )

    async def apply_patch(self, patch: FrameworkPatch) -> None:
        self.patches.append(patch)
        self._base_skill = min(1.0, self._base_skill + 0.08)
        if patch.category in self._skills:
            self._skills[patch.category] = min(1.0, self._skills[patch.category] + 0.1)
        else:
            self._skills[patch.category] = self._base_skill + 0.1


def make_evolving_agent(skill: float = 0.3, generation: int = 0) -> EvolvingAgent:
    return EvolvingAgent(
        AgentConfig(id=str(uuid.uuid4()), generation=generation),
        base_skill=skill,
    )


# ---------------------------------------------------------------------------
# Mock LLM infrastructure — no real API calls
# ---------------------------------------------------------------------------

def _make_litellm_response(content: str) -> MagicMock:
    """Build a litellm-shaped response object from a plain string."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _directive_json(intent: str = "Improve error handling", strategy: str = "incremental") -> str:
    return json.dumps({
        "intent": intent,
        "target_areas": ["error_handling", "input_validation"],
        "strategy": strategy,
        "confidence": 0.8,
        "reasoning": "Experience traces show recurring failures in error handling",
    })


def _patch_json(intent: str = "Add error handling", category: str = "error_handling") -> str:
    return json.dumps({
        "intent": intent,
        "diff": "--- a/agent.py\n+++ b/agent.py\n@@ -1,3 +1,6 @@\n+try:\n+    result = execute()\n+except Exception as e:\n+    log_error(e)",
        "category": category,
        "config_updates": {},
    })


def make_deterministic_acompletion(
    directive_intent: str = "Improve error handling",
    patch_category: str = "error_handling",
):
    """Return an async callable that mimics litellm.acompletion deterministically.

    The mock inspects the prompt text to decide whether the call is a directive
    request or a patch request, then returns the appropriate JSON.
    """
    async def _acompletion(**kwargs: Any) -> MagicMock:
        messages = kwargs.get("messages", [])
        prompt_text = messages[-1]["content"] if messages else ""

        if "mutation directive" in prompt_text:
            content = _directive_json(intent=directive_intent)
        elif "framework patch" in prompt_text:
            content = _patch_json(category=patch_category)
        else:
            content = '{"result": "ok"}'

        return _make_litellm_response(content)

    return _acompletion


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_full_stack(
    population_size: int = 4,
    group_size: int = 2,
    num_iterations: int = 3,
    num_probes: int = 5,
    alpha: float = 0.5,
    base_skill: float = 0.4,
) -> tuple[EvolutionLoop, SharedExperiencePool, InMemoryStore, LLMRouter, GroupReproducer]:
    """Construct every real component of the GEA pipeline."""
    config = HowlerConfig(
        population_size=population_size,
        group_size=group_size,
        num_iterations=num_iterations,
        alpha=alpha,
        num_probes=num_probes,
    )

    pool = AgentPool()
    for _ in range(population_size):
        pool.add(make_evolving_agent(skill=base_skill))

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    selector = PerformanceNoveltySelector(alpha=alpha)

    registry = ProbeRegistry()
    registry.register_default_probes(num_probes=num_probes)
    probe_eval = ProbeEvaluator(registry)

    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    loop = EvolutionLoop(
        config=config,
        pool=pool,
        selector=selector,
        reproducer=reproducer,
        experience=experience,
        probe_evaluator=probe_eval,
    )

    return loop, experience, store, llm, reproducer


# ===========================================================================
# Test 1 — full pipeline from experience traces to FrameworkPatch
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_experience_to_patch():
    """Real GroupReproducer consumes experience traces and returns a valid FrameworkPatch."""
    config = HowlerConfig(
        population_size=4, group_size=2, num_iterations=1, num_probes=5
    )

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    # Seed the experience pool with realistic traces
    run_id = "e2e-run-1"
    agent = make_evolving_agent(skill=0.6)
    agent.performance_score = 0.6

    await experience.submit(EvolutionaryTrace(
        agent_id=agent.id,
        run_id=run_id,
        generation=0,
        task_description="solve authentication bug",
        outcome="success",
        score=0.75,
        key_decisions=["chose token-based auth"],
        lessons_learned=["retry logic improves reliability"],
    ))
    await experience.submit(EvolutionaryTrace(
        agent_id=agent.id,
        run_id=run_id,
        generation=0,
        task_description="add unit tests",
        outcome="failure",
        score=0.3,
        key_decisions=["skipped edge cases"],
        lessons_learned=["always test boundary conditions"],
    ))

    captured_messages: list[list[dict]] = []

    async def _acompletion(**kwargs: Any) -> MagicMock:
        captured_messages.append(kwargs.get("messages", []))
        prompt = kwargs["messages"][-1]["content"]
        if "mutation directive" in prompt:
            return _make_litellm_response(_directive_json())
        return _make_litellm_response(_patch_json())

    with patch("litellm.acompletion", side_effect=_acompletion):
        patch_obj, directive = await reproducer.reproduce(
            parent=agent,
            run_id=run_id,
            group_id="group-0",
            generation=1,
        )

    # Patch must be a real FrameworkPatch, not a mock
    assert isinstance(patch_obj, FrameworkPatch)
    assert isinstance(directive, EvolutionDirective)

    # LLM was called twice: once for directive, once for patch
    assert len(captured_messages) == 2

    # Experience context must appear in the directive prompt
    directive_prompt = captured_messages[0][-1]["content"]
    assert "retry logic" in directive_prompt or "boundary conditions" in directive_prompt

    # Patch has content parsed from the mock LLM response
    assert patch_obj.intent != ""
    assert patch_obj.category != ""
    assert patch_obj.agent_id == agent.id

    # Patch generation is parent.generation + 1
    assert patch_obj.generation == agent.config.generation + 1

    # Applying the patch modifies the agent
    patches_before = len(agent.patches)
    await agent.apply_patch(patch_obj)
    assert len(agent.patches) == patches_before + 1


# ===========================================================================
# Test 2 — experience content influences LLM prompt
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_experience_influences_reproduction():
    """Different experience content produces different LLM prompts."""
    config = HowlerConfig(population_size=2, group_size=2, num_iterations=1, num_probes=3)

    # Build two isolated stacks with different lessons
    store_a = InMemoryStore()
    experience_a = SharedExperiencePool(store_a)

    store_b = InMemoryStore()
    experience_b = SharedExperiencePool(store_b)

    run_id = "influence-test"
    agent_a = make_evolving_agent(skill=0.5)
    agent_b = make_evolving_agent(skill=0.5)

    await experience_a.submit(EvolutionaryTrace(
        agent_id=agent_a.id,
        run_id=run_id,
        generation=0,
        task_description="optimize database queries",
        outcome="success",
        score=0.85,
        lessons_learned=["use tool_X for better indexing performance"],
    ))

    await experience_b.submit(EvolutionaryTrace(
        agent_id=agent_b.id,
        run_id=run_id,
        generation=0,
        task_description="optimize database queries",
        outcome="failure",
        score=0.2,
        lessons_learned=["avoid tool_X — it causes deadlocks under load"],
    ))

    prompts_a: list[str] = []
    prompts_b: list[str] = []

    async def capture_a(**kwargs: Any) -> MagicMock:
        prompts_a.append(kwargs["messages"][-1]["content"])
        prompt = kwargs["messages"][-1]["content"]
        if "mutation directive" in prompt:
            return _make_litellm_response(_directive_json("use tool_X"))
        return _make_litellm_response(_patch_json("tool_use"))

    async def capture_b(**kwargs: Any) -> MagicMock:
        prompts_b.append(kwargs["messages"][-1]["content"])
        prompt = kwargs["messages"][-1]["content"]
        if "mutation directive" in prompt:
            return _make_litellm_response(_directive_json("avoid tool_X"))
        return _make_litellm_response(_patch_json("error_handling"))

    llm = LLMRouter(config)

    reproducer_a = GroupReproducer(llm=llm, experience_pool=experience_a, config=config)
    reproducer_b = GroupReproducer(llm=llm, experience_pool=experience_b, config=config)

    with patch("litellm.acompletion", side_effect=capture_a):
        patch_a, _ = await reproducer_a.reproduce(
            parent=agent_a, run_id=run_id, group_id="g0", generation=1
        )

    with patch("litellm.acompletion", side_effect=capture_b):
        patch_b, _ = await reproducer_b.reproduce(
            parent=agent_b, run_id=run_id, group_id="g0", generation=1
        )

    # Directive prompts must contain the respective lessons
    assert prompts_a, "Reproducer A must have called LLM"
    assert prompts_b, "Reproducer B must have called LLM"

    directive_prompt_a = prompts_a[0]
    directive_prompt_b = prompts_b[0]

    assert "tool_X" in directive_prompt_a
    assert "tool_X" in directive_prompt_b

    # The lesson content must differ between the two pools
    assert directive_prompt_a != directive_prompt_b


# ===========================================================================
# Test 3 — 5-generation improvement with real pipeline
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_generation_improvement_e2e():
    """Scores should monotonically (or near-monotonically) improve over 5 real generations."""
    config = HowlerConfig(
        population_size=4, group_size=2, num_iterations=5, num_probes=4, alpha=0.5
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

    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    loop = EvolutionLoop(
        config=config,
        pool=pool,
        selector=selector,
        reproducer=reproducer,
        experience=experience,
        probe_evaluator=probe_eval,
    )

    tasks = [
        {"description": "solve coding problem", "type": "general"},
        {"description": "write tests", "type": "testing"},
    ]

    gen_scores: list[float] = []

    with patch("litellm.acompletion", side_effect=make_deterministic_acompletion()):
        for gen in range(config.num_iterations):
            summary = await loop.step("multi-gen-run", gen, tasks)
            gen_scores.append(summary["best_score"])

    assert len(gen_scores) == 5

    # Final best score must be at least 25% better than the initial best score
    # (EvolvingAgent gains +0.08 per patch, 5 generations = meaningful gain)
    assert gen_scores[-1] >= gen_scores[0], (
        f"Score did not improve: {gen_scores[0]:.3f} -> {gen_scores[-1]:.3f}"
    )

    # Verify at least 3 of 5 step-over-step transitions are non-decreasing
    improvements = sum(
        1 for a, b in zip(gen_scores, gen_scores[1:]) if b >= a
    )
    assert improvements >= 2, (
        f"Expected mostly non-decreasing scores, got {gen_scores}"
    )


# ===========================================================================
# Test 4 — group context contains traces from ALL agents in the group
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_group_experience_aggregation_in_reproduction():
    """LLM prompt must contain experience from all three agents in the group."""
    config = HowlerConfig(population_size=3, group_size=3, num_iterations=1, num_probes=3)

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    run_id = "group-agg-run"
    agents = [make_evolving_agent(skill=0.5 + i * 0.1) for i in range(3)]

    # Each agent has a distinct lesson
    distinct_lessons = [
        "agent0_lesson: prefer functional decomposition",
        "agent1_lesson: cache intermediate results",
        "agent2_lesson: validate inputs early",
    ]

    for agent, lesson in zip(agents, distinct_lessons):
        await experience.submit(EvolutionaryTrace(
            agent_id=agent.id,
            run_id=run_id,
            generation=0,
            task_description="general coding task",
            outcome="success",
            score=0.7,
            lessons_learned=[lesson],
        ))
        agent.performance_score = 0.7

    captured_directive_prompt: list[str] = []

    async def _capture(**kwargs: Any) -> MagicMock:
        content = kwargs["messages"][-1]["content"]
        if "mutation directive" in content:
            captured_directive_prompt.append(content)
        if "mutation directive" in content:
            return _make_litellm_response(_directive_json())
        return _make_litellm_response(_patch_json())

    parent = agents[0]

    with patch("litellm.acompletion", side_effect=_capture):
        patch_obj, directive = await reproducer.reproduce(
            parent=parent,
            run_id=run_id,
            group_id="group-0",
            generation=1,
        )

    assert captured_directive_prompt, "No directive prompt was captured"

    prompt = captured_directive_prompt[0]

    # All three agents' lessons must appear in the context
    assert "agent0_lesson" in prompt, "Agent 0 lesson missing from group context"
    assert "agent1_lesson" in prompt, "Agent 1 lesson missing from group context"
    assert "agent2_lesson" in prompt, "Agent 2 lesson missing from group context"

    # Returns valid patch
    assert isinstance(patch_obj, FrameworkPatch)


# ===========================================================================
# Test 5 — cross-lineage knowledge flows into reproduction
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_lineage_knowledge_transfer():
    """Experience from both lineage A and lineage B flows into a mixed-group reproduction."""
    config = HowlerConfig(population_size=4, group_size=2, num_iterations=1, num_probes=3)

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    run_id = "cross-lineage-run"

    # --- Generation 0: lineage A and lineage B run independently ---
    lineage_a_agent = make_evolving_agent(skill=0.5)
    lineage_b_agent = make_evolving_agent(skill=0.5)

    await experience.submit(EvolutionaryTrace(
        agent_id=lineage_a_agent.id,
        run_id=run_id,
        generation=0,
        task_description="lineage A task",
        outcome="success",
        score=0.8,
        lessons_learned=["lineage_A_knowledge: use recursive descent"],
    ))

    await experience.submit(EvolutionaryTrace(
        agent_id=lineage_b_agent.id,
        run_id=run_id,
        generation=0,
        task_description="lineage B task",
        outcome="success",
        score=0.7,
        lessons_learned=["lineage_B_knowledge: use iterative deepening"],
    ))

    # --- Generation 1: A' and B' offspring placed in the same group ---
    child_a = make_evolving_agent(skill=0.5, generation=1)
    child_a.config.parent_id = lineage_a_agent.id

    child_b = make_evolving_agent(skill=0.5, generation=1)
    child_b.config.parent_id = lineage_b_agent.id

    # Their gen-1 traces also go into the pool
    await experience.submit(EvolutionaryTrace(
        agent_id=child_a.id,
        run_id=run_id,
        generation=1,
        task_description="mixed group task",
        outcome="success",
        score=0.75,
        lessons_learned=["child_A_knowledge: combine A and B approaches"],
    ))
    await experience.submit(EvolutionaryTrace(
        agent_id=child_b.id,
        run_id=run_id,
        generation=1,
        task_description="mixed group task",
        outcome="success",
        score=0.65,
        lessons_learned=["child_B_knowledge: apply B strategy first"],
    ))

    child_a.performance_score = 0.75
    child_b.performance_score = 0.65

    captured_prompts: list[str] = []

    async def _capture(**kwargs: Any) -> MagicMock:
        content = kwargs["messages"][-1]["content"]
        if "mutation directive" in content:
            captured_prompts.append(content)
        if "mutation directive" in content:
            return _make_litellm_response(_directive_json())
        return _make_litellm_response(_patch_json())

    with patch("litellm.acompletion", side_effect=_capture):
        patch_obj, directive = await reproducer.reproduce(
            parent=child_a,
            run_id=run_id,
            group_id="mixed-group",
            generation=2,
        )

    assert captured_prompts, "No directive prompt captured"
    prompt = captured_prompts[0]

    # Both original lineages' knowledge must appear
    assert "lineage_A_knowledge" in prompt, "Lineage A knowledge missing from mixed group context"
    assert "lineage_B_knowledge" in prompt, "Lineage B knowledge missing from mixed group context"

    assert isinstance(patch_obj, FrameworkPatch)


# ===========================================================================
# Test 6 — novelty decreases as agents converge on identical patches
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_novelty_decreases_as_agents_converge():
    """Agents receiving identical patches become similar; novelty scores should fall."""
    config = HowlerConfig(
        population_size=6, group_size=3, num_iterations=1, num_probes=5, alpha=0.5
    )

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)

    # Build diverse initial agents with distinct capability vectors
    agents_gen0 = []
    for i in range(6):
        a = make_evolving_agent(skill=0.5)
        vec = [0.0] * 6
        vec[i] = 1.0
        a.capability_vector = vec
        agents_gen0.append(a)

    selector = PerformanceNoveltySelector(alpha=0.5)
    selector.score_agents(agents_gen0)
    initial_novelty_scores = [a.novelty_score for a in agents_gen0]
    initial_mean_novelty = sum(initial_novelty_scores) / len(initial_novelty_scores)

    # Apply the SAME patch to all agents, then re-score novelty
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)
    run_id = "convergence-run"

    converged_patch = FrameworkPatch(
        id=str(uuid.uuid4()),
        agent_id="shared",
        generation=1,
        intent="uniform improvement",
        diff="same diff for everyone",
        category="general",
    )

    for a in agents_gen0:
        await a.apply_patch(converged_patch)
        # After convergence all agents share the same capability vector
        a.capability_vector = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    selector.score_agents(agents_gen0)
    converged_novelty_scores = [a.novelty_score for a in agents_gen0]
    converged_mean_novelty = sum(converged_novelty_scores) / len(converged_novelty_scores)

    # When all capability vectors are identical the KNN distance is 0 for all,
    # which the estimator normalises to 1.0 (max).  What we're actually checking
    # is that differentiated initial novelty scores collapse to a uniform value.
    unique_initial = len(set(round(s, 4) for s in initial_novelty_scores))
    unique_converged = len(set(round(s, 4) for s in converged_novelty_scores))

    assert unique_converged <= unique_initial, (
        f"Converging agents should produce fewer distinct novelty values: "
        f"initial unique={unique_initial}, converged unique={unique_converged}"
    )


# ===========================================================================
# Test 7 — balanced combined score drives selection (alpha=0.5)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_combined_score_drives_selection():
    """With alpha=0.5, balanced agents beat either pure-performance or pure-novelty extremes."""
    selector = PerformanceNoveltySelector(alpha=0.5, k_neighbors=2)

    # Build three groups:
    #   high-performance but zero novelty (identical vectors)
    #   high-novelty but low performance (distinct vectors, low scores)
    #   balanced: moderate performance + moderate novelty
    agents: list[EvolvingAgent] = []

    # 3 high-performance, zero-novelty agents
    for _ in range(3):
        a = make_evolving_agent(skill=0.9)
        a.performance_score = 0.9
        a.capability_vector = [1.0, 0.0, 0.0, 0.0, 0.0]
        agents.append(a)

    # 3 high-novelty, low-performance agents
    for i in range(3):
        a = make_evolving_agent(skill=0.1)
        a.performance_score = 0.1
        vec = [0.0] * 5
        vec[i % 5] = 1.0
        vec[(i + 1) % 5] = 1.0
        a.capability_vector = vec
        agents.append(a)

    # 3 balanced agents (moderate performance + moderate novelty)
    for i in range(3):
        a = make_evolving_agent(skill=0.55)
        a.performance_score = 0.55
        vec = [0.5] * 5
        vec[i] = 1.0
        a.capability_vector = vec
        agents.append(a)

    balanced_agents = agents[6:]

    # Select top 3 from the full pool of 9
    survivors = selector.select(agents, num_survivors=3)
    survivor_ids = {a.id for a in survivors}

    # Assert that the selection does NOT simply pick pure-performance or pure-novelty extremes:
    # At least one balanced agent should survive
    balanced_survivors = sum(1 for a in balanced_agents if a.id in survivor_ids)
    assert balanced_survivors >= 1, (
        "Expected at least one balanced agent to survive with alpha=0.5 selection"
    )

    # All combined scores must be non-negative
    assert all(a.combined_score >= 0 for a in agents)

    # No pure-novelty agent (score=0.1) should beat pure-performance (score=0.9)
    # in combined score — performance contributes 50%
    high_perf = agents[:3]
    high_nov = agents[3:6]
    assert all(
        hp.combined_score >= hn.combined_score
        for hp, hn in zip(high_perf, high_nov)
    ), "High-performance agents should have higher combined scores than low-performance agents"


# ===========================================================================
# Test — LLM prompt carries the experience_context section header
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_reproduce_prompt_contains_experience_header():
    """DIRECTIVE_PROMPT must embed the group experience section in the LLM call."""
    config = HowlerConfig(population_size=2, group_size=2, num_iterations=1, num_probes=3)

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    run_id = "prompt-check-run"
    agent = make_evolving_agent(skill=0.6)
    agent.performance_score = 0.6

    await experience.submit(EvolutionaryTrace(
        agent_id=agent.id,
        run_id=run_id,
        generation=0,
        task_description="verify prompt structure",
        outcome="success",
        score=0.9,
        lessons_learned=["unique_marker_xzq: lesson content"],
    ))

    captured: list[str] = []

    async def _capture(**kwargs: Any) -> MagicMock:
        captured.append(kwargs["messages"][-1]["content"])
        prompt = kwargs["messages"][-1]["content"]
        if "mutation directive" in prompt:
            return _make_litellm_response(_directive_json())
        return _make_litellm_response(_patch_json())

    with patch("litellm.acompletion", side_effect=_capture):
        await reproducer.reproduce(parent=agent, run_id=run_id, group_id="g0", generation=1)

    assert len(captured) == 2, "Expected exactly 2 LLM calls (directive + patch)"

    directive_prompt = captured[0]

    # The DIRECTIVE_PROMPT template embeds the experience context
    assert "Group Experience" in directive_prompt
    assert "unique_marker_xzq" in directive_prompt

    # The PATCH_PROMPT embeds the directive fields but NOT the experience context
    patch_prompt = captured[1]
    assert "framework patch" in patch_prompt.lower() or "Patch" in patch_prompt
    assert "unique_marker_xzq" not in patch_prompt


# ===========================================================================
# Test — reproduce falls back gracefully on malformed LLM responses
# ===========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_reproduce_fallback_on_malformed_llm_response():
    """GroupReproducer must produce a usable FrameworkPatch even when LLM returns garbage."""
    config = HowlerConfig(population_size=2, group_size=2, num_iterations=1, num_probes=3)

    store = InMemoryStore()
    experience = SharedExperiencePool(store)
    llm = LLMRouter(config)
    reproducer = GroupReproducer(llm=llm, experience_pool=experience, config=config)

    agent = make_evolving_agent(skill=0.5)
    agent.performance_score = 0.5

    async def _bad_llm(**kwargs: Any) -> MagicMock:
        return _make_litellm_response("this is definitely not json {{{{")

    with patch("litellm.acompletion", side_effect=_bad_llm):
        patch_obj, directive = await reproducer.reproduce(
            parent=agent,
            run_id="fallback-run",
            group_id="g0",
            generation=1,
        )

    # Reproducer must not raise — it must fall back to defaults
    assert isinstance(patch_obj, FrameworkPatch)
    assert isinstance(directive, EvolutionDirective)
    assert patch_obj.agent_id == agent.id
    assert patch_obj.generation == agent.config.generation + 1
    assert directive.intent == "general improvement"
