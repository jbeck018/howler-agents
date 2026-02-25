"""Group reproducer - creates child agents via meta-LLM mutations."""

from __future__ import annotations

import json
import uuid

import structlog

from howler_agents.agents.base import Agent, FrameworkPatch
from howler_agents.config import HowlerConfig, LLMRole
from howler_agents.evolution.directive import EvolutionDirective
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.llm.router import LLMRouter

logger = structlog.get_logger()

DIRECTIVE_PROMPT = """You are an evolutionary meta-optimizer for AI agents.

Given the group's experience and the parent agent's current configuration,
generate a mutation directive that will improve the agent's capabilities.

## Group Experience
{experience_context}

## Parent Agent
- ID: {agent_id}
- Generation: {generation}
- Performance: {performance:.3f}
- Current config: {config}

## Instructions
Analyze the experience traces and generate a mutation directive.
Focus on areas where the agent struggled or where novel approaches could help.

Respond with JSON:
{{
  "intent": "brief description of the mutation",
  "target_areas": ["area1", "area2"],
  "strategy": "incremental|exploratory|targeted",
  "confidence": 0.0-1.0,
  "reasoning": "why this mutation should help"
}}"""

PATCH_PROMPT = """You are an AI agent framework mutator.

Apply the following mutation directive to the agent's framework configuration.

## Directive
- Intent: {intent}
- Target areas: {target_areas}
- Strategy: {strategy}

## Current Framework Config
{config}

## Instructions
Generate a framework patch (as a JSON diff) that implements this directive.
The patch should be minimal and focused.

Respond with JSON:
{{
  "intent": "{intent}",
  "diff": "unified diff or config change description",
  "category": "tool_use|planning|error_handling|communication|reasoning",
  "config_updates": {{}}
}}"""


class GroupReproducer:
    """Creates child agents from parent groups via meta-LLM mutations."""

    def __init__(
        self,
        llm: LLMRouter,
        experience_pool: SharedExperiencePool,
        config: HowlerConfig,
    ) -> None:
        self._llm = llm
        self._experience = experience_pool
        self._config = config

    async def reproduce(
        self,
        parent: Agent,
        run_id: str,
        group_id: str,
        generation: int,
    ) -> tuple[FrameworkPatch, EvolutionDirective]:
        """Generate a mutation for a parent agent."""
        experience_context = await self._experience.get_group_context(
            run_id=run_id, group_id=group_id, generation=generation
        )

        directive = await self._generate_directive(parent, experience_context)
        patch = await self._generate_patch(parent, directive)

        logger.info(
            "reproduction_complete",
            agent_id=parent.id,
            intent=directive.intent,
            category=patch.category,
        )

        return patch, directive

    async def _generate_directive(
        self, parent: Agent, experience_context: str
    ) -> EvolutionDirective:
        prompt = DIRECTIVE_PROMPT.format(
            experience_context=experience_context,
            agent_id=parent.id,
            generation=parent.config.generation,
            performance=parent.performance_score,
            config=json.dumps(parent.config.framework_config, indent=2),
        )

        response = await self._llm.complete(
            role=LLMRole.EVOLVING,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = json.loads(response)
            return EvolutionDirective(**data)
        except (json.JSONDecodeError, TypeError):
            return EvolutionDirective(intent="general improvement", strategy="incremental")

    async def _generate_patch(self, parent: Agent, directive: EvolutionDirective) -> FrameworkPatch:
        prompt = PATCH_PROMPT.format(
            intent=directive.intent,
            target_areas=", ".join(directive.target_areas),
            strategy=directive.strategy,
            config=json.dumps(parent.config.framework_config, indent=2),
        )

        response = await self._llm.complete(
            role=LLMRole.EVOLVING,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = json.loads(response)
            return FrameworkPatch(
                id=str(uuid.uuid4()),
                agent_id=parent.id,
                generation=parent.config.generation + 1,
                intent=data.get("intent", directive.intent),
                diff=data.get("diff", ""),
                category=data.get("category", "general"),
                config_updates=data.get("config_updates", {}),
            )
        except (json.JSONDecodeError, TypeError):
            return FrameworkPatch(
                agent_id=parent.id,
                generation=parent.config.generation + 1,
                intent=directive.intent,
                category="general",
            )
