"""LiteLLM-backed LLM router with role-based model dispatch."""

from __future__ import annotations

from typing import Any

import structlog

from howler_agents.config import HowlerConfig, LLMRole, RoleModelConfig

logger = structlog.get_logger()


class LLMRouter:
    """Routes LLM calls to the appropriate model based on role.

    Uses LiteLLM under the hood, which supports 100+ LLM providers
    through a unified interface. The model name string encodes the provider:
    - "claude-sonnet-4-20250514" -> Anthropic
    - "gpt-4o" -> OpenAI
    - "ollama/llama3" -> Ollama (local)
    """

    def __init__(self, config: HowlerConfig) -> None:
        self._role_map = config.role_models
        self._default_config = RoleModelConfig()

    async def complete(
        self,
        role: LLMRole,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Send a completion request routed by role."""
        import litellm

        config = self._role_map.get(role, self._default_config)

        logger.debug("llm_request", role=role.value, model=config.model)

        completion_kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", config.temperature),
            "max_tokens": kwargs.pop("max_tokens", config.max_tokens),
        }

        if config.api_key:
            completion_kwargs["api_key"] = config.api_key

        completion_kwargs.update(kwargs)

        response = await litellm.acompletion(**completion_kwargs)
        content = response.choices[0].message.content or ""

        logger.debug("llm_response", role=role.value, length=len(content))
        return content
