"""LLM router with role-based model dispatch.

Supports two backends:
- LiteLLM (default): Direct API calls to 100+ providers
- Claude Code CLI: Uses the local `claude` command, avoiding API rate limits

Set model to "claude-code/sonnet" (or haiku/opus) to use the CLI backend.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from howler_agents.config import HowlerConfig, LLMRole, RoleModelConfig
from howler_agents.llm.claude_code import (
    _SDK_BACKENDS,
    cli_complete,
    is_cli_model,
    sdk_complete,
)

logger = structlog.get_logger()

# Retry settings for rate limit errors
_MAX_RETRIES = 5
_BASE_DELAY_S = 2.0
_MAX_DELAY_S = 60.0


class LLMRouter:
    """Routes LLM calls to the appropriate model based on role.

    Supports two backends based on the model string:

    **LiteLLM backend** (default):
    - "claude-sonnet-4-20250514" -> Anthropic API
    - "gpt-4o" -> OpenAI API
    - "ollama/llama3" -> Ollama (local)

    **Local CLI backends** (no API rate limits):
    - "claude-code/sonnet" -> local `claude` CLI
    - "codex/default" -> OpenAI Codex CLI
    - "gemini-cli/default" -> Google Gemini CLI
    - "opencode/default" -> OpenCode CLI

    CLI backends use the tool's own subscription/auth instead of
    API keys, avoiding restrictive API rate limits.
    """

    def __init__(self, config: HowlerConfig, min_request_interval_s: float = 0.0) -> None:
        self._role_map = config.role_models
        self._default_config = RoleModelConfig()
        self._min_interval = min_request_interval_s
        self._last_request_time: float = 0.0

    async def complete(
        self,
        role: LLMRole,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Send a completion request routed by role.

        Automatically selects the backend based on the model string:
        - "claude-code/*" -> Claude Code CLI subprocess
        - anything else -> LiteLLM API call

        Retries automatically on rate limit errors with exponential backoff
        (LiteLLM backend only).
        """
        config = self._role_map.get(role, self._default_config)

        logger.debug("llm_request", role=role.value, model=config.model)

        # Route to local CLI backend
        if is_cli_model(config.model):
            prefix = config.model.split("/", 1)[0] if "/" in config.model else ""
            # SDK backend: structured query() instead of raw subprocess
            if prefix in _SDK_BACKENDS:
                return await sdk_complete(
                    model=config.model,
                    messages=messages,
                    **kwargs,
                )
            # Raw CLI subprocess backend (claude-code/, codex/, etc.)
            return await cli_complete(
                model=config.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", config.max_tokens),
            )

        # Default: LiteLLM backend
        return await self._litellm_complete(config, messages, **kwargs)

    async def _litellm_complete(
        self,
        config: RoleModelConfig,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Complete via LiteLLM with retry and rate limit spacing."""
        import litellm

        completion_kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", config.temperature),
            "max_tokens": kwargs.pop("max_tokens", config.max_tokens),
        }

        if config.api_key:
            completion_kwargs["api_key"] = config.api_key

        completion_kwargs.update(kwargs)

        # Rate limit spacing: ensure minimum interval between requests
        if self._min_interval > 0:
            import time

            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.acompletion(**completion_kwargs)
                content = response.choices[0].message.content or ""
                logger.debug("llm_response", role="litellm", length=len(content))
                return content
            except litellm.RateLimitError as exc:
                last_exc = exc
                delay = min(_BASE_DELAY_S * (2**attempt), _MAX_DELAY_S)
                logger.warning(
                    "rate_limit_retry",
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    delay_s=delay,
                    model=config.model,
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]
