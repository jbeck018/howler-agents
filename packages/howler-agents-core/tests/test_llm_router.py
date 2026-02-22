"""Tests for LLM router."""

from unittest.mock import AsyncMock, patch

import pytest

from howler_agents.config import HowlerConfig, LLMRole, RoleModelConfig
from howler_agents.llm.router import LLMRouter


@pytest.fixture
def router() -> LLMRouter:
    config = HowlerConfig()
    config.role_models[LLMRole.ACTING] = RoleModelConfig(model="mock/gpt-3.5-turbo")
    config.role_models[LLMRole.EVOLVING] = RoleModelConfig(model="mock/gpt-3.5-turbo")
    return LLMRouter(config)


@pytest.mark.asyncio
async def test_complete_routes_by_role(router: LLMRouter):
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "test response"

    with patch("litellm.acompletion", return_value=mock_response) as mock_llm:
        result = await router.complete(
            LLMRole.ACTING, [{"role": "user", "content": "hello"}]
        )
        assert result == "test response"
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model"] == "mock/gpt-3.5-turbo"
