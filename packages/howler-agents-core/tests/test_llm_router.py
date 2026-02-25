"""Tests for LLM router and CLI backends."""

from unittest.mock import AsyncMock, patch

import pytest

from howler_agents.config import HowlerConfig, LLMRole, RoleModelConfig
from howler_agents.llm.claude_code import (
    ClaudeCodeBackend,
    ClaudeSDKBackend,
    CodexBackend,
    GeminiCLIBackend,
    OpenCodeBackend,
    _is_api_error,
    _is_api_limit,
    _messages_to_prompt,
    cli_complete,
    detect_available_backend,
    is_cli_model,
    list_available_backends,
    sdk_complete,
)
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
        result = await router.complete(LLMRole.ACTING, [{"role": "user", "content": "hello"}])
        assert result == "test response"
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model"] == "mock/gpt-3.5-turbo"


# --------------------------------------------------------------------------- #
# CLI backend unit tests                                                       #
# --------------------------------------------------------------------------- #


class TestCLIModelDetection:
    def test_is_cli_model_claude_sdk(self) -> None:
        assert is_cli_model("claude-sdk/sonnet") is True
        assert is_cli_model("claude-sdk/haiku") is True
        assert is_cli_model("claude-sdk/opus") is True

    def test_is_cli_model_claude_code(self) -> None:
        assert is_cli_model("claude-code/sonnet") is True
        assert is_cli_model("claude-code/haiku") is True
        assert is_cli_model("claude-code/opus") is True

    def test_is_cli_model_codex(self) -> None:
        assert is_cli_model("codex/default") is True

    def test_is_cli_model_gemini(self) -> None:
        assert is_cli_model("gemini-cli/default") is True

    def test_is_cli_model_opencode(self) -> None:
        assert is_cli_model("opencode/default") is True

    def test_is_cli_model_api(self) -> None:
        assert is_cli_model("claude-sonnet-4-20250514") is False
        assert is_cli_model("gpt-4o") is False
        assert is_cli_model("ollama/llama3") is False


class TestCLIBackendCommands:
    def test_claude_code_build_cmd(self) -> None:
        backend = ClaudeCodeBackend()
        cmd = backend.build_cmd("Fix the bug", "sonnet")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "Fix the bug" in cmd
        assert "--model" in cmd
        assert "sonnet" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--no-session-persistence" in cmd

    def test_claude_code_build_cmd_stdin(self) -> None:
        backend = ClaudeCodeBackend()
        assert backend.supports_stdin() is True
        cmd = backend.build_cmd_stdin("sonnet")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "--dangerously-skip-permissions" in cmd

    def test_claude_code_clean_env_removes_claudecode(self) -> None:
        """CLAUDECODE must be DELETED from env, not set to empty."""
        backend = ClaudeCodeBackend()
        with patch.dict("os.environ", {"CLAUDECODE": "1", "HOME": "/home/test"}):
            env = backend.clean_env()
            assert "CLAUDECODE" not in env
            assert "HOME" in env

    def test_claude_code_clean_env_removes_node_options(self) -> None:
        """NODE_OPTIONS causes silent exit code 1."""
        backend = ClaudeCodeBackend()
        with patch.dict("os.environ", {"NODE_OPTIONS": "--inspect", "PATH": "/usr/bin"}):
            env = backend.clean_env()
            assert "NODE_OPTIONS" not in env
            assert "PATH" in env

    def test_codex_build_cmd(self) -> None:
        backend = CodexBackend()
        cmd = backend.build_cmd("Fix the bug", "default")
        assert cmd == ["codex", "exec", "Fix the bug"]

    def test_gemini_build_cmd(self) -> None:
        backend = GeminiCLIBackend()
        cmd = backend.build_cmd("Fix the bug", "default")
        assert cmd == ["gemini", "-p", "Fix the bug"]

    def test_opencode_build_cmd(self) -> None:
        backend = OpenCodeBackend()
        cmd = backend.build_cmd("Fix the bug", "default")
        assert cmd == ["opencode", "run", "-q", "Fix the bug"]


class TestMessagesToPrompt:
    def test_empty_messages(self) -> None:
        assert _messages_to_prompt([]) == ""

    def test_single_message(self) -> None:
        msgs = [{"role": "user", "content": "Hello world"}]
        assert _messages_to_prompt(msgs) == "Hello world"

    def test_multi_message(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Fix this bug"},
        ]
        result = _messages_to_prompt(msgs)
        assert "[System Instructions]" in result
        assert "You are helpful" in result
        assert "Fix this bug" in result

    def test_assistant_message(self) -> None:
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Thanks"},
        ]
        result = _messages_to_prompt(msgs)
        assert "[Previous Response]" in result
        assert "Hi there" in result


class TestBackendAvailability:
    def test_detect_available_returns_string_or_none(self) -> None:
        result = detect_available_backend()
        assert result is None or isinstance(result, str)

    def test_list_available_returns_list(self) -> None:
        result = list_available_backends()
        assert isinstance(result, list)

    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_detect_prefers_claude_sdk(self, _mock: AsyncMock) -> None:
        result = detect_available_backend()
        assert result is not None
        assert result.startswith("claude-sdk/")


@pytest.mark.asyncio
async def test_router_routes_to_cli_backend() -> None:
    """Router should dispatch to cli_complete for CLI model strings."""
    config = HowlerConfig()
    config.role_models[LLMRole.ACTING] = RoleModelConfig(model="claude-code/sonnet")
    router = LLMRouter(config)

    with patch(
        "howler_agents.llm.router.cli_complete",
        new_callable=AsyncMock,
        return_value="CLI response",
    ) as mock_cli:
        result = await router.complete(
            LLMRole.ACTING,
            [{"role": "user", "content": "test"}],
        )
        assert result == "CLI response"
        mock_cli.assert_called_once()


class TestCLIRetry:
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        """cli_complete retries on timeout with backoff."""
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            proc.communicate = AsyncMock(side_effect=TimeoutError("timed out"))
            proc.kill = AsyncMock()
            return proc

        with patch("howler_agents.llm.claude_code.asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("howler_agents.llm.claude_code.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="timed out"):
                await cli_complete(
                    "claude-code/sonnet",
                    [{"role": "user", "content": "test"}],
                    max_retries=1,
                    timeout=5,
                )
            # 1 initial + 1 retry = 2 calls
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """cli_complete returns result when retry succeeds."""
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                # First call times out
                proc.communicate = AsyncMock(side_effect=TimeoutError("timed out"))
                proc.kill = AsyncMock()
            else:
                # Second call succeeds
                proc.communicate = AsyncMock(
                    return_value=(b"fixed output", b"")
                )
                proc.returncode = 0
            return proc

        with patch("howler_agents.llm.claude_code.asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("howler_agents.llm.claude_code.asyncio.sleep", new_callable=AsyncMock):
            result = await cli_complete(
                "claude-code/sonnet",
                [{"role": "user", "content": "test"}],
                max_retries=1,
                timeout=5,
            )
            assert result == "fixed output"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self) -> None:
        """cli_complete does not retry when first call succeeds."""
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"response", b""))
            proc.returncode = 0
            return proc

        with patch("howler_agents.llm.claude_code.asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await cli_complete(
                "claude-code/sonnet",
                [{"role": "user", "content": "test"}],
                max_retries=2,
                timeout=5,
            )
            assert result == "response"
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_api_limit_fails_immediately(self) -> None:
        """cli_complete does not retry on API usage limit errors."""
        call_count = 0
        api_error = (
            b'API Error: 400 {"type":"error","error":{"type":"invalid_request_error",'
            b'"message":"You have reached your specified API usage limits."}}'
        )

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(api_error, b""))
            proc.returncode = 1
            return proc

        with patch("howler_agents.llm.claude_code.asyncio.create_subprocess_exec", side_effect=mock_exec):
            with pytest.raises(RuntimeError, match="API error"):
                await cli_complete(
                    "claude-code/sonnet",
                    [{"role": "user", "content": "test"}],
                    max_retries=2,
                    timeout=5,
                )
            # Should fail on first attempt, no retries
            assert call_count == 1


class TestAPIErrorDetection:
    def test_api_error_message(self) -> None:
        assert _is_api_error("API Error: 400 something") is True

    def test_api_error_rate_limit(self) -> None:
        assert _is_api_error('{"error":{"type":"rate_limit_error"}}') is True

    def test_api_error_normal_output(self) -> None:
        assert _is_api_error("diff --git a/f.py b/f.py") is False

    def test_api_limit_usage_reached(self) -> None:
        msg = "You have reached your specified API usage limits."
        assert _is_api_limit(msg) is True

    def test_api_limit_normal_error(self) -> None:
        assert _is_api_limit("rate_limit_error") is False


# --------------------------------------------------------------------------- #
# Claude Agent SDK backend tests                                               #
# --------------------------------------------------------------------------- #


class TestClaudeSDKBackend:
    def test_sdk_backend_registered(self) -> None:
        assert is_cli_model("claude-sdk/sonnet") is True

    def test_sdk_backend_available_checks_claude_binary(self) -> None:
        backend = ClaudeSDKBackend()
        assert backend.binary == "claude"

    def test_sdk_backend_build_cmd_raises(self) -> None:
        backend = ClaudeSDKBackend()
        with pytest.raises(NotImplementedError):
            backend.build_cmd("test", "sonnet")


class TestSDKComplete:
    @pytest.mark.asyncio
    async def test_sdk_complete_returns_result(self) -> None:
        """sdk_complete extracts result from ResultMessage."""
        from claude_agent_sdk import ResultMessage

        result_msg = ResultMessage(
            subtype="result",
            duration_ms=5000,
            duration_api_ms=4000,
            is_error=False,
            num_turns=1,
            session_id="test-session",
            total_cost_usd=0.01,
            usage={},
            result="diff --git a/f.py b/f.py\n-old\n+new",
        )

        async def mock_query(*, prompt, options=None):
            yield result_msg

        with patch("claude_agent_sdk.query", mock_query):
            result = await sdk_complete(
                "claude-sdk/sonnet",
                [{"role": "user", "content": "Fix the bug"}],
                timeout=30,
            )
            assert "diff --git" in result
            assert "+new" in result

    @pytest.mark.asyncio
    async def test_sdk_complete_timeout(self) -> None:
        """sdk_complete raises on timeout."""
        async def mock_query(*, prompt, options=None):
            import asyncio
            await asyncio.sleep(100)
            yield  # never reached

        with patch("claude_agent_sdk.query", mock_query), \
             pytest.raises(RuntimeError, match="timed out"):
            await sdk_complete(
                "claude-sdk/sonnet",
                [{"role": "user", "content": "test"}],
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_sdk_complete_api_error(self) -> None:
        """sdk_complete raises on API error in result."""
        from claude_agent_sdk import ResultMessage

        result_msg = ResultMessage(
            subtype="result",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=True,
            num_turns=1,
            session_id="test-session",
            total_cost_usd=None,
            usage={},
            result="API Error: 400 rate_limit_error",
        )

        async def mock_query(*, prompt, options=None):
            yield result_msg

        with patch("claude_agent_sdk.query", mock_query), \
             pytest.raises(RuntimeError, match="API error"):
            await sdk_complete(
                "claude-sdk/sonnet",
                [{"role": "user", "content": "test"}],
                timeout=30,
            )

    @pytest.mark.asyncio
    async def test_sdk_complete_no_result(self) -> None:
        """sdk_complete raises when no ResultMessage received."""
        async def mock_query(*, prompt, options=None):
            return
            yield  # makes this an async generator

        with patch("claude_agent_sdk.query", mock_query), \
             pytest.raises(RuntimeError, match="no result"):
            await sdk_complete(
                "claude-sdk/sonnet",
                [{"role": "user", "content": "test"}],
                timeout=30,
            )

    @pytest.mark.asyncio
    async def test_sdk_complete_error_result(self) -> None:
        """sdk_complete raises when result has is_error=True."""
        from claude_agent_sdk import ResultMessage

        result_msg = ResultMessage(
            subtype="result",
            duration_ms=2000,
            duration_api_ms=1500,
            is_error=True,
            num_turns=1,
            session_id="test-session",
            total_cost_usd=None,
            usage={},
            result="Something went wrong internally",
        )

        async def mock_query(*, prompt, options=None):
            yield result_msg

        with patch("claude_agent_sdk.query", mock_query), \
             pytest.raises(RuntimeError, match="returned error"):
            await sdk_complete(
                "claude-sdk/sonnet",
                [{"role": "user", "content": "test"}],
                timeout=30,
            )


@pytest.mark.asyncio
async def test_router_routes_to_sdk_backend() -> None:
    """Router should dispatch to sdk_complete for claude-sdk model strings."""
    config = HowlerConfig()
    config.role_models[LLMRole.ACTING] = RoleModelConfig(model="claude-sdk/sonnet")
    router = LLMRouter(config)

    with patch(
        "howler_agents.llm.router.sdk_complete",
        new_callable=AsyncMock,
        return_value="SDK response",
    ) as mock_sdk:
        result = await router.complete(
            LLMRole.ACTING,
            [{"role": "user", "content": "test"}],
        )
        assert result == "SDK response"
        mock_sdk.assert_called_once()
