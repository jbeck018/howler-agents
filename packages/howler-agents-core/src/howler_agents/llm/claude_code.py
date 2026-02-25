"""Local coding CLI backends for LLM completions.

Uses local coding CLI tools (Claude Code, Codex, Gemini CLI, OpenCode)
as LLM backends instead of direct API calls via LiteLLM. This avoids
API rate limits (e.g., 30K tokens/min) by using subscription-based
CLI tools which have much higher throughput.

Supported backends (set as model prefix):
- "claude-sdk/sonnet"   -> claude-agent-sdk query() (preferred)
- "claude-code/sonnet"  -> claude -p "..." --model sonnet
- "codex/default"       -> codex exec "..."
- "gemini-cli/default"  -> gemini -p "..."
- "opencode/default"    -> opencode run -q "..."

The LLMRouter detects the prefix and routes to the appropriate backend.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()

# Environment variables that must be REMOVED (not just emptied) when
# spawning Claude Code as a subprocess. These cause silent exit code 1.
_ENV_VARS_TO_REMOVE = {
    "CLAUDECODE",  # Nested execution detection
    "CLAUDE_CODE_ENTRYPOINT",  # SDK entrypoint marker
    "ANTHROPIC_MODEL",  # Forces API billing instead of subscription
    "ANTHROPIC_API_KEY",  # Forces API billing instead of subscription
    "NODE_OPTIONS",  # VSCode debugger injection
    "VSCODE_INSPECTOR_OPTIONS",  # VSCode inspector
}


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CLIBackend:
    """Configuration for a local coding CLI backend."""

    name: str
    binary: str
    env_overrides: dict[str, str]
    uses_stdin: bool = False  # If True, pipe prompt via stdin

    def build_cmd(self, prompt: str, model_suffix: str) -> list[str]:
        """Build the CLI command for non-interactive completion."""
        raise NotImplementedError

    def build_cmd_stdin(self, model_suffix: str) -> list[str]:
        """Build the CLI command for stdin-piped prompt.

        Override if the backend supports reading prompts from stdin.
        Returns the command without the prompt argument.
        """
        raise NotImplementedError

    def supports_stdin(self) -> bool:
        """Whether this backend supports reading prompts from stdin."""
        return False

    def available(self) -> bool:
        """Check if the CLI binary is on PATH."""
        return shutil.which(self.binary) is not None

    def clean_env(self) -> dict[str, str]:
        """Build a clean environment for subprocess execution.

        Starts from os.environ, removes problematic vars, applies overrides.
        """
        env = {k: v for k, v in os.environ.items() if k not in _ENV_VARS_TO_REMOVE}
        env.update(self.env_overrides)
        return env


class ClaudeCodeBackend(CLIBackend):
    """Claude Code: claude -p "..." --dangerously-skip-permissions"""

    def __init__(self) -> None:
        object.__setattr__(self, "name", "claude-code")
        object.__setattr__(self, "binary", "claude")
        # Use ANTHROPIC_MODEL env var instead of --model flag (bug #22362:
        # --model is ignored in -p mode). Overrides set at call time.
        object.__setattr__(self, "env_overrides", {})

    def build_cmd(self, prompt: str, model_suffix: str) -> list[str]:
        model = _CLAUDE_ALIASES.get(model_suffix, model_suffix)
        return [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
        ]

    def build_cmd_stdin(self, model_suffix: str) -> list[str]:
        """Pipe prompt via stdin: cat file | claude -p "query"."""
        model = _CLAUDE_ALIASES.get(model_suffix, model_suffix)
        return [
            "claude",
            "-p",
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
        ]

    def supports_stdin(self) -> bool:
        return True


class CodexBackend(CLIBackend):
    """OpenAI Codex CLI: codex exec "..." """

    def __init__(self) -> None:
        object.__setattr__(self, "name", "codex")
        object.__setattr__(self, "binary", "codex")
        object.__setattr__(self, "env_overrides", {})

    def build_cmd(self, prompt: str, model_suffix: str) -> list[str]:
        return ["codex", "exec", prompt]


class GeminiCLIBackend(CLIBackend):
    """Google Gemini CLI: gemini -p "..." """

    def __init__(self) -> None:
        object.__setattr__(self, "name", "gemini-cli")
        object.__setattr__(self, "binary", "gemini")
        object.__setattr__(self, "env_overrides", {})

    def build_cmd(self, prompt: str, model_suffix: str) -> list[str]:
        return ["gemini", "-p", prompt]


class OpenCodeBackend(CLIBackend):
    """OpenCode CLI: opencode run -q "..." """

    def __init__(self) -> None:
        object.__setattr__(self, "name", "opencode")
        object.__setattr__(self, "binary", "opencode")
        object.__setattr__(self, "env_overrides", {})

    def build_cmd(self, prompt: str, model_suffix: str) -> list[str]:
        return ["opencode", "run", "-q", prompt]


# ---------------------------------------------------------------------------
# Claude Agent SDK backend
# ---------------------------------------------------------------------------


class ClaudeSDKBackend(CLIBackend):
    """Claude Agent SDK: uses query() for structured, subscription-auth calls.

    Preferred over ClaudeCodeBackend because it:
    - Inherits subscription auth (no ANTHROPIC_API_KEY needed)
    - Provides structured typed responses (no stdout/stderr parsing)
    - Handles env var cleanup internally
    - Reports cost and duration metadata
    """

    def __init__(self) -> None:
        object.__setattr__(self, "name", "claude-sdk")
        object.__setattr__(self, "binary", "claude")
        object.__setattr__(self, "env_overrides", {})

    def build_cmd(self, prompt: str, model_suffix: str) -> list[str]:
        # Not used — SDK backend uses query() instead of subprocess
        raise NotImplementedError("ClaudeSDKBackend uses query(), not build_cmd()")


async def sdk_complete(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> str:
    """Complete a prompt using the Claude Agent SDK.

    Uses claude_agent_sdk.query() which spawns the Claude CLI with
    structured JSON streaming. Inherits subscription auth — no API key needed.

    Model format: "claude-sdk/sonnet" (or haiku, opus)
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk not installed. Install with: uv add claude-agent-sdk"
        ) from exc

    _, _, suffix = model.partition("/")
    if not suffix:
        suffix = "sonnet"
    model_name = _CLAUDE_ALIASES.get(suffix, suffix)

    prompt = _messages_to_prompt(messages)
    timeout = kwargs.get("timeout", 300)
    cwd = kwargs.get("cwd")
    system_prompt = kwargs.get("system_prompt")
    # Default max_turns=10: the SDK model may use internal Claude Code tools
    # (file reads, edits) which consume turns.  With max_turns=3, complex
    # prompts cause the model to exhaust its budget on tool calls, returning
    # length=0.  Setting 10 gives ample room for tool-use + text response.
    max_turns = kwargs.get("max_turns", 10)

    logger.info(
        "sdk_request",
        model=model_name,
        prompt_length=len(prompt),
        timeout=timeout,
    )

    # The SDK transport builds env as: {**os.environ, **options.env, ...}
    # So options.env can only ADD/OVERRIDE, not DELETE keys from os.environ.
    # We must temporarily remove problematic vars from os.environ itself
    # to prevent CLAUDECODE etc. from leaking into the subprocess.
    saved_env: dict[str, str] = {}
    for var in _ENV_VARS_TO_REMOVE:
        if var in os.environ:
            saved_env[var] = os.environ.pop(var)

    options = ClaudeAgentOptions(
        model=model_name,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
    )
    if cwd:
        options.cwd = cwd
    # Default system prompt prevents the SDK from using internal tools
    # (file reads/edits) which waste turns and produce empty text results.
    # Callers can override with a custom system_prompt if tools are needed.
    if system_prompt:
        options.system_prompt = system_prompt
    else:
        options.system_prompt = (
            "You are a code analysis assistant. Respond with text only. "
            "Do NOT read, write, or edit any files. Do NOT use any tools. "
            "Analyze the provided input and respond directly with your answer."
        )

    result_text: str | None = None
    assistant_texts: list[str] = []  # Collect text from AssistantMessages
    is_error = False

    try:
        async with asyncio.timeout(timeout):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    # Collect text blocks from assistant messages
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            assistant_texts.append(block.text)
                elif isinstance(message, ResultMessage):
                    result_text = message.result
                    is_error = message.is_error
                    logger.info(
                        "sdk_response",
                        length=len(result_text) if result_text else 0,
                        duration_ms=message.duration_ms,
                        cost_usd=message.total_cost_usd,
                        num_turns=message.num_turns,
                        is_error=is_error,
                    )
    except TimeoutError as exc:
        raise RuntimeError(f"claude-sdk timed out after {timeout}s") from exc
    except Exception as exc:
        err_msg = str(exc)
        if _is_api_limit(err_msg):
            raise RuntimeError(err_msg) from exc
        raise RuntimeError(f"claude-sdk error: {err_msg}") from exc
    finally:
        # Restore env vars so we don't affect the parent process permanently
        os.environ.update(saved_env)

    # Use ResultMessage.result if available, otherwise fall back to
    # collected AssistantMessage text blocks (the SDK may not populate
    # .result when Claude responds with text in an assistant turn).
    if not result_text and assistant_texts:
        result_text = "\n".join(assistant_texts)
        logger.info(
            "sdk_fallback_to_assistant_text",
            length=len(result_text),
            blocks=len(assistant_texts),
        )

    if result_text is None or (is_error and result_text):
        # Check for API errors in the result
        if result_text and _is_api_error(result_text):
            raise RuntimeError(f"claude-sdk API error: {result_text[:200]}")
        if result_text and is_error:
            raise RuntimeError(f"claude-sdk returned error: {result_text[:500]}")
        if result_text is None:
            raise RuntimeError("claude-sdk returned no result")

    return result_text


# Claude model alias mapping — maps user-facing suffixes to ANTHROPIC_MODEL values.
# These are used as env var values, so they must be valid model identifiers.
_CLAUDE_ALIASES: dict[str, str] = {
    "sonnet": "sonnet",
    "haiku": "haiku",
    "opus": "opus",
    "claude-sonnet-4-20250514": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
    "claude-opus-4-20250514": "opus",
    "default": "sonnet",
}

# Registry of all supported backends
_BACKENDS: dict[str, CLIBackend] = {
    "claude-sdk": ClaudeSDKBackend(),
    "claude-code": ClaudeCodeBackend(),
    "codex": CodexBackend(),
    "gemini-cli": GeminiCLIBackend(),
    "opencode": OpenCodeBackend(),
}

# Backends that use the SDK query() path instead of raw subprocess
_SDK_BACKENDS = {"claude-sdk"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_cli_model(model: str) -> bool:
    """Check if a model string targets a local CLI backend."""
    prefix = model.split("/", 1)[0] if "/" in model else ""
    return prefix in _BACKENDS


# Keep backward-compatible alias
is_claude_code_model = is_cli_model


def detect_available_backend() -> str | None:
    """Auto-detect the first available CLI backend on PATH.

    Returns a model prefix like "claude-sdk/sonnet" or None.
    Prefers claude-sdk over claude-code when Claude CLI is available.
    """
    for name, backend in _BACKENDS.items():
        if backend.available():
            suffix = "sonnet" if name in ("claude-sdk", "claude-code") else "default"
            return f"{name}/{suffix}"
    return None


def list_available_backends() -> list[str]:
    """List all CLI backends available on PATH."""
    return [name for name, b in _BACKENDS.items() if b.available()]


async def cli_complete(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> str:
    """Complete a prompt using a local coding CLI tool.

    Routes to the appropriate backend based on the model prefix:
    - "claude-code/sonnet" -> Claude Code
    - "codex/default" -> OpenAI Codex
    - "gemini-cli/default" -> Gemini CLI
    - "opencode/default" -> OpenCode

    Runs as an async subprocess so it doesn't block the event loop.
    Retries on timeout/error with exponential backoff to handle API
    rate limiting through CLI tools.
    """
    prefix, _, suffix = model.partition("/")
    backend = _BACKENDS.get(prefix)
    if backend is None:
        raise ValueError(f"Unknown CLI backend: {prefix!r}. Available: {list(_BACKENDS)}")

    if not suffix:
        suffix = "default"

    prompt = _messages_to_prompt(messages)

    logger.info(
        "cli_request",
        backend=backend.name,
        model_suffix=suffix,
        prompt_length=len(prompt),
    )

    # Build clean env: remove CLAUDECODE, NODE_OPTIONS, ANTHROPIC_MODEL, etc.
    # ANTHROPIC_MODEL forces API billing; use --model flag for subscription.
    env = backend.clean_env()

    # Scale timeout by prompt size: 300s base + 60s per 10K chars
    base_timeout = 300
    prompt_timeout = base_timeout + (len(prompt) // 10_000) * 60
    timeout = kwargs.get("timeout", max(prompt_timeout, 600))

    max_retries = kwargs.get("max_retries", 2)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Exponential backoff: 30s, 60s between retries
            delay = 30 * (2 ** (attempt - 1))
            logger.info(
                "cli_retry",
                backend=backend.name,
                attempt=attempt + 1,
                delay_s=delay,
            )
            await asyncio.sleep(delay)

        try:
            result = await _cli_exec_once(backend, prompt, suffix, env, timeout)
            if result is not None:
                return result
            # Empty response — retry
            last_error = RuntimeError(f"{backend.binary} returned empty response")
            continue
        except _CLITimeoutError:
            last_error = RuntimeError(
                f"{backend.binary} timed out after {timeout}s (attempt {attempt + 1})"
            )
            logger.warning(
                "cli_timeout",
                backend=backend.name,
                timeout_s=timeout,
                attempt=attempt + 1,
                max_retries=max_retries,
            )
            continue
        except _CLIProcessError as exc:
            err_msg = str(exc)
            # Hard API limit — retrying won't help, fail immediately
            if _is_api_limit(err_msg):
                logger.error(
                    "cli_api_limit",
                    backend=backend.name,
                    error=err_msg[:200],
                )
                raise RuntimeError(err_msg) from exc
            last_error = RuntimeError(err_msg)
            logger.warning(
                "cli_process_error",
                backend=backend.name,
                error=err_msg[:200],
                attempt=attempt + 1,
            )
            continue

    raise last_error or RuntimeError(f"{backend.binary} failed after {max_retries + 1} attempts")


class _CLITimeoutError(Exception):
    """Internal: CLI subprocess timed out."""


class _CLIProcessError(Exception):
    """Internal: CLI subprocess failed with non-zero exit and no usable output."""


# Patterns that indicate API errors (not useful partial output)
_API_ERROR_PATTERNS = (
    "API Error:",
    "api_error",
    "invalid_request_error",
    "rate_limit_error",
    "overloaded_error",
    "usage limits",
)

# Patterns that mean retrying is pointless (hard limits, not transient)
_API_LIMIT_PATTERNS = (
    "usage limits",
    "You have reached",
    "spending limit",
)


def _is_api_error(content: str) -> bool:
    """Check if CLI output is an API error message rather than useful content."""
    return any(p in content for p in _API_ERROR_PATTERNS)


def _is_api_limit(content: str) -> bool:
    """Check if API error indicates a hard usage limit (no point retrying)."""
    return any(p in content for p in _API_LIMIT_PATTERNS)


async def _cli_exec_once(
    backend: CLIBackend,
    prompt: str,
    suffix: str,
    env: dict[str, str],
    timeout: float,
) -> str | None:
    """Execute a single CLI call. Returns content or None (empty).

    Raises _CLITimeoutError on timeout, _CLIProcessError on fatal error.
    """
    cmd = backend.build_cmd(prompt, suffix)
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()

        # Claude Code may write response to stderr instead of stdout
        # (observed in non-interactive mode). Use whichever has content.
        content = out_text or err_text

        if proc.returncode != 0:
            logger.warning(
                "cli_error",
                backend=backend.name,
                returncode=proc.returncode,
                stderr=err_text[:500],
                stdout=out_text[:200] if out_text else "(empty)",
            )
            # Check for API error responses — these should NOT be treated
            # as valid content since they contain error messages, not patches.
            if content and _is_api_error(content):
                raise _CLIProcessError(f"{backend.binary} API error: {content[:200]}")
            # If there's content despite non-zero exit, return it anyway
            # (the CLI may return useful partial output with rc=1)
            if content:
                logger.info(
                    "cli_response_despite_error",
                    backend=backend.name,
                    length=len(content),
                    returncode=proc.returncode,
                )
                return content
            raise _CLIProcessError(
                f"{backend.binary} exited with code {proc.returncode}: {err_text[:200]}"
            )

        if not content:
            logger.warning(
                "cli_empty_response",
                backend=backend.name,
                prompt_length=len(prompt),
            )
            return None

        logger.info("cli_response", backend=backend.name, length=len(content))
        return content

    except TimeoutError as te:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        raise _CLITimeoutError(f"{backend.binary} timed out after {timeout}s") from te


# Backward-compatible aliases
claude_code_complete = cli_complete
claude_code_available = ClaudeCodeBackend().available
claude_sdk_available = ClaudeSDKBackend().available


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert a messages list to a single prompt string for CLI tools.

    CLI tools take a single prompt string, not a messages array.
    For simple cases (single user message), pass it directly.
    For multi-turn, format as a structured prompt.
    """
    if not messages:
        return ""

    # Single message — pass content directly
    if len(messages) == 1:
        return messages[0].get("content", "")

    # Multi-message — format with role labels
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System Instructions]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[Previous Response]\n{content}\n")
        else:
            parts.append(content)

    return "\n".join(parts)
