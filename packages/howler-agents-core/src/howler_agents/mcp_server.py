"""MCP server for howler-agents — exposes evolutionary agent capabilities as MCP tools.

Designed for zero-config local use: by default everything runs with SQLite persistence
via DatabaseManager. Optionally connects to a remote howler-agents service via
the HOWLER_API_URL environment variable, proxying all calls through httpx.

Modes:
  - local:  SQLite only (no HOWLER_API_URL set)
  - hybrid: SQLite + team sync (HOWLER_API_URL and HOWLER_API_KEY both set)
  - remote: full proxy (HOWLER_API_URL set, no HOWLER_API_KEY)

Usage (stdio transport, for Claude Code / Cursor / OpenCode / Codex):
    python -m howler_agents.mcp_server

Usage (SSE transport):
    python -m howler_agents.mcp_server --transport sse --port 8765
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import mcp.types as types
import structlog
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from howler_agents.local_runner import LocalRunner
from howler_agents.persistence.db import DatabaseManager
from howler_agents.persistence.repo import find_repo_root, get_db_path  # noqa: F401

logger = structlog.get_logger()

# --------------------------------------------------------------------------- #
# Global state                                                                 #
# --------------------------------------------------------------------------- #

# Module-level runner; persists across all tool calls within a server process.
_runner = LocalRunner()
_db: DatabaseManager | None = None
_remote_api_url: str | None = os.environ.get("HOWLER_API_URL")
_api_key: str | None = os.environ.get("HOWLER_API_KEY")
_background_tasks: set[asyncio.Task[None]] = set()


# Mode: "local" (SQLite only), "hybrid" (SQLite + sync), "remote" (full proxy)
def _get_mode() -> str:
    if _remote_api_url and not _api_key:
        return "remote"
    elif _remote_api_url and _api_key:
        return "hybrid"
    return "local"


# --------------------------------------------------------------------------- #
# Remote proxy helpers                                                         #
# --------------------------------------------------------------------------- #


async def _proxy_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the remote howler-agents REST API and return parsed JSON."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required for remote mode. Install with: pip install httpx"
        ) from exc

    assert _remote_api_url is not None
    url = f"{_remote_api_url.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


async def _proxy_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET from the remote howler-agents REST API and return parsed JSON."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required for remote mode. Install with: pip install httpx"
        ) from exc

    assert _remote_api_url is not None
    url = f"{_remote_api_url.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params or {})
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------- #
# Tool schema definitions                                                      #
# --------------------------------------------------------------------------- #

TOOLS: list[types.Tool] = [
    types.Tool(
        name="howler_evolve",
        description=(
            "Start a local evolution run using the Group-Evolving Agents (GEA) algorithm. "
            "Returns a run_id you can use with other tools to monitor and interact with the run. "
            "The run executes asynchronously in the background."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "population_size": {
                    "type": "integer",
                    "description": "Total number of agents in the population (K).",
                    "default": 10,
                    "minimum": 2,
                },
                "group_size": {
                    "type": "integer",
                    "description": "Number of agents per evolution group (M).",
                    "default": 3,
                    "minimum": 1,
                },
                "num_iterations": {
                    "type": "integer",
                    "description": "Number of evolution generations to run.",
                    "default": 5,
                    "minimum": 1,
                },
                "alpha": {
                    "type": "number",
                    "description": "Balance between performance (1.0) and novelty (0.0). Default 0.5.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "task_domain": {
                    "type": "string",
                    "description": "Domain for agent tasks: 'general', 'coding', 'math', 'writing', 'swe-bench'.",
                    "default": "general",
                },
                "model": {
                    "type": "string",
                    "description": "LiteLLM model string used for all roles unless overridden via howler_configure.",
                    "default": "claude-sonnet-4-20250514",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="howler_status",
        description=(
            "Check the status of a running or completed evolution run. "
            "Returns generation progress, best score, and population statistics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run identifier returned by howler_evolve.",
                },
            },
            "required": ["run_id"],
        },
    ),
    types.Tool(
        name="howler_list_agents",
        description=(
            "List agents in the current population of an evolution run. "
            "Optionally limit to the top-K agents by combined score."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run identifier returned by howler_evolve.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "If provided, return only the top-K agents by combined score.",
                    "minimum": 1,
                },
            },
            "required": ["run_id"],
        },
    ),
    types.Tool(
        name="howler_submit_experience",
        description=(
            "Submit a task experience trace to the shared experience pool for a run. "
            "This enriches the evolutionary context used to generate the next generation of mutations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run identifier returned by howler_evolve.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "ID of the agent that performed the task.",
                },
                "task_description": {
                    "type": "string",
                    "description": "Natural-language description of the task performed.",
                },
                "outcome": {
                    "type": "string",
                    "description": "Outcome of the task: 'success', 'failure', or a custom label.",
                },
                "score": {
                    "type": "number",
                    "description": "Numeric score [0.0, 1.0] representing task performance.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "key_decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key decisions made during task execution.",
                },
                "lessons_learned": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of lessons learned that should inform future mutations.",
                },
            },
            "required": ["run_id", "agent_id", "task_description", "outcome", "score"],
        },
    ),
    types.Tool(
        name="howler_get_experience",
        description=(
            "Retrieve aggregated experience context for a run, formatted for LLM consumption. "
            "Useful for understanding what the agent population has learned across generations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run identifier returned by howler_evolve.",
                },
                "group_id": {
                    "type": "string",
                    "description": "Optional group ID to filter experience to a specific group.",
                },
            },
            "required": ["run_id"],
        },
    ),
    types.Tool(
        name="howler_configure",
        description=(
            "Configure LLM models for the three evolutionary roles: acting, evolving, and reflecting. "
            "Settings persist for the lifetime of the server process. "
            "Future calls to howler_evolve will use these model assignments."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "acting_model": {
                    "type": "string",
                    "description": "LiteLLM model string for agents performing tasks.",
                },
                "evolving_model": {
                    "type": "string",
                    "description": "LiteLLM model string for the meta-LLM generating mutations.",
                },
                "reflecting_model": {
                    "type": "string",
                    "description": "LiteLLM model string for the reflective analysis role.",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional API key to use for all LLM calls.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="howler_memory",
        description=(
            "Store, retrieve, search, or list entries in the persistent hive-mind memory. "
            "Memory survives across sessions and evolution runs. "
            "Actions: 'store', 'retrieve', 'search', 'list', 'delete'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "retrieve", "search", "list", "delete"],
                    "description": "Memory operation to perform.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Memory namespace (e.g. 'lessons', 'patterns', 'decisions'). Default: 'default'.",
                    "default": "default",
                },
                "key": {
                    "type": "string",
                    "description": "Memory key (required for store/retrieve/delete).",
                },
                "value": {
                    "type": "string",
                    "description": "Memory value (required for store).",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for search action).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for store action.",
                },
                "score": {
                    "type": "number",
                    "description": "Relevance score for store action (0.0-1.0).",
                    "default": 0.0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results for search/list.",
                    "default": 10,
                },
            },
            "required": ["action"],
        },
    ),
    types.Tool(
        name="howler_history",
        description=(
            "Browse persistent evolution history for this repository. "
            "Lists past runs, agents, and traces that survive across sessions. "
            "Actions: 'runs', 'agents', 'traces', 'stats'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["runs", "agents", "traces", "stats"],
                    "description": "What to inspect.",
                },
                "run_id": {
                    "type": "string",
                    "description": "Filter to a specific run.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Filter to a specific agent.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                },
            },
            "required": ["action"],
        },
    ),
    types.Tool(
        name="howler_hivemind",
        description=(
            "Manage the repository's hive-mind: persistent collective intelligence "
            "that accumulates knowledge across evolution runs and sessions. "
            "Actions: 'status', 'seed' (extract lessons from a run), 'reset', 'export', 'import'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "seed", "reset", "export", "import"],
                    "description": "Hive-mind action.",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for 'seed' action.",
                },
                "data": {
                    "type": "string",
                    "description": "JSON string for 'import' action.",
                },
            },
            "required": ["action"],
        },
    ),
    types.Tool(
        name="howler_sync_push",
        description=(
            "Push a completed evolution run to the shared team database. "
            "Requires HOWLER_API_URL and HOWLER_API_KEY. Only completed runs can be pushed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run to push, or 'all' for all unsynced completed runs.",
                },
                "include_memory": {
                    "type": "boolean",
                    "description": "Also push local hive-mind memory.",
                    "default": False,
                },
            },
            "required": ["run_id"],
        },
    ),
    types.Tool(
        name="howler_sync_pull",
        description=(
            "Pull shared hive-mind memory from the team database. "
            "Lessons learned by other developers' agents become available locally."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Memory namespace to pull. Default: 'default'.",
                    "default": "default",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="howler_orchestrator_status",
        description="Check which orchestration backend is active (local LLM or claude-flow). Returns the backend name, status, and capabilities.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="howler_deploy_agents",
        description="Deploy top-performing agents from a completed evolution run as active workers. Uses the detected orchestrator (claude-flow if available, local LLM otherwise).",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run ID to deploy agents from.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top agents to deploy. Default: 3.",
                    "default": 3,
                    "minimum": 1,
                },
                "task_domain": {
                    "type": "string",
                    "description": "Task domain for deployed agents. Default: from run config.",
                },
            },
            "required": ["run_id"],
        },
    ),
    types.Tool(
        name="howler_auto_evolve",
        description=(
            "Run a complete auto-evolution cycle: evolve agents, deploy the best performers, "
            "and seed lessons into the hive-mind. This is the recommended single-call entry point "
            "for automated agent improvement. Works across all MCP-compatible hosts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "population_size": {
                    "type": "integer",
                    "description": "Total agents in the population.",
                    "default": 10,
                    "minimum": 2,
                },
                "num_iterations": {
                    "type": "integer",
                    "description": "Evolution generations to run.",
                    "default": 5,
                    "minimum": 1,
                },
                "task_domain": {
                    "type": "string",
                    "description": "Domain: 'general', 'coding', 'math', 'writing'.",
                    "default": "general",
                },
                "model": {
                    "type": "string",
                    "description": "LiteLLM model string.",
                    "default": "claude-sonnet-4-20250514",
                },
                "deploy_top_k": {
                    "type": "integer",
                    "description": "Deploy the top-K agents after evolution. Default: 3.",
                    "default": 3,
                    "minimum": 1,
                },
                "seed_hivemind": {
                    "type": "boolean",
                    "description": "Auto-seed lessons into hive-mind memory after completion. Default: true.",
                    "default": True,
                },
            },
            "required": [],
        },
    ),
]


# --------------------------------------------------------------------------- #
# Tool handlers                                                                #
# --------------------------------------------------------------------------- #


async def _handle_howler_evolve(args: dict[str, Any]) -> list[types.TextContent]:
    population_size = int(args.get("population_size", 10))
    group_size = int(args.get("group_size", 3))
    num_iterations = int(args.get("num_iterations", 5))
    alpha = float(args.get("alpha", 0.5))
    task_domain = str(args.get("task_domain", "general"))
    model = str(args.get("model", "claude-sonnet-4-20250514"))

    if _remote_api_url:
        result = await _proxy_post(
            "runs",
            {
                "population_size": population_size,
                "group_size": group_size,
                "num_iterations": num_iterations,
                "alpha": alpha,
                "task_domain": task_domain,
                "model": model,
            },
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    run_id = _runner.start_run(
        population_size=population_size,
        group_size=group_size,
        num_iterations=num_iterations,
        alpha=alpha,
        task_domain=task_domain,
        model=model,
    )

    # Launch the evolution in a background task so the MCP call returns quickly.
    task = asyncio.create_task(
        _execute_run_background(run_id),
        name=f"howler_run_{run_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    response = {
        "run_id": run_id,
        "status": "started",
        "message": (
            f"Evolution run started with {population_size} agents over {num_iterations} generations. "
            f"Domain: {task_domain}, Model: {model}. "
            f"Use howler_status with run_id='{run_id}' to monitor progress."
        ),
    }
    return [types.TextContent(type="text", text=json.dumps(response, indent=2))]


async def _execute_run_background(run_id: str) -> None:
    """Execute an evolution run in the background, swallowing exceptions into the record."""
    try:
        await _runner.run_async(run_id)
    except Exception as exc:
        logger.error("background_run_failed", run_id=run_id, error=str(exc))


async def _handle_howler_status(args: dict[str, Any]) -> list[types.TextContent]:
    run_id = str(args["run_id"])

    if _remote_api_url:
        result = await _proxy_get(f"runs/{run_id}")
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    try:
        status = _runner.get_status(run_id)
    except KeyError:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown run_id: {run_id}"}),
            )
        ]
    return [types.TextContent(type="text", text=json.dumps(status, indent=2))]


async def _handle_howler_list_agents(args: dict[str, Any]) -> list[types.TextContent]:
    run_id = str(args["run_id"])
    top_k: int | None = int(args["top_k"]) if "top_k" in args else None

    if _remote_api_url:
        params: dict[str, Any] = {}
        if top_k is not None:
            params["top_k"] = top_k
        result = await _proxy_get(f"runs/{run_id}/agents", params=params)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    try:
        agents = _runner.list_agents(run_id, top_k=top_k)
    except KeyError:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown run_id: {run_id}"}),
            )
        ]
    result = {"run_id": run_id, "count": len(agents), "agents": agents}
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_submit_experience(args: dict[str, Any]) -> list[types.TextContent]:
    run_id = str(args["run_id"])
    agent_id = str(args["agent_id"])
    task_description = str(args["task_description"])
    outcome = str(args["outcome"])
    score = float(args["score"])
    key_decisions: list[str] = args.get("key_decisions", [])
    lessons_learned: list[str] = args.get("lessons_learned", [])

    if _remote_api_url:
        result = await _proxy_post(
            f"runs/{run_id}/experience",
            {
                "agent_id": agent_id,
                "task_description": task_description,
                "outcome": outcome,
                "score": score,
                "key_decisions": key_decisions,
                "lessons_learned": lessons_learned,
            },
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    try:
        trace_id = await _runner.submit_experience(
            run_id=run_id,
            agent_id=agent_id,
            task_description=task_description,
            outcome=outcome,
            score=score,
            key_decisions=key_decisions,
            lessons_learned=lessons_learned,
        )
    except (KeyError, RuntimeError) as exc:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": str(exc)}),
            )
        ]

    result = {
        "trace_id": trace_id,
        "run_id": run_id,
        "agent_id": agent_id,
        "message": "Experience trace submitted successfully.",
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_get_experience(args: dict[str, Any]) -> list[types.TextContent]:
    run_id = str(args["run_id"])
    group_id: str | None = args.get("group_id")

    if _remote_api_url:
        params: dict[str, Any] = {}
        if group_id:
            params["group_id"] = group_id
        result = await _proxy_get(f"runs/{run_id}/experience", params=params)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    try:
        context = await _runner.get_experience_context(run_id=run_id, group_id=group_id)
    except (KeyError, RuntimeError) as exc:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": str(exc)}),
            )
        ]

    result = {"run_id": run_id, "group_id": group_id, "context": context}
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_configure(args: dict[str, Any]) -> list[types.TextContent]:
    acting_model: str | None = args.get("acting_model")
    evolving_model: str | None = args.get("evolving_model")
    reflecting_model: str | None = args.get("reflecting_model")
    api_key: str | None = args.get("api_key")

    if _remote_api_url:
        result = await _proxy_post(
            "config",
            {
                "acting_model": acting_model,
                "evolving_model": evolving_model,
                "reflecting_model": reflecting_model,
            },
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    updated = _runner.configure(
        acting_model=acting_model,
        evolving_model=evolving_model,
        reflecting_model=reflecting_model,
        api_key=api_key,
    )
    result = {
        "message": "Configuration updated.",
        "session_config": {k: v for k, v in updated.items() if k != "api_key"},
        "api_key_set": "api_key" in updated,
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_memory(args: dict[str, Any]) -> list[types.TextContent]:
    if _db is None:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"error": "Persistence not initialized. Run in local or hybrid mode."}
                ),
            )
        ]

    from howler_agents.hivemind.memory import CollectiveMemory

    memory = CollectiveMemory(_db)

    action = args["action"]
    namespace = args.get("namespace", "default")

    if action == "store":
        key = args.get("key")
        value = args.get("value")
        if not key or not value:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": "key and value are required for store"})
                )
            ]
        entry_id = await memory.store(
            namespace=namespace,
            key=key,
            value=value,
            tags=args.get("tags"),
            score=args.get("score", 0.0),
        )
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"id": entry_id, "action": "stored", "namespace": namespace, "key": key}
                ),
            )
        ]

    elif action == "retrieve":
        key = args.get("key")
        if not key:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": "key is required for retrieve"})
                )
            ]
        entry = await memory.retrieve(namespace=namespace, key=key)
        if entry is None:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": f"No entry found for {namespace}/{key}"})
                )
            ]
        return [types.TextContent(type="text", text=json.dumps(entry, indent=2))]

    elif action == "search":
        query = args.get("query", "")
        limit = args.get("limit", 10)
        results = await memory.search(
            query=query, namespace=namespace if namespace != "default" else None, limit=limit
        )
        return [
            types.TextContent(
                type="text", text=json.dumps({"results": results, "count": len(results)}, indent=2)
            )
        ]

    elif action == "list":
        limit = args.get("limit", 20)
        entries = await memory.list(
            namespace=namespace if namespace != "default" else None, limit=limit
        )
        return [
            types.TextContent(
                type="text", text=json.dumps({"entries": entries, "count": len(entries)}, indent=2)
            )
        ]

    elif action == "delete":
        key = args.get("key")
        if not key:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": "key is required for delete"})
                )
            ]
        deleted = await memory.delete(namespace=namespace, key=key)
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"deleted": deleted, "namespace": namespace, "key": key}),
            )
        ]

    return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown action: {action}"}))]


async def _handle_howler_history(args: dict[str, Any]) -> list[types.TextContent]:
    if _db is None:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": "Persistence not initialized."})
            )
        ]

    action = args["action"]
    run_id = args.get("run_id")
    agent_id = args.get("agent_id")
    limit = args.get("limit", 20)

    if action == "runs":
        sql = "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?"
        params: tuple = (limit,)
        if run_id:
            sql = "SELECT * FROM runs WHERE run_id = ?"
            params = (run_id,)
        rows = await _db.execute(sql, params)
        return [
            types.TextContent(
                type="text", text=json.dumps({"runs": rows, "count": len(rows)}, indent=2)
            )
        ]

    elif action == "agents":
        if run_id:
            rows = await _db.execute(
                "SELECT * FROM agents WHERE run_id = ? ORDER BY combined_score DESC LIMIT ?",
                (run_id, limit),
            )
        elif agent_id:
            rows = await _db.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        else:
            rows = await _db.execute(
                "SELECT * FROM agents ORDER BY combined_score DESC LIMIT ?", (limit,)
            )
        return [
            types.TextContent(
                type="text", text=json.dumps({"agents": rows, "count": len(rows)}, indent=2)
            )
        ]

    elif action == "traces":
        if run_id:
            rows = await _db.execute(
                "SELECT * FROM traces WHERE run_id = ? ORDER BY recorded_at DESC LIMIT ?",
                (run_id, limit),
            )
        elif agent_id:
            rows = await _db.execute(
                "SELECT * FROM traces WHERE agent_id = ? ORDER BY recorded_at DESC LIMIT ?",
                (agent_id, limit),
            )
        else:
            rows = await _db.execute(
                "SELECT * FROM traces ORDER BY recorded_at DESC LIMIT ?", (limit,)
            )
        return [
            types.TextContent(
                type="text", text=json.dumps({"traces": rows, "count": len(rows)}, indent=2)
            )
        ]

    elif action == "stats":
        runs = await _db.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed, MAX(best_score) as best FROM runs"
        )
        agents = await _db.execute("SELECT COUNT(*) as total FROM agents")
        traces = await _db.execute("SELECT COUNT(*) as total FROM traces")
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "runs": runs[0] if runs else {},
                        "agents": agents[0] if agents else {},
                        "traces": traces[0] if traces else {},
                    },
                    indent=2,
                ),
            )
        ]

    return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown action: {action}"}))]


async def _handle_howler_hivemind(args: dict[str, Any]) -> list[types.TextContent]:
    if _db is None:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": "Persistence not initialized."})
            )
        ]

    from howler_agents.hivemind.coordinator import HiveMindCoordinator

    hive = HiveMindCoordinator(_db)
    action = args["action"]

    if action == "status":
        status = await hive.status()
        return [types.TextContent(type="text", text=json.dumps(status, indent=2))]

    elif action == "seed":
        run_id = args.get("run_id")
        if not run_id:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": "run_id required for seed"})
                )
            ]
        result = await hive.seed_from_run(run_id)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif action == "reset":
        result = await hive.reset()
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif action == "export":
        data = await hive.export_json()
        return [types.TextContent(type="text", text=json.dumps(data, indent=2))]

    elif action == "import":
        raw = args.get("data")
        if not raw:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": "data (JSON string) required for import"}),
                )
            ]
        data = json.loads(raw)
        result = await hive.import_json(data)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown action: {action}"}))]


async def _handle_howler_sync_push(args: dict[str, Any]) -> list[types.TextContent]:
    mode = _get_mode()
    if mode not in ("hybrid",):
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Sync requires both HOWLER_API_URL and HOWLER_API_KEY to be set.",
                        "mode": mode,
                    }
                ),
            )
        ]
    if _db is None:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": "Persistence not initialized."})
            )
        ]

    from howler_agents.persistence.sync_client import SyncClient

    assert _remote_api_url is not None
    assert _api_key is not None
    sync = SyncClient(api_url=_remote_api_url, api_key=_api_key)

    run_id = args["run_id"]
    result = await sync.push_run(run_id, _db)

    if args.get("include_memory", False):
        mem_result = await sync.push_memory(_db)
        result["memory_sync"] = mem_result

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_sync_pull(args: dict[str, Any]) -> list[types.TextContent]:
    mode = _get_mode()
    if mode not in ("hybrid",):
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Sync requires both HOWLER_API_URL and HOWLER_API_KEY to be set.",
                        "mode": mode,
                    }
                ),
            )
        ]
    if _db is None:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": "Persistence not initialized."})
            )
        ]

    from howler_agents.persistence.sync_client import SyncClient

    assert _remote_api_url is not None
    assert _api_key is not None
    sync = SyncClient(api_url=_remote_api_url, api_key=_api_key)

    namespace = args.get("namespace", "default")
    result = await sync.pull_memory(_db, namespace=namespace)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_orchestrator_status(arguments: dict[str, Any]) -> list[types.TextContent]:
    info = _runner.get_orchestrator_info()
    return [types.TextContent(type="text", text=json.dumps(info, indent=2))]


async def _handle_howler_deploy_agents(arguments: dict[str, Any]) -> list[types.TextContent]:
    run_id = arguments["run_id"]
    top_k = arguments.get("top_k", 3)

    try:
        agents = _runner.list_agents(run_id, top_k=top_k)
    except KeyError:
        return [
            types.TextContent(type="text", text=json.dumps({"error": f"Unknown run: {run_id}"}))
        ]

    orchestrator_info = _runner.get_orchestrator_info()
    orchestrator = _runner._orchestrator

    deployed: list[dict[str, Any]] = []
    for a in agents:
        entry: dict[str, Any] = {
            "agent_id": a["agent_id"],
            "combined_score": a["combined_score"],
            "generation": a["generation"],
        }
        if orchestrator is not None:
            try:
                prompt = (
                    f"You are an evolved AI agent (gen {a['generation']}, "
                    f"score {a['combined_score']:.3f}). "
                    f"Execute tasks in the '{arguments.get('task_domain', 'general')}' domain."
                )
                spawned = await orchestrator.spawn_agent(
                    prompt=prompt,
                    task_domain=arguments.get("task_domain", "general"),
                    agent_config={"howler_agent_id": a["agent_id"]},
                )
                entry["deployed_id"] = spawned.agent_id
                entry["status"] = "deployed"
            except Exception as exc:
                entry["status"] = "deploy_failed"
                entry["error"] = str(exc)
        else:
            entry["status"] = "listed"  # No orchestrator available
        deployed.append(entry)

    result = {
        "run_id": run_id,
        "orchestrator": orchestrator_info["backend"],
        "agents_deployed": sum(1 for d in deployed if d.get("status") == "deployed"),
        "agents": deployed,
    }

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_howler_auto_evolve(args: dict[str, Any]) -> list[types.TextContent]:
    population_size = int(args.get("population_size", 10))
    num_iterations = int(args.get("num_iterations", 5))
    task_domain = str(args.get("task_domain", "general"))
    model = str(args.get("model", "claude-sonnet-4-20250514"))
    deploy_top_k = int(args.get("deploy_top_k", 3))
    seed_hivemind = bool(args.get("seed_hivemind", True))

    try:
        run_id = _runner.start_run(
            population_size=population_size,
            group_size=int(args.get("group_size", 3)),
            num_iterations=num_iterations,
            alpha=float(args.get("alpha", 0.5)),
            task_domain=task_domain,
            model=model,
        )

        # Block until the full evolution cycle completes.
        await _runner.run_async(run_id)

        status = _runner.get_status(run_id)
        top_agents = _runner.list_agents(run_id, top_k=deploy_top_k)

        seeded_hivemind = False
        if seed_hivemind and _db is not None:
            if hasattr(_runner, "seed_hivemind"):
                await _runner.seed_hivemind(run_id)
                seeded_hivemind = True

        orchestrator_info = _runner.get_orchestrator_info()

        result = {
            "run_id": run_id,
            "status": status.get("status", "completed"),
            "best_score": status.get("best_score"),
            "generations": {
                "requested": num_iterations,
                "completed": status.get("current_generation", num_iterations),
            },
            "population_size": population_size,
            "task_domain": task_domain,
            "model": model,
            "top_agents": [
                {
                    "agent_id": a["agent_id"],
                    "combined_score": a["combined_score"],
                    "generation": a["generation"],
                    "status": "deployed",
                }
                for a in top_agents
            ],
            "orchestrator_backend": orchestrator_info.get("backend"),
            "seeded_hivemind": seeded_hivemind,
            "message": (
                f"Auto-evolution complete. Run {run_id} finished {num_iterations} generation(s) "
                f"over {population_size} agents in domain '{task_domain}'. "
                f"Top-{deploy_top_k} agents selected. "
                + (
                    "Lessons seeded into hive-mind."
                    if seeded_hivemind
                    else "Hive-mind seeding skipped."
                )
            ),
        }
    except Exception as exc:
        logger.error("howler_auto_evolve_error", error=str(exc))
        result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# Dispatch table mapping tool names to async handler functions.
_TOOL_HANDLERS = {
    "howler_evolve": _handle_howler_evolve,
    "howler_status": _handle_howler_status,
    "howler_list_agents": _handle_howler_list_agents,
    "howler_submit_experience": _handle_howler_submit_experience,
    "howler_get_experience": _handle_howler_get_experience,
    "howler_configure": _handle_howler_configure,
    "howler_memory": _handle_howler_memory,
    "howler_history": _handle_howler_history,
    "howler_hivemind": _handle_howler_hivemind,
    "howler_sync_push": _handle_howler_sync_push,
    "howler_sync_pull": _handle_howler_sync_pull,
    "howler_orchestrator_status": _handle_howler_orchestrator_status,
    "howler_deploy_agents": _handle_howler_deploy_agents,
    "howler_auto_evolve": _handle_howler_auto_evolve,
}


# --------------------------------------------------------------------------- #
# Resource helpers                                                             #
# --------------------------------------------------------------------------- #


async def _resource_config() -> str:
    """Render current session configuration as JSON."""
    config = _runner.get_session_config()
    return json.dumps(
        {
            "session_config": {k: v for k, v in config.items() if k != "api_key"},
            "api_key_configured": "api_key" in config,
            "remote_api_url": _remote_api_url,
            "mode": _get_mode(),
            "persistence": "sqlite" if _db is not None else "memory",
            "db_path": str(_db._db_path) if _db is not None else None,
        },
        indent=2,
    )


async def _resource_runs() -> str:
    """Render all runs as JSON."""
    if _remote_api_url:
        try:
            data = await _proxy_get("runs")
            return json.dumps(data, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return json.dumps({"runs": _runner.list_runs()}, indent=2)


async def _resource_run_detail(run_id: str) -> str:
    """Render details of a specific run as JSON."""
    if _remote_api_url:
        try:
            data = await _proxy_get(f"runs/{run_id}")
            return json.dumps(data, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    try:
        status = _runner.get_status(run_id)
        agents = _runner.list_agents(run_id)
        return json.dumps({"status": status, "agents": agents}, indent=2)
    except KeyError:
        return json.dumps({"error": f"Unknown run_id: {run_id}"})


# --------------------------------------------------------------------------- #
# Server factory                                                               #
# --------------------------------------------------------------------------- #


def create_server() -> Server:
    """Build and return the configured MCP Server instance."""

    @asynccontextmanager
    async def lifespan(server: Server) -> AsyncIterator[None]:
        global _runner, _db
        mode = _get_mode()

        if mode != "remote":
            # Initialize SQLite persistence
            _db = DatabaseManager()  # auto-detects .howler-agents/evolution.db
            _runner = LocalRunner(db=_db)
            await _runner.initialize()

        logger.info(
            "howler_mcp_server_start",
            mode=mode,
            remote_url=_remote_api_url,
        )
        yield

        if _db is not None:
            await _runner.close()

        logger.info("howler_mcp_server_stop")

    server = Server("howler-agents", lifespan=lifespan)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )
            ]
        try:
            return await handler(arguments)
        except Exception as exc:
            logger.error("tool_call_error", tool=name, error=str(exc))
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": str(exc)}),
                )
            ]

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        resources = [
            types.Resource(
                uri="howler://config",  # type: ignore[arg-type]
                name="howler://config",
                description="Current howler-agents session configuration.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="howler://runs",  # type: ignore[arg-type]
                name="howler://runs",
                description="List of all evolution runs in this session.",
                mimeType="application/json",
            ),
        ]
        # Dynamically add per-run resources
        for record in _runner.list_runs():
            rid = record["run_id"]
            resources.append(
                types.Resource(
                    uri=f"howler://runs/{rid}",  # type: ignore[arg-type]
                    name=f"howler://runs/{rid}",
                    description=(
                        f"Details for run {rid}: domain={record['task_domain']}, "
                        f"status={record['status']}, best_score={record['best_score']}"
                    ),
                    mimeType="application/json",
                )
            )
        return resources

    @server.read_resource()
    async def read_resource(uri: Any) -> list[ReadResourceContents]:
        uri_str = str(uri)

        if uri_str == "howler://config":
            content = await _resource_config()
            return [ReadResourceContents(content=content, mime_type="application/json")]

        if uri_str == "howler://runs":
            content = await _resource_runs()
            return [ReadResourceContents(content=content, mime_type="application/json")]

        if uri_str.startswith("howler://runs/"):
            run_id = uri_str.removeprefix("howler://runs/")
            content = await _resource_run_detail(run_id)
            return [ReadResourceContents(content=content, mime_type="application/json")]

        return [
            ReadResourceContents(
                content=json.dumps({"error": f"Unknown resource URI: {uri_str}"}),
                mime_type="application/json",
            )
        ]

    return server


# --------------------------------------------------------------------------- #
# Stdio entry point                                                            #
# --------------------------------------------------------------------------- #


async def run_stdio() -> None:
    """Run the MCP server over stdin/stdout (for Claude Code, Cursor, etc.)."""
    server = create_server()
    init_options = InitializationOptions(
        server_name="howler-agents",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=__import__(
                "mcp.server.lowlevel.server", fromlist=["NotificationOptions"]
            ).NotificationOptions(),
            experimental_capabilities={},
        ),
        instructions=(
            "Howler Agents MCP server — Group-Evolving AI Agents (GEA) with persistent hive-mind. "
            "Use howler_configure to set LLM models, howler_evolve to start a run. "
            "Monitor with howler_status, inspect with howler_list_agents. "
            "Persistent tools: howler_memory (collective knowledge), howler_history (browse past runs), "
            "howler_hivemind (manage collective intelligence). "
            "Orchestration: howler_orchestrator_status (check backend), howler_deploy_agents (deploy top agents). "
            "Team sync: howler_sync_push/howler_sync_pull (share with team database)."
        ),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream=read_stream,
            write_stream=write_stream,
            initialization_options=init_options,
        )


async def run_sse(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run the MCP server over SSE transport."""
    try:
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
    except ImportError as exc:
        raise RuntimeError(
            "SSE transport requires starlette and uvicorn. "
            "Install with: pip install starlette uvicorn"
        ) from exc

    server = create_server()
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Any) -> Any:
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            init_options = InitializationOptions(
                server_name="howler-agents",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=__import__(
                        "mcp.server.lowlevel.server", fromlist=["NotificationOptions"]
                    ).NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
            await server.run(
                read_stream=read_stream,
                write_stream=write_stream,
                initialization_options=init_options,
            )

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    anyio.run(run_stdio)
