# Local Usage Guide

## Overview

Howler-agents implements Group-Evolving Agents (GEA) from arXiv:2602.04837 -- populations of AI agents that evolve together by sharing experience across lineages. It runs entirely locally using SQLite for persistence. No Docker, Postgres, or external services required.

The MCP server integrates directly with Claude Code, providing 11 tools for managing evolution runs, collective memory, and hive-mind intelligence. All data persists in a `.howler-agents/` directory at your repository root.

## Prerequisites

- Python 3.12+
- Claude Code CLI installed
- An LLM API key (Anthropic, OpenAI, or any LiteLLM-supported provider)

## Installation

### From Source (Development)

```bash
git clone https://github.com/jbeck018/howler-agents.git
cd howler-agents
pip install -e "packages/howler-agents-core[mcp]"
```

This installs the core library with MCP dependencies (`mcp`, `httpx`) and registers the `howler-agents` CLI command.

### From PyPI (Coming Soon)

```bash
pip install howler-agents-core[mcp]
```

### Verify the CLI

```bash
howler-agents --help
```

You should see subcommands: `serve`, `evolve`, `status`, `configure`.

## Register MCP Server

```bash
claude mcp add howler-agents -- howler-agents serve --transport stdio
```

This adds an entry to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "howler-agents": {
      "command": "howler-agents",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

Restart Claude Code to pick up the new server.

## Verify Setup

After restarting Claude Code, verify the tools are available by asking Claude Code:

> "List the available howler-agents tools"

You should see 11 tools starting with `howler_`. Alternatively, call `howler_configure` with no arguments to confirm the server is running and inspect the default configuration.

## Running Your First Evolution

### Via Claude Code (Recommended)

Ask Claude Code to run an evolution:

> "Run a howler evolution with 8 agents over 3 generations in the coding domain"

Claude Code will call `howler_evolve` with the appropriate parameters. The evolution runs asynchronously in the background. Use `howler_status` to monitor progress.

### Via MCP Tool Directly

Call `howler_evolve` with parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `population_size` | integer | 10 | Total agents in the population (K) |
| `group_size` | integer | 3 | Agents per evolution group (M) |
| `num_iterations` | integer | 5 | Number of evolution generations |
| `alpha` | number | 0.5 | Balance: 0.0 = pure novelty, 1.0 = pure performance |
| `task_domain` | string | "general" | Domain: "general", "coding", "math", "writing" |
| `model` | string | "claude-sonnet-4-20250514" | LiteLLM model string |

The tool returns a `run_id` immediately. The evolution executes in the background.

### Via CLI

```bash
howler-agents evolve \
  --population 8 \
  --iterations 3 \
  --domain coding \
  --model gpt-4o \
  --alpha 0.5
```

The CLI runs synchronously and prints a summary when complete. Add `--json-output` for machine-readable output.

## Configuration

### Model Configuration

Use `howler_configure` to assign different LLM models to the three evolutionary roles:

| Role | Description | Example |
|---|---|---|
| `acting_model` | Agents performing tasks | `gpt-4o` |
| `evolving_model` | Meta-LLM generating mutations | `claude-sonnet-4-20250514` |
| `reflecting_model` | Reflective analysis of experience | `anthropic/claude-haiku` |

Any LiteLLM model string works. If a role-specific model is not set, it falls back to the `model` parameter passed to `howler_evolve`.

Example:

```
howler_configure with:
  acting_model: "gpt-4o"
  evolving_model: "claude-sonnet-4-20250514"
```

Configuration persists for the lifetime of the MCP server process.

### API Keys

Set your API key via environment variable before starting Claude Code:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

Alternatively, pass `api_key` to `howler_configure` at runtime (not recommended for security reasons).

### Alpha Parameter

The `alpha` parameter controls the balance between performance and novelty in agent selection:

- `alpha=1.0` -- pure performance selection (exploit)
- `alpha=0.0` -- pure novelty selection (explore)
- `alpha=0.5` -- balanced (default, recommended)

## Persistence

All data is stored in `.howler-agents/evolution.db` at your repository root (SQLite with WAL mode). A `.gitignore` is automatically created inside `.howler-agents/` to exclude database files from version control.

### What Gets Stored

- **Runs**: Configuration, status, generation summaries, best scores
- **Agents**: System prompts, capability vectors, lineage, scores per generation
- **Traces**: Task outcomes, key decisions, lessons learned, scores
- **Memory**: Cross-run collective knowledge organized by namespace
- **Consensus**: Proposals and votes from the hive-mind

### Data Survives Sessions

The SQLite database persists across Claude Code sessions. When you restart Claude Code, all previous runs, agents, and collective memory are still available. The MCP server hydrates past runs from the database on startup.

Query persistent data with:

- `howler_history` with `action: "runs"` -- list past runs
- `howler_history` with `action: "stats"` -- aggregate statistics
- `howler_memory` with `action: "list"` -- browse collective knowledge

### Database Location

The database is placed at `<repo-root>/.howler-agents/evolution.db`. The repo root is detected by walking up from the current directory looking for a `.git` directory. If no git repo is found, it uses the current working directory.

## MCP Tools Reference

### howler_evolve

Start an evolution run. Returns a `run_id` for tracking.

| Parameter | Required | Type | Default | Description |
|---|---|---|---|---|
| `population_size` | no | integer | 10 | Total agents (K). Min: 2. |
| `group_size` | no | integer | 3 | Agents per group (M). Min: 1. |
| `num_iterations` | no | integer | 5 | Generations to run. Min: 1. |
| `alpha` | no | number | 0.5 | Performance vs novelty weight (0.0-1.0). |
| `task_domain` | no | string | "general" | Domain: general, coding, math, writing. |
| `model` | no | string | "claude-sonnet-4-20250514" | LiteLLM model string. |

**Example response:**

```json
{
  "run_id": "a1b2c3d4-...",
  "status": "started",
  "message": "Evolution run started with 10 agents over 5 generations."
}
```

---

### howler_status

Check progress of a running or completed evolution run.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `run_id` | yes | string | Run identifier from `howler_evolve`. |

**Returns:** Generation progress, best score, population size, mean score, timestamps, error (if any).

---

### howler_list_agents

List agents in a run's population, optionally filtered to the top performers.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `run_id` | yes | string | Run identifier. |
| `top_k` | no | integer | Return only the top-K agents by combined score. |

**Returns:** Array of agents with `agent_id`, `generation`, `group_id`, `performance_score`, `novelty_score`, `combined_score`, `capability_vector`, and `patches_applied`.

---

### howler_submit_experience

Submit a task experience trace to the shared experience pool. Enriches the evolutionary context used for the next generation of mutations.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `run_id` | yes | string | Run identifier. |
| `agent_id` | yes | string | ID of the agent that performed the task. |
| `task_description` | yes | string | Natural-language description of the task. |
| `outcome` | yes | string | Result: "success", "failure", or a custom label. |
| `score` | yes | number | Numeric score (0.0-1.0). |
| `key_decisions` | no | string[] | Key decisions made during execution. |
| `lessons_learned` | no | string[] | Lessons to inform future mutations. |

---

### howler_get_experience

Retrieve aggregated experience context for a run, formatted for LLM consumption.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `run_id` | yes | string | Run identifier. |
| `group_id` | no | string | Filter to a specific group. |

---

### howler_configure

Configure LLM models for the three evolutionary roles. Settings persist for the server process lifetime.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `acting_model` | no | string | LiteLLM model string for task agents. |
| `evolving_model` | no | string | LiteLLM model string for the mutation meta-LLM. |
| `reflecting_model` | no | string | LiteLLM model string for reflective analysis. |
| `api_key` | no | string | API key for LLM calls. |

---

### howler_memory

Persistent hive-mind memory. Survives across sessions and runs.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `action` | yes | string | One of: "store", "retrieve", "search", "list", "delete". |
| `namespace` | no | string | Namespace (e.g. "lessons", "patterns", "decisions"). Default: "default". |
| `key` | conditional | string | Required for store, retrieve, delete. |
| `value` | conditional | string | Required for store. |
| `query` | conditional | string | Required for search. |
| `tags` | no | string[] | Tags for store action. |
| `score` | no | number | Relevance score (0.0-1.0) for store. |
| `limit` | no | integer | Max results for search/list. Default: 10. |

**Example -- store a lesson:**

```
howler_memory with:
  action: "store"
  namespace: "lessons"
  key: "error-handling-pattern"
  value: "Always wrap LLM calls in try/catch with exponential backoff"
  score: 0.85
  tags: ["coding", "reliability"]
```

**Example -- search memory:**

```
howler_memory with:
  action: "search"
  query: "error handling"
  namespace: "lessons"
```

---

### howler_history

Browse persistent evolution history stored in SQLite.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `action` | yes | string | One of: "runs", "agents", "traces", "stats". |
| `run_id` | no | string | Filter to a specific run. |
| `agent_id` | no | string | Filter to a specific agent. |
| `limit` | no | integer | Max results. Default: 20. |

**Actions:**

- `"runs"` -- list past evolution runs with status, scores, timestamps
- `"agents"` -- list agents, optionally filtered by run_id or agent_id
- `"traces"` -- list experience traces with scores and lessons
- `"stats"` -- aggregate counts (total runs, agents, traces, best score)

---

### howler_hivemind

Manage the repository's collective intelligence.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `action` | yes | string | One of: "status", "seed", "reset", "export", "import". |
| `run_id` | conditional | string | Required for "seed" action. |
| `data` | conditional | string | JSON string for "import" action. |

**Actions:**

- `"status"` -- memory stats, pending consensus, completed runs count
- `"seed"` -- extract lessons and decisions from a completed run into hive-mind memory; auto-proposes consensus items for patterns appearing in 3+ traces
- `"reset"` -- clear all memory and consensus (destructive)
- `"export"` -- export all hive-mind data as JSON
- `"import"` -- import hive-mind data from a JSON string (uses UPSERT)

---

### howler_sync_push

Push a completed run to a shared team database. Requires `HOWLER_API_URL` and `HOWLER_API_KEY`.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `run_id` | yes | string | Run to push, or "all" for all unsynced completed runs. |
| `include_memory` | no | boolean | Also push local hive-mind memory. Default: false. |

---

### howler_sync_pull

Pull shared hive-mind memory from the team database into local SQLite.

| Parameter | Required | Type | Description |
|---|---|---|---|
| `namespace` | no | string | Memory namespace to pull. Default: "default". |

## Task Domains

Each domain provides a set of built-in probe tasks used during evolution:

| Domain | Task Types |
|---|---|
| `general` | Self-description, problem-solving strategies, best practices |
| `coding` | Code generation, debugging, refactoring |
| `math` | Calculus, algebra, geometry |
| `writing` | Persuasion, summarization, editing |

The agents are evaluated against these tasks each generation. Scores feed into the performance-novelty selection criterion.

## How Evolution Works

1. **Initialization**: A population of K agents is created, each assigned to a group.
2. **Evaluation**: Every agent runs probe tasks. Results produce a capability vector and performance score.
3. **Selection**: Agents are ranked by a combined score: `alpha * performance + (1 - alpha) * novelty`. Novelty is computed from the agent's capability vector distance to its nearest neighbors.
4. **Reproduction**: Top agents in each group produce offspring. The evolving LLM generates mutations (framework patches) informed by the shared experience pool.
5. **Experience sharing**: Lessons learned and key decisions are collected in the shared experience pool and, upon completion, seeded into the hive-mind for cross-run memory.
6. **Repeat**: Steps 2-5 repeat for `num_iterations` generations.

After a run completes, the hive-mind coordinator automatically extracts high-value lessons and proposes consensus items for frequently-recurring patterns.

## Team Sync (Optional)

For sharing evolution results across a team via a remote howler-agents service.

### Setup

Set environment variables before starting Claude Code:

```bash
export HOWLER_API_URL=https://your-howler-service.example.com
export HOWLER_API_KEY=your-api-key
```

Both variables must be set for sync mode. With only `HOWLER_API_URL` (no key), the server operates in full remote-proxy mode where all calls are forwarded to the remote API instead of using local SQLite.

### Push

Push a completed run (with agents and traces) to the team database:

```
howler_sync_push with:
  run_id: "a1b2c3d4-..."
  include_memory: true
```

### Pull

Download team knowledge into your local database:

```
howler_sync_pull with:
  namespace: "lessons"
```

### Conflict Resolution

When pulling, the sync client uses a score-wins strategy:

- If a memory key already exists locally with a **higher** score, the local version is kept.
- If the remote entry has a **higher** score, the local value is overwritten.
- The final score for a key is always `MAX(local_score, remote_score)`.

## Troubleshooting

### MCP server not starting

Verify the CLI is installed and on your PATH:

```bash
which howler-agents
python --version  # Must be 3.12+
```

If installed in a virtual environment, ensure Claude Code can find it. You may need to use the full path in `.mcp.json`:

```json
{
  "mcpServers": {
    "howler-agents": {
      "command": "/path/to/venv/bin/howler-agents",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

### Tools not visible in Claude Code

Restart Claude Code after modifying `.mcp.json`. The server must be registered and the process must start successfully for tools to appear.

### Database errors

If the database becomes corrupted, delete it and let the server recreate it on next startup:

```bash
rm .howler-agents/evolution.db
rm -f .howler-agents/evolution.db-wal .howler-agents/evolution.db-shm
```

All previous runs and memory will be lost.

### Model errors

Verify your API key is set in the environment:

```bash
echo $ANTHROPIC_API_KEY  # For Anthropic models
echo $OPENAI_API_KEY     # For OpenAI models
```

LiteLLM supports many providers. See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list and their required environment variables.

### Evolution run stuck or failed

Check the run status:

```
howler_status with run_id: "your-run-id"
```

If the status is "failed", the `error` field contains the failure reason. Common causes:

- Invalid API key or rate limiting from the LLM provider
- Network connectivity issues
- Model string not recognized by LiteLLM

If a run was interrupted by a process restart, it will automatically be marked as "failed" with the message "Interrupted (process restarted)" when the server next starts.

### Sync not working

Sync requires **both** `HOWLER_API_URL` and `HOWLER_API_KEY` to be set. Without both, `howler_sync_push` and `howler_sync_pull` will return an error indicating the current mode.

Check your mode:

```
howler_configure
```

The response includes a `mode` field: "local", "hybrid", or "remote".
