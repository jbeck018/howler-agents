import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/integrations",
  component: IntegrationsPage,
});

function IntegrationsPage() {
  return (
    <>
      <h1>Integrations</h1>
      <p>
        Howler Agents can be integrated into your AI coding workflow in several ways. The recommended
        approach is via the Model Context Protocol (MCP), which exposes the GEA system as a set of
        tools that any MCP-compatible editor or agent can call. For tools that do not support MCP,
        the REST API is available directly.
      </p>
      <p>
        The hosted service runs at <code>http://209.38.173.33</code>. All endpoints are under{" "}
        <code>/api/v1/</code> and accept either a <code>Bearer</code> JWT token or an{" "}
        <code>X-API-Key</code> header.
      </p>

      <h2>MCP Tool Reference</h2>
      <p>
        When the MCP server is running (either locally or pointed at the remote service), the
        following tools become available inside any connected coding tool:
      </p>
      <table>
        <thead>
          <tr>
            <th>Tool name</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>howler_evolve</code></td>
            <td>Create a new evolution run and optionally step it forward one or more generations</td>
          </tr>
          <tr>
            <td><code>howler_status</code></td>
            <td>Get the current state of a run including generation count and best score</td>
          </tr>
          <tr>
            <td><code>howler_step</code></td>
            <td>Advance an existing run by exactly one evolutionary generation</td>
          </tr>
          <tr>
            <td><code>howler_list_runs</code></td>
            <td>List all runs visible to the authenticated user</td>
          </tr>
          <tr>
            <td><code>howler_list_agents</code></td>
            <td>Return all agent states in a run at the current generation</td>
          </tr>
          <tr>
            <td><code>howler_best_agents</code></td>
            <td>Return the top-K agents ranked by the combined performance-novelty score</td>
          </tr>
          <tr>
            <td><code>howler_submit_experience</code></td>
            <td>Submit an experience trace (task, result, reflection) into the shared pool</td>
          </tr>
          <tr>
            <td><code>howler_get_traces</code></td>
            <td>Retrieve experience traces accumulated for a run</td>
          </tr>
        </tbody>
      </table>

      {/* ------------------------------------------------------------------ */}
      <h2>1. Claude Code Integration</h2>

      <h3>Via MCP (Recommended)</h3>
      <p>
        Claude Code supports MCP natively. Register the Howler Agents MCP server once and the tools
        are available in every conversation automatically.
      </p>

      <h4>Step 1: Install the package</h4>
      <pre><code>{`pip install howler-agents-core[mcp]`}</code></pre>

      <h4>Step 2: Register the server with Claude Code</h4>
      <p>To connect to the hosted service:</p>
      <pre><code>{`# Connect to the hosted service
claude mcp add howler-agents -- howler-agents serve --api-url http://209.38.173.33

# Or run the server locally (in-memory, no database required)
claude mcp add howler-agents -- howler-agents serve`}</code></pre>

      <h4>Step 3: Authenticate</h4>
      <p>
        The first time a tool is called against the remote service, pass your API key via the{" "}
        <code>HOWLER_API_KEY</code> environment variable or supply it through the MCP server
        configuration:
      </p>
      <pre><code>{`# Using an environment variable
HOWLER_API_KEY=your-key-here claude mcp add howler-agents -- howler-agents serve --api-url http://209.38.173.33

# Or export it in your shell profile so it is always set
export HOWLER_API_KEY=your-key-here`}</code></pre>

      <h4>Step 4: Use in a conversation</h4>
      <p>Once registered, ask Claude Code to use the tools directly:</p>
      <pre><code>{`# Example prompt
"Use howler_evolve to start a run with population_size=10, group_size=3, num_iterations=5,
then call howler_best_agents to show me the top 3 agents."`}</code></pre>

      <h3>Via API Key in CLAUDE.md</h3>
      <p>
        If you prefer not to run an MCP server, you can give Claude Code access to the REST API by
        adding an instruction block to your project's <code>CLAUDE.md</code>. Claude Code reads this
        file at the start of every session and treats its contents as standing instructions.
      </p>
      <pre><code>{`## Howler Agents REST API

The Howler Agents service is available at http://209.38.173.33/api/v1/.
Authenticate all requests with the header: X-API-Key: YOUR_API_KEY_HERE

### Common operations (use curl or httpx)

# Register and obtain an API key
curl -s -X POST http://209.38.173.33/api/v1/auth/register \\
  -H "Content-Type: application/json" \\
  -d '{"email":"you@example.com","password":"secret"}' | jq .

curl -s -X POST http://209.38.173.33/api/v1/auth/api-keys \\
  -H "Authorization: Bearer <jwt>" | jq .

# Create an evolution run
curl -s -X POST http://209.38.173.33/api/v1/runs \\
  -H "X-API-Key: YOUR_API_KEY_HERE" \\
  -H "Content-Type: application/json" \\
  -d '{"population_size":10,"group_size":3,"num_iterations":5,"alpha":0.5}' | jq .

# Step one generation
curl -s -X POST http://209.38.173.33/api/v1/runs/{RUN_ID}/step \\
  -H "X-API-Key: YOUR_API_KEY_HERE" | jq .

# Get best agents
curl -s http://209.38.173.33/api/v1/runs/{RUN_ID}/agents/best \\
  -H "X-API-Key: YOUR_API_KEY_HERE" | jq .

# Submit an experience trace
curl -s -X POST http://209.38.173.33/api/v1/runs/{RUN_ID}/experience \\
  -H "X-API-Key: YOUR_API_KEY_HERE" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_id": "agent-42",
    "task": "Fix the off-by-one error in the merge sort implementation",
    "result": "Patched loop bounds; all 47 tests now pass",
    "reflection": "The root cause was using < instead of <= in the termination condition",
    "score": 0.94
  }' | jq .`}</code></pre>

      {/* ------------------------------------------------------------------ */}
      <h2>2. Cursor Integration</h2>
      <p>
        Cursor supports MCP servers through a per-project or global configuration file. Add the
        Howler Agents server to <code>.cursor/mcp.json</code> at the root of your repository.
      </p>

      <h3>Project-level configuration</h3>
      <p>Create or edit <code>.cursor/mcp.json</code>:</p>
      <pre><code>{`{
  "mcpServers": {
    "howler-agents": {
      "command": "howler-agents",
      "args": ["serve"],
      "env": {
        "HOWLER_API_URL": "http://209.38.173.33",
        "HOWLER_API_KEY": "your-api-key-here"
      }
    }
  }
}`}</code></pre>

      <h3>Global configuration</h3>
      <p>
        To make Howler Agents available across all Cursor projects, add the same block to your
        global MCP configuration file at{" "}
        <code>~/.cursor/mcp.json</code> (create the file if it does not exist).
      </p>

      <h3>Local-only mode (no remote service)</h3>
      <pre><code>{`{
  "mcpServers": {
    "howler-agents": {
      "command": "howler-agents",
      "args": ["serve"]
    }
  }
}`}</code></pre>
      <p>
        Omitting <code>HOWLER_API_URL</code> causes the server to run entirely in-memory. No
        database or network access is required, making this suitable for offline experimentation.
      </p>

      {/* ------------------------------------------------------------------ */}
      <h2>3. OpenCode Integration</h2>
      <p>
        OpenCode reads MCP server definitions from its user configuration file. The exact path
        depends on your platform:
      </p>
      <table>
        <thead>
          <tr>
            <th>Platform</th>
            <th>Config file path</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Linux / macOS</td>
            <td><code>~/.config/opencode/config.json</code></td>
          </tr>
          <tr>
            <td>Windows</td>
            <td><code>%APPDATA%\opencode\config.json</code></td>
          </tr>
        </tbody>
      </table>

      <p>Add the following block inside your existing config file:</p>
      <pre><code>{`{
  "mcp": {
    "servers": {
      "howler-agents": {
        "command": "howler-agents",
        "args": ["serve"],
        "env": {
          "HOWLER_API_URL": "http://209.38.173.33",
          "HOWLER_API_KEY": "your-api-key-here"
        }
      }
    }
  }
}`}</code></pre>
      <p>
        Restart OpenCode after saving. The Howler Agents tools will appear in the tool-call panel
        automatically.
      </p>

      {/* ------------------------------------------------------------------ */}
      <h2>4. OpenAI Codex Integration</h2>
      <p>
        Codex (OpenAI's coding assistant) does not natively support the MCP protocol. Use the
        Howler Agents REST API directly from within Codex prompts, or call it from scripts that
        Codex generates.
      </p>

      <h3>Step 1: Register and create an API key</h3>
      <pre><code>{`# Register a new account
curl -s -X POST http://209.38.173.33/api/v1/auth/register \\
  -H "Content-Type: application/json" \\
  -d '{"email":"you@example.com","password":"your-password"}' | jq .

# Log in and capture the JWT
JWT=$(curl -s -X POST http://209.38.173.33/api/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email":"you@example.com","password":"your-password"}' | jq -r .access_token)

# Create a long-lived API key
curl -s -X POST http://209.38.173.33/api/v1/auth/api-keys \\
  -H "Authorization: Bearer $JWT" | jq .`}</code></pre>

      <h3>Step 2: Use the API key in Codex prompts</h3>
      <p>
        Store the API key as an environment variable in your shell or CI environment, then reference
        it in the scripts that Codex generates:
      </p>
      <pre><code>{`export HOWLER_API_KEY=your-api-key-here

# Example: a Python script Codex can generate or call
python - <<'EOF'
import os, httpx

BASE = "http://209.38.173.33/api/v1"
HEADERS = {"X-API-Key": os.environ["HOWLER_API_KEY"]}

# Create a run
run = httpx.post(f"{BASE}/runs", headers=HEADERS, json={
    "population_size": 10,
    "group_size": 3,
    "num_iterations": 5,
    "alpha": 0.5,
}).json()
run_id = run["id"]
print("Created run:", run_id)

# Step through generations
for _ in range(5):
    state = httpx.post(f"{BASE}/runs/{run_id}/step", headers=HEADERS).json()
    print(f"  Generation {state['current_generation']}: best={state['best_score']:.3f}")

# Retrieve top agents
best = httpx.get(f"{BASE}/runs/{run_id}/agents/best", headers=HEADERS).json()
for agent in best[:3]:
    print("Top agent:", agent["id"], "score:", agent["score"])
EOF`}</code></pre>

      <h3>Step 3: Provide context in your Codex system prompt</h3>
      <p>
        Include a brief description of the API in your Codex system prompt so the model knows how
        to call it:
      </p>
      <pre><code>{`You have access to the Howler Agents REST API at http://209.38.173.33/api/v1/.
Authenticate requests with the header X-API-Key: $HOWLER_API_KEY.

Key endpoints:
  POST /runs                          - Create an evolution run
  POST /runs/{id}/step                - Step one generation
  GET  /runs/{id}/agents/best         - Get top-K evolved agents
  POST /runs/{id}/experience          - Submit an experience trace
  GET  /runs/{id}/traces              - List experience traces

Use these endpoints when the user asks you to evolve agents, retrieve evolved
strategies, or submit code review feedback as experience traces.`}</code></pre>

      {/* ------------------------------------------------------------------ */}
      <h2>5. Windsurf, Continue.dev, and Other MCP-Compatible Tools</h2>
      <p>
        Any editor or agent framework that implements the MCP client specification can connect to
        the Howler Agents server. The configuration format varies slightly by tool but the server
        invocation is always the same: <code>howler-agents serve</code>.
      </p>

      <h3>Windsurf</h3>
      <p>
        Add the server to <code>~/.codeium/windsurf/mcp_config.json</code>:
      </p>
      <pre><code>{`{
  "mcpServers": {
    "howler-agents": {
      "command": "howler-agents",
      "args": ["serve"],
      "env": {
        "HOWLER_API_URL": "http://209.38.173.33",
        "HOWLER_API_KEY": "your-api-key-here"
      }
    }
  }
}`}</code></pre>

      <h3>Continue.dev</h3>
      <p>
        Add the server to your Continue configuration at <code>~/.continue/config.json</code>:
      </p>
      <pre><code>{`{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "howler-agents",
          "args": ["serve"],
          "env": {
            "HOWLER_API_URL": "http://209.38.173.33",
            "HOWLER_API_KEY": "your-api-key-here"
          }
        }
      }
    ]
  }
}`}</code></pre>

      <h3>Any other MCP client</h3>
      <p>
        The server speaks the standard MCP stdio transport. Launch it as a subprocess with:
      </p>
      <pre><code>{`howler-agents serve [--api-url http://209.38.173.33]`}</code></pre>
      <p>
        The process reads JSON-RPC messages from stdin and writes responses to stdout, with log
        lines going to stderr. No additional flags or configuration files are required for local
        mode.
      </p>

      {/* ------------------------------------------------------------------ */}
      <h2>6. Local Development Mode</h2>
      <p>
        The MCP server can run entirely in-memory without any database, external service, or
        network access. This is the fastest way to experiment with the GEA system on your local
        machine.
      </p>

      <h3>Installation</h3>
      <pre><code>{`pip install howler-agents-core[mcp]`}</code></pre>

      <h3>Start the server</h3>
      <pre><code>{`howler-agents serve`}</code></pre>
      <p>
        When <code>HOWLER_API_URL</code> is not set, the server creates an <code>InMemoryStore</code>{" "}
        backed evolution loop. All state is held in the process and is discarded when the server
        stops. This is appropriate for:
      </p>
      <ul>
        <li>Evaluating Howler Agents before committing to a database setup</li>
        <li>Running integration tests in CI without external dependencies</li>
        <li>Offline development when the hosted service is unreachable</li>
        <li>Rapid iteration on custom agent prompts and probe tasks</li>
      </ul>

      <h3>Verify the server is responding</h3>
      <p>
        The server exposes a <code>howler_status</code> tool that returns immediately. In a separate
        terminal (or from your editor after connecting via MCP), trigger a call:
      </p>
      <pre><code>{`# Using the Python MCP client SDK (optional verification step)
python - <<'EOF'
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="howler-agents", args=["serve"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

asyncio.run(main())
EOF`}</code></pre>

      {/* ------------------------------------------------------------------ */}
      <h2>7. Using Howler Agents with Agent Skills</h2>
      <p>
        Claude Code supports custom sub-agents defined as Markdown files under{" "}
        <code>.claude/agents/</code>. You can create a dedicated Howler evolver agent that
        orchestrates evolution runs as part of your coding workflow.
      </p>

      <h3>Create the agent skill file</h3>
      <p>
        Save the following as <code>.claude/agents/howler-evolver.md</code> in your project root:
      </p>
      <pre><code>{`---
name: howler-evolver
description: >
  Runs evolutionary improvement loops over agent strategies using the
  Howler Agents GEA system. Call this agent when you want to evolve better
  coding agents, submit code review feedback as experience traces, or
  retrieve the best-performing agent strategies for a given task domain.
tools:
  - howler_evolve
  - howler_step
  - howler_status
  - howler_list_agents
  - howler_best_agents
  - howler_submit_experience
  - howler_get_traces
---

# Howler Evolver Agent

You are a specialist agent that uses the Howler Agents GEA (Group-Evolving
Agents) system to continuously improve coding agent strategies through
evolutionary selection and shared experience.

## Workflow

### 1. Start an evolution run

When asked to evolve agents for a task domain, call howler_evolve with
appropriate parameters. Good defaults for code-review tasks:

  population_size: 10   # Total agents in the population
  group_size: 3         # Agents per experience-sharing group
  num_iterations: 5     # Generations to run
  alpha: 0.5            # Balance between performance and novelty

### 2. Submit code review experiences

After each code review cycle in the broader workflow, submit the outcome
as an experience trace using howler_submit_experience. Include:

  - task: A short description of what was reviewed
  - result: The concrete outcome (tests passed, bugs found, etc.)
  - reflection: What the reviewing agent learned
  - score: A 0–1 float representing review quality

### 3. Retrieve evolved strategies

After the run completes, call howler_best_agents to obtain the top-ranked
agent configurations. Extract their system prompt snippets and apply them
to the coding agents in the current workflow.

### 4. Interpret results

When reporting results to the user:
  - State the final best_score and how many generations were run
  - Summarise the top-3 agents and what makes each strategy distinct
  - Suggest which strategy best fits the task domain at hand

## Example invocation sequence

1. howler_evolve(population_size=10, group_size=3, num_iterations=5, alpha=0.5)
   -> Returns run_id

2. howler_submit_experience(run_id=<id>, agent_id="agent-0",
     task="Review PR #42 for type safety",
     result="Found 3 missing None checks; all fixed",
     reflection="Prioritise checking return types of external calls",
     score=0.91)

3. howler_step(run_id=<id>)   # repeat until done
   howler_status(run_id=<id>) # check progress

4. howler_best_agents(run_id=<id>, top_k=3)
   -> Return strategies to caller`}</code></pre>

      <h3>How this integrates with a coding workflow</h3>
      <p>
        The evolver agent works as a background specialist. A typical end-to-end flow looks like
        this:
      </p>
      <ol>
        <li>
          A <strong>coordinator agent</strong> receives a task (e.g., "improve code review quality
          across the repository").
        </li>
        <li>
          The coordinator delegates to the <strong>howler-evolver</strong> agent to start a GEA run.
        </li>
        <li>
          As other agents in the workflow perform code reviews, they call{" "}
          <code>howler_submit_experience</code> to log outcomes into the shared experience pool.
        </li>
        <li>
          The evolver periodically calls <code>howler_step</code> to advance the evolutionary
          process, incorporating all submitted experience.
        </li>
        <li>
          When the run completes, the coordinator calls <code>howler_best_agents</code> to retrieve
          the winning strategies and applies them to future review cycles.
        </li>
      </ol>

      {/* ------------------------------------------------------------------ */}
      <h2>8. Docker Compose for Local Full Stack</h2>
      <p>
        For a persistent local environment that mirrors the hosted service, run the full stack with
        Docker Compose. This gives you a PostgreSQL database with <code>pgvector</code> for
        durable experience storage, plus the Howler service REST API.
      </p>

      <h3>docker-compose.yml</h3>
      <pre><code>{`services:
  howler-service:
    image: registry.digitalocean.com/orochi-registry/howler-service:latest
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql+asyncpg://howler:howler@postgres:5432/howler_agents
      HOWLER_LLM_ACTING_MODEL: claude-sonnet-4-20250514
      HOWLER_LLM_EVOLVING_MODEL: claude-sonnet-4-20250514
      HOWLER_LLM_REFLECTING_MODEL: claude-sonnet-4-20250514
      ANTHROPIC_API_KEY: \${ANTHROPIC_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: howler
      POSTGRES_PASSWORD: howler
      POSTGRES_DB: howler_agents
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U howler -d howler_agents"]
      interval: 5s
      timeout: 5s
      retries: 10
    restart: unless-stopped

volumes:
  postgres-data:`}</code></pre>

      <h3>Start the stack</h3>
      <pre><code>{`# Export your LLM provider key
export ANTHROPIC_API_KEY=your-key-here

# Pull images and start services
docker compose up -d

# Wait for the health check, then verify the API is up
curl -s http://localhost:8080/health | jq .`}</code></pre>

      <h3>Register a local account and get an API key</h3>
      <pre><code>{`# Register
curl -s -X POST http://localhost:8080/api/v1/auth/register \\
  -H "Content-Type: application/json" \\
  -d '{"email":"dev@local","password":"dev"}' | jq .

# Log in and capture JWT
JWT=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email":"dev@local","password":"dev"}' | jq -r .access_token)

# Create API key
curl -s -X POST http://localhost:8080/api/v1/auth/api-keys \\
  -H "Authorization: Bearer $JWT" | jq .`}</code></pre>

      <h3>Point the MCP server at your local instance</h3>
      <pre><code>{`# Claude Code
claude mcp add howler-agents -- howler-agents serve --api-url http://localhost:8080

# Or for Cursor / other tools, update mcp.json
{
  "mcpServers": {
    "howler-agents": {
      "command": "howler-agents",
      "args": ["serve"],
      "env": {
        "HOWLER_API_URL": "http://localhost:8080",
        "HOWLER_API_KEY": "your-local-api-key"
      }
    }
  }
}`}</code></pre>

      <h3>Stop and clean up</h3>
      <pre><code>{`docker compose down          # stop containers, keep volumes
docker compose down -v       # stop containers and delete all data`}</code></pre>

      {/* ------------------------------------------------------------------ */}
      <h2>Authentication Reference</h2>
      <p>All API requests require one of the following authentication methods:</p>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Header</th>
            <th>When to use</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>JWT Bearer token</td>
            <td><code>Authorization: Bearer &lt;token&gt;</code></td>
            <td>Short-lived sessions; obtained from <code>/auth/login</code></td>
          </tr>
          <tr>
            <td>API Key</td>
            <td><code>X-API-Key: &lt;key&gt;</code></td>
            <td>Long-lived integrations; obtained from <code>/auth/api-keys</code></td>
          </tr>
        </tbody>
      </table>

      <blockquote>
        <p>
          <strong>Note:</strong> API keys do not expire by default. Rotate them from the{" "}
          <code>/api/v1/auth/api-keys</code> endpoint if a key is compromised. Store keys in
          environment variables or a secrets manager; never commit them to source control.
        </p>
      </blockquote>

      <h2>Troubleshooting</h2>
      <table>
        <thead>
          <tr>
            <th>Symptom</th>
            <th>Likely cause</th>
            <th>Resolution</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>MCP server not found by editor</td>
            <td><code>howler-agents</code> not on PATH</td>
            <td>
              Run <code>which howler-agents</code> to confirm installation. Use the full binary path
              in the MCP config if needed.
            </td>
          </tr>
          <tr>
            <td><code>401 Unauthorized</code> from REST API</td>
            <td>Missing or expired credentials</td>
            <td>
              Check that <code>HOWLER_API_KEY</code> is set and that the key was created against
              the correct server URL.
            </td>
          </tr>
          <tr>
            <td>Tools not appearing in editor</td>
            <td>MCP config syntax error or wrong key name</td>
            <td>
              Validate the JSON config file with <code>jq . mcp.json</code>. Ensure the server
              name matches what the editor expects.
            </td>
          </tr>
          <tr>
            <td>Docker Compose service not starting</td>
            <td>Postgres health check failing</td>
            <td>
              Run <code>docker compose logs postgres</code> to inspect errors. Ensure port 5432 is
              not already in use on the host.
            </td>
          </tr>
          <tr>
            <td>Local mode loses data on restart</td>
            <td>In-memory store is ephemeral by design</td>
            <td>
              Switch to the full Docker Compose stack if persistence is required.
            </td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
