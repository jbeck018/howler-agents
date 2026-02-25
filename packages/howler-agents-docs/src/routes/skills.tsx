import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/skills",
  component: SkillsPage,
});

function SkillsPage() {
  return (
    <>
      <h1>Skills (Slash Commands)</h1>
      <p>
        Howler Agents includes several Claude Code skills that combine the GEA
        evolutionary system with the hive-mind collective intelligence. These
        are invoked as slash commands in Claude Code.
      </p>

      <h2>Setup</h2>
      <pre><code>{`/howler-setup`}</code></pre>
      <p>
        Initializes the local environment: installs the package, registers the MCP server,
        creates the <code>.howler-agents/</code> directory, and verifies connectivity.
        Run this first before using any other howler skill.
      </p>

      <h2>Core Skills</h2>

      <h3>/howler-agents</h3>
      <p>
        <strong>Best-first-pass solution.</strong> Combines hive-mind collective intelligence
        with GEA evolution to produce the best possible solution on any task in a single pass.
      </p>
      <p>Protocol:</p>
      <ol>
        <li>Queries hive-mind memory for relevant lessons and patterns from past runs</li>
        <li>Runs GEA evolution with a configurable depth (quick / standard / deep)</li>
        <li>Extracts the winning agent's strategy and experience traces</li>
        <li>Synthesizes the final solution from collective intelligence</li>
        <li>Stores new lessons back to hive-mind for future runs</li>
      </ol>
      <table>
        <thead><tr><th>Depth</th><th>Agents</th><th>Generations</th><th>Use when</th></tr></thead>
        <tbody>
          <tr><td>quick</td><td>3</td><td>2</td><td>Fast iteration, small fixes</td></tr>
          <tr><td>standard</td><td>6</td><td>3</td><td>Default for most tasks</td></tr>
          <tr><td>deep</td><td>10</td><td>5</td><td>Complex multi-file changes</td></tr>
        </tbody>
      </table>
      <pre><code>{`# Example
/howler-agents Fix the authentication token refresh logic in auth.ts`}</code></pre>

      <h3>/howler-agents-wiggam</h3>
      <p>
        <strong>Iterative refinement loop.</strong> Combines the Ralph Wiggum iterative loop
        technique with howler-agents hive-mind + GEA evolution. Each iteration runs a full
        howler-agents pass, sees previous work in files, and uses collective intelligence
        to improve until a completion promise is met.
      </p>
      <p>
        Each iteration gets smarter because it reads previous iterations' lessons from hive-mind
        memory, plus sees all file changes from prior iterations in the workspace.
      </p>
      <pre><code>{`# Example
/howler-agents-wiggam Fix all failing tests in the auth module \\
  --completion-promise "ALL TESTS PASSING" \\
  --max-iterations 15`}</code></pre>
      <table>
        <thead><tr><th>Option</th><th>Default</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td><code>--completion-promise</code></td><td>(required)</td><td>Statement that must be TRUE to exit</td></tr>
          <tr><td><code>--max-iterations</code></td><td>10</td><td>Maximum loop iterations</td></tr>
          <tr><td><code>--depth</code></td><td>quick</td><td>Per-iteration evolution depth</td></tr>
          <tr><td><code>--domain</code></td><td>coding</td><td>Task domain</td></tr>
        </tbody>
      </table>

      <h3>/howler-init</h3>
      <p>
        <strong>Repository intelligence seeding.</strong> Analyzes the current repository and
        seeds the hive-mind with structured knowledge about architecture, patterns, dependencies,
        test coverage, and conventions. Run this once when setting up howler-agents in a new repo
        to give the evolutionary system full context.
      </p>
      <p>Generates intelligence entries for:</p>
      <ul>
        <li>Repository structure and package layout</li>
        <li>Architecture patterns and module boundaries</li>
        <li>Coding conventions (naming, imports, error handling)</li>
        <li>Test infrastructure and coverage patterns</li>
        <li>Build system and CI configuration</li>
      </ul>
      <pre><code>{`# Full repo analysis
/howler-init

# Focus on a specific area
/howler-init --focus auth --depth deep`}</code></pre>

      <h2>Evolution & Monitoring Skills</h2>

      <h3>/howler-evolve</h3>
      <p>
        Start a full GEA evolution run with team coordination. Spawns a team of specialized
        sub-agents (coordinator, evaluator, reproducer) that collaborate through Claude Code's
        team system.
      </p>
      <pre><code>{`/howler-evolve --domain coding --population 10 --iterations 5`}</code></pre>

      <h3>/howler-auto-evolve</h3>
      <p>
        One-shot evolution run with automatic deployment of the best evolved agents.
        Runs the full evolution loop and deploys winners without manual intervention.
      </p>

      <h3>/howler-status</h3>
      <p>
        Check the status of a running or completed evolution. Shows generation progress,
        population scores, and top agent rankings.
      </p>
      <pre><code>{`/howler-status`}</code></pre>

      <h3>/howler-memory</h3>
      <p>
        Browse, search, and manage the hive-mind collective memory. Supports listing entries
        by namespace, searching by query, and viewing detailed memory entries.
      </p>
      <pre><code>{`/howler-memory search "authentication patterns"
/howler-memory list --namespace lessons`}</code></pre>

      <h3>/howler-sync</h3>
      <p>
        Synchronize evolution data between local SQLite storage and the remote Postgres
        database. Push local results to share with the team, or pull remote data to
        benefit from others' evolution runs.
      </p>
      <pre><code>{`/howler-sync push
/howler-sync pull`}</code></pre>

      <h2>How Skills Work Together</h2>
      <p>
        The skills form a progressive workflow:
      </p>
      <ol>
        <li><code>/howler-setup</code> -- Install and configure (once per project)</li>
        <li><code>/howler-init</code> -- Seed repo intelligence (once per project)</li>
        <li><code>/howler-agents</code> -- Solve tasks with collective intelligence (daily use)</li>
        <li><code>/howler-agents-wiggam</code> -- Iterate on complex tasks until done (as needed)</li>
        <li><code>/howler-memory</code> -- Browse accumulated knowledge (as needed)</li>
        <li><code>/howler-sync</code> -- Share with team (periodically)</li>
      </ol>
      <p>
        Each skill builds on the hive-mind memory. Over time, the collective intelligence
        grows as more tasks are solved, making every subsequent invocation more effective.
      </p>
    </>
  );
}
