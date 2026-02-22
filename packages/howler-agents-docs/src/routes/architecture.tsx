import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/architecture",
  component: ArchitecturePage,
});

function ArchitecturePage() {
  return (
    <>
      <h1>Architecture</h1>
      <p>Howler Agents follows a modular architecture with clear separation of concerns.</p>

      <h2>Core Modules</h2>
      <table>
        <thead><tr><th>Module</th><th>Responsibility</th></tr></thead>
        <tbody>
          <tr><td><code>agents/</code></td><td>Agent base class, pool management, framework patches</td></tr>
          <tr><td><code>selection/</code></td><td>Performance + novelty scoring and selection</td></tr>
          <tr><td><code>experience/</code></td><td>Shared experience pool with pluggable stores</td></tr>
          <tr><td><code>evolution/</code></td><td>Main loop, reproducer, evolution directives</td></tr>
          <tr><td><code>probes/</code></td><td>Capability vector evaluation</td></tr>
          <tr><td><code>llm/</code></td><td>LiteLLM-backed role-based model routing</td></tr>
        </tbody>
      </table>

      <h2>Evolution Loop</h2>
      <p>Each generation follows this cycle:</p>
      <ol>
        <li><strong>Evaluate</strong>: All agents run tasks and probes</li>
        <li><strong>Score</strong>: Combined performance + novelty criterion</li>
        <li><strong>Select</strong>: Top agents survive as parents</li>
        <li><strong>Aggregate</strong>: Group experience traces compiled</li>
        <li><strong>Reproduce</strong>: Meta-LLM generates mutations for children</li>
      </ol>

      <h2>Experience Stores</h2>
      <p>The experience store is pluggable via the <code>ExperienceStore</code> protocol:</p>
      <ul>
        <li><code>InMemoryStore</code> - For testing and development</li>
        <li><code>PostgresStore</code> - Durable production backend with pgvector</li>
        <li><code>RedisStore</code> - Hot cache for frequently accessed traces</li>
      </ul>

      <h2>LLM Routing</h2>
      <p>
        LiteLLM provides a unified interface to 100+ LLM providers. Each evolutionary
        role (Acting, Evolving, Reflecting) can be mapped to a different model:
      </p>
      <pre><code>{`HOWLER_LLM_ACTING_MODEL=claude-sonnet-4-20250514
HOWLER_LLM_EVOLVING_MODEL=gpt-4o
HOWLER_LLM_REFLECTING_MODEL=ollama/llama3`}</code></pre>
    </>
  );
}
