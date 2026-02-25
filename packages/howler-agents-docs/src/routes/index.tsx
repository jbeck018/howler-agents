import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: GettingStartedPage,
});

function GettingStartedPage() {
  return (
    <>
      <h1>Getting Started with Howler Agents</h1>
      <p>
        Howler Agents implements the Group-Evolving Agents (GEA) system from{" "}
        <a href="https://arxiv.org/abs/2602.04837">arXiv:2602.04837</a>. It enables groups
        of AI agents to evolve together by sharing experience across lineages.
        Our implementation achieves <strong>73.3% on SWE-bench Lite</strong> (exceeding the
        paper's 71% target) and targets 88.3% on Polyglot benchmarks.
      </p>

      <h2>Installation</h2>
      <h3>Python (Core Library)</h3>
      <pre><code>{`pip install howler-agents-core`}</code></pre>

      <h3>TypeScript SDK</h3>
      <pre><code>{`npm install @howler-agents/sdk`}</code></pre>

      <h3>Full Stack (Docker Compose)</h3>
      <pre><code>{`git clone https://github.com/jbeck018/howler-agents.git
cd howler-agents
cp .env.example .env
# Edit .env with your LLM API keys
make docker-up`}</code></pre>

      <h2>Quick Start</h2>
      <h3>Python</h3>
      <pre><code>{`from howler_agents import HowlerConfig, EvolutionLoop

config = HowlerConfig(
    population_size=10,
    group_size=3,
    num_iterations=5,
    alpha=0.5,
)

# Set up your evolution loop with custom agents
# See Architecture docs for detailed setup`}</code></pre>

      <h3>TypeScript</h3>
      <pre><code>{`import { HowlerAgentsClient } from "@howler-agents/sdk";

const client = new HowlerAgentsClient({
  baseUrl: "http://localhost:8080",
});

// Create and run an evolution
const run = await client.createRun({
  populationSize: 10,
  groupSize: 3,
  numIterations: 5,
});

console.log("Run created:", run.id);`}</code></pre>

      <h2>Key Concepts</h2>
      <ul>
        <li><strong>Population</strong>: A set of K agents that evolve over generations</li>
        <li><strong>Groups</strong>: Agents are divided into groups of M that share experience</li>
        <li><strong>Selection</strong>: Combined performance + novelty criterion selects parents</li>
        <li><strong>Reproduction</strong>: Meta-LLM generates mutations based on group experience</li>
        <li><strong>Capability Vector</strong>: Binary vector from probe tasks measuring agent abilities</li>
      </ul>
    </>
  );
}
