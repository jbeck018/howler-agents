import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/sdk-guides",
  component: SdkGuidesPage,
});

function SdkGuidesPage() {
  return (
    <>
      <h1>SDK Guides</h1>

      <h2>TypeScript SDK</h2>
      <pre><code>{`import { HowlerAgentsClient } from "@howler-agents/sdk";

const client = new HowlerAgentsClient({
  baseUrl: "http://localhost:8080",
});

// Create a run
const run = await client.createRun({
  populationSize: 10,
  groupSize: 3,
  numIterations: 5,
  alpha: 0.5,
  taskDomain: "swe-bench",
});

// Step through generations
for (let i = 0; i < run.totalGenerations; i++) {
  const updated = await client.stepEvolution(run.id);
  console.log(\`Generation \${updated.currentGeneration}: best=\${updated.bestScore}\`);
}

// Get the best agents
const best = await client.getBestAgents(run.id, 3);
console.log("Top agents:", best);`}</code></pre>

      <h2>Python SDK</h2>
      <pre><code>{`from howler_agents import HowlerConfig, EvolutionLoop
from howler_agents.agents.pool import AgentPool
from howler_agents.experience.pool import SharedExperiencePool
from howler_agents.experience.store.memory import InMemoryStore
from howler_agents.selection.criterion import PerformanceNoveltySelector
from howler_agents.evolution.reproducer import GroupReproducer
from howler_agents.probes.evaluator import ProbeEvaluator
from howler_agents.probes.registry import ProbeRegistry
from howler_agents.llm.router import LLMRouter

config = HowlerConfig(population_size=10, group_size=3, num_iterations=5)
store = InMemoryStore()
experience = SharedExperiencePool(store)
llm = LLMRouter(config)
selector = PerformanceNoveltySelector(alpha=config.alpha)
reproducer = GroupReproducer(llm, experience, config)
registry = ProbeRegistry()
registry.register_default_probes()
probes = ProbeEvaluator(registry)

pool = AgentPool()
# Add your custom agents to the pool...

loop = EvolutionLoop(config, pool, selector, reproducer, experience, probes)
results = await loop.run("my-run", tasks=[{"description": "..."}])`}</code></pre>
    </>
  );
}
