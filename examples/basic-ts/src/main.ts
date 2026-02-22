/**
 * Basic example: Create and run an evolution with the Howler Agents SDK.
 */

import { HowlerAgentsClient } from "@howler-agents/sdk";

const SERVICE_URL = process.env.HOWLER_SERVICE_URL ?? "http://localhost:8080";

async function main() {
  const client = new HowlerAgentsClient({ baseUrl: SERVICE_URL });

  // Check service health
  const health = await client.health();
  console.log("Service status:", health.status);

  // Create an evolution run
  const run = await client.createRun({
    populationSize: 10,
    groupSize: 3,
    numIterations: 5,
    alpha: 0.5,
    taskDomain: "general",
  });
  console.log(`Created run: ${run.id} (status: ${run.status})`);

  // Step through each generation
  for (let gen = 0; gen < run.totalGenerations; gen++) {
    const updated = await client.stepEvolution(run.id);
    console.log(
      `Generation ${updated.currentGeneration}/${updated.totalGenerations} - ` +
      `Status: ${updated.status}`
    );
  }

  // Get the best agents
  const bestAgents = await client.getBestAgents(run.id, 3);
  console.log("\nTop 3 agents:");
  for (const agent of bestAgents) {
    console.log(`  ${agent.id}: combined=${agent.combinedScore.toFixed(3)}`);
  }

  // List all runs
  const { runs, total } = await client.listRuns();
  console.log(`\nTotal runs: ${total}`);
}

main().catch(console.error);
