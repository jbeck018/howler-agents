import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/api-reference",
  component: ApiReferencePage,
});

function ApiReferencePage() {
  return (
    <>
      <h1>API Reference</h1>

      <h2>REST Endpoints</h2>
      <table>
        <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td><code>POST</code></td><td><code>/api/v1/runs</code></td><td>Create a new evolution run</td></tr>
          <tr><td><code>GET</code></td><td><code>/api/v1/runs/&#123;id&#125;</code></td><td>Get run details</td></tr>
          <tr><td><code>GET</code></td><td><code>/api/v1/runs</code></td><td>List all runs</td></tr>
          <tr><td><code>POST</code></td><td><code>/api/v1/runs/&#123;id&#125;/step</code></td><td>Step one generation</td></tr>
          <tr><td><code>GET</code></td><td><code>/api/v1/runs/&#123;id&#125;/agents</code></td><td>List agents in a run</td></tr>
          <tr><td><code>GET</code></td><td><code>/api/v1/runs/&#123;id&#125;/agents/best</code></td><td>Get top-K agents</td></tr>
          <tr><td><code>POST</code></td><td><code>/api/v1/runs/&#123;id&#125;/experience</code></td><td>Submit experience trace</td></tr>
          <tr><td><code>POST</code></td><td><code>/api/v1/runs/&#123;id&#125;/probes</code></td><td>Submit probe results</td></tr>
          <tr><td><code>GET</code></td><td><code>/health</code></td><td>Health check</td></tr>
        </tbody>
      </table>

      <h2>gRPC Service</h2>
      <p>The gRPC service is defined in <code>proto/howler_agents/v1/service.proto</code> and exposes the same operations with streaming support:</p>
      <pre><code>{`service HowlerAgentsService {
  rpc CreateRun(CreateRunRequest) returns (CreateRunResponse);
  rpc GetRun(GetRunRequest) returns (GetRunResponse);
  rpc ListRuns(ListRunsRequest) returns (ListRunsResponse);
  rpc StepEvolution(StepEvolutionRequest) returns (StepEvolutionResponse);
  rpc RunEvolution(RunEvolutionRequest) returns (RunEvolutionResponse);
  rpc StreamEvolution(StreamEvolutionRequest) returns (stream EvolutionEvent);
  rpc GetAgentGroup(GetAgentGroupRequest) returns (GetAgentGroupResponse);
  rpc GetBestAgents(GetBestAgentsRequest) returns (GetBestAgentsResponse);
  rpc SubmitExperience(SubmitExperienceRequest) returns (SubmitExperienceResponse);
  rpc SubmitProbeResults(SubmitProbeResultsRequest) returns (SubmitProbeResultsResponse);
}`}</code></pre>
    </>
  );
}
