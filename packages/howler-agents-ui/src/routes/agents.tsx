import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Card, CardContent } from "../components/ui/card";
import { CapabilityVector } from "../components/capability-vector";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { formatScore } from "@/lib/utils";
import { useRunsQuery, useAgentsQuery, type Agent } from "../lib/api";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/agents",
  component: AgentsPage,
});

function AgentsPage() {
  const { data: runsData, isLoading, error } = useRunsQuery({ limit: 10 });
  const latestRun = runsData?.runs?.find((r) => r.status === "completed") ?? runsData?.runs?.[0];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Agents</h2>
      {error && (
        <p className="text-sm text-destructive">Cannot connect to service.</p>
      )}
      {isLoading && (
        <p className="text-sm text-muted-foreground">Loading...</p>
      )}
      {latestRun ? (
        <AgentsList runId={latestRun.id} />
      ) : (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No runs found. Create an evolution run to see agents.
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function AgentsList({ runId }: { runId: string }) {
  const { data: agents, isLoading } = useAgentsQuery(runId);

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading agents...</p>;

  return (
    <Card>
      <CardContent className="pt-6">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Agent</TableHead>
              <TableHead>Gen</TableHead>
              <TableHead>Performance</TableHead>
              <TableHead>Novelty</TableHead>
              <TableHead>Combined</TableHead>
              <TableHead>Capabilities</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(!agents || agents.length === 0) && (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                  No agents in this run yet. Step the evolution to create agents.
                </TableCell>
              </TableRow>
            )}
            {agents?.map((agent: Agent) => (
              <TableRow key={agent.id}>
                <TableCell className="font-mono text-xs text-primary">
                  {agent.id.slice(0, 8)}...
                </TableCell>
                <TableCell>{agent.generation}</TableCell>
                <TableCell>{formatScore(agent.performance_score)}</TableCell>
                <TableCell>{formatScore(agent.novelty_score)}</TableCell>
                <TableCell className="font-bold">{formatScore(agent.combined_score)}</TableCell>
                <TableCell>
                  <CapabilityVector vector={agent.capability_vector || []} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
