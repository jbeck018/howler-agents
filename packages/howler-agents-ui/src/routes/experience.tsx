import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { formatScore, formatDate } from "@/lib/utils";
import { useRunsQuery, useTracesQuery, type Trace } from "../lib/api";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/experience",
  component: ExperiencePage,
});

function ExperiencePage() {
  const { data: runsData, isLoading, error } = useRunsQuery({ limit: 10 });
  const latestRun = runsData?.runs?.find((r) => r.status === "completed") ?? runsData?.runs?.[0];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Experience Explorer</h2>
      {error && (
        <p className="text-sm text-destructive">Cannot connect to service.</p>
      )}
      {isLoading && (
        <p className="text-sm text-muted-foreground">Loading...</p>
      )}
      {latestRun ? (
        <TracesList runId={latestRun.id} />
      ) : (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No runs found. Create an evolution run to explore experience traces.
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function TracesList({ runId }: { runId: string }) {
  const { data: traces, isLoading } = useTracesQuery(runId);

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading traces...</p>;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Evolutionary Traces</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Agent</TableHead>
              <TableHead>Gen</TableHead>
              <TableHead>Task</TableHead>
              <TableHead>Outcome</TableHead>
              <TableHead>Score</TableHead>
              <TableHead>Recorded</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(!traces || traces.length === 0) && (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                  No traces recorded yet. Submit experience to see data here.
                </TableCell>
              </TableRow>
            )}
            {traces?.map((trace: Trace) => (
              <TableRow key={trace.id}>
                <TableCell className="font-mono text-xs text-primary">
                  {trace.agent_id.slice(0, 8)}...
                </TableCell>
                <TableCell>{trace.generation}</TableCell>
                <TableCell className="max-w-[200px] truncate text-sm">
                  {trace.task_description}
                </TableCell>
                <TableCell className="max-w-[200px] truncate text-sm">
                  {trace.outcome}
                </TableCell>
                <TableCell className="font-bold">{formatScore(trace.score)}</TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {formatDate(trace.recorded_at ?? undefined)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
