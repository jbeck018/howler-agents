import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { PerformanceChart } from "../components/performance-chart";
import { EventStream } from "../components/event-stream";
import { useRunQuery, useAgentsQuery, useStepEvolutionMutation } from "../lib/api";
import { Play } from "lucide-react";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: RunDetailPage,
});

const statusVariant: Record<string, "default" | "secondary" | "success" | "warning" | "destructive"> = {
  pending: "secondary",
  running: "warning",
  completed: "success",
  failed: "destructive",
};

function RunDetailPage() {
  const { runId } = Route.useParams();
  const { data: run, isLoading, error } = useRunQuery(runId);
  const { data: agents } = useAgentsQuery(runId);
  const stepMutation = useStepEvolutionMutation(runId);

  if (isLoading) return <p className="text-muted-foreground">Loading run...</p>;
  if (error || !run) return <p className="text-destructive">Run not found or service unavailable.</p>;

  const performanceData = Array.from({ length: run.current_generation }, (_, i) => ({
    generation: i,
    bestScore: run.best_score * ((i + 1) / run.current_generation),
    meanScore: run.best_score * ((i + 1) / run.current_generation) * 0.75,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <h2 className="text-2xl font-bold">Run {run.id.slice(0, 8)}...</h2>
        <Badge variant={statusVariant[run.status] ?? "default"}>{run.status}</Badge>
        <span className="text-sm text-muted-foreground">
          Generation {run.current_generation} / {run.total_generations}
        </span>
        {run.status !== "completed" && (
          <Button
            size="sm"
            onClick={() => stepMutation.mutate()}
            disabled={stepMutation.isPending}
          >
            <Play className="mr-1 h-3 w-3" />
            {stepMutation.isPending ? "Stepping..." : "Step"}
          </Button>
        )}
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <PerformanceChart data={performanceData} />
        <EventStream events={[]} />
      </div>
      {agents && agents.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Agents ({agents.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs text-muted-foreground">
              {JSON.stringify(agents.slice(0, 5), null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Run Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="text-xs text-muted-foreground">
            {JSON.stringify(run.config, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
