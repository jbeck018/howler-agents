import * as React from "react";
import { createRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { RunsTable } from "../components/runs-table";
import { Plus } from "lucide-react";
import { useRunsQuery, useCreateRunMutation } from "../lib/api";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs",
  component: RunsPage,
});

function RunsPage() {
  const { data, isLoading, error } = useRunsQuery({ limit: 100 });
  const createRun = useCreateRunMutation();
  const navigate = useNavigate();

  const handleCreate = async () => {
    const run = await createRun.mutateAsync({
      population_size: 10,
      group_size: 3,
      num_iterations: 5,
      alpha: 0.5,
      num_probes: 20,
      task_domain: "general",
    });
    navigate({ to: "/runs/$runId", params: { runId: run.id } });
  };

  const runs = (data?.runs ?? []).map((r) => ({
    id: r.id,
    status: r.status,
    currentGeneration: r.current_generation,
    totalGenerations: r.total_generations,
    bestScore: r.best_score,
    createdAt: r.created_at ?? undefined,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Evolution Runs</h2>
        <Button onClick={handleCreate} disabled={createRun.isPending}>
          <Plus className="mr-2 h-4 w-4" />
          {createRun.isPending ? "Creating..." : "New Run"}
        </Button>
      </div>
      <Card>
        <CardContent className="pt-6">
          {error && (
            <p className="text-sm text-destructive mb-4">
              Cannot connect to service. Ensure backend is running.
            </p>
          )}
          {isLoading ? (
            <p className="text-sm text-muted-foreground py-8 text-center">Loading runs...</p>
          ) : (
            <RunsTable runs={runs} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
