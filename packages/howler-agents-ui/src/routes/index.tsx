import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Activity, Bot, Dna, TrendingUp } from "lucide-react";
import { StatsCard } from "../components/stats-card";
import { RunsTable } from "../components/runs-table";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { useRunsQuery, useHealthQuery } from "../lib/api";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: DashboardPage,
});

function DashboardPage() {
  const { data: health } = useHealthQuery();
  const { data: runsData, isLoading, error } = useRunsQuery({ limit: 50 });

  const runs = runsData?.runs ?? [];
  const totalRuns = runsData?.total ?? 0;
  const activeRuns = runs.filter((r) => r.status === "running").length;
  const bestScore = runs.length > 0 ? Math.max(...runs.map((r) => r.best_score)) : 0;
  const recentRuns = runs.slice(0, 5);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        {health && (
          <span className="text-xs text-muted-foreground">
            Service: {health.status}
          </span>
        )}
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatsCard title="Total Runs" value={totalRuns} icon={Dna} description="All time" />
        <StatsCard title="Active Runs" value={activeRuns} icon={Activity} description="Currently running" />
        <StatsCard title="Total Agents" value={"--"} icon={Bot} description="Connect to service" />
        <StatsCard
          title="Best Score"
          value={bestScore > 0 ? `${(bestScore * 100).toFixed(1)}%` : "--"}
          icon={TrendingUp}
          description="Across all runs"
        />
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Recent Runs</CardTitle>
        </CardHeader>
        <CardContent>
          {error && (
            <p className="text-sm text-destructive mb-4">
              Failed to connect to service. Start the backend with `make dev-service`.
            </p>
          )}
          {isLoading ? (
            <p className="text-sm text-muted-foreground py-4">Loading...</p>
          ) : (
            <RunsTable
              runs={recentRuns.map((r) => ({
                id: r.id,
                status: r.status,
                currentGeneration: r.current_generation,
                totalGenerations: r.total_generations,
                bestScore: r.best_score,
                createdAt: r.created_at ?? undefined,
              }))}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
