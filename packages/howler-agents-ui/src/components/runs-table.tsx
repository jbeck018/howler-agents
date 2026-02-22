import * as React from "react";
import { Link } from "@tanstack/react-router";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { formatDate, formatScore } from "@/lib/utils";

interface Run {
  id: string;
  status: string;
  currentGeneration: number;
  totalGenerations: number;
  bestScore: number;
  createdAt?: string;
}

interface RunsTableProps {
  runs: Run[];
}

const statusVariant: Record<string, "default" | "secondary" | "success" | "warning" | "destructive"> = {
  pending: "secondary",
  running: "warning",
  completed: "success",
  failed: "destructive",
};

export function RunsTable({ runs }: RunsTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Progress</TableHead>
          <TableHead>Best Score</TableHead>
          <TableHead>Created</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow key={run.id}>
            <TableCell>
              <Link to="/runs/$runId" params={{ runId: run.id }} className="text-primary hover:underline font-mono text-xs">
                {run.id.slice(0, 8)}...
              </Link>
            </TableCell>
            <TableCell>
              <Badge variant={statusVariant[run.status] ?? "default"}>{run.status}</Badge>
            </TableCell>
            <TableCell className="text-muted-foreground">
              {run.currentGeneration} / {run.totalGenerations}
            </TableCell>
            <TableCell>{formatScore(run.bestScore)}</TableCell>
            <TableCell className="text-muted-foreground text-xs">{formatDate(run.createdAt)}</TableCell>
          </TableRow>
        ))}
        {runs.length === 0 && (
          <TableRow>
            <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
              No evolution runs yet. Create one to get started.
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  );
}
