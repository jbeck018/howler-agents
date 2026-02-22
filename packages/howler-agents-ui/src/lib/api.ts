/**
 * API client and TanStack Query hooks for Howler Agents.
 *
 * Types are imported from the SDK which mirrors the proto definitions.
 * When Orval/ConnectRPC generation is set up, these hooks will be replaced
 * by generated ones.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { refreshAccessToken } from "./auth";

// Base URL for the API - configurable via environment
const API_BASE = (typeof window !== "undefined" && (window as any).__HOWLER_API_URL__) || "http://localhost:8080";

// --- Types matching the backend Pydantic schemas ---

export interface RunConfig {
  population_size: number;
  group_size: number;
  num_iterations: number;
  alpha: number;
  num_probes: number;
  llm_config: Record<string, string>;
  task_domain: string;
  task_config: Record<string, unknown>;
}

export interface EvolutionRun {
  id: string;
  config: RunConfig;
  status: "pending" | "running" | "completed" | "failed";
  current_generation: number;
  total_generations: number;
  best_score: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface RunListResponse {
  runs: EvolutionRun[];
  total: number;
}

export interface Agent {
  id: string;
  run_id: string;
  generation: number;
  parent_id: string | null;
  group_id: string | null;
  performance_score: number;
  novelty_score: number;
  combined_score: number;
  capability_vector: number[];
  created_at: string | null;
}

export interface Trace {
  id: string;
  agent_id: string;
  run_id: string;
  generation: number;
  task_description: string;
  outcome: string;
  score: number;
  key_decisions: string[];
  lessons_learned: string[];
  recorded_at: string | null;
}

export interface TraceSubmit {
  agent_id: string;
  task_description: string;
  outcome: string;
  score: number;
  key_decisions: string[];
  lessons_learned: string[];
}

// --- Fetch helper ---

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const token = localStorage.getItem("howler_access_token");
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
  });
  if (res.status === 401) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      return apiFetch(path, init);
    }
    localStorage.removeItem("howler_access_token");
    localStorage.removeItem("howler_refresh_token");
    window.location.href = "/login";
    throw new Error("Session expired");
  }
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// --- Query Keys ---

export const queryKeys = {
  runs: ["runs"] as const,
  run: (id: string) => ["runs", id] as const,
  agents: (runId: string) => ["runs", runId, "agents"] as const,
  bestAgents: (runId: string) => ["runs", runId, "agents", "best"] as const,
  traces: (runId: string) => ["runs", runId, "traces"] as const,
  health: ["health"] as const,
  stats: ["stats"] as const,
};

// --- Query Hooks ---

export function useHealthQuery() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiFetch<{ status: string }>("/health"),
    retry: 1,
    staleTime: 10_000,
  });
}

export function useRunsQuery(params?: { limit?: number; offset?: number; status?: string }) {
  const searchParams = new URLSearchParams();
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  if (params?.status) searchParams.set("status", params.status);
  const query = searchParams.toString();

  return useQuery({
    queryKey: [...queryKeys.runs, params],
    queryFn: () => apiFetch<RunListResponse>(`/api/v1/runs${query ? `?${query}` : ""}`),
    staleTime: 5_000,
  });
}

export function useRunQuery(runId: string) {
  return useQuery({
    queryKey: queryKeys.run(runId),
    queryFn: () => apiFetch<EvolutionRun>(`/api/v1/runs/${runId}`),
    enabled: !!runId,
    staleTime: 5_000,
  });
}

export function useAgentsQuery(runId: string) {
  return useQuery({
    queryKey: queryKeys.agents(runId),
    queryFn: () => apiFetch<Agent[]>(`/api/v1/runs/${runId}/agents`),
    enabled: !!runId,
    staleTime: 5_000,
  });
}

export function useBestAgentsQuery(runId: string, topK: number = 5) {
  return useQuery({
    queryKey: queryKeys.bestAgents(runId),
    queryFn: () => apiFetch<Agent[]>(`/api/v1/runs/${runId}/agents/best?top_k=${topK}`),
    enabled: !!runId,
    staleTime: 5_000,
  });
}

export function useTracesQuery(runId: string) {
  return useQuery({
    queryKey: queryKeys.traces(runId),
    queryFn: () => apiFetch<Trace[]>(`/api/v1/runs/${runId}/traces`),
    enabled: !!runId,
    staleTime: 5_000,
  });
}

// --- Mutation Hooks ---

export function useCreateRunMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (config: Partial<RunConfig>) =>
      apiFetch<EvolutionRun>("/api/v1/runs", {
        method: "POST",
        body: JSON.stringify({ config }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.runs });
    },
  });
}

export function useStepEvolutionMutation(runId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<EvolutionRun>(`/api/v1/runs/${runId}/step`, {
        method: "POST",
        body: "{}",
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.run(runId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.runs });
    },
  });
}

export function useSubmitExperienceMutation(runId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (trace: TraceSubmit) =>
      apiFetch<{ accepted: boolean }>(`/api/v1/runs/${runId}/experience`, {
        method: "POST",
        body: JSON.stringify(trace),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.run(runId) });
    },
  });
}

// --- Auth Mutation Hooks ---

export interface LoginPayload {
  email: string;
  password: string;
}

export interface RegisterPayload {
  email: string;
  password: string;
  orgName: string;
}

export function useLoginMutation() {
  return useMutation({
    mutationFn: ({ email, password }: LoginPayload) =>
      apiFetch<{ access_token: string; refresh_token?: string }>("/api/v1/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      }),
  });
}

export function useRegisterMutation() {
  return useMutation({
    mutationFn: ({ email, password, orgName }: RegisterPayload) =>
      apiFetch<{ access_token: string; refresh_token?: string }>("/api/v1/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password, org_name: orgName }),
      }),
  });
}
