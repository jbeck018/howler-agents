/**
 * Core types for Howler Agents SDK.
 * These mirror the proto definitions and will eventually be replaced
 * by generated types from buf generate.
 */

export interface RunConfig {
  populationSize: number;
  groupSize: number;
  numIterations: number;
  alpha: number;
  numProbes: number;
  llmConfig: Record<string, string>;
  taskDomain: string;
  taskConfig: Record<string, unknown>;
}

export interface Agent {
  id: string;
  runId: string;
  generation: number;
  parentId?: string;
  groupId?: string;
  performanceScore: number;
  noveltyScore: number;
  combinedScore: number;
  capabilityVector: number[];
  patches: FrameworkPatch[];
  createdAt?: string;
}

export interface AgentGroup {
  id: string;
  runId: string;
  generation: number;
  agents: Agent[];
  groupPerformance: number;
}

export interface FrameworkPatch {
  id: string;
  agentId: string;
  generation: number;
  intent: string;
  diff: string;
  category: string;
  appliedAt?: string;
}

export interface EvolutionaryTrace {
  id: string;
  agentId: string;
  runId: string;
  generation: number;
  taskDescription: string;
  outcome: string;
  score: number;
  keyDecisions: string[];
  lessonsLearned: string[];
  recordedAt?: string;
}

export interface EvolutionRun {
  id: string;
  config: RunConfig;
  status: "pending" | "running" | "completed" | "failed";
  currentGeneration: number;
  totalGenerations: number;
  groups: AgentGroup[];
  bestAgent?: Agent;
  bestScore: number;
  createdAt?: string;
  updatedAt?: string;
}

export type EvolutionEventType =
  | "generation_started"
  | "agent_evaluated"
  | "selection_completed"
  | "reproduction_completed"
  | "generation_completed"
  | "run_completed"
  | "run_failed";

export interface EvolutionEvent {
  runId: string;
  type: EvolutionEventType;
  data: Record<string, unknown>;
  timestamp: string;
}
