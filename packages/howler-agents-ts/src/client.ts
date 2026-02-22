/**
 * Main Howler Agents client.
 */

import type {
  RunConfig,
  EvolutionRun,
  Agent,
  AgentGroup,
  EvolutionaryTrace,
  EvolutionEvent,
} from "./types.js";
import { HowlerError } from "./errors/base.js";
import { ConnectionError } from "./errors/connection.js";
import { NotFoundError } from "./errors/not-found.js";
import { AuthenticationError } from "./errors/authentication.js";

/**
 * Response returned from login, register, and token refresh endpoints.
 */
export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  user: {
    user_id: string;
    org_id: string;
    email: string;
    role: string;
  };
}

/**
 * Response returned when creating an API key.
 */
export interface ApiKeyResponse {
  id: string;
  key: string;
  name: string;
  created_at: string;
}

/**
 * User profile returned from the /auth/me endpoint.
 */
export interface UserResponse {
  user_id: string;
  org_id: string;
  email: string;
  role: string;
}

export interface HowlerClientOptions {
  /** Base URL of the Howler Agents service (REST) */
  baseUrl?: string;
  /** API key for authentication (ha_live_... format) */
  apiKey?: string;
  /** JWT access token for authentication */
  accessToken?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
}

export class HowlerAgentsClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;

  constructor(options: HowlerClientOptions) {
    this.baseUrl = (options.baseUrl ?? "").replace(/\/$/, "");
    this.timeout = options.timeout ?? 30_000;
    this.headers = {
      "Content-Type": "application/json",
    };
    if (options.accessToken) {
      this.headers["Authorization"] = `Bearer ${options.accessToken}`;
    } else if (options.apiKey) {
      this.headers["Authorization"] = `Bearer ${options.apiKey}`;
    }
  }

  // --- Auth ---

  /**
   * Authenticate with email and password, returning tokens and user info.
   */
  async login(email: string, password: string): Promise<AuthResponse> {
    return this.post<AuthResponse>("/api/v1/auth/login", { email, password });
  }

  /**
   * Register a new user and organisation, returning tokens and user info.
   */
  async register(
    email: string,
    password: string,
    orgName: string
  ): Promise<AuthResponse> {
    return this.post<AuthResponse>("/api/v1/auth/register", {
      email,
      password,
      org_name: orgName,
    });
  }

  /**
   * Exchange a refresh token for a new access/refresh token pair.
   */
  async refreshToken(refreshToken: string): Promise<AuthResponse> {
    return this.post<AuthResponse>("/api/v1/auth/refresh", {
      refresh_token: refreshToken,
    });
  }

  /**
   * Create a named API key. Requires an authenticated client (accessToken or apiKey).
   */
  async createApiKey(name: string): Promise<ApiKeyResponse> {
    return this.post<ApiKeyResponse>("/api/v1/auth/api-keys", { name });
  }

  /**
   * Fetch the currently authenticated user's profile.
   */
  async me(): Promise<UserResponse> {
    return this.get<UserResponse>("/api/v1/auth/me");
  }

  // --- Run Management ---

  async createRun(config: Partial<RunConfig>): Promise<EvolutionRun> {
    return this.post<EvolutionRun>("/api/v1/runs", { config });
  }

  async getRun(runId: string): Promise<EvolutionRun> {
    return this.get<EvolutionRun>(`/api/v1/runs/${runId}`);
  }

  async listRuns(options?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<{ runs: EvolutionRun[]; total: number }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", String(options.limit));
    if (options?.offset) params.set("offset", String(options.offset));
    if (options?.status) params.set("status", options.status);
    const query = params.toString();
    return this.get(`/api/v1/runs${query ? `?${query}` : ""}`);
  }

  // --- Evolution Control ---

  async stepEvolution(runId: string): Promise<EvolutionRun> {
    return this.post<EvolutionRun>(`/api/v1/runs/${runId}/step`, {});
  }

  // --- Agent Queries ---

  async listAgents(runId: string): Promise<Agent[]> {
    return this.get<Agent[]>(`/api/v1/runs/${runId}/agents`);
  }

  async getBestAgents(
    runId: string,
    topK: number = 5
  ): Promise<Agent[]> {
    return this.get<Agent[]>(
      `/api/v1/runs/${runId}/agents/best?top_k=${topK}`
    );
  }

  // --- Experience ---

  async submitExperience(
    runId: string,
    trace: Omit<EvolutionaryTrace, "id" | "runId" | "recordedAt">
  ): Promise<{ accepted: boolean }> {
    return this.post(`/api/v1/runs/${runId}/experience`, trace);
  }

  async submitProbeResults(
    runId: string,
    agentId: string,
    results: boolean[]
  ): Promise<{ capabilityVector: number[] }> {
    return this.post(`/api/v1/runs/${runId}/probes`, {
      agent_id: agentId,
      results,
    });
  }

  // --- Streaming ---

  async *streamEvolution(
    runId: string
  ): AsyncGenerator<EvolutionEvent, void, unknown> {
    const url = `${this.baseUrl}/api/v1/runs/${runId}/stream`;
    const response = await fetch(url, {
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout * 10),
    });

    if (!response.ok) {
      throw new ConnectionError(`Stream failed: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new ConnectionError("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data && data !== "[DONE]") {
              yield JSON.parse(data) as EvolutionEvent;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // --- Health ---

  async health(): Promise<{ status: string }> {
    return this.get<{ status: string }>("/health");
  }

  // --- Internal HTTP ---

  private async get<T>(path: string): Promise<T> {
    const response = await this.fetch(path, { method: "GET" });
    return response.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await this.fetch(path, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return response.json() as Promise<T>;
  }

  private async fetch(path: string, init: RequestInit): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    let response: Response;
    try {
      response = await fetch(url, {
        ...init,
        headers: { ...this.headers, ...init.headers },
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (error) {
      if (error instanceof TypeError) {
        throw new ConnectionError(`Failed to connect to ${this.baseUrl}`);
      }
      throw error;
    }

    if (!response.ok) {
      if (response.status === 401) {
        const text = await response.text().catch(() => "");
        throw new AuthenticationError(
          text ? `Authentication failed: ${text}` : "Authentication failed"
        );
      }
      if (response.status === 404) {
        throw new NotFoundError(`Resource not found: ${path}`);
      }
      const text = await response.text().catch(() => "");
      throw new HowlerError(
        `HTTP ${response.status}: ${text}`,
        response.status
      );
    }

    return response;
  }
}
