/**
 * @howler-agents/sdk - TypeScript SDK for Howler Agents
 *
 * Group-Evolving Agents: groups of AI agents evolve together
 * by sharing experience across lineages.
 */

// Client
export { HowlerAgentsClient } from "./client.js";
export type {
  HowlerClientOptions,
  AuthResponse,
  ApiKeyResponse,
  UserResponse,
} from "./client.js";

// Transport
export { createTransport } from "./transport/transport.js";
export type { TransportOptions } from "./transport/transport.js";

// Errors
export { HowlerError } from "./errors/base.js";
export { ConnectionError } from "./errors/connection.js";
export { NotFoundError } from "./errors/not-found.js";
export { AuthenticationError } from "./errors/authentication.js";

// Types (manually defined until proto generation is set up)
export type {
  RunConfig,
  Agent,
  AgentGroup,
  FrameworkPatch,
  EvolutionaryTrace,
  EvolutionRun,
  EvolutionEvent,
} from "./types.js";
