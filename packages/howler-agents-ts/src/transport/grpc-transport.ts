/**
 * Node.js transport configuration (gRPC).
 *
 * Returns transport config suitable for @connectrpc/connect-node
 * when proto-generated service stubs are available.
 */

import type { TransportOptions } from "./transport.js";

export interface GrpcTransportConfig {
  type: "grpc";
  baseUrl: string;
  headers: Record<string, string>;
}

export function createGrpcTransport(options: TransportOptions): GrpcTransportConfig {
  return {
    type: "grpc" as const,
    baseUrl: options.baseUrl,
    headers: options.apiKey
      ? { Authorization: `Bearer ${options.apiKey}` }
      : {},
  };
}
