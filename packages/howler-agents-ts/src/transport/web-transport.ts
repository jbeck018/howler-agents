/**
 * Browser transport configuration (gRPC-Web).
 *
 * Returns transport config suitable for @connectrpc/connect-web
 * when proto-generated service stubs are available.
 */

import type { TransportOptions } from "./transport.js";

export interface WebTransportConfig {
  type: "grpc-web";
  baseUrl: string;
  headers: Record<string, string>;
}

export function createWebTransport(options: TransportOptions): WebTransportConfig {
  return {
    type: "grpc-web" as const,
    baseUrl: options.baseUrl,
    headers: options.apiKey
      ? { Authorization: `Bearer ${options.apiKey}` }
      : {},
  };
}
