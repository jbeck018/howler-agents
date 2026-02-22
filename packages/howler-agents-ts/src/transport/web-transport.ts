/**
 * ConnectRPC transport for browsers (gRPC-Web).
 */

import type { TransportOptions } from "./transport.js";

export function createWebTransport(options: TransportOptions) {
  // Will use @connectrpc/connect-web when proto stubs are generated
  return {
    type: "grpc-web" as const,
    baseUrl: options.baseUrl,
    headers: options.apiKey
      ? { Authorization: `Bearer ${options.apiKey}` }
      : {},
  };
}
