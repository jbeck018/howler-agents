/**
 * ConnectRPC transport for Node.js (gRPC).
 */

import type { TransportOptions } from "./transport.js";

export function createGrpcTransport(options: TransportOptions) {
  // Will use @connectrpc/connect-node when proto stubs are generated
  return {
    type: "grpc" as const,
    baseUrl: options.baseUrl,
    headers: options.apiKey
      ? { Authorization: `Bearer ${options.apiKey}` }
      : {},
  };
}
