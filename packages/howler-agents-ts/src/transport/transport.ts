/**
 * Transport factory for ConnectRPC.
 * Auto-detects browser vs Node.js environment.
 */

export interface TransportOptions {
  baseUrl: string;
  apiKey?: string;
}

export function createTransport(options: TransportOptions) {
  // In a real setup, this would use @connectrpc/connect-web or connect-node
  // based on the environment. For now, return a config object.
  const isBrowser =
    typeof globalThis !== "undefined" &&
    typeof (globalThis as Record<string, unknown>).window !== "undefined";

  return {
    baseUrl: options.baseUrl,
    transport: isBrowser ? "web" : "node",
    headers: options.apiKey
      ? { Authorization: `Bearer ${options.apiKey}` }
      : {},
  };
}
