/**
 * Transport configuration for Howler Agents SDK.
 *
 * The primary SDK client (HowlerAgentsClient) uses fetch-based REST.
 * This module provides transport configuration that can be used with
 * ConnectRPC when proto-generated stubs become available.
 */

export interface TransportOptions {
  baseUrl: string;
  apiKey?: string;
}

export interface TransportConfig {
  baseUrl: string;
  transport: "web" | "node";
  headers: Record<string, string>;
}

export function createTransport(options: TransportOptions): TransportConfig {
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
