/**
 * Custom fetcher for Orval-generated hooks.
 * Configures the base URL and default headers.
 */

const API_BASE = (typeof window !== "undefined" && (window as any).__HOWLER_API_URL__) || "http://localhost:8080";

export async function customFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const fullUrl = url.startsWith("http") ? url : `${API_BASE}${url}`;
  const response = await fetch(fullUrl, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`${response.status}: ${response.statusText}`);
  }

  return response.json();
}
