import * as React from "react";

// --- Types ---

export interface AuthUser {
  user_id: string;
  org_id: string;
  email: string;
  role: string;
}

export interface AuthState {
  user: AuthUser | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface AuthActions {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, orgName: string) => Promise<void>;
  logout: () => void;
}

export type AuthContextValue = AuthState & AuthActions;

// --- Helpers ---

const ACCESS_TOKEN_KEY = "howler_access_token";
const REFRESH_TOKEN_KEY = "howler_refresh_token";

const API_BASE =
  (typeof window !== "undefined" && (window as any).__HOWLER_API_URL__) ||
  "http://localhost:8080";

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const base64 = token.split(".")[1];
    if (!base64) return null;
    // Pad base64 to a multiple of 4 characters
    const padded = base64.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(padded);
    return JSON.parse(json) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function userFromPayload(payload: Record<string, unknown>): AuthUser | null {
  const user_id = payload["sub"] ?? payload["user_id"];
  const org_id = payload["org_id"];
  const email = payload["email"];
  const role = payload["role"] ?? "member";
  if (
    typeof user_id === "string" &&
    typeof org_id === "string" &&
    typeof email === "string" &&
    typeof role === "string"
  ) {
    return { user_id, org_id, email, role };
  }
  return null;
}

function isTokenExpired(payload: Record<string, unknown>): boolean {
  const exp = payload["exp"];
  if (typeof exp !== "number") return false;
  // Token is considered expired if it expires within the next 5 minutes
  return Date.now() / 1000 > exp - 300;
}

async function refreshAccessToken(): Promise<string | null> {
  const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
  if (!refreshToken) return null;

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      access_token?: string;
      refresh_token?: string;
    };
    if (!data.access_token) return null;
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    if (data.refresh_token) {
      localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token);
    }
    return data.access_token;
  } catch {
    return null;
  }
}

// Exported so api.ts can call it on 401
export { refreshAccessToken };

// --- Context ---

const AuthContext = React.createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = React.useState<AuthState>({
    user: null,
    accessToken: null,
    isAuthenticated: false,
    isLoading: true,
  });

  // On mount, restore session from localStorage
  React.useEffect(() => {
    const storedToken = localStorage.getItem(ACCESS_TOKEN_KEY);
    if (!storedToken) {
      setState((s) => ({ ...s, isLoading: false }));
      return;
    }

    const payload = decodeJwtPayload(storedToken);
    if (!payload) {
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
      setState((s) => ({ ...s, isLoading: false }));
      return;
    }

    if (isTokenExpired(payload)) {
      // Attempt a silent refresh before giving up
      refreshAccessToken().then((newToken) => {
        if (!newToken) {
          localStorage.removeItem(ACCESS_TOKEN_KEY);
          localStorage.removeItem(REFRESH_TOKEN_KEY);
          setState({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
          return;
        }
        const newPayload = decodeJwtPayload(newToken);
        const user = newPayload ? userFromPayload(newPayload) : null;
        setState({ user, accessToken: newToken, isAuthenticated: !!user, isLoading: false });
      });
      return;
    }

    const user = userFromPayload(payload);
    setState({ user, accessToken: storedToken, isAuthenticated: !!user, isLoading: false });
  }, []);

  // Proactive token refresh — schedule a refresh 5 min before expiry
  React.useEffect(() => {
    if (!state.accessToken) return;
    const payload = decodeJwtPayload(state.accessToken);
    if (!payload) return;
    const exp = payload["exp"];
    if (typeof exp !== "number") return;

    const msUntilRefresh = (exp - 300) * 1000 - Date.now();
    if (msUntilRefresh <= 0) return;

    const timer = setTimeout(async () => {
      const newToken = await refreshAccessToken();
      if (newToken) {
        const newPayload = decodeJwtPayload(newToken);
        const user = newPayload ? userFromPayload(newPayload) : null;
        setState((s) => ({ ...s, user, accessToken: newToken, isAuthenticated: !!user }));
      } else {
        // Refresh failed — clear auth
        localStorage.removeItem(ACCESS_TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        setState({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
      }
    }, msUntilRefresh);

    return () => clearTimeout(timer);
  }, [state.accessToken]);

  // --- Actions ---

  const login = React.useCallback(async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<string, unknown>;
      throw new Error((body["detail"] as string) ?? `Login failed: ${res.status}`);
    }
    const data = (await res.json()) as {
      access_token: string;
      refresh_token?: string;
    };
    localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    if (data.refresh_token) {
      localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token);
    }
    const payload = decodeJwtPayload(data.access_token);
    const user = payload ? userFromPayload(payload) : null;
    setState({ user, accessToken: data.access_token, isAuthenticated: !!user, isLoading: false });
  }, []);

  const register = React.useCallback(
    async (email: string, password: string, orgName: string) => {
      const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, org_name: orgName }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as Record<string, unknown>;
        throw new Error((body["detail"] as string) ?? `Registration failed: ${res.status}`);
      }
      const data = (await res.json()) as {
        access_token: string;
        refresh_token?: string;
      };
      localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
      if (data.refresh_token) {
        localStorage.setItem(REFRESH_TOKEN_KEY, data.refresh_token);
      }
      const payload = decodeJwtPayload(data.access_token);
      const user = payload ? userFromPayload(payload) : null;
      setState({ user, accessToken: data.access_token, isAuthenticated: !!user, isLoading: false });
    },
    []
  );

  const logout = React.useCallback(() => {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    setState({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
  }, []);

  const value: AuthContextValue = React.useMemo(
    () => ({ ...state, login, register, logout }),
    [state, login, register, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = React.useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
