import * as React from "react";
import { createRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { useAuth } from "../lib/auth";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/login",
  component: LoginPage,
});

type Tab = "signin" | "register";

function LoginPage() {
  const [activeTab, setActiveTab] = React.useState<Tab>("signin");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [orgName, setOrgName] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [isPending, setIsPending] = React.useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();

  function handleTabChange(tab: Tab) {
    setActiveTab(tab);
    setError(null);
  }

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsPending(true);
    try {
      await login(email, password);
      navigate({ to: "/" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign in failed");
    } finally {
      setIsPending(false);
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsPending(true);
    try {
      await register(email, password, orgName);
      navigate({ to: "/" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setIsPending(false);
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="w-full max-w-md px-4">
        <div className="mb-8 flex items-center justify-center gap-3">
          <img src="/favicon.png" alt="Howler" className="h-10 w-10" />
          <span className="text-2xl font-bold text-primary">Howler Agents</span>
        </div>

        <Card>
          <CardHeader className="pb-4">
            {/* Tab bar */}
            <div className="flex gap-1 rounded-[var(--radius)] border bg-muted p-1">
              <button
                type="button"
                onClick={() => handleTabChange("signin")}
                className={[
                  "flex-1 rounded-[var(--radius)] py-1.5 text-sm font-medium transition-colors",
                  activeTab === "signin"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                ].join(" ")}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => handleTabChange("register")}
                className={[
                  "flex-1 rounded-[var(--radius)] py-1.5 text-sm font-medium transition-colors",
                  activeTab === "register"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                ].join(" ")}
              >
                Register
              </button>
            </div>

            <CardTitle className="mt-4 text-base">
              {activeTab === "signin" ? "Welcome back" : "Create an account"}
            </CardTitle>
            <CardDescription>
              {activeTab === "signin"
                ? "Sign in to your Howler Agents account."
                : "Register a new account and organization."}
            </CardDescription>
          </CardHeader>

          <CardContent>
            {activeTab === "signin" ? (
              <form onSubmit={handleSignIn} className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="signin-email" className="text-xs font-medium text-muted-foreground">
                    Email
                  </label>
                  <input
                    id="signin-email"
                    type="email"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full rounded-[var(--radius)] border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="you@example.com"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="signin-password" className="text-xs font-medium text-muted-foreground">
                    Password
                  </label>
                  <input
                    id="signin-password"
                    type="password"
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full rounded-[var(--radius)] border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="••••••••"
                  />
                </div>
                {error && (
                  <p className="text-sm text-destructive">{error}</p>
                )}
                <Button type="submit" className="w-full" disabled={isPending}>
                  {isPending ? "Signing in..." : "Sign In"}
                </Button>
              </form>
            ) : (
              <form onSubmit={handleRegister} className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="register-email" className="text-xs font-medium text-muted-foreground">
                    Email
                  </label>
                  <input
                    id="register-email"
                    type="email"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full rounded-[var(--radius)] border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="you@example.com"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="register-password" className="text-xs font-medium text-muted-foreground">
                    Password
                  </label>
                  <input
                    id="register-password"
                    type="password"
                    autoComplete="new-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full rounded-[var(--radius)] border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="••••••••"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="register-org" className="text-xs font-medium text-muted-foreground">
                    Organization Name
                  </label>
                  <input
                    id="register-org"
                    type="text"
                    autoComplete="organization"
                    required
                    value={orgName}
                    onChange={(e) => setOrgName(e.target.value)}
                    className="w-full rounded-[var(--radius)] border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="Acme Corp"
                  />
                </div>
                {error && (
                  <p className="text-sm text-destructive">{error}</p>
                )}
                <Button type="submit" className="w-full" disabled={isPending}>
                  {isPending ? "Creating account..." : "Create Account"}
                </Button>
              </form>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
