import * as React from "react";
import { createRoute, useNavigate } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { useHealthQuery } from "../lib/api";
import { useAuth } from "../lib/auth";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage,
});

const LLM_ROLES = ["Acting", "Evolving", "Reflecting"] as const;

function SettingsPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate({ to: "/login" });
  }

  const [models, setModels] = React.useState<Record<string, string>>({
    Acting: "claude-sonnet-4-20250514",
    Evolving: "claude-sonnet-4-20250514",
    Reflecting: "claude-sonnet-4-20250514",
  });
  const [saved, setSaved] = React.useState(false);
  const [connectionStatus, setConnectionStatus] = React.useState<"idle" | "ok" | "error">("idle");
  const healthQuery = useHealthQuery();

  function handleModelChange(role: string, value: string) {
    setModels((prev) => ({ ...prev, [role]: value }));
    setSaved(false);
  }

  function handleSave() {
    localStorage.setItem("howler_llm_config", JSON.stringify(models));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function handleTestConnection() {
    healthQuery.refetch().then((result) => {
      setConnectionStatus(result.data?.status === "ok" ? "ok" : "error");
      setTimeout(() => setConnectionStatus("idle"), 3000);
    }).catch(() => {
      setConnectionStatus("error");
      setTimeout(() => setConnectionStatus("idle"), 3000);
    });
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Settings</h2>
      {user && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Account</CardTitle>
            <CardDescription>Your logged-in account details.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
              <div>
                <span className="text-xs font-medium text-muted-foreground">Email</span>
                <p className="mt-0.5 font-mono text-sm">{user.email}</p>
              </div>
              <div>
                <span className="text-xs font-medium text-muted-foreground">Organization ID</span>
                <p className="mt-0.5 font-mono text-sm">{user.org_id}</p>
              </div>
              <div>
                <span className="text-xs font-medium text-muted-foreground">User ID</span>
                <p className="mt-0.5 font-mono text-sm">{user.user_id}</p>
              </div>
              <div>
                <span className="text-xs font-medium text-muted-foreground">Role</span>
                <p className="mt-0.5 font-mono text-sm capitalize">{user.role}</p>
              </div>
            </div>
            <Button variant="destructive" size="sm" onClick={handleLogout}>
              Sign Out
            </Button>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">LLM Configuration</CardTitle>
          <CardDescription>Configure which models to use for each evolutionary role.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {LLM_ROLES.map((role) => (
              <div key={role} className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground">{role} Model</label>
                <input
                  type="text"
                  value={models[role]}
                  onChange={(e) => handleModelChange(role, e.target.value)}
                  className="w-full rounded-[var(--radius)] border bg-background px-3 py-2 text-sm"
                />
              </div>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={handleSave}>Save Configuration</Button>
            {saved && <span className="text-sm text-green-500">Saved</span>}
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Service Connection</CardTitle>
          <CardDescription>Test connectivity to the backend service.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Status: {healthQuery.data?.status === "ok" ? "Connected" : "Not connected"}
          </p>
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={handleTestConnection}>
              Test Connection
            </Button>
            {connectionStatus === "ok" && (
              <span className="text-sm text-green-500">Connection successful</span>
            )}
            {connectionStatus === "error" && (
              <span className="text-sm text-destructive">Connection failed</span>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
