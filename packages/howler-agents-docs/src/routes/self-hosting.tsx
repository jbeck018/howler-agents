import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/self-hosting",
  component: SelfHostingPage,
});

function SelfHostingPage() {
  return (
    <>
      <h1>Self-Hosting</h1>

      <h2>Docker Compose (Recommended)</h2>
      <pre><code>{`# Clone and configure
git clone https://github.com/your-org/howler-agents.git
cd howler-agents
cp .env.example .env
# Edit .env with your API keys

# Start all services
make docker-up

# Services:
# - REST API: http://localhost:8080
# - gRPC: localhost:50051
# - Dashboard: http://localhost:3000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379`}</code></pre>

      <h2>Kubernetes</h2>
      <p>Kustomize-based deployment with staging and production overlays:</p>
      <pre><code>{`# Staging
kubectl apply -k deploy/k8s/overlays/staging

# Production
kubectl apply -k deploy/k8s/overlays/production`}</code></pre>

      <h2>Configuration</h2>
      <p>All configuration is via environment variables. See <code>.env.example</code> for the full list.</p>
    </>
  );
}
