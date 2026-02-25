import * as React from "react";
import { Link, Outlet, createRootRoute } from "@tanstack/react-router";
import { Book, Code, Rocket, Server, Cpu, FlaskConical, Plug, Zap } from "lucide-react";
import "../styles.css";

const navItems = [
  { to: "/", label: "Getting Started", icon: Rocket },
  { to: "/architecture", label: "Architecture", icon: Cpu },
  { to: "/api-reference", label: "API Reference", icon: Code },
  { to: "/sdk-guides", label: "SDK Guides", icon: Book },
  { to: "/skills", label: "Skills", icon: Zap },
  { to: "/self-hosting", label: "Self-Hosting", icon: Server },
  { to: "/paper-results", label: "Paper Results", icon: FlaskConical },
  { to: "/integrations", label: "Integrations", icon: Plug },
] as const;

export const Route = createRootRoute({
  component: DocsLayout,
});

function DocsLayout() {
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 flex h-screen w-64 flex-col border-r bg-card">
        <div className="flex items-center gap-3 border-b p-4">
          <span className="text-lg font-bold text-primary">Howler Agents</span>
          <span className="rounded-full border px-2 py-0.5 text-[10px] text-muted-foreground">docs</span>
        </div>
        <nav className="flex-1 space-y-1 p-3">
          {navItems.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className="flex items-center gap-3 rounded-[var(--radius)] px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground [&.active]:text-primary"
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>
      <main className="flex-1 p-8">
        <div className="prose mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
