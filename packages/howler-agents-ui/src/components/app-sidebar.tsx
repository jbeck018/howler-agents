import * as React from "react";
import { Link } from "@tanstack/react-router";
import { Activity, Bot, FlaskConical, LayoutDashboard, Settings, Dna } from "lucide-react";

const navItems: { to: "/" | "/runs" | "/agents" | "/experience" | "/settings"; label: string; icon: React.FC<any> }[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/runs", label: "Evolution Runs", icon: Dna },
  { to: "/agents", label: "Agents", icon: Bot },
  { to: "/experience", label: "Experience", icon: FlaskConical },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function AppSidebar() {
  return (
    <aside className="flex h-screen w-64 flex-col border-r bg-card">
      <div className="flex items-center gap-3 border-b p-4">
        <img src="/favicon.png" alt="Howler" className="h-8 w-8" />
        <span className="text-lg font-bold text-primary">Howler Agents</span>
      </div>
      <nav className="flex-1 space-y-1 p-3">
        {navItems.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            className="flex items-center gap-3 rounded-[var(--radius)] px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground [&.active]:bg-accent [&.active]:text-primary"
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="border-t p-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <Activity className="h-3 w-3 text-success" />
          System Online
        </div>
      </div>
    </aside>
  );
}
