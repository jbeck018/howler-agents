import * as React from "react";
import { Outlet, createRootRoute, useLocation, useNavigate } from "@tanstack/react-router";
import { AppSidebar } from "../components/app-sidebar";
import { AppHeader } from "../components/app-header";
import { useAuth } from "../lib/auth";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const isLoginPage = location.pathname === "/login";

  React.useEffect(() => {
    if (isLoading) return;
    if (!isAuthenticated && !isLoginPage) {
      navigate({ to: "/login" });
    } else if (isAuthenticated && isLoginPage) {
      navigate({ to: "/" });
    }
  }, [isAuthenticated, isLoading, isLoginPage, navigate]);

  // While auth state is resolving, render nothing to avoid a flash
  if (isLoading) {
    return null;
  }

  // Login page gets its own full-screen layout (no sidebar/header)
  if (isLoginPage) {
    return <Outlet />;
  }

  // Redirect is in progress — render nothing
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="flex h-screen">
      <AppSidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <AppHeader />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
