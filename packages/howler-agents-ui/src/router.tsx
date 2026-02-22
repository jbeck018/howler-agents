import { createRouter } from "@tanstack/react-router";
import { Route as rootRoute } from "./routes/__root";
import { Route as indexRoute } from "./routes/index";
import { Route as runsRoute } from "./routes/runs";
import { Route as runDetailRoute } from "./routes/runs.$runId";
import { Route as agentsRoute } from "./routes/agents";
import { Route as experienceRoute } from "./routes/experience";
import { Route as settingsRoute } from "./routes/settings";
import { Route as loginRoute } from "./routes/login";

const routeTree = rootRoute.addChildren([
  loginRoute,
  indexRoute,
  runsRoute,
  runDetailRoute,
  agentsRoute,
  experienceRoute,
  settingsRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
