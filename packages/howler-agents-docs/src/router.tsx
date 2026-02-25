import { createRouter } from "@tanstack/react-router";
import { Route as rootRoute } from "./routes/__root";
import { Route as indexRoute } from "./routes/index";
import { Route as architectureRoute } from "./routes/architecture";
import { Route as apiReferenceRoute } from "./routes/api-reference";
import { Route as sdkGuidesRoute } from "./routes/sdk-guides";
import { Route as selfHostingRoute } from "./routes/self-hosting";
import { Route as paperResultsRoute } from "./routes/paper-results";
import { Route as integrationsRoute } from "./routes/integrations";
import { Route as skillsRoute } from "./routes/skills";

const routeTree = rootRoute.addChildren([
  indexRoute,
  architectureRoute,
  apiReferenceRoute,
  sdkGuidesRoute,
  skillsRoute,
  selfHostingRoute,
  paperResultsRoute,
  integrationsRoute,
]);

export const router = createRouter({ routeTree, basepath: "/howler-agents" });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
