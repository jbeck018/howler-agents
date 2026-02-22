import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import { HowlerAgentsClient } from "../src/client.js";
import { NotFoundError } from "../src/errors/not-found.js";
import { ConnectionError } from "../src/errors/connection.js";
import { AuthenticationError } from "../src/errors/authentication.js";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

const mockRun = {
  id: "run-1",
  config: {
    populationSize: 10,
    groupSize: 3,
    numIterations: 5,
    alpha: 0.5,
    numProbes: 20,
    llmConfig: {},
    taskDomain: "general",
    taskConfig: {},
  },
  status: "pending",
  currentGeneration: 0,
  totalGenerations: 5,
  groups: [],
  bestScore: 0,
};

const mockAuthResponse = {
  access_token: "eyJhbGciOiJIUzI1NiJ9.test.token",
  refresh_token: "rt_test_refresh_token",
  user: {
    user_id: "user-123",
    org_id: "org-456",
    email: "test@example.com",
    role: "admin",
  },
};

const mockApiKeyResponse = {
  id: "key-789",
  key: "ha_live_testapikey123",
  name: "my-key",
  created_at: "2026-02-20T00:00:00Z",
};

const handlers = [
  http.get("http://localhost:8080/health", () =>
    HttpResponse.json({ status: "ok" })
  ),
  http.post("http://localhost:8080/api/v1/runs", () =>
    HttpResponse.json(mockRun, { status: 201 })
  ),
  http.get("http://localhost:8080/api/v1/runs/run-1", () =>
    HttpResponse.json(mockRun)
  ),
  http.get("http://localhost:8080/api/v1/runs/not-found", () =>
    HttpResponse.json({ detail: "Not found" }, { status: 404 })
  ),
  http.get("http://localhost:8080/api/v1/runs", () =>
    HttpResponse.json({ runs: [mockRun], total: 1 })
  ),
  http.get("http://localhost:8080/api/v1/runs/run-1/agents", () =>
    HttpResponse.json([])
  ),
  http.get("http://localhost:8080/api/v1/runs/run-1/agents/best", () =>
    HttpResponse.json([])
  ),

  // Auth handlers
  http.post("http://localhost:8080/api/v1/auth/login", async ({ request }) => {
    const body = (await request.json()) as { email: string; password: string };
    if (body.password === "wrongpassword") {
      return HttpResponse.json(
        { detail: "Invalid credentials" },
        { status: 401 }
      );
    }
    return HttpResponse.json(mockAuthResponse);
  }),
  http.post(
    "http://localhost:8080/api/v1/auth/register",
    async ({ request }) => {
      const body = (await request.json()) as {
        email: string;
        password: string;
        org_name: string;
      };
      return HttpResponse.json({ ...mockAuthResponse, user: { ...mockAuthResponse.user, email: body.email } }, { status: 201 });
    }
  ),
  http.post(
    "http://localhost:8080/api/v1/auth/refresh",
    async ({ request }) => {
      const body = (await request.json()) as { refresh_token: string };
      if (body.refresh_token !== mockAuthResponse.refresh_token) {
        return HttpResponse.json(
          { detail: "Invalid refresh token" },
          { status: 401 }
        );
      }
      return HttpResponse.json({
        ...mockAuthResponse,
        access_token: "eyJhbGciOiJIUzI1NiJ9.new.token",
        refresh_token: "rt_new_refresh_token",
      });
    }
  ),
  http.post(
    "http://localhost:8080/api/v1/auth/api-keys",
    async ({ request }) => {
      const authHeader = request.headers.get("Authorization");
      if (!authHeader) {
        return HttpResponse.json(
          { detail: "Not authenticated" },
          { status: 401 }
        );
      }
      const body = (await request.json()) as { name: string };
      return HttpResponse.json({ ...mockApiKeyResponse, name: body.name }, { status: 201 });
    }
  ),
  http.get("http://localhost:8080/api/v1/auth/me", ({ request }) => {
    const authHeader = request.headers.get("Authorization");
    if (!authHeader) {
      return HttpResponse.json(
        { detail: "Not authenticated" },
        { status: 401 }
      );
    }
    return HttpResponse.json(mockAuthResponse.user);
  }),
];

const server = setupServer(...handlers);

describe("HowlerAgentsClient", () => {
  let client: HowlerAgentsClient;

  beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
  afterAll(() => server.close());
  afterEach(() => server.resetHandlers());

  beforeAll(() => {
    client = new HowlerAgentsClient({ baseUrl: "http://localhost:8080" });
  });

  it("should check health", async () => {
    const result = await client.health();
    expect(result.status).toBe("ok");
  });

  it("should create a run", async () => {
    const run = await client.createRun({ populationSize: 10 });
    expect(run.id).toBe("run-1");
    expect(run.status).toBe("pending");
  });

  it("should get a run", async () => {
    const run = await client.getRun("run-1");
    expect(run.id).toBe("run-1");
  });

  it("should throw NotFoundError for missing run", async () => {
    await expect(client.getRun("not-found")).rejects.toThrow(NotFoundError);
  });

  it("should list runs", async () => {
    const result = await client.listRuns({ limit: 10 });
    expect(result.runs).toHaveLength(1);
    expect(result.total).toBe(1);
  });

  it("should list agents for a run", async () => {
    const agents = await client.listAgents("run-1");
    expect(agents).toEqual([]);
  });

  it("should get best agents", async () => {
    const agents = await client.getBestAgents("run-1", 5);
    expect(agents).toEqual([]);
  });
});

describe("HowlerAgentsClient — auth", () => {
  let anonClient: HowlerAgentsClient;
  let authedClient: HowlerAgentsClient;

  beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
  afterAll(() => server.close());
  afterEach(() => server.resetHandlers());

  beforeAll(() => {
    anonClient = new HowlerAgentsClient({ baseUrl: "http://localhost:8080" });
    authedClient = new HowlerAgentsClient({
      baseUrl: "http://localhost:8080",
      accessToken: mockAuthResponse.access_token,
    });
  });

  describe("login", () => {
    it("returns tokens and user on valid credentials", async () => {
      const result = await anonClient.login("test@example.com", "correctpassword");
      expect(result.access_token).toBe(mockAuthResponse.access_token);
      expect(result.refresh_token).toBe(mockAuthResponse.refresh_token);
      expect(result.user.user_id).toBe("user-123");
      expect(result.user.email).toBe("test@example.com");
      expect(result.user.role).toBe("admin");
    });

    it("throws AuthenticationError on wrong credentials (401)", async () => {
      await expect(
        anonClient.login("test@example.com", "wrongpassword")
      ).rejects.toThrow(AuthenticationError);
    });

    it("throws AuthenticationError with statusCode 401", async () => {
      try {
        await anonClient.login("test@example.com", "wrongpassword");
        expect.fail("Expected AuthenticationError to be thrown");
      } catch (err) {
        expect(err).toBeInstanceOf(AuthenticationError);
        expect((err as AuthenticationError).statusCode).toBe(401);
      }
    });
  });

  describe("register", () => {
    it("returns tokens and user for a new registration", async () => {
      const result = await anonClient.register(
        "new@example.com",
        "password123",
        "My Org"
      );
      expect(result.access_token).toBeDefined();
      expect(result.refresh_token).toBeDefined();
      expect(result.user.email).toBe("new@example.com");
    });
  });

  describe("refreshToken", () => {
    it("returns new token pair for a valid refresh token", async () => {
      const result = await anonClient.refreshToken(
        mockAuthResponse.refresh_token
      );
      expect(result.access_token).toBe("eyJhbGciOiJIUzI1NiJ9.new.token");
      expect(result.refresh_token).toBe("rt_new_refresh_token");
    });

    it("throws AuthenticationError for an invalid refresh token", async () => {
      await expect(
        anonClient.refreshToken("bad_refresh_token")
      ).rejects.toThrow(AuthenticationError);
    });
  });

  describe("createApiKey", () => {
    it("creates an API key when authenticated", async () => {
      const result = await authedClient.createApiKey("my-key");
      expect(result.id).toBe("key-789");
      expect(result.name).toBe("my-key");
      expect(result.key).toMatch(/^ha_live_/);
      expect(result.created_at).toBeDefined();
    });

    it("throws AuthenticationError when not authenticated", async () => {
      await expect(anonClient.createApiKey("my-key")).rejects.toThrow(
        AuthenticationError
      );
    });
  });

  describe("me", () => {
    it("returns user profile when authenticated", async () => {
      const result = await authedClient.me();
      expect(result.user_id).toBe("user-123");
      expect(result.org_id).toBe("org-456");
      expect(result.email).toBe("test@example.com");
      expect(result.role).toBe("admin");
    });

    it("throws AuthenticationError when not authenticated", async () => {
      await expect(anonClient.me()).rejects.toThrow(AuthenticationError);
    });
  });

  describe("401 response handling", () => {
    it("throws AuthenticationError (not generic HowlerError) on any 401", async () => {
      server.use(
        http.get("http://localhost:8080/api/v1/runs/run-1", () =>
          HttpResponse.json({ detail: "Token expired" }, { status: 401 })
        )
      );
      await expect(authedClient.getRun("run-1")).rejects.toThrow(
        AuthenticationError
      );
    });
  });
});
