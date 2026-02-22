# Group-Evolving Agents (GEA) — Deep Dive & Cross-Language Package Architecture

> Based on: **"Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing"**  
> Weng, Antoniades, Nathani, Zhang, Pu, Wang — arXiv:2602.04837 (Feb 2026)

---

## Table of Contents

1. [Paper Deep Dive](#paper-deep-dive)
2. [Why Not a Rust-Compiled Core](#why-not-rust)
3. [Architecture Overview](#architecture-overview)
4. [Layer 1: Python Core Library (`gea-core`)](#layer-1-python-core-library)
5. [Layer 2: Service Layer (gRPC + REST)](#layer-2-service-layer)
6. [Layer 3: Language SDKs](#layer-3-language-sdks)
7. [Layer 4: Experience Store Schema](#layer-4-experience-store-schema)
8. [Layer 5: Deployment](#layer-5-deployment)
9. [Critical Implementation Decisions](#critical-implementation-decisions)
10. [Monorepo Structure](#monorepo-structure)
11. [Reproducing Paper Results](#reproducing-paper-results)

---

## Paper Deep Dive

### Core Problem GEA Solves

Prior open-ended self-improving systems (Darwin Gödel Machine, etc.) evolve agents along **isolated tree branches**. A breakthrough discovered in branch A never reaches branch B. Useful diversity is generated but then *wasted* — it dies in its lineage without propagating.

GEA's observation: unlike biological organisms, AI agents can **directly share** trajectories, tools, and learned artifacts. There's no reason to enforce genetic isolation. The unit of evolution should be the **group**, not the individual.

### The Five Core Mechanisms

#### 1. Performance-Novelty Parent Selection

At each iteration, a parent group is selected by jointly optimizing two signals:

- **Performance**: raw task success rate — reflects immediate competence and likely offspring quality
- **Novelty**: KNN distance in probe-task behavior space — encourages exploration beyond already-optimized regions

The two signals are combined via a weighted criterion. Neither alone is sufficient: performance-only collapses to local optima; novelty-only never converges.

```
score(agent) = α · performance(agent) + (1 - α) · novelty(agent)
novelty(agent) = mean distance to K nearest neighbors in capability vector space
```

#### 2. Shared Experience Pool

All agents in the parent group contribute their full evolutionary traces into a single aggregated pool:
- Code patches applied at each iteration
- Tool discoveries and integrations
- Successful workflows
- Failure modes and how they were resolved
- Probe task outcomes (binary capability vectors)

Nothing is siloed. The pool is the collective memory of the group.

#### 3. Group Reproduction via Experience-Conditioned Patching

The pooled experience is fed to a **meta-LLM** (the paper uses GPT-o1 for reflection, Claude Haiku/Sonnet for acting) which:
1. Analyzes aggregate traces to identify patterns, redundancies, and gaps
2. Generates **evolution directives** — high-level intent for the next generation
3. Produces **framework-level patches** — concrete code/config/workflow changes

Critically, these patches target **workflows and tool usage**, not model-specific prompting, making them model-agnostic and transferable.

#### 4. Probe Task Characterization

Each agent is characterized by a **binary capability vector** across a fixed suite of probe tasks. This vector is:
- The input to KNN novelty calculation
- A fingerprint of what the agent *can* and *cannot* do
- Used to track which capabilities propagate across generations

Binary representation (task solved / not solved) was found to better characterize coding capabilities than continuous scores.

#### 5. Cumulative Tool Integration

The best GEA agent integrates **8 out of 9** key tool innovations discovered across the population, with tools originating from 5 different parent agents. The equivalent DGM (tree-structured) agent captures only 5, because 4 useful discoveries die in isolated branches.

GEA's best agent draws experience from **17 unique ancestors** (28.3% of the population) — nearly double DGM's 9.

### Benchmark Results

| Metric | GEA | DGM (prior SOTA) | Human-designed SOTA |
|---|---|---|---|
| SWE-bench Verified | **71.0%** | 56.7% | 71.8% (OpenHands + GPT-5) |
| Polyglot | **88.3%** | 68.3% | 52.0% (Aider + GPT-5) |
| Bug fix iterations | **1.4** | 5.0 | — |
| Unique ancestors in best agent | **17** | 9 | — |
| Worst-case top-5 performance | **58.3%** | 56.7% (single best) | — |

### Key Insight on Task Complexity

GEA adapts its evolutionary behavior to task complexity:
- **Polyglot** (single-file edits): larger, more concentrated patches → 88.3% in just 4 iterations
- **SWE-bench** (multi-file coordination): smaller, distributed patches → 71.0% requiring 8 iterations

This suggests the patch size and distribution are emergent properties of the task structure, not hardcoded parameters.

---

## Why Not Rust

> **TL;DR: The bottleneck is LLM API latency, not computation. gRPC is the right cross-language contract.**

The idea of compiling a Rust core and using it across languages via FFI is appealing but wrong for this problem. Here's the full reasoning:

### Where GEA Spends Its Time

```
Evolution iteration wall-clock time breakdown (estimated):
  ├── LLM API calls (patch generation, reflection)  ~85%
  ├── Agent task execution (SWE-bench, Polyglot)    ~12%
  ├── KNN novelty computation                        ~1%
  ├── Experience aggregation / serialization         ~1%
  └── Other                                          ~1%
```

Rust excels at compute-bound work. When 97% of your time is waiting on HTTP responses from Anthropic/OpenAI, a faster core saves nothing measurable.

### The FFI Distribution Problem

A Rust native library (`.so` / `.dylib` / `.dll`) requires:
- Pre-compiled binaries for every OS × architecture target (linux-x86_64, linux-aarch64, darwin-arm64, darwin-x86_64, windows-x86_64, ...)
- Platform-specific packaging in each language's package manager
- FFI boundary management across async runtimes (tokio in Rust vs asyncio in Python vs goroutines in Go)
- ABI stability maintenance — internal Rust changes can silently break consumers

gRPC gives you the same cross-language contract with none of this overhead, and the proto file is self-documenting.

### When Rust *Would* Make Sense

If you were running GEA at massive scale — thousands of concurrent evolution runs — and profiling confirmed that novelty vector math was a bottleneck, a targeted Rust module compiled to WebAssembly or a native library for that specific function would be worth it. That's an optimization problem for later, not a foundation decision now.

### Verdict

**gRPC service architecture. Python core. Thin language SDKs.** This is the right structure.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Language SDKs (thin clients)                 │
│   Python │ TypeScript │ Go │ Rust │ Java │ Ruby │ C# │ ...      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  gRPC (primary) / REST (fallback)
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      GEA Service Layer                            │
│         FastAPI + gRPC gateway — stateless, horizontally         │
│         scalable, holds all LLM API keys server-side             │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
┌──────────▼──────────┐          ┌────────────▼──────────────────┐
│   gea-core          │          │   Experience Store             │
│   Python library    │          │   Postgres + pgvector          │
│   (pip installable  │          │   Redis (hot cache)            │
│    standalone too)  │          └───────────────────────────────┘
└──────────┬──────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                       GEA Core Modules                            │
│                                                                   │
│  AgentPool          — manages the living population              │
│  PerformanceNoveltySeletor — balanced parent group selection     │
│  SharedExperiencePool — aggregated group traces                  │
│  GroupReproducer    — parent group → child group via meta-LLM    │
│  ProbeEvaluator     — builds binary capability vectors           │
│  LLMRouter          — assigns right model to right role          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Python Core Library

**Install:** `pip install gea-core`  
**Use case:** Direct Python usage or as the engine behind the service.

### Directory Structure

```
gea-core/
├── gea/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract Agent — subclass with your implementation
│   │   ├── pool.py          # AgentPool — manages the living population
│   │   └── patch.py         # FrameworkPatch — represents a code/workflow delta
│   │
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── performance.py   # TaskPerformanceScorer
│   │   ├── novelty.py       # KNNNoveltyEstimator — behavior space distance
│   │   └── criterion.py     # PerformanceNoveltySelector — combines both signals
│   │
│   ├── experience/
│   │   ├── __init__.py
│   │   ├── trace.py         # EvolutionaryTrace — one agent's full history
│   │   ├── pool.py          # SharedExperiencePool — group-level aggregation
│   │   └── store/
│   │       ├── base.py      # ExperienceStore interface (pluggable)
│   │       ├── memory.py    # InMemoryStore — for testing
│   │       ├── redis.py     # RedisStore — fast production cache
│   │       └── postgres.py  # PostgresStore — durable, queryable
│   │
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── directive.py     # EvolutionDirective — LLM-generated intent
│   │   ├── reproducer.py    # GroupReproducer — core group reproduction logic
│   │   └── loop.py          # EvolutionLoop — main orchestration
│   │
│   ├── probes/
│   │   ├── __init__.py
│   │   ├── base.py          # ProbeTask interface
│   │   ├── evaluator.py     # ProbeEvaluator — builds binary capability vectors
│   │   └── registry.py     # ProbeRegistry — register custom probe tasks
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # LLMBackend interface
│   │   ├── anthropic.py     # Claude Haiku/Sonnet (acting + evolving)
│   │   ├── openai.py        # GPT-o1 (reflection — per paper)
│   │   └── router.py        # LLMRouter — different models for different roles
│   │
│   └── config.py            # GEAConfig — K, M, alpha, iterations, etc.
│
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
└── README.md
```

### Key Abstractions

#### `Agent` (base.py)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentConfig:
    """JSON-serializable config — language-agnostic at service boundary."""
    agent_id: str
    workflow: dict[str, Any]
    tools: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

class Agent(ABC):
    """
    Subclass this with your actual agent implementation.
    GEA doesn't care what the agent *does*, only:
    - how it performs on tasks
    - what experience it produces
    - its capability vector across probe tasks
    """
    def __init__(self, config: AgentConfig):
        self.config = config
        self.capability_vector: list[float] = []
        self.performance_score: float = 0.0

    @abstractmethod
    async def run_task(self, task: dict) -> dict:
        """Execute one task. Return result dict."""
        ...

    @abstractmethod
    async def apply_patch(self, patch: "FrameworkPatch") -> "Agent":
        """Apply an evolution patch. Return the updated agent."""
        ...

    def to_json(self) -> dict:
        """Serialize for the service boundary."""
        return {
            "agent_id": self.config.agent_id,
            "config": self.config.__dict__,
            "capability_vector": self.capability_vector,
            "performance_score": self.performance_score,
        }
```

#### `PerformanceNoveltySelector` (criterion.py)
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class PerformanceNoveltySelector:
    def __init__(self, k_neighbors: int = 4, alpha: float = 0.5):
        self.k = k_neighbors
        self.alpha = alpha  # weight: 0=pure novelty, 1=pure performance

    def select_parent_group(
        self,
        agents: list[Agent],
        group_size: int
    ) -> list[Agent]:
        vectors = np.array([a.capability_vector for a in agents])
        performances = np.array([a.performance_score for a in agents])

        # KNN novelty: mean distance to k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(vectors)
        distances, _ = nbrs.kneighbors(vectors)
        novelty_scores = distances[:, 1:].mean(axis=1)  # exclude self

        # Normalize both signals to [0, 1]
        perf_norm = (performances - performances.min()) / (performances.ptp() + 1e-8)
        nov_norm = (novelty_scores - novelty_scores.min()) / (novelty_scores.ptp() + 1e-8)

        combined = self.alpha * perf_norm + (1 - self.alpha) * nov_norm
        top_indices = np.argsort(combined)[-group_size:]
        return [agents[i] for i in top_indices]
```

#### `SharedExperiencePool` (pool.py)
```python
@dataclass
class EvolutionaryTrace:
    agent_id: str
    iteration: int
    patches_applied: list[FrameworkPatch]
    probe_outcomes: dict[str, bool]      # probe_task_id -> solved?
    experience_narrative: str            # LLM-readable context
    parent_ids: list[str]               # tracks full lineage

class SharedExperiencePool:
    """
    Aggregates traces from all agents in a parent group.
    This is the core of GEA's experience sharing mechanism.
    """
    def __init__(self):
        self._traces: list[EvolutionaryTrace] = []

    def add_trace(self, trace: EvolutionaryTrace) -> None:
        self._traces.append(trace)

    def aggregate(self) -> str:
        """
        Produce a single LLM-readable context string from all traces.
        This gets fed to the meta-LLM for evolution directive generation.
        """
        sections = []
        for trace in self._traces:
            sections.append(f"""
## Agent {trace.agent_id} (Iteration {trace.iteration})
Capabilities solved: {[k for k,v in trace.probe_outcomes.items() if v]}
Capabilities missing: {[k for k,v in trace.probe_outcomes.items() if not v]}
Experience: {trace.experience_narrative}
Key patches: {[p.summary for p in trace.patches_applied]}
            """)
        return "\n".join(sections)
```

#### `LLMRouter` (router.py)
```python
class LLMRole(str, Enum):
    ACTING = "acting"       # Execute tasks, apply patches (Claude Haiku — fast, cheap)
    EVOLVING = "evolving"   # Generate patches (Claude Sonnet — balanced)
    REFLECTING = "reflecting"  # Meta-analysis of traces (GPT-o1 — deep reasoning)

class LLMRouter:
    """
    Routes different evolution roles to different LLM backends.
    Per paper: Claude Haiku/Sonnet for acting/evolving, GPT-o1 for reflection.
    """
    def __init__(self, role_map: dict[LLMRole, LLMBackend]):
        self._map = role_map

    async def complete(self, role: LLMRole, prompt: str, **kwargs) -> str:
        backend = self._map[role]
        return await backend.complete(prompt, **kwargs)
```

---

## Layer 2: Service Layer

The service is **stateless** — all state lives in Postgres + Redis. This enables horizontal scaling and zero-downtime deployments.

### Proto Definition

**`proto/gea/v1/gea.proto`**

```protobuf
syntax = "proto3";
package gea.v1;

option go_package = "github.com/you/gea-go/gen/gea/v1";
option java_package = "io.gea.v1";

service GEAService {
  // Lifecycle
  rpc CreateRun          (CreateRunRequest)      returns (EvolutionRun);
  rpc GetRun             (GetRunRequest)         returns (EvolutionRun);
  rpc ListRuns           (ListRunsRequest)       returns (ListRunsResponse);

  // Evolution control
  rpc StepEvolution      (StepRequest)           returns (StepResult);
  rpc RunEvolution       (RunRequest)            returns (RunResult);  // blocks until done

  // Agent operations
  rpc GetAgentGroup      (GroupRequest)          returns (AgentGroup);
  rpc GetBestAgents      (BestAgentsRequest)     returns (AgentGroup);

  // Experience submission (agents call this after task execution)
  rpc SubmitExperience   (SubmitExperienceRequest) returns (ExperienceAck);
  rpc SubmitProbeResults (ProbeResultsRequest)     returns (ExperienceAck);

  // Streaming (for real-time monitoring)
  rpc StreamEvolution    (StreamRequest)         returns (stream EvolutionEvent);
}

// ─── Config ───────────────────────────────────────────────────────────────────

message RunConfig {
  int32  group_size        = 1;  // K in paper — default 2
  int32  novelty_neighbors = 2;  // M for KNN — default 4
  float  alpha             = 3;  // perf/novelty balance — default 0.5
  int32  max_iterations    = 4;
  string task_domain       = 5;  // "swe-bench" | "polyglot" | custom
  map<string, string> llm_roles = 6;  // role -> model name overrides
}

// ─── Core Types ───────────────────────────────────────────────────────────────

message Agent {
  string          agent_id          = 1;
  bytes           serialized_config = 2;  // JSON blob — opaque to service
  repeated float  capability_vector = 3;
  float           performance_score = 4;
  float           novelty_score     = 5;
  repeated string parent_ids        = 6;
  int32           iteration         = 7;
  string          version           = 8;  // always include for future compat
}

message FrameworkPatch {
  string patch_id   = 1;
  string agent_id   = 2;
  bytes  diff       = 3;  // unified diff or structured JSON patch
  string patch_type = 4;  // "workflow" | "tool" | "config" | "prompt"
  string summary    = 5;
  int64  created_at = 6;
}

message EvolutionaryTrace {
  string                    agent_id              = 1;
  int32                     iteration             = 2;
  repeated FrameworkPatch   patches_applied       = 3;
  map<string, bool>         probe_outcomes        = 4;
  string                    experience_narrative  = 5;  // LLM-readable
  repeated string           parent_ids            = 6;
}

// ─── Run ──────────────────────────────────────────────────────────────────────

message EvolutionRun {
  string    run_id     = 1;
  RunConfig config     = 2;
  string    status     = 3;  // "created" | "running" | "completed" | "failed"
  int32     iteration  = 4;
  int64     created_at = 5;
  int64     updated_at = 6;
}

// ─── Events (streaming) ───────────────────────────────────────────────────────

message EvolutionEvent {
  string run_id         = 1;
  int32  iteration      = 2;
  string event_type     = 3;  // "iteration_complete" | "patch_applied" | "error"
  float  best_perf      = 4;
  float  mean_perf      = 5;
  AgentGroup top_agents = 6;
  string message        = 7;
}
```

### Service Directory

```
gea-service/
├── server.py              # Entry point — starts gRPC + FastAPI
├── grpc_handlers.py       # Implements all GEAService RPC methods
├── rest_router.py         # REST API (mirrors gRPC surface exactly)
│   ├── POST /runs
│   ├── GET  /runs/{run_id}
│   ├── POST /runs/{run_id}/step
│   ├── POST /runs/{run_id}/experience
│   └── GET  /runs/{run_id}/agents/best
├── auth.py                # API key / JWT middleware
├── middleware.py          # Logging, tracing, rate limiting
├── Dockerfile
└── pyproject.toml
```

### Key Service Behavior

The meta-LLM reflection step (GPT-o1 in the paper) **runs server-side**. SDKs submit experience and receive patches back. This means:
- LLM API keys never leave the server
- Model swaps happen without SDK updates
- Reflection logic can be improved without requiring client upgrades

---

## Layer 3: Language SDKs

Every SDK follows the same pattern: a generated gRPC stub + a thin ergonomic wrapper. The proto is the source of truth. Run `scripts/gen-proto.sh` to regenerate all stubs after proto changes.

### TypeScript / Node.js (`@gea/sdk`)

```typescript
// src/client.ts
import { GEAServiceClient } from '../generated/gea/v1/gea_grpc_pb';
import { credentials } from '@grpc/grpc-js';

export interface GEAClientConfig {
  endpoint: string;  // e.g. "grpc://localhost:50051"
  apiKey?: string;
  tls?: boolean;
}

export class GEAClient {
  private client: GEAServiceClient;

  constructor(config: GEAClientConfig) {
    const creds = config.tls
      ? credentials.createSsl()
      : credentials.createInsecure();
    this.client = new GEAServiceClient(config.endpoint, creds);
  }

  async createRun(config: RunConfig): Promise<EvolutionRun> { ... }
  async stepEvolution(runId: string): Promise<StepResult> { ... }
  async getBestAgents(runId: string, topK: number): Promise<Agent[]> { ... }
  async submitExperience(trace: EvolutionaryTrace): Promise<void> { ... }

  // Async generator for streaming
  async *streamEvolution(runId: string): AsyncGenerator<EvolutionEvent> {
    const stream = this.client.streamEvolution({ runId });
    for await (const event of stream) {
      yield event;
    }
  }
}

// Usage
const gea = new GEAClient({ endpoint: 'grpc://localhost:50051' });
const run = await gea.createRun({ groupSize: 2, noveltyNeighbors: 4 });

for await (const event of gea.streamEvolution(run.runId)) {
  console.log(`Iteration ${event.iteration}: best=${event.bestPerf.toFixed(3)}`);
}

const agents = await gea.getBestAgents(run.runId, 3);
```

**Package structure:**
```
gea-ts/
├── src/
│   ├── client.ts       # GEAClient
│   ├── models.ts       # TypeScript types (mirrors proto)
│   ├── stream.ts       # Async streaming helpers
│   └── errors.ts       # GEAError hierarchy
├── generated/          # Auto-generated from .proto — never edit manually
├── examples/
│   └── basic-run.ts
├── tests/
├── package.json
└── tsconfig.json
```

### Go (`github.com/you/gea-go`)

```go
// client.go
package gea

import (
    "context"
    pb "github.com/you/gea-go/gen/gea/v1"
    "google.golang.org/grpc"
)

type Client struct {
    conn   *grpc.ClientConn
    client pb.GEAServiceClient
}

func NewClient(endpoint string, opts ...Option) (*Client, error) {
    conn, err := grpc.Dial(endpoint, grpc.WithInsecure())
    if err != nil {
        return nil, err
    }
    return &Client{conn: conn, client: pb.NewGEAServiceClient(conn)}, nil
}

func (c *Client) CreateRun(ctx context.Context, cfg *RunConfig) (*EvolutionRun, error) { ... }
func (c *Client) StepEvolution(ctx context.Context, runID string) (*StepResult, error) { ... }
func (c *Client) GetBestAgents(ctx context.Context, runID string, topK int) ([]*Agent, error) { ... }
func (c *Client) SubmitExperience(ctx context.Context, trace *EvolutionaryTrace) error { ... }

// StreamEvolution returns a channel for receiving events
func (c *Client) StreamEvolution(ctx context.Context, runID string) (<-chan *EvolutionEvent, error) {
    ch := make(chan *EvolutionEvent, 10)
    stream, err := c.client.StreamEvolution(ctx, &pb.StreamRequest{RunId: runID})
    if err != nil {
        return nil, err
    }
    go func() {
        defer close(ch)
        for {
            event, err := stream.Recv()
            if err != nil { return }
            ch <- event
        }
    }()
    return ch, nil
}

// Usage
client, _ := gea.NewClient("localhost:50051")
run, _ := client.CreateRun(ctx, &gea.RunConfig{GroupSize: 2})

events, _ := client.StreamEvolution(ctx, run.RunId)
for event := range events {
    fmt.Printf("Iteration %d: best=%.3f\n", event.Iteration, event.BestPerf)
}
```

**Package structure:**
```
gea-go/
├── client.go
├── models.go       # Go types + JSON tags
├── options.go      # Functional options pattern
├── errors.go
├── gen/            # Auto-generated protobuf — never edit manually
├── examples/
│   └── basic/main.go
├── go.mod
└── go.sum
```

### Rust (`gea-rs`)

```rust
// src/client.rs
use tonic::transport::Channel;
use crate::proto::gea_service_client::GEAServiceClient;

pub struct GEAClient {
    client: GEAServiceClient<Channel>,
}

impl GEAClient {
    pub async fn connect(endpoint: &str) -> Result<Self, tonic::transport::Error> {
        let channel = Channel::from_shared(endpoint.to_string())?.connect().await?;
        Ok(Self { client: GEAServiceClient::new(channel) })
    }

    pub async fn create_run(&mut self, config: RunConfig) -> Result<EvolutionRun, tonic::Status> {
        let response = self.client.create_run(tonic::Request::new(config.into())).await?;
        Ok(response.into_inner().into())
    }

    pub async fn stream_evolution(
        &mut self,
        run_id: &str,
    ) -> Result<impl Stream<Item = Result<EvolutionEvent, tonic::Status>>, tonic::Status> {
        let request = StreamRequest { run_id: run_id.to_string() };
        let response = self.client.stream_evolution(request).await?;
        Ok(response.into_inner())
    }
}

// Usage
let mut client = GEAClient::connect("http://localhost:50051").await?;
let run = client.create_run(RunConfig { group_size: 2, novelty_neighbors: 4, ..Default::default() }).await?;

let mut stream = client.stream_evolution(&run.run_id).await?;
while let Some(event) = stream.next().await {
    let event = event?;
    println!("Iteration {}: best={:.3}", event.iteration, event.best_perf);
}
```

**Package structure:**
```
gea-rs/
├── src/
│   ├── lib.rs
│   ├── client.rs
│   ├── models.rs
│   └── errors.rs
├── proto/              # Symlinked from repo root proto/
├── build.rs            # tonic-build compiles proto at build time
├── examples/
│   └── basic_run.rs
├── Cargo.toml
└── Cargo.lock
```

### Python SDK (`gea-py`)

The Python SDK can use the service *or* call `gea-core` directly for in-process use.

```python
# gea-py/gea_py/client.py
import grpc
from gea.v1 import gea_pb2, gea_pb2_grpc
from typing import AsyncIterator

class GEAClient:
    """
    Thin client wrapping the gRPC service.
    For pure Python use, import gea-core directly instead.
    """
    def __init__(self, endpoint: str, api_key: str | None = None):
        self._channel = grpc.aio.insecure_channel(endpoint)
        self._stub = gea_pb2_grpc.GEAServiceStub(self._channel)

    async def create_run(self, config: dict) -> dict: ...
    async def step_evolution(self, run_id: str) -> dict: ...
    async def get_best_agents(self, run_id: str, top_k: int = 5) -> list[dict]: ...
    async def submit_experience(self, trace: dict) -> None: ...

    async def stream_evolution(self, run_id: str) -> AsyncIterator[dict]:
        request = gea_pb2.StreamRequest(run_id=run_id)
        async for event in self._stub.StreamEvolution(request):
            yield self._event_to_dict(event)
```

### Java (`io.gea:gea-java`)

```java
// GEAClient.java
public class GEAClient implements AutoCloseable {
    private final ManagedChannel channel;
    private final GEAServiceGrpc.GEAServiceStub asyncStub;
    private final GEAServiceGrpc.GEAServiceBlockingStub blockingStub;

    public GEAClient(String endpoint) {
        this.channel = ManagedChannelBuilder.forTarget(endpoint).usePlaintext().build();
        this.asyncStub = GEAServiceGrpc.newStub(channel);
        this.blockingStub = GEAServiceGrpc.newBlockingStub(channel);
    }

    public EvolutionRun createRun(RunConfig config) {
        return blockingStub.createRun(CreateRunRequest.newBuilder().setConfig(config).build());
    }

    public Iterator<EvolutionEvent> streamEvolution(String runId) {
        return blockingStub.streamEvolution(StreamRequest.newBuilder().setRunId(runId).build());
    }

    @Override
    public void close() { channel.shutdown(); }
}
```

---

## Layer 4: Experience Store Schema

```sql
-- Enable pgvector for KNN novelty computation
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Core Tables ──────────────────────────────────────────────────────────────

CREATE TABLE evolution_runs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  config        JSONB NOT NULL,
  status        TEXT NOT NULL DEFAULT 'created',
  iteration     INT NOT NULL DEFAULT 0,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE agents (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id            UUID NOT NULL REFERENCES evolution_runs(id),
  iteration         INT NOT NULL,
  serialized_config JSONB NOT NULL,           -- opaque agent config
  capability_vector vector(64),               -- probe task binary outcomes
  performance_score FLOAT NOT NULL DEFAULT 0,
  novelty_score     FLOAT NOT NULL DEFAULT 0,
  parent_ids        UUID[] NOT NULL DEFAULT '{}',  -- full lineage tracking
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE evolutionary_traces (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id              UUID NOT NULL REFERENCES agents(id),
  run_id                UUID NOT NULL REFERENCES evolution_runs(id),
  iteration             INT NOT NULL,
  patches_applied       JSONB[] NOT NULL DEFAULT '{}',
  probe_outcomes        JSONB NOT NULL DEFAULT '{}',  -- task_id -> bool
  experience_narrative  TEXT,                          -- LLM-readable context
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE framework_patches (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id      UUID NOT NULL REFERENCES evolution_runs(id),
  agent_id    UUID NOT NULL REFERENCES agents(id),
  iteration   INT NOT NULL,
  diff        BYTEA NOT NULL,    -- unified diff
  patch_type  TEXT NOT NULL,     -- 'workflow' | 'tool' | 'config' | 'prompt'
  summary     TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── Indexes ──────────────────────────────────────────────────────────────────

-- KNN novelty computation — this is the hot query path
CREATE INDEX idx_agents_capability_vector
  ON agents USING ivfflat (capability_vector vector_cosine_ops)
  WITH (lists = 100);

-- Common query patterns
CREATE INDEX idx_agents_run_iteration ON agents (run_id, iteration);
CREATE INDEX idx_agents_performance ON agents (run_id, performance_score DESC);
CREATE INDEX idx_traces_agent ON evolutionary_traces (agent_id);
CREATE INDEX idx_traces_run_iteration ON evolutionary_traces (run_id, iteration);

-- ─── Helper Views ─────────────────────────────────────────────────────────────

-- Best agents per run with full ancestry depth
CREATE VIEW agent_lineage_stats AS
SELECT
  a.id,
  a.run_id,
  a.performance_score,
  a.novelty_score,
  array_length(a.parent_ids, 1) AS lineage_depth,
  (SELECT COUNT(DISTINCT unnest) FROM unnest(a.parent_ids)) AS unique_ancestors
FROM agents a;

-- Evolution progress per run
CREATE VIEW run_progress AS
SELECT
  run_id,
  iteration,
  MAX(performance_score) AS best_performance,
  AVG(performance_score) AS mean_performance,
  STDDEV(performance_score) AS performance_stddev,
  COUNT(*) AS population_size
FROM agents
GROUP BY run_id, iteration
ORDER BY run_id, iteration;
```

---

## Layer 5: Deployment

### Local Development

**`docker-compose.yml`**
```yaml
version: '3.9'

services:
  gea-service:
    build:
      context: ./gea-service
      dockerfile: Dockerfile
    ports:
      - "50051:50051"   # gRPC
      - "8080:8080"     # REST
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://gea:gea@postgres:5432/gea
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: gea
      POSTGRES_USER: gea
      POSTGRES_PASSWORD: gea
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gea"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s

volumes:
  postgres_data:
```

### Kubernetes (Production)

The service is fully stateless — scale it horizontally freely. All state lives in Postgres + Redis.

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gea-service
spec:
  replicas: 3  # Scale based on concurrent evolution runs
  selector:
    matchLabels:
      app: gea-service
  template:
    metadata:
      labels:
        app: gea-service
    spec:
      containers:
      - name: gea-service
        image: your-registry/gea-service:latest
        ports:
        - containerPort: 50051  # gRPC
        - containerPort: 8080   # REST
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: gea-secrets
              key: anthropic-api-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gea-secrets
              key: openai-api-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gea-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 10
        readinessProbe:
          grpc:
            port: 50051
---
apiVersion: v1
kind: Service
metadata:
  name: gea-service
spec:
  selector:
    app: gea-service
  ports:
  - name: grpc
    port: 50051
    targetPort: 50051
  - name: rest
    port: 8080
    targetPort: 8080
```

---

## Critical Implementation Decisions

### 1. Serialization: JSON everywhere at the boundary

`serialized_config` and `experience_blob` are JSON at the service boundary. This means:
- Any language can inspect, construct, and debug agent configs without language-specific deserialization
- The service is truly agnostic to what an "agent" is internally
- The Python core uses Pydantic models; the boundary is always JSON

### 2. Meta-LLM lives server-side, always

The reflection step that generates evolution directives runs in the service. SDKs submit experience and receive patches. This pattern ensures:
- LLM API keys never leave the server
- Model upgrades happen without client SDK updates
- The reflection prompt and logic can be A/B tested without client changes

### 3. Probe tasks run client-side, results submit server-side

Languages register probe task *results* (binary outcomes) via the API — they don't run probes inside the service. This lets a Go agent run its own probe suite and submit the outcome vector. The service only sees the vector, not the tasks themselves. This is what makes the system truly cross-language.

### 4. Proto versioning discipline

Add a `version` field to every message now, even if unused. Never remove proto fields — deprecate them. gRPC's backward-compatibility rules let you add fields freely but removing them is breaking. Design the schema conservatively and version the API in the package path (`gea/v1`).

### 5. KNN novelty uses pgvector in production

For the novelty computation, pgvector's `ivfflat` index on the capability vector enables fast approximate KNN search even at thousands of agents. This keeps novelty computation in the database rather than pulling all vectors into memory.

### 6. Experience narrative is LLM-readable, not schema-bound

The `experience_narrative` field is a free-form text summary written by the meta-LLM after each iteration. Don't try to structure this — its value is precisely that it's richly descriptive and the next meta-LLM call can reason over it freely. The structured data (probe outcomes, patches) lives in typed fields; the narrative is the connective tissue.

---

## Monorepo Structure

```
gea/
├── packages/
│   ├── gea-core/          # Python library — pip install gea-core
│   │   ├── gea/
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   ├── gea-service/       # gRPC + REST service
│   │   ├── server.py
│   │   ├── grpc_handlers.py
│   │   ├── rest_router.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   │
│   ├── gea-py/            # Python SDK (service client)
│   │   ├── gea_py/
│   │   └── pyproject.toml
│   │
│   ├── gea-ts/            # TypeScript SDK
│   │   ├── src/
│   │   ├── generated/
│   │   └── package.json
│   │
│   ├── gea-go/            # Go SDK
│   │   ├── client.go
│   │   ├── gen/
│   │   └── go.mod
│   │
│   ├── gea-rs/            # Rust SDK
│   │   ├── src/
│   │   ├── build.rs
│   │   └── Cargo.toml
│   │
│   └── gea-java/          # Java SDK
│       ├── src/
│       └── pom.xml
│
├── proto/
│   └── gea/v1/gea.proto   # Source of truth — never edit generated files
│
├── migrations/            # SQL migration files
│   ├── 001_initial.sql
│   └── 002_add_lineage_views.sql
│
├── examples/
│   ├── swe-bench/         # Reproduce paper's SWE-bench results
│   └── polyglot/          # Reproduce Polyglot results
│
├── scripts/
│   └── gen-proto.sh       # Regenerates all SDK stubs from proto
│                          # Run this after any proto changes
├── k8s/                   # Kubernetes manifests
├── docker-compose.yml
└── README.md
```

### Proto Generation Script

**`scripts/gen-proto.sh`**
```bash
#!/usr/bin/env bash
set -e

PROTO_DIR="./proto"
PROTO_FILE="$PROTO_DIR/gea/v1/gea.proto"

echo "Generating Python..."
python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out=./packages/gea-service/generated \
  --grpc_python_out=./packages/gea-service/generated \
  "$PROTO_FILE"

echo "Generating TypeScript..."
protoc \
  -I "$PROTO_DIR" \
  --plugin=protoc-gen-ts=./node_modules/.bin/protoc-gen-ts \
  --ts_out=./packages/gea-ts/generated \
  --grpc_out=./packages/gea-ts/generated \
  "$PROTO_FILE"

echo "Generating Go..."
protoc \
  -I "$PROTO_DIR" \
  --go_out=./packages/gea-go/gen \
  --go-grpc_out=./packages/gea-go/gen \
  "$PROTO_FILE"

echo "Generating Rust..."
# Rust uses tonic-build via build.rs — triggered by cargo build
echo "  (handled by build.rs in gea-rs)"

echo "Generating Java..."
protoc \
  -I "$PROTO_DIR" \
  --java_out=./packages/gea-java/src/main/java \
  --grpc-java_out=./packages/gea-java/src/main/java \
  "$PROTO_FILE"

echo "Done. All stubs regenerated."
```

---

## Reproducing Paper Results

### SWE-bench Verified (71.0% target)

```python
# examples/swe-bench/run.py
import asyncio
from gea import EvolutionLoop, GEAConfig
from gea.llm import LLMRouter, LLMRole
from gea.llm.anthropic import AnthropicBackend
from gea.llm.openai import OpenAIBackend
from .swe_agent import SWEBenchAgent  # your SWE-bench agent implementation

async def main():
    config = GEAConfig(
        group_size=2,          # K=2 per paper
        novelty_neighbors=4,   # M=4 per paper
        max_iterations=30,     # 30 iterations on SWE-bench per paper
        alpha=0.5,
    )

    router = LLMRouter({
        LLMRole.ACTING:     AnthropicBackend(model="claude-haiku-4-5"),
        LLMRole.EVOLVING:   AnthropicBackend(model="claude-sonnet-4-6"),
        LLMRole.REFLECTING: OpenAIBackend(model="o1"),
    })

    loop = EvolutionLoop(
        agent_class=SWEBenchAgent,
        config=config,
        llm_router=router,
    )

    result = await loop.run()
    print(f"Final best: {result.best_performance:.1%}")
    print(f"Unique ancestors in best agent: {result.best_agent.unique_ancestor_count}")

asyncio.run(main())
```

### Polyglot (88.3% target)

```python
# examples/polyglot/run.py
config = GEAConfig(
    group_size=2,
    novelty_neighbors=4,
    max_iterations=20,   # 20 iterations on Polyglot per paper
    alpha=0.5,
)
# Otherwise identical — GEA adapts patch size to task complexity automatically
```

---

## Summary

The architecture is a four-layer system:

**`gea-core`** (Python) holds all the algorithmic logic — agent abstraction, selection criterion, experience pool, group reproduction, LLM routing. It's installable standalone and also runs as the engine behind the service.

**`gea-service`** exposes the core over gRPC (primary) and REST (fallback). It is stateless and horizontally scalable. The meta-LLM reflection step lives here. All LLM API keys stay server-side.

**Language SDKs** are thin gRPC clients (~500–1000 lines each, mostly generated). The proto file is the cross-language contract. Any language with a gRPC library can participate — agents in Go, TypeScript, Rust, Java all submit experience and receive patches through the same protocol.

**Postgres + pgvector + Redis** hold all state. pgvector's `ivfflat` index enables fast KNN novelty computation. Redis caches hot reads. Postgres provides durable, queryable evolutionary history — the full 17-ancestor lineage the paper describes is queryable with a single SQL query.

The key design principle throughout: **the service doesn't need to understand what an agent is or does**. It only orchestrates the group evolution logic. Agent configs are opaque JSON blobs. Experience narratives are free-form text. The service's job is selection, aggregation, and patch generation — everything else is the agent's problem.
