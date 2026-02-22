-- Howler Agents: Initial schema
-- Requires: PostgreSQL 16+ with pgvector extension

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Evolution runs
CREATE TABLE evolution_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    current_generation INTEGER NOT NULL DEFAULT 0,
    total_generations INTEGER NOT NULL,
    best_agent_id UUID,
    best_score DOUBLE PRECISION DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent groups
CREATE TABLE agent_groups (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES evolution_runs(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    group_performance DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_groups_run_id ON agent_groups(run_id);
CREATE INDEX idx_agent_groups_generation ON agent_groups(run_id, generation);

-- Agents
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES evolution_runs(id) ON DELETE CASCADE,
    group_id UUID REFERENCES agent_groups(id) ON DELETE SET NULL,
    generation INTEGER NOT NULL,
    parent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    performance_score DOUBLE PRECISION DEFAULT 0,
    novelty_score DOUBLE PRECISION DEFAULT 0,
    combined_score DOUBLE PRECISION DEFAULT 0,
    capability_vector VECTOR(64),
    framework_config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agents_run_id ON agents(run_id);
CREATE INDEX idx_agents_group_id ON agents(group_id);
CREATE INDEX idx_agents_generation ON agents(run_id, generation);
CREATE INDEX idx_agents_combined_score ON agents(run_id, combined_score DESC);

-- IVFFlat index for KNN novelty search on capability vectors
CREATE INDEX idx_agents_capability_vector ON agents
    USING ivfflat (capability_vector vector_l2_ops) WITH (lists = 10);

-- Evolutionary traces (experience records)
CREATE TABLE evolutionary_traces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES evolution_runs(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    task_description TEXT NOT NULL,
    outcome TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    key_decisions TEXT[] DEFAULT '{}',
    lessons_learned TEXT[] DEFAULT '{}',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_traces_agent_id ON evolutionary_traces(agent_id);
CREATE INDEX idx_traces_run_id ON evolutionary_traces(run_id);
CREATE INDEX idx_traces_generation ON evolutionary_traces(run_id, generation);

-- Framework patches (mutations)
CREATE TABLE framework_patches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    intent TEXT NOT NULL,
    diff TEXT NOT NULL,
    category TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_patches_agent_id ON framework_patches(agent_id);

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_evolution_runs_updated_at
    BEFORE UPDATE ON evolution_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
