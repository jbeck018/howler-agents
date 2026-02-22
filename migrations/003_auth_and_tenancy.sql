-- Howler Agents: Auth and multi-tenancy
-- Migration 003

-- Organizations (tenants)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Organization membership with role
CREATE TABLE org_members (
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
    PRIMARY KEY (org_id, user_id)
);

-- API keys (only hash + prefix stored, full key returned once at creation)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL,
    key_prefix TEXT NOT NULL,  -- first 8 chars for lookup
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_org_id ON api_keys(org_id);
CREATE INDEX idx_api_keys_key_prefix ON api_keys(key_prefix);

-- Add org_id to existing tables for tenant isolation
ALTER TABLE evolution_runs ADD COLUMN org_id UUID REFERENCES organizations(id);
ALTER TABLE agents ADD COLUMN org_id UUID REFERENCES organizations(id);
ALTER TABLE evolutionary_traces ADD COLUMN org_id UUID REFERENCES organizations(id);

CREATE INDEX idx_evolution_runs_org_id ON evolution_runs(org_id);
CREATE INDEX idx_agents_org_id ON agents(org_id);
CREATE INDEX idx_evolutionary_traces_org_id ON evolutionary_traces(org_id);

-- Row-Level Security for tenant isolation
-- FORCE is required so RLS applies to the table owner too
ALTER TABLE evolution_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE evolution_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents FORCE ROW LEVEL SECURITY;
ALTER TABLE evolutionary_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE evolutionary_traces FORCE ROW LEVEL SECURITY;

-- Use current_setting(..., true) to return NULL instead of error when unset.
-- Short-circuit AND prevents the UUID cast on empty/NULL values.
CREATE POLICY tenant_isolation_runs ON evolution_runs
    USING (
        current_setting('app.current_org_id', true) IS NOT NULL
        AND current_setting('app.current_org_id', true) != ''
        AND org_id = current_setting('app.current_org_id', true)::uuid
    );

CREATE POLICY tenant_isolation_agents ON agents
    USING (
        current_setting('app.current_org_id', true) IS NOT NULL
        AND current_setting('app.current_org_id', true) != ''
        AND org_id = current_setting('app.current_org_id', true)::uuid
    );

CREATE POLICY tenant_isolation_traces ON evolutionary_traces
    USING (
        current_setting('app.current_org_id', true) IS NOT NULL
        AND current_setting('app.current_org_id', true) != ''
        AND org_id = current_setting('app.current_org_id', true)::uuid
    );
