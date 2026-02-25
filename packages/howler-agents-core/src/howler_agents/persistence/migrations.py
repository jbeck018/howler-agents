"""Schema migrations for the howler-agents SQLite database."""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 2

_DDL_STATEMENTS = [
    # Schema version tracking
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # Evolution runs
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id              TEXT PRIMARY KEY,
        status              TEXT NOT NULL DEFAULT 'pending',
        task_domain         TEXT NOT NULL DEFAULT 'general',
        population_size     INTEGER NOT NULL DEFAULT 10,
        group_size          INTEGER NOT NULL DEFAULT 3,
        num_iterations      INTEGER NOT NULL DEFAULT 5,
        alpha               REAL NOT NULL DEFAULT 0.5,
        model               TEXT NOT NULL DEFAULT 'claude-sonnet-4-20250514',
        current_generation  INTEGER NOT NULL DEFAULT 0,
        best_score          REAL NOT NULL DEFAULT 0.0,
        best_agent_id       TEXT,
        mean_score          REAL NOT NULL DEFAULT 0.0,
        generation_summaries TEXT NOT NULL DEFAULT '[]',
        started_at          TEXT NOT NULL,
        finished_at         TEXT,
        error               TEXT
    )
    """,
    # Agents
    """
    CREATE TABLE IF NOT EXISTS agents (
        agent_id            TEXT PRIMARY KEY,
        run_id              TEXT NOT NULL REFERENCES runs(run_id),
        generation          INTEGER NOT NULL DEFAULT 0,
        parent_id           TEXT,
        group_id            TEXT,
        framework_config    TEXT NOT NULL DEFAULT '{}',
        performance_score   REAL NOT NULL DEFAULT 0.0,
        novelty_score       REAL NOT NULL DEFAULT 0.0,
        combined_score      REAL NOT NULL DEFAULT 0.0,
        capability_vector   TEXT NOT NULL DEFAULT '[]',
        patches_count       INTEGER NOT NULL DEFAULT 0,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # Evolutionary traces (experience pool)
    """
    CREATE TABLE IF NOT EXISTS traces (
        id                  TEXT PRIMARY KEY,
        agent_id            TEXT NOT NULL,
        run_id              TEXT NOT NULL REFERENCES runs(run_id),
        group_id            TEXT NOT NULL DEFAULT '',
        generation          INTEGER NOT NULL DEFAULT 0,
        task_description    TEXT NOT NULL DEFAULT '',
        outcome             TEXT NOT NULL DEFAULT '',
        score               REAL NOT NULL DEFAULT 0.0,
        key_decisions       TEXT NOT NULL DEFAULT '[]',
        lessons_learned     TEXT NOT NULL DEFAULT '[]',
        patches_applied     TEXT NOT NULL DEFAULT '[]',
        parent_ids          TEXT NOT NULL DEFAULT '[]',
        recorded_at         TEXT NOT NULL
    )
    """,
    # Collective memory
    """
    CREATE TABLE IF NOT EXISTS memory (
        id              TEXT PRIMARY KEY,
        namespace       TEXT NOT NULL DEFAULT 'default',
        key             TEXT NOT NULL,
        value           TEXT NOT NULL,
        embedding       TEXT,
        tags            TEXT NOT NULL DEFAULT '[]',
        score           REAL NOT NULL DEFAULT 0.0,
        access_count    INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
        expires_at      TEXT,
        UNIQUE(namespace, key)
    )
    """,
    # Consensus records
    """
    CREATE TABLE IF NOT EXISTS consensus (
        id          TEXT PRIMARY KEY,
        topic       TEXT NOT NULL,
        decision    TEXT NOT NULL,
        votes       TEXT NOT NULL DEFAULT '[]',
        confidence  REAL NOT NULL DEFAULT 0.0,
        generation  INTEGER NOT NULL DEFAULT 0,
        run_id      TEXT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_traces_run        ON traces(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_traces_agent      ON traces(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_traces_generation ON traces(run_id, generation)",
    "CREATE INDEX IF NOT EXISTS idx_agents_run        ON agents(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_namespace  ON memory(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_memory_key        ON memory(namespace, key)",
    "CREATE INDEX IF NOT EXISTS idx_consensus_topic   ON consensus(topic)",
]


_V2_MIGRATIONS = [
    # Add group_id column to existing traces tables (no-op on fresh databases)
    "ALTER TABLE traces ADD COLUMN group_id TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_traces_group ON traces(run_id, group_id)",
]


async def run_migrations(db: object) -> None:  # db: DatabaseManager (avoid circular import)
    """Create tables and indexes, then record the schema version."""
    from howler_agents.persistence.db import DatabaseManager

    assert isinstance(db, DatabaseManager)

    for statement in _DDL_STATEMENTS:
        await db.execute_write(statement.strip())

    # Check current schema version
    rows = await db.execute(
        "SELECT MAX(version) AS v FROM schema_version",
    )
    current_version = rows[0]["v"] if rows and rows[0]["v"] is not None else 0

    # Apply incremental migrations
    if current_version < 2:
        for stmt in _V2_MIGRATIONS:
            try:
                await db.execute_write(stmt)
            except Exception:
                pass  # Column already exists on fresh databases

    if current_version < SCHEMA_VERSION:
        await db.execute_write(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        log.info("migration_applied", version=SCHEMA_VERSION)
    else:
        log.debug("schema_already_current", version=SCHEMA_VERSION)
