"""Sync client: push/pull between local SQLite and a remote Postgres-backed API."""

from __future__ import annotations

from typing import Any

import structlog

from howler_agents.persistence.db import DatabaseManager

log = structlog.get_logger(__name__)


class SyncClient:
    """Push/pull between local SQLite and a remote Howler API.

    Push: completed runs (run + agents + traces) to the remote.
    Pull: hive-mind memory from the remote into local SQLite.
    """

    def __init__(self, api_url: str, api_key: str | None = None) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def push_run(self, run_id: str, db: DatabaseManager) -> dict[str, Any]:
        """Push a completed run (run + agents + traces) to the remote API."""
        import httpx

        run_data = await db.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        )
        if not run_data:
            return {"error": f"Run {run_id} not found"}
        if run_data[0]["status"] != "completed":
            return {
                "error": (
                    f"Run {run_id} is not completed"
                    f" (status={run_data[0]['status']})"
                )
            }

        agents = await db.execute(
            "SELECT * FROM agents WHERE run_id = ?", (run_id,)
        )
        traces = await db.execute(
            "SELECT * FROM traces WHERE run_id = ?", (run_id,)
        )

        payload: dict[str, Any] = {
            "run": run_data[0],
            "agents": agents,
            "traces": traces,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._api_url}/api/v1/sync/runs",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()

        log.info("push_run_complete", run_id=run_id, agents=len(agents), traces=len(traces))
        return result

    async def push_memory(
        self, db: DatabaseManager, namespace: str = "default"
    ) -> dict[str, Any]:
        """Push local memory entries for *namespace* to the remote API."""
        import httpx

        entries = await db.execute(
            "SELECT * FROM memory WHERE namespace = ?", (namespace,)
        )
        if not entries:
            log.debug("push_memory_empty", namespace=namespace)
            return {"pushed": 0}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._api_url}/api/v1/sync/memory",
                json={"entries": entries},
                headers=self._headers(),
            )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()

        log.info("push_memory_complete", namespace=namespace, entries=len(entries))
        return result

    async def pull_memory(
        self, db: DatabaseManager, namespace: str = "default"
    ) -> dict[str, Any]:
        """Pull memory entries from the remote and merge them into local SQLite.

        Uses a score-wins strategy: remote values only replace local ones when
        the remote score is higher.
        """
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self._api_url}/api/v1/sync/memory",
                params={"namespace": namespace},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        imported = 0
        for entry in data.get("entries", []):
            await db.execute_write(
                """
                INSERT INTO memory (id, namespace, key, value, tags, score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value      = CASE WHEN excluded.score > memory.score
                                      THEN excluded.value
                                      ELSE memory.value END,
                    score      = MAX(memory.score, excluded.score),
                    updated_at = excluded.updated_at
                """,
                (
                    entry["id"],
                    entry["namespace"],
                    entry["key"],
                    entry["value"],
                    entry.get("tags", "[]"),
                    entry.get("score", 0.0),
                    entry.get("created_at", ""),
                    entry.get("updated_at", ""),
                ),
            )
            imported += 1

        log.info("pull_memory_complete", namespace=namespace, imported=imported)
        return {"pulled": imported, "namespace": namespace}

    async def status(self) -> dict[str, Any]:
        """Check connectivity to the remote API."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._api_url}/health")
            connected = resp.status_code == 200
            log.debug("sync_status_ok", url=self._api_url, connected=connected)
            return {"connected": connected, "url": self._api_url}
        except Exception as exc:
            log.warning("sync_status_failed", url=self._api_url, error=str(exc))
            return {"connected": False, "url": self._api_url, "error": str(exc)}
