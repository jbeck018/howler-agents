"""Top-level hive-mind coordinator: orchestrates memory and consensus."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import structlog

from howler_agents.hivemind.consensus import ConsensusEngine
from howler_agents.hivemind.memory import CollectiveMemory
from howler_agents.persistence.db import DatabaseManager

log = structlog.get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class HiveMindCoordinator:
    """Top-level coordinator for hive-mind operations.

    Connects :class:`CollectiveMemory` and :class:`ConsensusEngine` and
    provides high-level operations such as seeding from evolution runs.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self.memory = CollectiveMemory(db)
        self.consensus = ConsensusEngine(db, self.memory)

    async def status(self) -> dict[str, Any]:
        """Return hive-mind status: memory stats, pending consensus, run history."""
        mem_stats = await self.memory.stats()
        pending = await self.consensus.list(status="pending")

        runs = await self._db.execute(
            "SELECT COUNT(*) as count, MAX(best_score) as best FROM runs WHERE status = 'completed'"
        )

        return {
            "memory": mem_stats,
            "pending_consensus": len(pending),
            "completed_runs": runs[0]["count"] if runs else 0,
            "best_score_ever": runs[0]["best"] if runs else 0.0,
        }

    async def seed_from_run(self, run_id: str) -> dict[str, Any]:
        """Extract high-value lessons and decisions from a completed run.

        Scans traces for the run, identifies frequently-occurring lessons and
        decisions, stores them as memory entries, and auto-proposes consensus
        items for patterns that appear in three or more traces.
        """
        traces = await self._db.execute(
            "SELECT * FROM traces WHERE run_id = ? ORDER BY score DESC",
            (run_id,),
        )
        if not traces:
            log.warning("seed_from_run_no_traces", run_id=run_id)
            return {"seeded": 0, "proposed": 0}

        lessons: dict[str, float] = {}
        decisions: dict[str, float] = {}

        for t in traces:
            for lesson in json.loads(t["lessons_learned"]):
                if lesson and (lesson not in lessons or t["score"] > lessons[lesson]):
                    lessons[lesson] = t["score"]
            for decision in json.loads(t["key_decisions"]):
                if decision and (decision not in decisions or t["score"] > decisions[decision]):
                    decisions[decision] = t["score"]

        seeded = 0

        for lesson, score in sorted(lessons.items(), key=lambda x: -x[1])[:10]:
            await self.memory.store(
                namespace="lessons",
                key=f"run_{run_id[:8]}_{seeded}",
                value=lesson,
                score=score,
                tags=["auto-seeded", run_id[:8]],
            )
            seeded += 1

        for decision, score in sorted(decisions.items(), key=lambda x: -x[1])[:5]:
            await self.memory.store(
                namespace="decisions",
                key=f"run_{run_id[:8]}_{seeded}",
                value=decision,
                score=score,
                tags=["auto-seeded", run_id[:8]],
            )
            seeded += 1

        lesson_counts: Counter[str] = Counter()
        for t in traces:
            for lesson in json.loads(t["lessons_learned"]):
                if lesson:
                    lesson_counts[lesson] += 1

        proposed = 0
        for lesson, count in lesson_counts.most_common(5):
            if count >= 3:
                await self.consensus.propose(
                    topic=f"Adopt lesson: {lesson[:50]}",
                    proposal=lesson,
                    evidence=[run_id],
                    run_id=run_id,
                )
                proposed += 1

        log.info("seed_from_run_complete", run_id=run_id, seeded=seeded, proposed=proposed)
        return {"seeded": seeded, "proposed": proposed}

    async def reset(self) -> dict[str, Any]:
        """Clear all hive-mind state (memory + consensus). Destructive."""
        mem_count = await self._db.execute_write("DELETE FROM memory")
        con_count = await self._db.execute_write("DELETE FROM consensus")
        log.warning("hivemind_reset", memory_deleted=mem_count, consensus_deleted=con_count)
        return {"memory_deleted": mem_count, "consensus_deleted": con_count}

    async def export_json(self) -> dict[str, Any]:
        """Export all hive-mind data as a JSON-serialisable dict."""
        memories = await self._db.execute("SELECT * FROM memory ORDER BY namespace, key")
        consensus_items = await self._db.execute("SELECT * FROM consensus ORDER BY created_at")
        return {"memory": memories, "consensus": consensus_items}

    async def import_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """Import hive-mind data from a JSON dict. Uses UPSERT for memory entries."""
        imported_mem = 0
        imported_con = 0

        for entry in data.get("memory", []):
            raw_tags = entry.get("tags", "[]")
            tags = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
            await self.memory.store(
                namespace=entry["namespace"],
                key=entry["key"],
                value=entry["value"],
                tags=tags,
                score=entry.get("score", 0.0),
            )
            imported_mem += 1

        for item in data.get("consensus", []):
            await self._db.execute_write(
                """
                INSERT OR IGNORE INTO consensus
                    (id, topic, decision, votes, confidence, generation, run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["id"],
                    item["topic"],
                    item["decision"],
                    item.get("votes", "[]"),
                    item.get("confidence", 0.0),
                    item.get("generation", 0),
                    item.get("run_id"),
                    item.get("created_at", _now_iso()),
                ),
            )
            imported_con += 1

        log.info(
            "hivemind_import_complete",
            memory_imported=imported_mem,
            consensus_imported=imported_con,
        )
        return {"memory_imported": imported_mem, "consensus_imported": imported_con}
