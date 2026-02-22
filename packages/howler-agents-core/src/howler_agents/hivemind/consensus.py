"""Consensus engine: propose, vote, and resolve hive-mind decisions."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from howler_agents.hivemind.memory import CollectiveMemory
from howler_agents.persistence.db import DatabaseManager

log = structlog.get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ConsensusEngine:
    """Propose, vote on, and resolve consensus items.

    Accepted proposals (confidence >= 0.6) are promoted to collective memory
    under the ``consensus`` namespace.
    """

    _ACCEPT_THRESHOLD = 0.6

    def __init__(self, db: DatabaseManager, memory: CollectiveMemory) -> None:
        self._db = db
        self._memory = memory

    async def propose(
        self,
        topic: str,
        proposal: str,
        evidence: list[str] | None = None,
        run_id: str | None = None,
        generation: int = 0,
    ) -> str:
        """Create a new consensus proposal and return its ID."""
        consensus_id = str(uuid.uuid4())
        initial_vote = {"direction": "for", "evidence": evidence or []}
        votes_json = json.dumps([initial_vote])

        await self._db.execute_write(
            """
            INSERT INTO consensus (id, topic, decision, votes, confidence, generation, run_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                consensus_id,
                topic,
                proposal,
                votes_json,
                0.0,
                generation,
                run_id,
                _now_iso(),
            ),
        )

        log.info("consensus_proposed", id=consensus_id, topic=topic)
        return consensus_id

    async def vote(
        self,
        consensus_id: str,
        direction: str,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        """Cast a vote ('for' or 'against') on an existing consensus item.

        Returns the updated vote tallies and recalculated confidence.
        """
        if direction not in {"for", "against"}:
            raise ValueError(f"direction must be 'for' or 'against', got {direction!r}")

        rows = await self._db.execute("SELECT * FROM consensus WHERE id = ?", (consensus_id,))
        if not rows:
            raise LookupError(f"Consensus item {consensus_id!r} not found")

        item = rows[0]
        votes: list[dict[str, Any]] = json.loads(item["votes"])
        votes.append({"direction": direction, "evidence": evidence or []})

        votes_for = sum(1 for v in votes if v["direction"] == "for")
        votes_against = sum(1 for v in votes if v["direction"] == "against")
        total = votes_for + votes_against
        confidence = votes_for / total if total else 0.0

        await self._db.execute_write(
            "UPDATE consensus SET votes = ?, confidence = ? WHERE id = ?",
            (json.dumps(votes), confidence, consensus_id),
        )

        log.debug(
            "consensus_voted",
            id=consensus_id,
            direction=direction,
            for_=votes_for,
            against=votes_against,
            confidence=confidence,
        )
        return {
            "consensus_id": consensus_id,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "confidence": confidence,
        }

    async def resolve(self, consensus_id: str) -> dict[str, Any]:
        """Resolve a consensus item based on current votes.

        Items with confidence >= 0.6 are accepted and promoted to collective
        memory. Returns the resolution outcome.
        """
        rows = await self._db.execute("SELECT * FROM consensus WHERE id = ?", (consensus_id,))
        if not rows:
            raise LookupError(f"Consensus item {consensus_id!r} not found")

        item = rows[0]
        accepted = item["confidence"] >= self._ACCEPT_THRESHOLD

        if accepted:
            await self._memory.store(
                namespace="consensus",
                key=item["topic"],
                value=item["decision"],
                score=item["confidence"],
                tags=["consensus-accepted"],
            )
            log.info("consensus_accepted", id=consensus_id, topic=item["topic"])
        else:
            log.info("consensus_rejected", id=consensus_id, topic=item["topic"])

        return {
            "consensus_id": consensus_id,
            "topic": item["topic"],
            "accepted": accepted,
            "confidence": item["confidence"],
        }

    async def auto_resolve_all(self, min_votes: int = 3) -> list[dict[str, Any]]:
        """Auto-resolve all items that have accumulated at least *min_votes* votes."""
        rows = await self._db.execute("SELECT * FROM consensus")

        results: list[dict[str, Any]] = []
        for item in rows:
            votes = json.loads(item["votes"])
            if len(votes) >= min_votes:
                outcome = await self.resolve(item["id"])
                results.append(outcome)

        log.info("auto_resolve_complete", resolved=len(results))
        return results

    async def list(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List consensus items, optionally filtered by status.

        Status is derived from confidence: 'pending' means no votes yet or
        unresolved; this method returns all rows ordered by creation time.
        """
        if status == "pending":
            rows = await self._db.execute(
                "SELECT * FROM consensus WHERE confidence = 0.0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        else:
            rows = await self._db.execute(
                "SELECT * FROM consensus ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return rows
