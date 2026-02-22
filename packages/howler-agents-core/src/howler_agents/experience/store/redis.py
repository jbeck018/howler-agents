"""Redis hot-cache experience store."""

from __future__ import annotations

import json
from datetime import timezone
from typing import Any

from howler_agents.experience.trace import EvolutionaryTrace


class RedisStore:
    """Hot cache experience store backed by Redis."""

    def __init__(self, redis_client: Any, prefix: str = "howler:traces") -> None:
        self._redis = redis_client
        self._prefix = prefix

    def _key(self, *parts: str) -> str:
        return ":".join([self._prefix, *parts])

    def _serialize(self, trace: EvolutionaryTrace) -> str:
        return json.dumps({
            "id": trace.id,
            "agent_id": trace.agent_id,
            "run_id": trace.run_id,
            "generation": trace.generation,
            "task_description": trace.task_description,
            "outcome": trace.outcome,
            "score": trace.score,
            "key_decisions": trace.key_decisions,
            "lessons_learned": trace.lessons_learned,
            "recorded_at": trace.recorded_at.isoformat(),
        })

    def _deserialize(self, data: str) -> EvolutionaryTrace:
        from datetime import datetime
        d = json.loads(data)
        d["recorded_at"] = datetime.fromisoformat(d["recorded_at"]).replace(tzinfo=timezone.utc)
        return EvolutionaryTrace(**d)

    async def save(self, trace: EvolutionaryTrace) -> None:
        data = self._serialize(trace)
        pipe = self._redis.pipeline()
        pipe.hset(self._key("by_id"), trace.id, data)
        pipe.lpush(self._key("agent", trace.agent_id), data)
        pipe.lpush(self._key("run", trace.run_id), data)
        pipe.lpush(self._key("gen", trace.run_id, str(trace.generation)), data)
        await pipe.execute()

    async def get_by_agent(self, agent_id: str) -> list[EvolutionaryTrace]:
        items = await self._redis.lrange(self._key("agent", agent_id), 0, -1)
        return [self._deserialize(item) for item in items]

    async def get_by_run(self, run_id: str, limit: int = 100) -> list[EvolutionaryTrace]:
        items = await self._redis.lrange(self._key("run", run_id), 0, limit - 1)
        return [self._deserialize(item) for item in items]

    async def get_by_generation(self, run_id: str, generation: int) -> list[EvolutionaryTrace]:
        items = await self._redis.lrange(self._key("gen", run_id, str(generation)), 0, -1)
        return [self._deserialize(item) for item in items]

    async def delete_by_run(self, run_id: str) -> int:
        key = self._key("run", run_id)
        count = await self._redis.llen(key)
        await self._redis.delete(key)
        return count
