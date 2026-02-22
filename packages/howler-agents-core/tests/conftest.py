"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make _helpers importable from test files
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import MockAgent, make_agent  # noqa: E402

from howler_agents.agents.base import AgentConfig  # noqa: E402
from howler_agents.config import HowlerConfig  # noqa: E402
from howler_agents.experience.store.memory import InMemoryStore  # noqa: E402

# Re-export for backward compat
__all__ = ["MockAgent", "make_agent", "AgentConfig"]


@pytest.fixture
def config() -> HowlerConfig:
    return HowlerConfig(population_size=6, group_size=3, num_iterations=2, alpha=0.5, num_probes=5)


@pytest.fixture
def memory_store() -> InMemoryStore:
    return InMemoryStore()
