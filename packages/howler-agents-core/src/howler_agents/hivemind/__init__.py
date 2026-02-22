"""Hive-mind package: collective memory, consensus engine, and coordinator."""

from __future__ import annotations

from howler_agents.hivemind.consensus import ConsensusEngine
from howler_agents.hivemind.coordinator import HiveMindCoordinator
from howler_agents.hivemind.memory import CollectiveMemory

__all__ = [
    "CollectiveMemory",
    "ConsensusEngine",
    "HiveMindCoordinator",
]
