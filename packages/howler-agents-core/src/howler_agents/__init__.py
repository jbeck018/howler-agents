"""Howler Agents - Group-Evolving Agents core library."""

__all__ = ["EvolutionLoop", "HowlerConfig"]
__version__ = "0.3.1"


def __getattr__(name: str):
    """Lazy imports — avoid loading numpy/sklearn/scipy on CLI startup."""
    if name == "HowlerConfig":
        from howler_agents.config import HowlerConfig

        return HowlerConfig
    if name == "EvolutionLoop":
        from howler_agents.evolution.loop import EvolutionLoop

        return EvolutionLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
