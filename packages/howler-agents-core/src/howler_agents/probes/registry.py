"""Probe task registry."""

from __future__ import annotations

from typing import Any


class ProbeRegistry:
    """Registry for probe tasks used in capability vector computation."""

    def __init__(self) -> None:
        self._probes: list[dict[str, Any]] = []

    def register(self, probe: dict[str, Any]) -> None:
        """Register a probe task."""
        self._probes.append(probe)

    def get_probes(self) -> list[dict[str, Any]]:
        """Return all registered probes."""
        return list(self._probes)

    @property
    def count(self) -> int:
        return len(self._probes)

    def register_default_probes(self, num_probes: int = 20) -> None:
        """Register a set of default probe tasks for general capability assessment."""
        default_probes = [
            {"description": "Simple arithmetic: compute 2+2", "type": "arithmetic", "expected": "4"},
            {"description": "String reversal: reverse 'hello'", "type": "string_ops", "expected": "olleh"},
            {"description": "List sorting: sort [3,1,2]", "type": "sorting", "expected": "[1,2,3]"},
            {"description": "JSON parsing: extract 'name' from {\"name\": \"test\"}", "type": "json", "expected": "test"},
            {"description": "Error handling: handle division by zero", "type": "error_handling"},
            {"description": "Code generation: write a fibonacci function", "type": "code_gen"},
            {"description": "Debugging: find the bug in off-by-one loop", "type": "debugging"},
            {"description": "Refactoring: simplify nested conditionals", "type": "refactoring"},
            {"description": "Testing: write a unit test for add(a,b)", "type": "testing"},
            {"description": "Documentation: write docstring for sort function", "type": "documentation"},
            {"description": "API design: design a REST endpoint for users", "type": "api_design"},
            {"description": "Data transformation: convert CSV to JSON", "type": "data_transform"},
            {"description": "Pattern matching: extract emails from text", "type": "regex"},
            {"description": "Concurrency: identify race condition", "type": "concurrency"},
            {"description": "Security: identify SQL injection vulnerability", "type": "security"},
            {"description": "Performance: optimize O(n^2) to O(n log n)", "type": "optimization"},
            {"description": "Architecture: suggest microservice boundary", "type": "architecture"},
            {"description": "Git: resolve merge conflict", "type": "version_control"},
            {"description": "Database: write an indexed query", "type": "database"},
            {"description": "Deployment: write a Dockerfile", "type": "devops"},
        ]
        for probe in default_probes[:num_probes]:
            self.register(probe)
