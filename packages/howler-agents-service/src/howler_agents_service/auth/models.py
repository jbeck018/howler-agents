"""Auth domain models."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass
class CurrentUser:
    user_id: UUID
    org_id: UUID
    email: str
    role: str  # "owner" | "admin" | "member"
