"""JWT token creation and verification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from howler_agents_service.settings import settings


def _now_utc() -> datetime:
    return datetime.now(UTC)


def create_access_token(
    user_id: UUID,
    org_id: UUID,
    email: str,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    now = _now_utc()
    payload = {
        "sub": str(user_id),
        "org": str(org_id),
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + expires_delta,
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    user_id: UUID,
    org_id: UUID,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT refresh token."""
    if expires_delta is None:
        expires_delta = timedelta(days=settings.refresh_token_expire_days)
    now = _now_utc()
    payload = {
        "sub": str(user_id),
        "org": str(org_id),
        "iat": now,
        "exp": now + expires_delta,
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises jwt.PyJWTError on failure."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
