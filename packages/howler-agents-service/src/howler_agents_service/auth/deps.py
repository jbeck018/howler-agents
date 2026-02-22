"""FastAPI auth dependencies."""

from __future__ import annotations

from datetime import UTC
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from howler_agents_service.auth.jwt import decode_token
from howler_agents_service.auth.models import CurrentUser
from howler_agents_service.auth.passwords import verify_password
from howler_agents_service.db.deps import SessionDep
from howler_agents_service.db.models import ApiKeyModel, OrgMemberModel, UserModel


async def _user_from_token(token: str, session: AsyncSession) -> CurrentUser:
    """Decode a Bearer JWT and return the CurrentUser."""
    try:
        payload = decode_token(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Not an access token")

    try:
        user_id = UUID(payload["sub"])
        org_id = UUID(payload["org"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Malformed token payload") from exc

    return CurrentUser(
        user_id=user_id,
        org_id=org_id,
        email=payload.get("email", ""),
        role=payload.get("role", "member"),
    )


async def _user_from_api_key(raw_key: str, session: AsyncSession) -> CurrentUser:
    """Validate an API key and return the CurrentUser."""
    if not raw_key.startswith("ha_live_"):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    # The prefix used for lookup is the 8 chars immediately after "ha_live_"
    # so the full prefix stored is the first 16 chars of the raw key
    key_prefix = raw_key[:16]

    from sqlalchemy import select

    result = await session.execute(
        select(ApiKeyModel).where(
            ApiKeyModel.key_prefix == key_prefix,
            ApiKeyModel.revoked_at.is_(None),
        )
    )
    api_key_row = result.scalars().first()

    if api_key_row is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check expiry
    from datetime import datetime

    if api_key_row.expires_at and api_key_row.expires_at < datetime.now(UTC):
        raise HTTPException(status_code=401, detail="API key expired")

    if not verify_password(raw_key, api_key_row.key_hash):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Fetch the org membership to get role
    member_result = await session.execute(
        select(OrgMemberModel).where(
            OrgMemberModel.org_id == api_key_row.org_id,
        )
    )
    member = member_result.scalars().first()
    role = member.role if member else "member"

    # Fetch one user in the org to satisfy CurrentUser (API keys are org-level)
    user_result = await session.execute(
        select(UserModel)
        .join(OrgMemberModel, OrgMemberModel.user_id == UserModel.id)
        .where(OrgMemberModel.org_id == api_key_row.org_id)
        .limit(1)
    )
    user = user_result.scalars().first()
    if user is None:
        raise HTTPException(status_code=401, detail="No user associated with API key org")

    return CurrentUser(
        user_id=user.id,
        org_id=api_key_row.org_id,
        email=user.email,
        role=role,
    )


async def get_current_user(request: Request, session: SessionDep) -> CurrentUser:
    """
    Resolve the current authenticated user.

    Checks Authorization: Bearer <token> first, then X-API-Key header.
    On success, sets app.current_org_id on the DB session for RLS.
    """
    auth_header = request.headers.get("Authorization", "")
    api_key_header = request.headers.get("X-API-Key", "")

    current_user: CurrentUser | None = None

    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        current_user = await _user_from_token(token, session)
    elif api_key_header:
        current_user = await _user_from_api_key(api_key_header, session)
    else:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Set the org context for Row-Level Security.
    # Using SET (session-scoped) instead of SET LOCAL (transaction-scoped)
    # so the setting persists across commit/refresh cycles within the same
    # request. Each request gets its own DB connection from the pool, so
    # this is safe — the setting is reset when the connection is returned.
    # The org_id is always a UUID from a validated JWT, never user input.
    org_id_str = str(current_user.org_id)
    await session.execute(text(f"SET app.current_org_id = '{org_id_str}'"))

    return current_user


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]


def require_role(*roles: str):
    """Dependency factory that enforces role membership."""

    async def _check(current_user: CurrentUserDep) -> CurrentUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{current_user.role}' is not permitted. Required: {list(roles)}",
            )
        return current_user

    return Depends(_check)
