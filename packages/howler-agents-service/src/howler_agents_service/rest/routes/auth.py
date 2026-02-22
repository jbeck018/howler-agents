"""Auth endpoints: register, login, refresh, API keys, /me."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from howler_agents_service.auth.deps import CurrentUserDep
from howler_agents_service.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from howler_agents_service.auth.passwords import verify_password
from howler_agents_service.db.deps import AuthRepoDep, SessionDep

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str | None = None
    org_name: str
    org_slug: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("org_slug")
    @classmethod
    def slug_format(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError("Slug must contain only lowercase letters, digits, and hyphens")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class CreateApiKeyRequest(BaseModel):
    name: str
    expires_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key: str  # returned only once
    key_prefix: str
    created_at: datetime | None = None


class MeResponse(BaseModel):
    user_id: str
    org_id: str
    email: str
    role: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    request: RegisterRequest, repo: AuthRepoDep, session: SessionDep
) -> TokenResponse:
    """Create a new user and organization, returning JWT tokens."""
    # Check for existing email
    existing = await repo.get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Check for slug collision
    existing_org = await repo.get_org_by_slug(request.org_slug)
    if existing_org:
        raise HTTPException(status_code=409, detail="Organization slug already taken")

    org = await repo.create_org(name=request.org_name, slug=request.org_slug)
    user = await repo.create_user(
        email=request.email,
        password=request.password,
        display_name=request.display_name,
    )
    await repo.add_member(org_id=org.id, user_id=user.id, role="owner")
    await session.commit()

    access = create_access_token(user_id=user.id, org_id=org.id, email=user.email, role="owner")
    refresh = create_refresh_token(user_id=user.id, org_id=org.id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, repo: AuthRepoDep) -> TokenResponse:
    """Verify credentials and return JWT tokens."""
    user = await repo.get_user_by_email(request.email)
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    member = await repo.get_member_by_user(user.id)
    if not member:
        raise HTTPException(status_code=403, detail="User has no organization membership")

    access = create_access_token(
        user_id=user.id,
        org_id=member.org_id,
        email=user.email,
        role=member.role,
    )
    refresh = create_refresh_token(user_id=user.id, org_id=member.org_id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, repo: AuthRepoDep) -> TokenResponse:
    """Exchange a refresh token for a new access token."""
    try:
        payload = decode_token(request.refresh_token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    try:
        user_id = UUID(payload["sub"])
        _org_id = UUID(payload["org"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Malformed token payload")

    _user = await repo.get_user_by_email("")  # we need user by id — fetch via member
    # fetch member to get current role
    member = await repo.get_member_by_user(user_id)
    if not member:
        raise HTTPException(status_code=401, detail="User not found")

    # We need the user's email — get it through a direct lookup
    from sqlalchemy import select

    from howler_agents_service.db.models import UserModel

    result = await repo._session.execute(select(UserModel).where(UserModel.id == user_id))
    db_user = result.scalars().first()
    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")

    access = create_access_token(
        user_id=db_user.id,
        org_id=member.org_id,
        email=db_user.email,
        role=member.role,
    )
    new_refresh = create_refresh_token(user_id=db_user.id, org_id=member.org_id)
    return TokenResponse(access_token=access, refresh_token=new_refresh)


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=201)
async def create_api_key(
    request: CreateApiKeyRequest,
    current_user: CurrentUserDep,
    repo: AuthRepoDep,
    session: SessionDep,
) -> ApiKeyResponse:
    """Create an API key for the current user's org. Key is returned only once."""
    api_key_model, raw_key = await repo.create_api_key(
        org_id=current_user.org_id,
        name=request.name,
        expires_at=request.expires_at,
    )
    await session.commit()
    await session.refresh(api_key_model)
    return ApiKeyResponse(
        id=str(api_key_model.id),
        name=api_key_model.name,
        key=raw_key,
        key_prefix=api_key_model.key_prefix,
        created_at=api_key_model.created_at,
    )


@router.get("/me", response_model=MeResponse)
async def me(current_user: CurrentUserDep) -> MeResponse:
    """Return information about the currently authenticated user."""
    return MeResponse(
        user_id=str(current_user.user_id),
        org_id=str(current_user.org_id),
        email=current_user.email,
        role=current_user.role,
    )
