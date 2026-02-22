"""Repository for auth-related DB operations."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from howler_agents_service.auth.passwords import hash_password
from howler_agents_service.db.models import ApiKeyModel, OrgMemberModel, OrganizationModel, UserModel


class AuthRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_org(self, name: str, slug: str) -> OrganizationModel:
        """Create a new organization."""
        org = OrganizationModel(name=name, slug=slug)
        self._session.add(org)
        await self._session.flush()
        await self._session.refresh(org)
        return org

    async def get_org_by_slug(self, slug: str) -> OrganizationModel | None:
        result = await self._session.execute(
            select(OrganizationModel).where(OrganizationModel.slug == slug)
        )
        return result.scalars().first()

    async def create_user(self, email: str, password: str, display_name: str | None = None) -> UserModel:
        """Create a new user with a bcrypt-hashed password."""
        user = UserModel(
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
        )
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def get_user_by_email(self, email: str) -> UserModel | None:
        result = await self._session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        return result.scalars().first()

    async def add_member(self, org_id: UUID, user_id: UUID, role: str = "member") -> OrgMemberModel:
        """Add a user as a member of an organization."""
        member = OrgMemberModel(org_id=org_id, user_id=user_id, role=role)
        self._session.add(member)
        await self._session.flush()
        return member

    async def get_member(self, org_id: UUID, user_id: UUID) -> OrgMemberModel | None:
        result = await self._session.execute(
            select(OrgMemberModel).where(
                OrgMemberModel.org_id == org_id,
                OrgMemberModel.user_id == user_id,
            )
        )
        return result.scalars().first()

    async def get_member_by_user(self, user_id: UUID) -> OrgMemberModel | None:
        """Get the first membership for a user (users typically belong to one org)."""
        result = await self._session.execute(
            select(OrgMemberModel).where(OrgMemberModel.user_id == user_id)
        )
        return result.scalars().first()

    async def create_api_key(
        self,
        org_id: UUID,
        name: str,
        expires_at: datetime | None = None,
    ) -> tuple[ApiKeyModel, str]:
        """
        Create an API key for an org.

        Returns (ApiKeyModel, raw_key). The raw_key is only returned once —
        only the hash is persisted.
        """
        # Generate a random key with the ha_live_ prefix
        random_part = secrets.token_urlsafe(32)
        raw_key = f"ha_live_{random_part}"

        # First 16 chars of the raw key used as the lookup prefix
        key_prefix = raw_key[:16]

        api_key = ApiKeyModel(
            org_id=org_id,
            key_hash=hash_password(raw_key),
            key_prefix=key_prefix,
            name=name,
            expires_at=expires_at,
        )
        self._session.add(api_key)
        await self._session.flush()
        await self._session.refresh(api_key)
        return api_key, raw_key

    async def get_api_key_by_prefix(self, key_prefix: str) -> ApiKeyModel | None:
        result = await self._session.execute(
            select(ApiKeyModel).where(
                ApiKeyModel.key_prefix == key_prefix,
                ApiKeyModel.revoked_at.is_(None),
            )
        )
        return result.scalars().first()
