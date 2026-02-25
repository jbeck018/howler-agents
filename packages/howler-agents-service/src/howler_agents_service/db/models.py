"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Auth / tenancy models
# ---------------------------------------------------------------------------


class OrganizationModel(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    slug = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    members = relationship(
        "OrgMemberModel", back_populates="organization", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "ApiKeyModel", back_populates="organization", cascade="all, delete-orphan"
    )


class UserModel(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    display_name = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    memberships = relationship(
        "OrgMemberModel", back_populates="user", cascade="all, delete-orphan"
    )


class OrgMemberModel(Base):
    __tablename__ = "org_members"

    org_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role = Column(String, nullable=False, default="member")

    organization = relationship("OrganizationModel", back_populates="members")
    user = relationship("UserModel", back_populates="memberships")


class ApiKeyModel(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    key_hash = Column(Text, nullable=False)
    key_prefix = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    organization = relationship("OrganizationModel", back_populates="api_keys")


class EvolutionRunModel(Base):
    __tablename__ = "evolution_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    config = Column(JSONB, nullable=False)
    status = Column(String, nullable=False, default="pending")
    current_generation = Column(Integer, nullable=False, default=0)
    total_generations = Column(Integer, nullable=False)
    best_agent_id = Column(UUID(as_uuid=True), nullable=True)
    best_score = Column(Float, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    agents = relationship("AgentModel", back_populates="run", cascade="all, delete-orphan")
    groups = relationship("AgentGroupModel", back_populates="run", cascade="all, delete-orphan")


class AgentGroupModel(Base):
    __tablename__ = "agent_groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True), ForeignKey("evolution_runs.id", ondelete="CASCADE"), nullable=False
    )
    generation = Column(Integer, nullable=False)
    group_performance = Column(Float, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    run = relationship("EvolutionRunModel", back_populates="groups")
    agents = relationship("AgentModel", back_populates="group")


class AgentModel(Base):
    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    run_id = Column(
        UUID(as_uuid=True), ForeignKey("evolution_runs.id", ondelete="CASCADE"), nullable=False
    )
    group_id = Column(
        UUID(as_uuid=True), ForeignKey("agent_groups.id", ondelete="SET NULL"), nullable=True
    )
    generation = Column(Integer, nullable=False)
    parent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    performance_score = Column(Float, default=0)
    novelty_score = Column(Float, default=0)
    combined_score = Column(Float, default=0)
    capability_vector: list[float] = Column(ARRAY(Float), default=list)  # type: ignore[assignment]
    framework_config = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    run = relationship("EvolutionRunModel", back_populates="agents")
    group = relationship("AgentGroupModel", back_populates="agents")
    patches = relationship(
        "FrameworkPatchModel", back_populates="agent", cascade="all, delete-orphan"
    )
    traces = relationship(
        "EvolutionaryTraceModel", back_populates="agent", cascade="all, delete-orphan"
    )


class FrameworkPatchModel(Base):
    __tablename__ = "framework_patches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    generation = Column(Integer, nullable=False)
    intent = Column(Text, nullable=False)
    diff = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    applied_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    agent = relationship("AgentModel", back_populates="patches")


class EvolutionaryTraceModel(Base):
    __tablename__ = "evolutionary_traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    run_id = Column(
        UUID(as_uuid=True), ForeignKey("evolution_runs.id", ondelete="CASCADE"), nullable=False
    )
    group_id = Column(Text, nullable=False, default="")
    generation = Column(Integer, nullable=False)
    task_description = Column(Text, nullable=False)
    outcome = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    key_decisions: list[str] = Column(ARRAY(Text), default=list)  # type: ignore[assignment]
    lessons_learned: list[str] = Column(ARRAY(Text), default=list)  # type: ignore[assignment]
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    agent = relationship("AgentModel", back_populates="traces")
