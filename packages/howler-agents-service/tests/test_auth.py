"""Auth endpoint tests using mock-first (London School TDD)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from howler_agents_service.auth.deps import get_current_user
from howler_agents_service.auth.models import CurrentUser
from howler_agents_service.db.deps import get_auth_repo, get_runs_repo, get_session
from howler_agents_service.rest.routes.auth import router as auth_router
from howler_agents_service.rest.routes.health import router as health_router
from howler_agents_service.rest.routes.runs import router as runs_router

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_user(email: str = "alice@example.com", role: str = "owner") -> MagicMock:
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = email
    user.password_hash = "$2b$12$fakehash"
    user.display_name = "Alice"
    user.created_at = datetime.now(UTC)
    return user


def _make_org(name: str = "Acme", slug: str = "acme") -> MagicMock:
    org = MagicMock()
    org.id = uuid.uuid4()
    org.name = name
    org.slug = slug
    org.created_at = datetime.now(UTC)
    return org


def _make_member(org_id, user_id, role: str = "owner") -> MagicMock:
    member = MagicMock()
    member.org_id = org_id
    member.user_id = user_id
    member.role = role
    return member


def _make_api_key(org_id) -> tuple[MagicMock, str]:
    raw_key = "ha_live_testkey1234567890abcdef"
    model = MagicMock()
    model.id = uuid.uuid4()
    model.org_id = org_id
    model.key_hash = "hash"
    model.key_prefix = raw_key[:16]
    model.name = "test-key"
    model.created_at = datetime.now(UTC)
    model.expires_at = None
    model.revoked_at = None
    return model, raw_key


class FakeAuthRepo:
    """In-memory auth repo for testing."""

    def __init__(self):
        self._users: dict[str, Any] = {}
        self._orgs: dict[str, Any] = {}
        self._members: list[Any] = []
        self._api_keys: list[Any] = []

    async def get_user_by_email(self, email: str):
        return self._users.get(email)

    async def get_org_by_slug(self, slug: str):
        return next((o for o in self._orgs.values() if o.slug == slug), None)

    async def create_org(self, name: str, slug: str):
        org = _make_org(name=name, slug=slug)
        self._orgs[str(org.id)] = org
        return org

    async def create_user(self, email: str, password: str, display_name: str | None = None):
        user = _make_user(email=email)
        user.display_name = display_name
        self._users[email] = user
        return user

    async def add_member(self, org_id, user_id, role: str = "member"):
        member = _make_member(org_id=org_id, user_id=user_id, role=role)
        self._members.append(member)
        return member

    async def get_member(self, org_id, user_id):
        return next(
            (m for m in self._members if m.org_id == org_id and m.user_id == user_id),
            None,
        )

    async def get_member_by_user(self, user_id):
        return next((m for m in self._members if m.user_id == user_id), None)

    async def create_api_key(self, org_id, name: str, expires_at=None):
        model, raw = _make_api_key(org_id)
        model.name = name
        model.expires_at = expires_at
        self._api_keys.append(model)
        return model, raw

    async def get_api_key_by_prefix(self, key_prefix: str):
        return next((k for k in self._api_keys if k.key_prefix == key_prefix), None)


class FakeRunsRepo:
    def __init__(self):
        self._runs: dict[str, Any] = {}

    async def create(self, config, total_generations, org_id=None):
        run = MagicMock()
        run.id = uuid.uuid4()
        run.config = config
        run.status = "pending"
        run.current_generation = 0
        run.total_generations = total_generations
        run.best_score = 0
        run.org_id = org_id
        run.created_at = datetime.now(UTC)
        run.updated_at = datetime.now(UTC)
        self._runs[str(run.id)] = run
        return run

    async def get(self, run_id):
        return self._runs.get(str(run_id))

    async def list(self, limit=20, offset=0, status=None):
        runs = list(self._runs.values())
        if status:
            runs = [r for r in runs if r.status == status]
        return runs[offset : offset + limit], len(runs)

    async def update_status(self, run_id, status, **kwargs):
        run = self._runs.get(str(run_id))
        if run:
            run.status = status
            for k, v in kwargs.items():
                setattr(run, k, v)
        return run


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


def _make_test_app(auth_repo: FakeAuthRepo | None = None, runs_repo: FakeRunsRepo | None = None):
    """Build a test FastAPI app with all auth + runs routes."""
    app = FastAPI(title="test")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, tags=["health"])
    app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
    app.include_router(runs_router, prefix="/api/v1", tags=["runs"])

    _auth_repo = auth_repo or FakeAuthRepo()
    _runs_repo = runs_repo or FakeRunsRepo()

    # Stub out the DB session so auth deps don't hit real DB
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()

    app.dependency_overrides[get_session] = lambda: fake_session
    app.dependency_overrides[get_auth_repo] = lambda: _auth_repo
    app.dependency_overrides[get_runs_repo] = lambda: _runs_repo

    return app, _auth_repo, _runs_repo


@pytest.fixture
def auth_app():
    app, auth_repo, runs_repo = _make_test_app()
    return app, auth_repo, runs_repo


@pytest.fixture
def client(auth_app):
    app, auth_repo, runs_repo = auth_app
    return TestClient(app), auth_repo, runs_repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register(
    client: TestClient, email="alice@example.com", password="secret123", slug="acme"
) -> dict:
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "org_name": "Acme Inc",
            "org_slug": slug,
        },
    )
    return resp


def _valid_token(user_id=None, org_id=None, email="alice@example.com", role="owner") -> str:
    from howler_agents_service.auth.jwt import create_access_token

    uid = user_id or uuid.uuid4()
    oid = org_id or uuid.uuid4()
    return create_access_token(uid, oid, email, role)


# ---------------------------------------------------------------------------
# Register tests
# ---------------------------------------------------------------------------


def test_register_creates_user_and_org(client):
    tc, _auth_repo, _ = client
    resp = _register(tc)
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_register_duplicate_email_returns_409(client):
    tc, _auth_repo, _ = client
    _register(tc, email="bob@example.com", slug="bob-org")
    resp = _register(tc, email="bob@example.com", slug="bob-org2")
    assert resp.status_code == 409
    assert "Email" in resp.json()["detail"]


def test_register_duplicate_slug_returns_409(client):
    tc, _auth_repo, _ = client
    _register(tc, email="user1@example.com", slug="shared-slug")
    resp = _register(tc, email="user2@example.com", slug="shared-slug")
    assert resp.status_code == 409
    assert "slug" in resp.json()["detail"]


def test_register_short_password_returns_422(client):
    tc, _, _ = client
    resp = tc.post(
        "/api/v1/auth/register",
        json={
            "email": "x@x.com",
            "password": "short",
            "org_name": "X",
            "org_slug": "x-org",
        },
    )
    assert resp.status_code == 422


def test_register_invalid_slug_returns_422(client):
    tc, _, _ = client
    resp = tc.post(
        "/api/v1/auth/register",
        json={
            "email": "x@x.com",
            "password": "longpassword",
            "org_name": "X",
            "org_slug": "Invalid Slug!",
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login tests
# ---------------------------------------------------------------------------


def test_login_returns_tokens(client):
    tc, auth_repo, _ = client
    # Pre-populate the auth_repo with a known user
    from howler_agents_service.auth.passwords import hash_password

    user = _make_user(email="login@example.com")
    user.password_hash = hash_password("mypassword")
    org = _make_org(slug="login-org")
    member = _make_member(org_id=org.id, user_id=user.id, role="owner")
    auth_repo._users["login@example.com"] = user
    auth_repo._members.append(member)

    resp = tc.post(
        "/api/v1/auth/login", json={"email": "login@example.com", "password": "mypassword"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_login_wrong_password_returns_401(client):
    tc, auth_repo, _ = client
    from howler_agents_service.auth.passwords import hash_password

    user = _make_user(email="fail@example.com")
    user.password_hash = hash_password("correctpassword")
    auth_repo._users["fail@example.com"] = user

    resp = tc.post(
        "/api/v1/auth/login", json={"email": "fail@example.com", "password": "wrongpassword"}
    )
    assert resp.status_code == 401


def test_login_unknown_email_returns_401(client):
    tc, _, _ = client
    resp = tc.post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "anything"}
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Token refresh tests
# ---------------------------------------------------------------------------


def test_refresh_returns_new_access_token(client):
    tc, auth_repo, _ = client
    # Set up a user + member in the repo
    user = _make_user()
    org = _make_org()
    member = _make_member(org_id=org.id, user_id=user.id, role="owner")
    auth_repo._users[user.email] = user
    auth_repo._members.append(member)

    from howler_agents_service.auth.jwt import create_refresh_token

    refresh = create_refresh_token(user_id=user.id, org_id=org.id)

    # Stub the session execute to return the user
    _app, _ar, _rr = tc.app, auth_repo, None
    # We need the session.execute to return the user — patch AuthRepo._session

    async def fake_execute(*args, **kwargs):
        result = MagicMock()
        result.scalars.return_value.first.return_value = user
        return result

    with patch.object(auth_repo, "_session", create=True) as mock_session:
        mock_session.execute = AsyncMock(side_effect=fake_execute)
        resp = tc.post("/api/v1/auth/refresh", json={"refresh_token": refresh})

    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data


def test_refresh_with_access_token_returns_401(client):
    tc, _, _ = client
    access = _valid_token()
    resp = tc.post("/api/v1/auth/refresh", json={"refresh_token": access})
    assert resp.status_code == 401


def test_refresh_with_invalid_token_returns_401(client):
    tc, _, _ = client
    resp = tc.post("/api/v1/auth/refresh", json={"refresh_token": "not.a.token"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /me endpoint
# ---------------------------------------------------------------------------


def test_me_returns_user_info(client):
    tc, _, _ = client
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    token = _valid_token(user_id=user_id, org_id=org_id, email="me@example.com", role="admin")

    # Override get_current_user to bypass DB lookup
    app = tc.app
    current_user = CurrentUser(user_id=user_id, org_id=org_id, email="me@example.com", role="admin")
    app.dependency_overrides[get_current_user] = lambda: current_user

    resp = tc.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "me@example.com"
    assert data["role"] == "admin"
    assert data["user_id"] == str(user_id)
    assert data["org_id"] == str(org_id)


def test_me_without_token_returns_401(client):
    tc, _, _ = client
    resp = tc.get("/api/v1/auth/me")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# API key creation tests
# ---------------------------------------------------------------------------


def test_create_api_key_returns_key_once(client):
    tc, _auth_repo, _ = client
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    app = tc.app
    current_user = CurrentUser(
        user_id=user_id, org_id=org_id, email="owner@example.com", role="owner"
    )
    app.dependency_overrides[get_current_user] = lambda: current_user

    token = _valid_token(user_id=user_id, org_id=org_id, role="owner")
    resp = tc.post(
        "/api/v1/auth/api-keys",
        json={"name": "my-key"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["key"].startswith("ha_live_")
    assert "id" in data
    assert data["name"] == "my-key"


def test_create_api_key_without_auth_returns_401(client):
    tc, _, _ = client
    resp = tc.post("/api/v1/auth/api-keys", json={"name": "my-key"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Protected endpoint access tests
# ---------------------------------------------------------------------------


def test_protected_endpoint_with_valid_token(client):
    tc, _auth_repo, _runs_repo = client
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    app = tc.app
    current_user = CurrentUser(
        user_id=user_id, org_id=org_id, email="user@example.com", role="member"
    )
    app.dependency_overrides[get_current_user] = lambda: current_user

    token = _valid_token(user_id=user_id, org_id=org_id)
    resp = tc.get("/api/v1/runs", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_protected_endpoint_with_no_token_returns_401(client):
    tc, _, _ = client
    resp = tc.get("/api/v1/runs")
    assert resp.status_code == 401


def test_protected_endpoint_with_invalid_token_returns_401(client):
    tc, _, _ = client
    resp = tc.get("/api/v1/runs", headers={"Authorization": "Bearer invalid.token.here"})
    assert resp.status_code == 401


def test_health_endpoint_is_public(client):
    """Health check must remain public — no auth required."""
    tc, _, _ = client
    resp = tc.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API key usage on protected endpoint
# ---------------------------------------------------------------------------


def test_api_key_auth_on_protected_endpoint(client):
    tc, _, _ = client
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    app = tc.app
    current_user = CurrentUser(
        user_id=user_id, org_id=org_id, email="apiuser@example.com", role="member"
    )
    app.dependency_overrides[get_current_user] = lambda: current_user

    resp = tc.get("/api/v1/runs", headers={"X-API-Key": "ha_live_somekeyvalue"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth utility unit tests
# ---------------------------------------------------------------------------


def test_create_and_decode_access_token():
    """JWT round-trip test."""
    from howler_agents_service.auth.jwt import create_access_token, decode_token

    uid = uuid.uuid4()
    oid = uuid.uuid4()
    token = create_access_token(uid, oid, "test@example.com", "owner")
    payload = decode_token(token)
    assert payload["sub"] == str(uid)
    assert payload["org"] == str(oid)
    assert payload["email"] == "test@example.com"
    assert payload["role"] == "owner"
    assert payload["type"] == "access"


def test_create_and_decode_refresh_token():
    from howler_agents_service.auth.jwt import create_refresh_token, decode_token

    uid = uuid.uuid4()
    oid = uuid.uuid4()
    token = create_refresh_token(uid, oid)
    payload = decode_token(token)
    assert payload["sub"] == str(uid)
    assert payload["type"] == "refresh"


def test_expired_token_raises():
    from datetime import timedelta

    import jwt as pyjwt

    from howler_agents_service.auth.jwt import create_access_token, decode_token

    uid = uuid.uuid4()
    oid = uuid.uuid4()
    token = create_access_token(uid, oid, "x@x.com", "member", expires_delta=timedelta(seconds=-1))
    with pytest.raises(pyjwt.ExpiredSignatureError):
        decode_token(token)


def test_hash_and_verify_password():
    from howler_agents_service.auth.passwords import hash_password, verify_password

    pw = "super-secret-password"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed)
    assert not verify_password("wrong", hashed)
