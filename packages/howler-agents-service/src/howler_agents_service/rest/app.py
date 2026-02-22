"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from howler_agents_service.db.engine import close_db, init_db
from howler_agents_service.rest.routes.agents import router as agents_router
from howler_agents_service.rest.routes.auth import router as auth_router
from howler_agents_service.rest.routes.experience import router as experience_router
from howler_agents_service.rest.routes.health import router as health_router
from howler_agents_service.rest.routes.runs import router as runs_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield
    await close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Howler Agents API",
        description="Group-Evolving Agents service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public routes
    app.include_router(health_router, tags=["health"])

    # Auth routes (register/login are public; /me and /api-keys are protected inside the router)
    app.include_router(auth_router, prefix="/api/v1", tags=["auth"])

    # Protected API routes
    app.include_router(runs_router, prefix="/api/v1", tags=["runs"])
    app.include_router(agents_router, prefix="/api/v1", tags=["agents"])
    app.include_router(experience_router, prefix="/api/v1", tags=["experience"])

    return app
