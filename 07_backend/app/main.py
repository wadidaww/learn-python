"""
app/main.py
============
FastAPI application entry point.

Features:
  - Health endpoint
  - CORS middleware
  - Lifespan events (startup/shutdown)
  - Versioned API router mounting
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from contextlib import asynccontextmanager
    from collections.abc import AsyncGenerator
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from app.routers import users as users_router

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan handler (startup / shutdown)."""
        print("Starting up: initialising database pool…")
        yield
        print("Shutting down: closing connections…")

    def create_app() -> FastAPI:
        """Application factory – create and configure FastAPI app."""
        _app = FastAPI(
            title="Learn Python API",
            version="0.1.0",
            description="Production-ready FastAPI demonstration",
            lifespan=lifespan,
        )

        # CORS
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Routers
        _app.include_router(users_router.router, prefix="/api/v1")

        # Health endpoint
        @_app.get("/health", tags=["meta"])
        async def health() -> dict[str, str]:
            """Service health check."""
            return {"status": "ok", "version": "0.1.0"}

        return _app

    app = create_app()
