"""API Key Authentication Middleware for ARX Terminal Backend.

Validates the X-API-Key header on all protected routes.
Public routes (/health, OPTIONS) bypass auth.

Set ARX_API_KEY environment variable on Railway/Render backend host.
The frontend sends it as: X-API-Key: <value>
"""

import os
import hmac
import logging
from typing import Set
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Base public routes that bypass authentication
PUBLIC_PATHS: Set[str] = {"/health"}
DEV_DOCS_PATHS: Set[str] = {"/docs", "/openapi.json", "/redoc"}


def is_production() -> bool:
    """Authoritatively determine whether server is executing under production posture."""
    env = os.getenv("ENVIRONMENT", "").strip().lower()
    railway_env = os.getenv("RAILWAY_ENVIRONMENT", "").strip().lower()
    railway_env_name = os.getenv("RAILWAY_ENVIRONMENT_NAME", "").strip().lower()
    
    # Explicit non-production environments
    for val in (env, railway_env, railway_env_name):
        if val in ("dev", "development", "test", "staging", "preview"):
            return False
            
    if env == "production" or railway_env == "production" or railway_env_name == "production":
        return True

    # If running inside pytest without explicit ENVIRONMENT override, treat as non-production
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    # Default to production fail-closed for safety
    return True


def get_configured_api_key() -> str:
    """Retrieve server-side API key dynamically from environment."""
    return os.getenv("ARX_API_KEY", "").strip()


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces X-API-Key header authentication on all protected API routes.
    Fails closed in production if ARX_API_KEY is unset or empty.
    Bypassed for CORS preflight (OPTIONS) and /health.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        
        # Always pass through CORS preflight (OPTIONS)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Health endpoint is public in all environments
        if path == "/health":
            return await call_next(request)

        prod_mode = is_production()

        # Non-production interactive documentation bypass
        if not prod_mode and path in DEV_DOCS_PATHS:
            return await call_next(request)

        configured_key = get_configured_api_key()

        # FAIL-CLOSED IN PRODUCTION: If ARX_API_KEY is missing, reject all requests.
        # Never allow silent pass-through when operating in production.
        if not configured_key:
            if prod_mode:
                logger.critical(
                    "SECURITY FAIL-CLOSED: ARX_API_KEY is not configured in production environment. "
                    "Rejecting request to %s.", path
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal Server Misconfiguration",
                        "message": "Authentication is mandatory in production, but ARX_API_KEY is not configured.",
                    },
                )
            else:
                logger.debug("ARX_API_KEY not set — API key enforcement skipped (dev mode).")
                return await call_next(request)

        incoming_key = request.headers.get("X-API-Key", "").strip()

        if not incoming_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Missing X-API-Key header. Requests must originate from ARX Terminal.",
                },
            )

        # Constant-time comparison to prevent timing-based side-channel attacks
        if not _secure_compare(incoming_key, configured_key):
            logger.warning(
                "Invalid API key attempt from %s on %s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "message": "Invalid API key.",
                },
            )

        return await call_next(request)


def _secure_compare(a: str, b: str) -> bool:
    """Constant-time string comparison using Python standard library hmac.compare_digest."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
