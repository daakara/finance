"""API Key Authentication Middleware for ARX Terminal Backend.

Validates the X-API-Key header on all protected routes.
Public routes (/health, OPTIONS) bypass auth.

Set ARX_API_KEY environment variable on Railway/Render backend host.
The frontend sends it as: X-API-Key: <value>
"""

import os
import hmac
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Routes that do NOT require authentication
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

ARX_API_KEY = os.getenv("ARX_API_KEY", "")


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces X-API-Key header authentication on all protected API routes.
    Bypassed for CORS preflight, /health, and documentation paths.
    """

    async def dispatch(self, request: Request, call_next):
        # Always pass through CORS preflight and public paths
        if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # If ARX_API_KEY is not configured server-side, skip enforcement
        # This allows local development without needing the env var set.
        if not ARX_API_KEY:
            logger.debug("ARX_API_KEY not set — API key enforcement skipped (dev mode).")
            return await call_next(request)

        incoming_key = request.headers.get("X-API-Key", "")

        if not incoming_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Missing X-API-Key header. Requests must originate from ARX Terminal.",
                },
            )

        # Constant-time comparison to prevent timing-based side-channel attacks
        if not _secure_compare(incoming_key, ARX_API_KEY):
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


