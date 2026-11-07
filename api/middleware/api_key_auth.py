"""API Key Authentication & Operation Scope Middleware for ARX Terminal Backend.

Architectural Guarantees (SEC-001 Recovery, Issue #203):
1. PUBLIC_RESEARCH: Preserves unauthenticated access to intentionally public market research endpoints
   (market data, instruments, analytics, volatility, regimes, ETF, fundamentals, macro, screener reads,
   public Radar/VCP explain & snapshots, and Smart Money public evidence).
2. PRIVILEGED_ADMIN: Destructive and administrative operations (e.g. cache purging, governance epochs)
   require dedicated administrative credentials (e.g. X-Admin-Key) and fail closed in production.
   Ordinary API keys are strictly forbidden from authorizing administrative operations.
3. INTERNAL_SERVICE: Background worker and scheduler triggers require dedicated internal secrets
   (e.g. VCP_SCHEDULER_SECRET, TELEMETRY_INTERNAL_SECRET) and fail closed.
4. AUTHENTICATED_USER: Tenant and user operations (/portfolio, /journal, /cockpit) are resolved
   via application RequestContextResolver (X-User-Id / session selectors).
5. FAIL-CLOSED ON UNKNOWN: Any unknown or unclassified operation fails closed in production (HTTP 500
   when ARX_API_KEY is unset; HTTP 401 when missing; HTTP 403 when invalid).
"""

import os
import re
import hmac
import logging
from typing import Set, Tuple, List, Pattern
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Base public documentation paths
DEV_DOCS_PATHS: Set[str] = {"/docs", "/openapi.json", "/redoc"}

# Exact specifications for intentionally public read-only research operations
PUBLIC_RESEARCH_SPECS: List[Tuple[str, str]] = [
    ("GET", "/health"),
    ("GET", "/api/v1/market/instruments/{symbol}"),
    ("GET", "/api/v1/security-master/instruments/{symbol}"),
    ("GET", "/api/v1/security-master/status"),
    ("POST", "/api/v1/security-master/instruments/batch"),
    ("GET", "/api/v1/analytics/setups"),
    ("GET", "/api/v1/analytics/setups/{symbol}"),
    ("GET", "/api/v1/analytics/{symbol}"),
    ("GET", "/api/v1/volatility/{symbol}"),
    ("GET", "/api/v1/regimes/current"),
    ("GET", "/api/v1/regimes/{symbol}"),
    ("GET", "/api/v1/macro/ribbon"),
    ("GET", "/api/v1/etf/profile/{symbol}"),
    ("GET", "/api/v1/etf/sectors/{symbol}"),
    ("GET", "/api/v1/fundamentals"),
    ("GET", "/api/v1/fundamentals/"),
    ("GET", "/api/v1/fundamentals/{symbol}"),
    ("GET", "/api/v1/fundamentals/system/status"),
    ("GET", "/api/v1/screener/run"),
    ("POST", "/api/v1/screener/run"),
    ("GET", "/api/v1/screener/position-size"),
    ("GET", "/api/v1/screener/phase26/validation-report"),
    ("GET", "/api/v1/screener/vcp/snapshot"),
    ("GET", "/api/v1/screener/vcp/explain/{symbol}"),
    ("GET", "/api/v1/screener/vcp/coordination"),
    ("GET", "/api/v1/screener/smart-money/snapshot"),
    ("GET", "/api/v1/smart-money/overview"),
    ("GET", "/api/v1/smart-money/evidence/{symbol}"),
    ("GET", "/api/v1/smart-money/filings/{symbol}"),
    ("GET", "/api/v1/smart-money/sec-filings/{symbol}"),
    ("GET", "/api/v1/smart-money/finra-darkpool/{symbol}"),
    ("GET", "/api/v1/smart-money/congress"),
    ("GET", "/api/v1/smart-money/options-flow"),
    ("GET", "/api/v1/governance/epoch-2/status"),
    ("GET", "/api/v1/governance/evaluation-summary"),
    ("GET", "/api/v1/governance/prospective-ledger"),
    ("GET", "/api/v1/telemetry/audit"),
    ("GET", "/api/v1/telemetry/denominator"),
    ("GET", "/api/v1/telemetry/epochs"),
    ("GET", "/api/v1/telemetry/corporate-events/denominator"),
]

# Privileged operations guarded by route-level administrative, internal, or user tenant secrets
DELEGATED_AUTH_SPECS: List[Tuple[str, str]] = [
    # Privileged administrative operations
    ("POST", "/api/v1/cache/clear"),
    ("POST", "/api/v1/governance/epoch-2/activate"),
    ("POST", "/api/v1/governance/epoch-2/certify-release"),
    ("POST", "/api/v1/governance/epoch-2/revoke-runtime"),
    # Internal service & background worker operations
    ("POST", "/api/v1/screener/vcp/natural-scan"),
    ("POST", "/api/v1/screener/vcp/scan"),
    ("POST", "/api/v1/screener/smart-money/scan"),
    ("POST", "/api/v1/telemetry/persist"),
    ("POST", "/api/v1/telemetry/corporate-events/events"),
    ("POST", "/api/v1/telemetry/corporate-events/persist"),
    # Authenticated user operations (RequestContextResolver / workspace tenancy)
    ("GET", "/api/v1/portfolio"),
    ("POST", "/api/v1/portfolio"),
    ("PUT", "/api/v1/portfolio/{symbol}"),
    ("DELETE", "/api/v1/portfolio/{symbol}"),
    ("POST", "/api/v1/portfolio/holdings/{holding_id}/exit"),
    ("POST", "/api/v1/portfolio/migrate"),
    ("GET", "/api/v1/journal/trades"),
    ("POST", "/api/v1/journal/trades"),
    ("POST", "/api/v1/journal/fill"),
    ("POST", "/api/v1/journal/exit"),
    ("POST", "/api/v1/journal/close"),
    ("GET", "/api/v1/journal/telemetry"),
    ("GET", "/api/v1/cockpit/state"),
    ("POST", "/api/v1/cockpit/profile"),
    ("POST", "/api/v1/cockpit/action"),
    ("POST", "/api/v1/cockpit/actions"),
]


def _compile_route_pattern(pattern: str) -> Pattern:
    """Compile route path pattern into regex handling path parameters and optional trailing slashes."""
    norm = pattern.rstrip("/")
    escaped = re.escape(norm)
    regex_pattern = re.sub(r"\\\{[a-zA-Z0-9_]+\\\}", r"[a-zA-Z0-9_\\-\\.]+", escaped)
    return re.compile(rf"^{regex_pattern}/?$")


COMPILED_PUBLIC_PATTERNS: List[Tuple[str, Pattern]] = [
    (method, _compile_route_pattern(spec))
    for method, spec in PUBLIC_RESEARCH_SPECS
]

COMPILED_DELEGATED_PATTERNS: List[Tuple[str, Pattern]] = [
    (method, _compile_route_pattern(spec))
    for method, spec in DELEGATED_AUTH_SPECS
]


def is_public_research_operation(method: str, path: str) -> bool:
    """Determine whether an operation is classified as intentionally public research."""
    m = method.upper()
    for spec_method, pattern in COMPILED_PUBLIC_PATTERNS:
        if spec_method == m and pattern.match(path):
            return True
    return False


def is_delegated_auth_operation(method: str, path: str) -> bool:
    """Determine whether an operation has dedicated route-level administrative or tenant auth."""
    m = method.upper()
    for spec_method, pattern in COMPILED_DELEGATED_PATTERNS:
        if spec_method == m and pattern.match(path):
            return True
    return False


def is_production() -> bool:
    """Authoritatively determine whether server is executing under production posture."""
    env = os.getenv("ENVIRONMENT", "").strip().lower()
    railway_env = os.getenv("RAILWAY_ENVIRONMENT", "").strip().lower()
    railway_env_name = os.getenv("RAILWAY_ENVIRONMENT_NAME", "").strip().lower()

    for val in (env, railway_env, railway_env_name):
        if val in ("dev", "development", "test", "staging", "preview"):
            return False

    if env == "production" or railway_env == "production" or railway_env_name == "production":
        return True

    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    return True


def get_configured_api_key() -> str:
    """Retrieve server-side API key dynamically from environment."""
    return os.getenv("ARX_API_KEY", "").strip()


def _secure_compare(a: str, b: str) -> bool:
    """Constant-time string comparison using Python standard library hmac.compare_digest."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces authorization scope boundaries across API endpoints.
    - Public research endpoints are permitted unauthenticated.
    - Delegated auth endpoints proceed to router-level credential validation.
    - Unknown or unclassified endpoints fail closed in production.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"

        # Always pass through CORS preflight (OPTIONS)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Non-production interactive documentation bypass
        prod_mode = is_production()
        if not prod_mode and path in DEV_DOCS_PATHS:
            return await call_next(request)

        # 1. Public Research: Intentionally open read-only endpoints
        if is_public_research_operation(request.method, path):
            return await call_next(request)

        # 2. Delegated Auth: Router-level dedicated administrative, internal, or tenant auth
        if is_delegated_auth_operation(request.method, path):
            # If an external caller supplies X-API-Key and ARX_API_KEY is configured,
            # validate that the key is not corrupted/invalid.
            configured_key = get_configured_api_key()
            incoming_key = request.headers.get("X-API-Key", "").strip()
            if configured_key and incoming_key and not _secure_compare(incoming_key, configured_key):
                logger.warning(
                    "Invalid API key on delegated route %s from %s",
                    path,
                    request.client.host if request.client else "unknown",
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden", "message": "Invalid API key."},
                    headers={"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"},
                )
            return await call_next(request)

        # 3. Unknown / Sensitive Route: FAIL CLOSED
        configured_key = get_configured_api_key()

        if not configured_key:
            if prod_mode:
                logger.critical(
                    "SECURITY FAIL-CLOSED: ARX_API_KEY is not configured in production environment. "
                    "Rejecting request to unknown or sensitive route %s.", path
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal Server Misconfiguration",
                        "message": "Authentication is mandatory in production, but ARX_API_KEY is not configured.",
                    },
                    headers={"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"},
                )
            else:
                logger.debug("ARX_API_KEY not set — pass-through for unclassified route in dev mode: %s", path)
                return await call_next(request)

        incoming_key = request.headers.get("X-API-Key", "").strip()
        if not incoming_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Missing X-API-Key header. Requests must originate from ARX Terminal.",
                },
                headers={"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"},
            )

        if not _secure_compare(incoming_key, configured_key):
            logger.warning(
                "Invalid API key attempt from %s on %s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "message": "Invalid API key."},
                headers={"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"},
            )

        return await call_next(request)
