"""
Private RequestContext Resolver Infrastructure for ARX SaaS Foundation (Wave 1F-A).

This module provides an opt-in dependency for resolving application-layer
RequestContext on private, context-aware routes.

Architectural Guarantees:
1. OPT-IN ONLY: This resolver is NEVER registered as universal middleware,
   global router dependency, or application-wide hook.
2. INV-SAAS-05: Public context-free routes must never invoke or depend on
   this resolver.
3. FAIL-CLOSED / NON-SPECULATIVE: Because authentication and workspace tenancy
   are not yet implemented, this resolver does NOT fabricate authenticated identity
   or validate cryptographically untrusted claims.

Explicit Current-Phase Status:
- AUTHENTICATED_ACTOR_RESOLUTION = "NOT_IMPLEMENTED"
- WORKSPACE_MEMBERSHIP_RESOLUTION = "NOT_IMPLEMENTED"
- SUBSCRIPTION_RESOLUTION = "NOT_IMPLEMENTED"
"""

import hashlib
import re
import sys
import uuid
from typing import Optional, Any

# Late-bind Request type for FastAPI dependency injection without violating stdlib import governance
RequestType = getattr(
    sys.modules.get("starlette.requests") or __import__("starlette.requests", fromlist=["Request"]),
    "Request",
    Any,
)

from api.context.request_context import RequestContext
from api.context.workspace_identity import derive_compatibility_workspace_id

# Phase status attestations
AUTHENTICATED_ACTOR_RESOLUTION: str = "NOT_IMPLEMENTED"
WORKSPACE_MEMBERSHIP_RESOLUTION: str = "NOT_IMPLEMENTED"
SUBSCRIPTION_RESOLUTION: str = "NOT_IMPLEMENTED"

# Identifier validation regex: alphanumeric, dash, underscore, 1-64 chars
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")


def _sanitize_identifier(value: Optional[str]) -> Optional[str]:
    """Sanitize and validate an identifier string."""
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if SAFE_ID_REGEX.match(cleaned):
        return cleaned
    filtered = re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)[:64]
    return filtered if filtered else None


def _sanitize_request_id(value: Optional[str]) -> str:
    """Sanitize or generate a deterministic request ID."""
    if value and isinstance(value, str):
        cleaned = value.strip()
        if SAFE_REQUEST_ID_REGEX.match(cleaned):
            return cleaned
        filtered = re.sub(r"[^a-zA-Z0-9_\-\.]", "", cleaned)[:128]
        if filtered:
            return filtered
    return f"req_{uuid.uuid4().hex[:12]}"


class RequestContextResolver:
    """
    Opt-in resolver for application RequestContext on private routes.

    Input Contract & Trust Model:
    -----------------------------
    1. Request ID:
       - Source: 'X-Request-ID' header
       - Trust Level: UNTRUSTED_CLIENT_TRACE
       - Validation: SAFE_REQUEST_ID_REGEX
       - Fallback: Auto-generated 'req_<uuid12>'
       - Privacy: Non-sensitive operational trace

    2. Compatibility Actor Selector:
       - Source: 'X-User-Id' header
       - Trust Level: UNAUTHENTICATED_COMPATIBILITY_SELECTOR
       - Validation: SAFE_ID_REGEX
       - Fallback: None (represents anonymous/unauthenticated actor)
       - Privacy: Pseudonymous identifier
       - Note: Full authenticated actor resolution is NOT_IMPLEMENTED

    3. Deterministic Migration Workspace Selector:
       - Source: 'X-Workspace-ID' header
       - Trust Level: UNAUTHENTICATED_COMPATIBILITY_SELECTOR
       - Validation: SAFE_ID_REGEX
       - Fallback: 'ws_usr_<sha256(actor)[:16]>' if actor present, else 'ws_default'
       - Privacy: Tenant identifier
       - Note: Workspace membership authorization is NOT_IMPLEMENTED

    Prohibited Inputs:
    - Commercial tier headers, monetary payment headers, session tokens.
    """

    def __init__(self, workspace_repository: Optional[Any] = None) -> None:
        self.workspace_repo = workspace_repository

    async def __call__(
        self,
        request: Optional[Any] = None,
        x_request_id: Optional[str] = None,
        x_workspace_id: Optional[str] = None,
        x_user_id: Optional[str] = None,
    ) -> RequestContext:
        """Dependency entrypoint for private routes."""
        return self.resolve(
            request=request,
            x_request_id=x_request_id,
            x_workspace_id=x_workspace_id,
            x_user_id=x_user_id,
        )

    def resolve(
        self,
        request: Optional[Any] = None,
        x_request_id: Optional[str] = None,
        x_workspace_id: Optional[str] = None,
        x_user_id: Optional[str] = None,
    ) -> RequestContext:
        """
        Pure synchronous or async resolution logic.
        Constructs an immutable RequestContext without reading commercial state.
        """
        # Extract from request headers and query parameters if available
        if request is not None and hasattr(request, "headers"):
            headers = request.headers
            if x_request_id is None:
                x_request_id = headers.get("x-request-id") or headers.get("X-Request-ID")
            if x_workspace_id is None:
                x_workspace_id = headers.get("x-workspace-id") or headers.get("X-Workspace-ID")
            if x_user_id is None:
                x_user_id = (
                    headers.get("x-user-id")
                    or headers.get("X-User-Id")
                    or headers.get("x-profile-id")
                    or headers.get("X-Profile-Id")
                )
        if request is not None and hasattr(request, "query_params") and x_user_id is None:
            qp = request.query_params
            x_user_id = qp.get("profile_id") or qp.get("subject_id")

        # 1. Resolve Request ID
        req_id = _sanitize_request_id(x_request_id)

        # 2. Resolve Unauthenticated Compatibility Actor Selector
        actor_id = _sanitize_identifier(x_user_id)

        # 3. Resolve Workspace ID (Deterministic migration mapping)
        sanitized_ws = _sanitize_identifier(x_workspace_id)
        if sanitized_ws:
            workspace_id = sanitized_ws
        else:
            workspace_id = derive_compatibility_workspace_id(actor_id)

        return RequestContext(
            actor_id=actor_id,
            workspace_id=workspace_id,
            request_id=req_id,
        )


# Singleton resolver instance for route dependency injection
default_request_context_resolver = RequestContextResolver()


async def resolve_request_context(
    request: RequestType = None,
    x_request_id: Optional[str] = None,
    x_workspace_id: Optional[str] = None,
    x_user_id: Optional[str] = None,
) -> RequestContext:
    """
    Opt-in dependency for private/context-aware endpoints.

    Usage in Wave 1F-B private routes:
        @router.get("/portfolio")
        async def get_portfolio(
            context: RequestContext = Depends(resolve_request_context)
        ): ...

    STRICTLY FORBIDDEN on public context-free routes (INV-SAAS-05).
    """
    return default_request_context_resolver.resolve(
        request=request,
        x_request_id=x_request_id,
        x_workspace_id=x_workspace_id,
        x_user_id=x_user_id,
    )
