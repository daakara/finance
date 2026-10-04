"""Application context package for ARX SaaS Foundation."""

from api.context.request_context import RequestContext, create_default_context
from api.context.resolver import (
    RequestContextResolver,
    resolve_request_context,
    default_request_context_resolver,
)

from api.context.workspace_identity import (
    derive_compatibility_workspace_id,
    is_valid_workspace_id,
    resolve_workspace_id,
    WORKSPACE_ID_AUTHORITY,
    WORKSPACE_ID_PREFIX,
)

__all__ = [
    "RequestContext",
    "create_default_context",
    "RequestContextResolver",
    "resolve_request_context",
    "default_request_context_resolver",
    "derive_compatibility_workspace_id",
    "is_valid_workspace_id",
    "resolve_workspace_id",
    "WORKSPACE_ID_AUTHORITY",
    "WORKSPACE_ID_PREFIX",
]
