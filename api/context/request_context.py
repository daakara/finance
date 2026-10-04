"""
Application-layer RequestContext contract for ARX SaaS Foundation.

This module defines the immutable identity and tenancy context passed through
future application services. It is strictly identity and tenancy focused and
contains zero commercial plan, billing, pricing, session, or quantitative state.
"""

from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass(frozen=True)
class RequestContext:
    """
    Immutable application-layer request context.

    Public fields:
    - actor_id: Optional string identifying the acting user (None represents an anonymous actor).
    - workspace_id: Required string identifying the tenant/workspace context.
    - request_id: Required string identifying the trace/request boundary.
    """

    actor_id: Optional[str]
    workspace_id: str
    request_id: str

    def __post_init__(self) -> None:
        """Validate required field presence and types upon construction."""
        if self.actor_id is not None:
            if not isinstance(self.actor_id, str) or not self.actor_id.strip():
                raise ValueError("actor_id must be None or a non-empty string.")

        if not isinstance(self.workspace_id, str) or not self.workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string.")

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string.")


def create_default_context(request_id: Optional[str] = None) -> RequestContext:
    """
    Construct a neutral/default RequestContext for tests or future application-layer use.

    Note: This helper is not wired to existing routes and does not alter
    current API route execution or default anonymous behavior.
    """
    return RequestContext(
        actor_id=None,
        workspace_id="ws_default",
        request_id=request_id.strip() if (request_id and isinstance(request_id, str) and request_id.strip()) else str(uuid.uuid4()),
    )
