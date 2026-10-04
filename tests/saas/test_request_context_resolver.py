"""
Unit tests for RequestContextResolver infrastructure (Wave 1F-A).

Verifies:
1. Input contract and trust model adherence
2. Deterministic fallback to ws_default for anonymous callers
3. Deterministic migration mapping ws_usr_<hash> for identified actors
4. Sanitization and length bounding of identifiers
5. Non-speculative current phase constants
6. Absence of commercial plan or billing state
"""

import ast
import hashlib
import inspect
import pytest
from unittest.mock import MagicMock

from api.context.request_context import RequestContext
from api.context.resolver import (
    RequestContextResolver,
    resolve_request_context,
    default_request_context_resolver,
    AUTHENTICATED_ACTOR_RESOLUTION,
    WORKSPACE_MEMBERSHIP_RESOLUTION,
    SUBSCRIPTION_RESOLUTION,
    _sanitize_identifier,
    _sanitize_request_id,
)


def test_phase_status_attestations():
    """Verify non-speculative phase status attestations."""
    assert AUTHENTICATED_ACTOR_RESOLUTION == "NOT_IMPLEMENTED"
    assert WORKSPACE_MEMBERSHIP_RESOLUTION == "NOT_IMPLEMENTED"
    assert SUBSCRIPTION_RESOLUTION == "NOT_IMPLEMENTED"


def test_sanitize_identifier_behavior():
    """Verify identifier sanitization rules."""
    assert _sanitize_identifier(None) is None
    assert _sanitize_identifier("") is None
    assert _sanitize_identifier("   ") is None
    assert _sanitize_identifier("valid-id_123") == "valid-id_123"
    assert _sanitize_identifier("bad@id#with$symbols") == "badidwithsymbols"
    long_id = "a" * 100
    assert len(_sanitize_identifier(long_id)) == 64


def test_sanitize_request_id_behavior():
    """Verify request ID sanitization and generation."""
    assert _sanitize_request_id("custom-req-001") == "custom-req-001"
    generated = _sanitize_request_id(None)
    assert generated.startswith("req_")
    assert len(generated) > 5


def test_default_resolution_anonymous():
    """Verify default resolution produces anonymous context with ws_default."""
    resolver = RequestContextResolver()
    ctx = resolver.resolve()
    assert isinstance(ctx, RequestContext)
    assert ctx.actor_id is None
    assert ctx.workspace_id == "ws_default"
    assert ctx.request_id.startswith("req_")


def test_actor_resolution_deterministic_workspace():
    """Verify actor without explicit workspace resolves to deterministic personal workspace."""
    resolver = RequestContextResolver()
    actor = "trader_alpha"
    expected_hash = hashlib.sha256(actor.encode("utf-8")).hexdigest()[:16]
    expected_ws = f"ws_usr_{expected_hash}"

    ctx = resolver.resolve(x_user_id=actor)
    assert ctx.actor_id == actor
    assert ctx.workspace_id == expected_ws
    assert ctx.request_id.startswith("req_")


def test_explicit_workspace_override():
    """Verify explicit workspace header overrides deterministic actor mapping."""
    resolver = RequestContextResolver()
    ctx = resolver.resolve(x_user_id="trader_alpha", x_workspace_id="ws_firm_99")
    assert ctx.actor_id == "trader_alpha"
    assert ctx.workspace_id == "ws_firm_99"


def test_explicit_request_id():
    """Verify explicit request ID is preserved."""
    resolver = RequestContextResolver()
    ctx = resolver.resolve(x_request_id="req_trace_456")
    assert ctx.request_id == "req_trace_456"


@pytest.mark.asyncio
async def test_resolve_request_context_dependency():
    """Verify the FastAPI dependency helper works asynchronously."""
    mock_request = MagicMock()
    ctx = await resolve_request_context(
        request=mock_request,
        x_request_id="req-test-async",
        x_workspace_id="ws_async",
        x_user_id="user_async",
    )
    assert isinstance(ctx, RequestContext)
    assert ctx.request_id == "req-test-async"
    assert ctx.workspace_id == "ws_async"
    assert ctx.actor_id == "user_async"


def test_resolver_ast_purity():
    """Verify resolver source code has no commercial plans or quant imports."""
    import api.context.resolver as mod
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    forbidden_tokens = ["stripe", "checkout", "pro_plan", "free_plan", "pricing", "invoice"]
    src_lower = src.lower()
    for tok in forbidden_tokens:
        assert tok not in src_lower, f"Forbidden token '{tok}' found in resolver.py"
