"""
Request and Correlation Context Management for ARX Observability.

Enforces:
1. Strict RFC-4122 UUIDv4 validation for all incoming identifiers.
2. Complete replacement of adversarial, oversized, or non-v4 inputs.
3. Thread-safe and asyncio-task-safe ContextVar isolation across concurrent operations.
4. Token-based lifecycle management to prevent cross-request context leakage.
"""

import uuid
from typing import Optional
from contextvars import ContextVar, Token

# Internal asyncio context variables
_REQUEST_ID_CTX: ContextVar[Optional[str]] = ContextVar("arx_request_id", default=None)
_CORRELATION_ID_CTX: ContextVar[Optional[str]] = ContextVar("arx_correlation_id", default=None)


def validate_or_generate_uuidv4(val: Optional[str]) -> str:
    """
    Validates that the provided string is a strictly compliant RFC-4122 UUIDv4.
    If the input is missing, non-string, whitespace, oversized, non-v4 (e.g. v1),
    or contains newline/injected characters, a new safe UUIDv4 is generated.
    """
    if not isinstance(val, str):
        return str(uuid.uuid4())

    cleaned = val.strip()
    # RFC-4122 string representation is exactly 36 chars with hyphens
    if len(cleaned) != 36:
        return str(uuid.uuid4())

    try:
        parsed = uuid.UUID(cleaned, version=4)
        if parsed.version == 4 and str(parsed) == cleaned.lower():
            return str(parsed)
    except (ValueError, AttributeError, TypeError):
        pass

    return str(uuid.uuid4())


def get_request_id() -> Optional[str]:
    """Returns the current request ID from context or None if outside a request lifecycle."""
    return _REQUEST_ID_CTX.get()


def set_request_id(request_id: Optional[str]) -> Token:
    """Sets the active request ID in context and returns the restoration token."""
    return _REQUEST_ID_CTX.set(request_id)


def reset_request_id(token: Token) -> None:
    """Restores the previous request ID context using the token."""
    _REQUEST_ID_CTX.reset(token)


def get_correlation_id() -> Optional[str]:
    """Returns the current correlation ID from context or None if outside a request lifecycle."""
    return _CORRELATION_ID_CTX.get()


def set_correlation_id(correlation_id: Optional[str]) -> Token:
    """Sets the active correlation ID in context and returns the restoration token."""
    return _CORRELATION_ID_CTX.set(correlation_id)


def reset_correlation_id(token: Token) -> None:
    """Restores the previous correlation ID context using the token."""
    _CORRELATION_ID_CTX.reset(token)
