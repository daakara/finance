"""
HTTP Correlation and Request Tracing Middleware for ARX Observability.

Enforces:
1. Validation of incoming X-Request-ID or safe generation of UUIDv4.
2. Propagation of X-Correlation-ID or safe generation of UUIDv4.
3. Injection of both headers into response headers.
4. Non-interfering, thread-safe ContextVar assignment and cleanup.
5. Structured HTTP completion and failure event logging.
6. Fail-open telemetry safety: telemetry failures never mutate or halt API execution.
"""

import time
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .context import (
    validate_or_generate_uuidv4,
    set_request_id,
    reset_request_id,
    set_correlation_id,
    reset_correlation_id,
)
from .monitoring import capture_exception

logger = logging.getLogger("arx.observability.http")


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI / Starlette middleware for distributed request and correlation tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract headers and validate against strict UUIDv4 format
        raw_req_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        raw_corr_id = request.headers.get("x-correlation-id") or request.headers.get("X-Correlation-ID")

        req_id = validate_or_generate_uuidv4(raw_req_id)
        corr_id = validate_or_generate_uuidv4(raw_corr_id)

        # Bind context variables for the duration of this request task
        t_req = set_request_id(req_id)
        t_corr = set_correlation_id(corr_id)

        start_time = time.perf_counter()
        response: Response = None

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Ensure response contains correlation headers
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Correlation-ID"] = corr_id

            # Emit structured request completion event (fail-safe)
            try:
                event_name = "http_request_completed" if response.status_code < 500 else "http_request_failed"
                log_level = logging.INFO if response.status_code < 400 else (logging.WARNING if response.status_code < 500 else logging.ERROR)

                logger.log(
                    log_level,
                    f"{request.method} {request.url.path} responded {response.status_code} in {duration_ms}ms",
                    extra={
                        "event_name": event_name,
                        "route": request.url.path,
                        "method": request.method,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                        "request_id": req_id,
                        "correlation_id": corr_id,
                    },
                )
            except Exception as log_err:
                # Telemetry failure safety: logging failures must never crash request
                pass

            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            try:
                capture_exception(
                    exc,
                    context={
                        "route": request.url.path,
                        "method": request.method,
                        "request_id": req_id,
                        "correlation_id": corr_id,
                    },
                )
            except Exception:
                pass
            try:
                logger.error(
                    f"Unhandled exception during {request.method} {request.url.path}: {exc}",
                    exc_info=True,
                    extra={
                        "event_name": "application_exception",
                        "route": request.url.path,
                        "method": request.method,
                        "status_code": 500,
                        "duration_ms": duration_ms,
                        "request_id": req_id,
                        "correlation_id": corr_id,
                    },
                )
            except Exception:
                pass

            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Internal details have been masked for security.",
                },
                headers={
                    "X-Request-ID": req_id,
                    "X-Correlation-ID": corr_id,
                },
            )

        finally:
            # Guarantee token reset to prevent context leakage across interleaved tasks
            reset_request_id(t_req)
            reset_correlation_id(t_corr)
