"""High-Performance Redis & Memory-Fallback Rate Limiter Middleware for FastAPI."""

import time
import os
import logging
from typing import Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Try to initialize Redis connection
REDIS_URL = os.getenv("REDIS_URL", "")
redis_client = None

if REDIS_URL:
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        redis_client.ping()
        logger.info("Connected to Redis distributed rate limiter successfully.")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis at {REDIS_URL}, falling back to in-memory sliding window: {e}")
        redis_client = None

# Fallback in-memory store: { client_ip: [timestamp1, timestamp2, ...] }
in_memory_rate_store: Dict[str, list] = {}


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter backed by Redis with zero-downtime in-memory fallback.
    - Default Global: 120 req / min per IP
    - Analytics: 60 req / min per IP
    - Screener / Heavy: 30 req / min per IP
    """

    def __init__(self, app, default_limit: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds

    def _get_route_limit(self, path: str) -> int:
        if "/api/v1/screener" in path:
            return 30
        elif "/api/v1/analytics" in path:
            return 60
        elif "/api/v1/volatility" in path:
            return 60
        return self.default_limit

    async def dispatch(self, request: Request, call_next) -> Response:
        # Exclude health checks, CORS pre-flights (OPTIONS), and static documentation
        if request.method == "OPTIONS" or request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        # Extract client IP (handling reverse proxy forwarded headers)
        forwarded = request.headers.get("x-forwarded-for")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown_client")

        limit = self._get_route_limit(request.url.path)
        current_time = int(time.time())
        window_start = current_time - self.window_seconds

        # 1. Distributed Redis Check
        if redis_client:
            try:
                key = f"ratelimit:{client_ip}:{request.url.path.split('/')[3] if len(request.url.path.split('/')) > 3 else 'root'}"
                # Use Redis Sorted Set (ZSET) for precise sliding window
                pipeline = redis_client.pipeline()
                pipeline.zremrangebyscore(key, 0, window_start)
                pipeline.zadd(key, {str(current_time) + ":" + str(time.perf_counter()): current_time})
                pipeline.zcard(key)
                pipeline.expire(key, self.window_seconds + 5)
                results = pipeline.execute()

                request_count = results[2]
                if request_count > limit:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Too Many Requests",
                            "message": f"Rate limit of {limit} requests/minute exceeded for this endpoint.",
                            "retry_after_seconds": self.window_seconds,
                        },
                        headers={"Retry-After": str(self.window_seconds)},
                    )
            except Exception as e:
                logger.warning(f"Redis rate limit check failed: {e}. Executing with in-memory fallback.")

        # 2. In-Memory Sliding Window Fallback
        if not redis_client:
            store_key = f"{client_ip}:{request.url.path.split('/')[3] if len(request.url.path.split('/')) > 3 else 'root'}"
            timestamps = in_memory_rate_store.get(store_key, [])
            # Prune expired timestamps
            valid_timestamps = [t for t in timestamps if t > window_start]
            if len(valid_timestamps) >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "message": f"Rate limit of {limit} requests/minute exceeded for this endpoint.",
                        "retry_after_seconds": self.window_seconds,
                    },
                    headers={"Retry-After": str(self.window_seconds)},
                )
            valid_timestamps.append(current_time)
            in_memory_rate_store[store_key] = valid_timestamps

        response = await call_next(request)
        return response
