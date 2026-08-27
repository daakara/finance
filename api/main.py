"""FastAPI Backend Application Entry Point."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analytics, volatility, screener, regimes, cache, smart_money
from api.middleware.rate_limiter import RedisRateLimitMiddleware

app = FastAPI(
    title="Financial Market Analysis API",
    description="High-performance async REST API for multi-asset analytics, GARCH volatility forecasting, and Hidden Gems screening.",
    version="1.0.0",
)

# 1. Strict CORS Whitelist Configuration
ALLOWED_ORIGINS = [
    "https://finance-xp8.pages.dev",
    "http://localhost:3000",
    "http://localhost:3005",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3005",
    "http://localhost:8000",
]

# Allow custom environment-based origins if specified
extra_origin = os.getenv("ALLOWED_ORIGIN", "")
if extra_origin and extra_origin not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.pages\.dev|https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Distributed Redis & Memory-Fallback Rate Limiter Middleware
app.add_middleware(RedisRateLimitMiddleware, default_limit=120, window_seconds=60)

# 3. API Routers
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(volatility.router, prefix="/api/v1/volatility", tags=["Volatility & Forecasting"])
app.include_router(screener.router, prefix="/api/v1/screener", tags=["Screener"])
app.include_router(regimes.router, prefix="/api/v1/regimes", tags=["Market Regimes"])
app.include_router(cache.router, prefix="/api/v1/cache", tags=["Cache Management"])
app.include_router(smart_money.router, prefix="/api/v1/smart-money", tags=["Smart Money & Flow"])


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "online", "version": "1.0.0"}

