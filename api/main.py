"""FastAPI Backend Application Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analytics, volatility, screener, regimes, cache
from api.middleware.rate_limiter import RedisRateLimitMiddleware

app = FastAPI(
    title="Financial Market Analysis API",
    description="High-performance async REST API for multi-asset analytics, GARCH volatility forecasting, and Hidden Gems screening.",
    version="1.0.0",
)

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "online", "version": "1.0.0"}
