"""FastAPI Backend Application Entry Point."""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analytics, volatility, screener, regimes, cache, smart_money
from api.middleware.rate_limiter import RedisRateLimitMiddleware
from api.middleware.api_key_auth import ApiKeyAuthMiddleware

logger = logging.getLogger("api.main")

# Detect production vs. local development
IS_PRODUCTION = os.getenv("ENVIRONMENT", "production").lower() == "production"


async def warmup_core_assets():
    """Background task to pre-fetch and warm up SQLite cache for core universe assets on container boot."""
    await asyncio.sleep(2)
    try:
        from analyst_dashboard.data.market_db import MarketDatabaseEngine
        import yfinance as yf
        db = MarketDatabaseEngine()
        core_symbols = ["LNTH", "CIEN", "NVDA", "AAPL", "MSFT", "PLTR", "SPY", "QQQ"]
        for sym in core_symbols:
            try:
                existing = db.get_daily_candles(sym, limit=5)
                if not existing:
                    hist = yf.Ticker(sym).history(period="1y", interval="1d")
                    if not hist.empty:
                        db.save_daily_candles(sym, hist)
                        logger.info(f"Successfully pre-warmed SQLite cache for {sym}")
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Pre-warming skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(warmup_core_assets())
    yield
    task.cancel()


app = FastAPI(
    title="ARX Terminal API",
    description="High-performance async REST API for multi-asset analytics, GARCH volatility forecasting, and Hidden Gems screening.",
    version="1.0.0",
    lifespan=lifespan,
    # Disable interactive docs in production to prevent API surface exposure
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# 1. Strict CORS Whitelist Configuration
ALLOWED_ORIGINS = [
    "https://www.arxterminal.com",
    "https://arxterminal.com",
    "https://finance-xp8.pages.dev",
    "http://localhost:3000",
    "http://localhost:3005",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3005",
    "http://localhost:8000",
]

# Allow a single additional origin set via environment variable (e.g. staging preview)
extra_origin = os.getenv("ALLOWED_ORIGIN", "")
if extra_origin and extra_origin not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Tightened: only exact arxterminal.com subdomains and the specific pages.dev project.
    # No longer matches arbitrary *.pages.dev or *.vercel.app wildcards.
    allow_origin_regex=r"https://(www\.)?arxterminal\.com|https://finance-xp8\.pages\.dev|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization", "Accept", "Origin", "User-Agent"],
)

# 2. Backend Security Headers Middleware (HSTS, nosniff, frame protection)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    return response

# 3. API Key Authentication (enforced in production when ARX_API_KEY env var is set)
app.add_middleware(ApiKeyAuthMiddleware)

# 4. Distributed Redis & Memory-Fallback Rate Limiter Middleware
app.add_middleware(RedisRateLimitMiddleware, default_limit=120, window_seconds=60)

# 5. Global Production Error Masking Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    if IS_PRODUCTION:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Internal details have been masked for security.",
            },
        )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
        },
    )

# 6. API Routers
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(volatility.router, prefix="/api/v1/volatility", tags=["Volatility & Forecasting"])
app.include_router(screener.router, prefix="/api/v1/screener", tags=["Screener"])
app.include_router(regimes.router, prefix="/api/v1/regimes", tags=["Market Regimes"])
app.include_router(cache.router, prefix="/api/v1/cache", tags=["Cache Management"])
app.include_router(smart_money.router, prefix="/api/v1/smart-money", tags=["Smart Money & Flow"])


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "online"}
