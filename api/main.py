"""FastAPI Backend Application Entry Point."""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analytics, volatility, screener, regimes, cache, smart_money
from api.middleware.rate_limiter import RedisRateLimitMiddleware

logger = logging.getLogger("api.main")


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
    title="Financial Market Analysis API",
    description="High-performance async REST API for multi-asset analytics, GARCH volatility forecasting, and Hidden Gems screening.",
    version="1.0.0",
    lifespan=lifespan,
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

