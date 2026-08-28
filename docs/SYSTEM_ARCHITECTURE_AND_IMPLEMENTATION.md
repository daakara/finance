# 🏗️ Master System Architecture & Implementation Specification

*A comprehensive technical breakdown of the full-stack architecture, data flow, invariant circuit breakers, and persistent storage infrastructure.*

---

## 1. High-Level Technology Stack

`
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     FULL SYSTEM STACK MAP                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 🌐 FRONTEND EDGE TIER                                                                            │
│    • Framework: Next.js 14.2 (App Router) + React 18 + TypeScript                                │
│    • Styling & Motion: TailwindCSS + Lucide Icons + Custom Glow / Pulse Animations               │
│    • Charts & Canvas: TradingView Lightweight Charts (Sub-millisecond Canvas Rendering)          │
│    • Deployment: Cloudflare Pages (77 SSG Static Pages, Global Edge CDN, Zero Cold Starts)       │
│                                                                                                  │
│ ⚡ BACKEND & API SERVICES TIER                                                                    │
│    • Framework: Python 3.11 + FastAPI + Uvicorn Async Workers                                    │
│    • Rate Limiting: Redis Middleware (RedisRateLimitMiddleware) with Memory Fallback             │
│    • Ingestion Bridges: Yahoo Finance (yfinance), EODHD API, FRED Federal Reserve Macro API      │
│    • Deployment: Railway Container (DOCKERFILE, Auto-Restart, 2 Worker Threads)                 │
│                                                                                                  │
│ 🧮 QUANTITATIVE & CONTROL THEORY TIER                                                             │
│    • Volatility Models: Cornish-Fisher Modified VaR (95%/99%), GARCH(1,1), Realized 14-ATR       │
│    • Execution Engines: Mark Minervini VCP Pattern Detector, Linda Raschke 20 EMA Pullback       │
│    • Safety Systems: Self-Healing Runtime Invariant Circuit Breaker (_enforce_execution_invariants)│
│    • Political Flow: Congressional STOCK Act (PL 112-105) + SEC Form 4 Insider Alignment Index   │
│                                                                                                  │
│ 💽 STORAGE & PERSISTENCE TIER                                                                    │
│    • Backend Persistent Store: Railway Persistent Volume (web-volume mounted at /root)           │
│    • Market Database: SQLite NVMe Store (/root/.finance_market_store.db)                         │
│    • Outcome History Ledger: SQLite NVMe Store (/root/.finance_platform_history.db)              │
│    • Client-Side State Memory: Browser localStorage / IndexedDB (Portfolios, Capital, Alerts)    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
`

---

## 2. End-to-End Data & Execution Flow

`
                                  ┌───────────────────────────┐
                                  │   USER BROWSER / EDGE     │
                                  │ (Cloudflare Pages Static) │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 │ User Selects Ticker (e.g.    │ LocalStorage Hydration:      │
                 │ NVDA, NOW, COIN, ISRG)       │ Account Equity (,000)     │
                 ▼                              ▼                              │
┌─────────────────────────────────┐   ┌──────────────────────────────────┐     │
│   FastAPI Analytics Router      │   │   PositionSizerModal             │     │
│   (api/routes/analytics.py)     │   │   (PositionSizerModal.tsx)       │     │
└────────────────┬────────────────┘   └──────────────────────────────────┘     │
                 │                                                             │
                 ▼                                                             │
┌─────────────────────────────────┐                                            │
│   Persistent Market Database    │                                            │
│   (MarketDatabaseEngine)        │                                            │
│   • Check /root/.market_store.db│                                            │
│   • Fetch Yahoo / EODHD if cold │                                            │
└────────────────┬────────────────┘                                            │
                 │                                                             │
                 ▼                                                             │
┌─────────────────────────────────┐                                            │
│   Quantitative Analyzers        │                                            │
│   • Cornish-Fisher VaR (Risk)   │                                            │
│   • Minervini VCP (Execution)   │                                            │
│   • Factor DNA (Piotroski F)    │                                            │
└────────────────┬────────────────┘                                            │
                 │                                                             │
                 ▼                                                             │
┌─────────────────────────────────┐                                            │
│   RUNTIME INVARIANT BREAKER     │                                            │
│   _enforce_execution_invariants │                                            │
│   • Stop Loss [-3.5%, -7.0%]    │                                            │
│   • Target 1 [+4.0%, +24.5%]    │                                            │
│   • Stage 4 Pivot <= Spot * 1.16│                                            │
│   • Risk:Reward in [1.20, 3.85] │                                            │
└────────────────┬────────────────┘                                            │
                 │                                                             │
                 ├──────────────────────────────┐                              │
                 ▼                              ▼                              │
┌─────────────────────────────────┐   ┌──────────────────────────────────┐     │
│   History Database Engine       │   │   REST JSON Payload Return       │     │
│   trade_recommendation_history  │   │   (Optimal Execution + Risk +    │     │
│   (Logged to NVMe for Outcomes) │   │    Smart Money Flow Confluence)  │     │
└─────────────────────────────────┘   └─────────────────┬────────────────┘     │
                                                        │                      │
                                                        ▼                      │
                                      ┌──────────────────────────────────┐     │
                                      │   OptimalEntryExitCard           │◄────┘
                                      │   • 🏛️ Smart Money Inflow Badge │
                                      │   • ⚠️ Macro Volatility Buffer   │
                                      │   • 🎯 1-Click Sizer & Alerting  │
                                      └──────────────────────────────────┘
`

---

## 3. Mathematical & Control Invariants

The platform operates under strict mathematical boundaries to guarantee zero corrupted signals:

`
┌──────────────────────────────┬──────────────────────────────┬────────────────────────────────────────┐
│ Invariant Name               │ Permitted Numerical Range    │ Failure Action & Circuit Breaker       │
├──────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ Swing Stop Loss              │ [-3.5%, -7.0%] of Spot       │ Clamped automatically to valid corridor│
│ Day Trader Stop Loss         │ [-0.9%, -2.2%] of Spot       │ Clamped automatically to intraday ATR  │
│ Target 1 (TP1)               │ [+4.0%, +24.5%] of Spot      │ Clamped to prevent split data blowouts │
│ Target 2 (TP2)               │ [TP1 + 3.0%, +35.0%]         │ Enforced monotonic spacing > TP1       │
│ Risk-to-Reward Ratio         │ [1.20 : 1.0, 3.85 : 1.0]     │ Clamped to prevent fake 10:1+ spikes   │
│ Stage 4 Breakout Pivot       │ <= Spot * 1.16               │ Capped at 50-SMA baseline              │
│ Price Consensus Drift        │ .00 Across Registries      │ Hard assertion in static audit suite   │
└──────────────────────────────┴──────────────────────────────┴────────────────────────────────────────┘
`


---

## 4. Cloudflare Pages & Railway Infrastructure Setup Details

### 🌐 Cloudflare Pages (Frontend Edge Deployment)
* **Production URL**: https://finance-xp8.pages.dev
* **Framework Preset**: Next.js (Static Export / SSG)
* **Build Command**: cd frontend && npm install && npm run build
* **Build Output Directory**: rontend/out
* **Environment Variables**:
  * NEXT_PUBLIC_API_URL: https://web-production-e370b.up.railway.app
  * NODE_VERSION: 20.x
* **Static Generation (SSG)**: 77 static routes pre-rendered during build, including:
  * / (Main Analytics Terminal with 4 workspaces)
  * /screener (Hidden Gems & Quant Screener)
  * /portfolio (Anonymous IndexedDB Portfolio Tracker)
  * /smart-money & /smart-money/late-filers (STOCK Act Radar)
  * /compare & /compare/[pair] (7 normalized comparison pairs)
  * /stock/[ticker] (42 pre-cached equity ticker hubs)
  * /strategy/[type] (5 strategy hubs: Minervini, Magic Formula, etc.)
  * /politician/[slug] & /committee/[slug] (Congressional trackers)

---

### ⚡ Railway (Backend API & Persistent Volume Deployment)
* **Production API URL**: https://web-production-e370b.up.railway.app
* **Service Name**: web (tranquil-radiance / production)
* **Builder**: Dockerfile (python:3.11-slim)
* **Start Command**: sh -c 'uvicorn api.main:app --host 0.0.0.0 --port  --workers 2'
* **Restart Policy**: ON_FAILURE (max 10 retries)
* **Persistent Volume Setup**:
  * **Volume Name**: web-volume
  * **Mount Path**: /root
  * **Persistent Files on Volume**:
    * /root/.finance_market_store.db (Historical OHLCV candles, fundamental snapshots)
    * /root/.finance_platform_history.db (Trade recommendation outcome ledger, GARCH history)
* **Environment Variables**:
  * PORT: 8000 (Injected automatically by Railway)
  * DATA_DIR: /root (Points SQLite engines to the persistent volume)
  * REDIS_URL: Optional (Falls back to in-memory rate limiting if omitted)
  * EODHD_API_KEY: Optional (High-throughput candle fallback)
  * FRED_API_KEY: Optional (Direct Federal Reserve macro yields)
