# Project: daakara/finance (ARX Terminal)

## Architecture
- **Backend**: FastAPI 0.110+ on Python 3.11 with Starlette, Uvicorn, SQLite (`MarketDatabaseEngine`, `HistoryDatabaseEngine`), and external providers (Yahoo Finance, FRED API, SEC EDGAR, FINRA Transparency).
- **Middleware**: Redis Sliding Window Rate Limiter (with in-memory fallback), API Key Authentication (HMAC constant-time), Security Headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options), CORS Middleware with strict origin whitelist.
- **Quant Engines**:
  - `OptimalExecutionEngine`: Mark Minervini VCP Stage 2 execution ladders (Stop, Entry Min/Max, TP1, TP2, R:R >= 1.85:1).
  - `TraderArchetypes`: 5-Persona Investor Models (Warren Buffett, Nancy Pelosi, Stanley Druckenmiller, Jim Simons, David Gardner).
  - `AdvancedRiskAnalyzer`: Cornish-Fisher 95/99% Modified VaR, Sortino downside deviation, GMM market regimes.
  - `SmartMoneyEngine`: Congressional STOCK Act disclosures (45-day statutory violation penalty), SEC Form 4 insider transactions, options flow sweeps.
  - `ConfluenceEngine`: 4-Pillar composite conviction scoring & Fractional Half-Kelly position sizing.
  - `GemScreener`: Peter Lynch GARP & Greenblatt Magic Formula screeners.
- **Frontend**: Next.js 14.2 App Router (SSG `output: "export"`, `trailingSlash: true`, 98 static routes), React 18, TailwindCSS, TradingView Lightweight Charts v4 with Cyber Dark & Paper Light themes, SSR fallback shells (`TerminalSsrShell`, `SmartMoneySsrShell`, `CompareSsrShell`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | FastAPI Route Integrity & FINRA Fix | Fix `api/routes/smart_money.py` `get_finra_darkpool` endpoint to call `finra_fetcher.get_ats_metrics` and return `{"symbol": sym, "metrics": ...}` | M1 | Survey Backend |
| 2 | Regimes Route Collision Fix | Separate `/api/v1/regimes/current` and `/api/v1/regimes/{symbol}` to prevent querying ticker "CURRENT" | M1 | Survey Backend |
| 3 | Confluence Binary Gap Risk Fix | Adjust `confluence_engine.py` binary catalyst scoring penalty (<24h to earnings) so score strictly drops below 50.0 | M1 | Survey Backend / Quant |
| 4 | Production Error Masking | Standardize exception handling in `analytics.py` and `regimes.py` to prevent leaking raw tracebacks in production | M1 | Survey Backend |
| 5 | SSL & Import Sanitization | Add module-level `import os` and remove global `CURL_DISABLE_SSL_VERIFY = '1'` in `gem_fetchers.py` | M1 | Survey Backend |
| 6 | SQLite WAL & Concurrency Resilience | Enable `PRAGMA journal_mode = WAL;`, `busy_timeout = 5000;`, and connection retry logic with backoff in `db_engine.py` & `market_db.py` | M1 | Survey Backend |
| 7 | Rate Limiter Resilience & Eviction | Fix Redis timeout fallback logic so in-memory sliding window executes on Redis failure; add LRU eviction to `in_memory_rate_store` | M1 | Survey Backend |
| 8 | Lifespan Startup Consolidation | Remove deprecated `@app.on_event("startup")` and consolidate background pre-warming into `lifespan(app)` | M1 | Survey Backend |
| 9 | Minervini Execution Ladder Monotonicity & R:R Floor | Enforce strict monotonicity ($Stop < Entry_{min} \le Entry_{max} < TP1 < TP2$) and mandatory $R:R \ge 1.85:1$ in `optimal_execution.py` | M2 | Survey Quant |
| 10 | Master Catalog Crypto Key Fix | Fix `getMasterAsset()` in `masterCatalog.ts` to preserve `BTC-USD`, `ETH-USD`, `SOL-USD` keys without stripping `-USD` | M2 | Survey Quant |
| 11 | Sortino Downside Deviation Standardization | Standardize downside deviation calculation across `AdvancedRiskAnalyzer`, `RiskAnalysisEngine`, and `PortfolioMetrics` to root-mean-square over full sample $N$ | M2 | Survey Quant |
| 12 | 5-Persona Investor Alignment & Anti-Hallucination | Sector-aware theses for David Gardner (freight/logistics), dynamic regime-aware theses for Druckenmiller, and expanded Pelosi policy coverage | M2 | Survey Quant |
| 13 | Cornish-Fisher VaR Invariant Test | Add invariant test verifying $\text{MVaR}_{99} \ge \text{Parametric VaR}_{95}$ under negative skewness | M2 | Survey Quant |
| 14 | STOCK Act Forensics & Late-Filer Audit | Verify statutory filing lag penalties (-32 pts for >45d) and committee jurisdiction alignment scoring | M2 | Survey Quant |
| 15 | Static Route Spot Price Hydration | Wire `CATALOG_BASELINE_PRICES` into `/stock/[ticker]/page.tsx` and `/compare/[pair]/page.tsx` instead of hardcoded $100.00 | M3 | Survey Frontend |
| 16 | Screener Genuine R:R Math | Harmonize Screener generator target/stop percentages (+10.5% / -4.5%) to naturally yield $\ge 1.85:1$ without synthetic clamp | M3 | Survey Frontend |
| 17 | Brand Nomenclature Parity | Harmonize legacy `"Finance Terminal"` references in secondary layout metadata and JSON-LD schemas to canonical `"ARX Terminal"` | M3 | Survey Frontend |
| 18 | ESLint Non-Interactive Config | Scaffold `.eslintrc.json` in `frontend/` so `npm run lint` executes cleanly without prompt | M3 | Survey Frontend |
| 19 | SSR Fallback Shell Verification | Verify pre-rendered semantic HTML shells (`TerminalSsrShell`, `SmartMoneySsrShell`, `CompareSsrShell`) for headless search bots | M3 | Survey Frontend |
| 20 | Viewport & Theme Verification | Verify 320px-1750px responsiveness, modal z-index supremacy (`z-[1000]` > `z-[999]`), and Cyber Dark / Paper Light contrast | M3 | Survey Frontend |
| 21 | Comprehensive E2E Testing Suite | Build 4-tier opaque-box test suite covering 100% of features in tests/ and frontend contract tests | M4 | Survey All |
| 22 | Final Acceptance Gate & Adversarial Hardening | Pass 100% of Python and Next.js test suites, run Tier 5 adversarial coverage hardening with challenger loop | M5 | Survey All |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Full-Stack Production Readiness & Security Audit | Features 1-8: FastAPI routes, middleware, SQLite data engines, error masking, rate limiter, FINRA fix, Confluence fix | none | DONE |
| M2 | Quantitative Models & Financial Domain Invariants | Features 9-14: Minervini ladders R:R >= 1.85, 5-persona models, Sortino standardization, Cornish-Fisher VaR, STOCK Act forensics, masterCatalog crypto fix | M1 | IN_PROGRESS |
| M3 | Frontend UI/UX & Coherent Insights Review | Features 15-20: Next.js 98 static routes, SSR shells, spot price hydration, screener R:R math, brand parity, ESLint config, theme & viewport checks | M2 | PLANNED |
| M4 | E2E Testing Track Infrastructure & Test Suite | Feature 21: Comprehensive test infra, 4-tier test cases, TEST_READY.md publication | parallel to M1-M3 | PLANNED |
| M5 | Final Milestone: 100% Test Pass & Adversarial Hardening | Feature 22: Phase 1 (100% test pass) + Phase 2 (Tier 5 adversarial hardening) + Forensic Audit Gate | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
### API ↔ Frontend
- `GET /api/v1/analytics/{symbol}`: Returns JSON with keys `symbol`, `price`, `technicals`, `risk`, `fundamentals`, `archetypes`, `execution`, `confluence`, `smart_money`.
- `GET /api/v1/smart-money/finra-darkpool/{symbol}`: Returns `{"symbol": str, "metrics": {"symbol": str, "ats_share_pct": float, "off_exchange_short_pct": float, "is_darkpool_heavy": bool, "summary": str}}`.
- `GET /api/v1/regimes/current`: Returns `{"symbol": "SPY", "regime": str, "annualized_volatility": float, "sortino_ratio": float, "recommendation": str, ...}`.
- `GET /api/v1/regimes/{symbol}`: Returns regime metrics for specified ticker.

### Quant Engines ↔ Analytics Pipeline
- `OptimalExecutionEngine.calculate_execution_plan(symbol, price, atr_14, highs_52w, lows_52w)`:
  - `stop_loss < entry_min <= entry_max < take_profit_1 < take_profit_2`
  - `risk_reward_ratio = round((take_profit_1 - price) / (price - stop_loss), 2) >= 1.85`
- `ConfluenceEngine.calculate_confluence(...)`:
  - `confluenceScore: float [0, 100]`
  - Imminent binary risk (<24h to earnings) with moderate fundamentals strictly reduces score < 50.0.

## Code Layout
- `api/`: FastAPI application, routes (`analytics.py`, `volatility.py`, `screener.py`, `regimes.py`, `smart_money.py`, `cache.py`), middleware (`rate_limiter.py`, `auth.py`, `security.py`, `cors.py`).
- `analyst_dashboard/`: Core quantitative analytics, data fetchers (`fred_fetcher.py`, `finra_fetcher.py`, `gem_fetchers.py`), analyzers (`optimal_execution.py`, `trader_archetypes.py`, `advanced_risk_analyzer.py`, `confluence_engine.py`, `gem_screener.py`, `smart_money.py`), and storage (`db_engine.py`, `market_db.py`).
- `frontend/`: Next.js 14 App Router application (`app/`, `components/`, `lib/`, `public/`).
- `tests/`: Pytest unit and integration test suites.
