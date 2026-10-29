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
│    • Deployment: Cloudflare Pages (149 SSG Static Pages, Global Edge CDN, Zero Cold Starts)      │
│                                                                                                  │
│ ⚡ BACKEND & API SERVICES TIER                                                                    │
│    • Framework: Python 3.11 / 3.12 + FastAPI + Uvicorn Async Workers                             │
│    • Rate Limiting: Redis Middleware (RedisRateLimitMiddleware) with Memory Fallback             │
│    • Ingestion Bridges: Yahoo Finance (yfinance), EODHD API, FRED Federal Reserve Macro API      │
│    • Deployment: Railway Container (DOCKERFILE, Auto-Restart, 2 Worker Threads)                 │
│                                                                                                  │
│ 🧮 QUANTITATIVE & CONTROL THEORY TIER                                                             │
│    • Volatility Models: Cornish-Fisher Modified VaR (95%/99%), GARCH(1,1), Realized 14-ATR       │
│    • Execution Engines: Mark Minervini VCP Pattern Detector, Linda Raschke 20 EMA Pullback       │
│    • Safety Systems: Self-Healing Runtime Invariant Circuit Breakers & Liquidity Hazard Gate     │
│    • Political Flow: Congressional STOCK Act (PL 112-105) + SEC Form 4 Insider Alignment Index   │
│    • Governance & Control: GOV-001 Cross-Sprint Integrity Validator & SCS-v1 Action Engine       │
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

---

## 5. Market-Wide Discovery Architecture & Research Ratification (ARX-DOC-2026-10-29)

*Formal architectural baseline incorporation of Directive ARX-MW-004H Product Owner ratification and evidence reconciliation.*

### 5.1 Architecture Direction Ratification
- **Ratification Verdict:** `ARCHITECTURE_DIRECTION_RATIFIED`
- **Canonical Architecture:** **Model E+ (Enhanced Security Registry Core) with Derived F1 Opportunity Projections**.
- **Ratification Reference:** [`ARX_MW004H_FINAL_ARCHITECTURE_RATIFICATION_AND_CANDIDATE_FREEZE.md`](file:///c:/Users/akara/Documents/Projects/finance-arx-market-wide-top35/docs/research/market_wide_poc/ARX_MW004H_FINAL_ARCHITECTURE_RATIFICATION_AND_CANDIDATE_FREEZE.md)
- **Investigation Closure:** The comparative architecture investigation between Model E+ and Model F1 is officially concluded. No further comparative architecture studies are authorized without a new, material, reproducible counterexample.

### 5.2 Preserved Git & Research Identities
- **Current Mainline Reference:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` (`origin/main`, PR #129 sizing isolation merged).
- **Historical Research Baseline Anchor:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53` (Anchor for PR #128 and MW-004A..H investigation).
- **Research Implementation Candidate Commit:** `f154032152710a51b209d80a4a21dfb655d312e4` (Branch `research/arx-mw002-independent-discovery`).
- **Research Ratification Record Commit:** `ac074ea487a55c276ee64db59f3aa8f278d2b992`.
- **Draft PR #128 Remote Reference:** `3fff68108593f304275cdf18b696c8697f5b8257` (`origin/poc/arx-market-wide-top35`).  
  *Preservation Policy:* PR #128 is strictly retained as an open, unmerged research reference. It is **not** merged, closed, or described as containing the final corrective implementation.

### 5.3 Bounded Market Store Dataset Reconciliation
- **MW-004G Baseline File Hash:** `ab31fdffa0b8b20af714bf12becf55f9ea81d17018fdda81c77e3c7bb1ade615` (4,714,496 bytes).
- **MW-004H Frozen Candidate File Hash:** `00c0690efebfa5e419beb3890c3bd251b2182b0994f50a4c32359bb18834d273` (5,705,728 bytes).
- **Logical Data Parity vs. File-Byte Identity:**
  - `asset_ohlcv_daily` row count: **40,310 rows** in both baselines (delta = +0).
  - Logical candle data content hash: `27934736360afd0fbd266c20c3663191a0a46426bd2ecc4494f96ae43805f41e` (100% bitwise identical).
  - Added multi-pillar research tables verified empty (0 rows each):
    - `asset_catalyst_registry`: 0 rows
    - `asset_factor_snapshots`: 0 rows
    - `asset_ohlcv_provenance`: 0 rows
    - `insider_disclosures`: 0 rows
    - `sqlite_sequence`: 0 rows
  - The SQLite file-level size increase (+991,232 bytes) is verified to result solely from SQLite B-tree page pre-allocation for these 5 empty schema definitions. Zero price bars were modified.

### 5.4 Seven Corrective Verifications & Test Reproducibility
The 7 corrective findings are verified on the frozen candidate (`f154032`):
1. `DEFECT-001` (Ranking Immutability): Deep-copy candidate isolation in `top35_selector.py`.
2. `DEFECT-002` (Research Authority Containment): Dataclass barriers, serialization firewall (`to_dict()`), and `ConfluenceEngine` 0-share suppression.
3. `FN-001` (Missing Evidence Integrity): Auto-enforcement and validation of required missing tokens after deserialization.
4. `FN-002` (Cohort Qualification Revalidation): Predicate enforcement in selector ($R:R \ge 2.0$, candle count $\ge 50$, Stage $\ne 4$).
5. `FN-003` (Score Provenance Transparency): `ScoreProvenance` enum distinguishing authentic 0.0 scores from missing-data 0.0 values.
6. `FN-004` (Trust Boundary Firewall): Rejection of forged execution permissions in `verify_research_trust_boundary()`.
7. `FN-005` (Replay Classification Binding): Rejection of `EXACT_ORIGINAL_REPLAY` without PIT SEC filings; `HISTORICAL_RECONSTRUCTION` enforced.

*Reproducible Clean-Checkout Test Commands:*
```bash
python -m pytest tests/test_arx_mw002_adversarial_falsification.py  # 18 passed in 4.92s
python tests/test_arx_mw002_adversarial_falsification.py            # 18 passed in 2.74s
python scratch/test_independent_candidate_verification.py           # 7/7 checks passed
python scratch/verify_mw004h_manifest_integrity.py                  # 100% bitwise pass
```

### 5.5 Production & Governance Boundary Safeguards
- **Production Quarantine:** Market-wide discovery remains quarantined under `/poc-market-wide`.
- **Explicit Restrictions:** No production Radar integration, no full-market scanner scheduling, no live financial execution, no commercial/SEC egress, no prospective learning updates, and no Sprint 5 execution (`SPRINT5_EXECUTION_AUTHORIZED = FALSE`).
- **Separation of Concerns:** Locally verified research contracts are strictly decoupled from unverified production financial consumers. Release authorization requires separate gate clearance.
