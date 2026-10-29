# ARX TERMINAL — ARCHITECTURE & ENGINE CONTRACT RECONCILIATION
## Multi-Engine Topology, Cross-Engine Ownership, Data Flow & Financial Authority Boundaries
**Review Date:** 29 October 2026  
**Mainline Baseline SHA:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Governing Directive:** `MASTER DIRECTIVE ARX-DOC-2026-10-29 (Section 4)`  
**Document Authority:** `CANONICAL_ACTIVE / FAIL_CLOSED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Summary & Engine Topology

ARX Terminal coordinates twelve specialized computational engines. To prevent Cross-Boundary Blindness, data contamination, and unauthorized capital allocation, each engine operates under an explicit data contract and strict separation between **Research Observation Authority** and **Financial Execution Authority**.

```
[ Research Observation Plane ]
  ├── 1. Technical Analysis Engine ─────┐
  ├── 2. Minervini VCP Setup Engine ────┤
  ├── 3. Fundamentals & Valuation ─────┼──> [ 10. Confluence Synthesis & Research Cockpit ]
  ├── 4. Macro & Regime Engine ─────────┤         │ (Non-executable Research Opportunities)
  ├── 5. Smart Money Engine (HELD) ─────┤         │
  ├── 6. Corporate Events Engine (HELD) ┘         ▼
  └── 12. Market-Wide Discovery (Model E+ POC) ──> [ 8. Radar & Opportunity Presentation ]
                                                         │
==================== STRICT GOVERNANCE & PROVENANCE FIREWALL ====================
                                                         │
[ Financial Governance & Execution Plane ]               ▼
  ├── 11. Portfolio & Provenance Engine ───────> [ 9. Risk Governor & Position Sizer ]
  │     (Verified Broker Telemetry vs Paper)             │ (Mandatory Fail-Closed Zero-Size)
  │                                                      ▼
  └────────────────────────────────────────────> [ Execution Boundary: HOLD_EXEC_001 ]
                                                   (Zero Live Order Egress Authorized)
```

---

## 2. Comprehensive 12-Engine Contract Reconciliation

### Engine 1: Technical Analysis Engine
- **Purpose & Scope:** Compute moving averages (20 EMA, 50 SMA, 200 SMA), VWAP, ATR(14), RSI(14), Cornish-Fisher VaR, and dual-horizon candlestick geometry.
- **Authoritative Producer:** `engines/technical_engine.py`, `api/routes/analytics.py`, `api/routes/market.py`.
- **Primary Inputs:** Daily and intraday OHLCV time-series from primary exchange feeds / EODHD fallback.
- **Input Provenance:** Direct exchange tape timestamps. Requires $\ge 50$ historical daily bars for baseline trend analysis; 200 bars for macro moving averages.
- **Output Contracts:** `TechnicalIndicators` envelope containing trend status, volatility metrics, and support/resistance levels.
- **Freshness Semantics:** Intraday bars cache TTL = 60s; Daily bars cache TTL = 3600s.
- **Error & Fallback:** If historical bars $< 50$, emits `INSUFFICIENT_DATA` error state. No silent moving average zero-fills.
- **Financial Authority:** Research signal only. Does not authorize order placement.
- **Production Status:** `DEPLOYED_STABLE` on Railway backend (`de5d246`).

### Engine 2: Minervini VCP (Volatility Contraction Pattern) Setup Engine
- **Purpose & Scope:** Algorithmic identification of progressive volatility contraction sequences, volume dry-up at pivots, and Cup-with-Handle patterns based on Mark Minervini Trend Template criteria.
- **Authoritative Producer:** `engines/technical_engine.py` (VCP module), holdout test corpus.
- **Primary Inputs:** Daily OHLCV adjusted price series, market index benchmark ($SPY / $QQQ) for Relative Strength rating.
- **Evaluation Cutoffs:** Stage 2 uptrend prerequisite: Price $> 150 \text{ SMA} > 200 \text{ SMA}$, 52-week low $+30\%$, 52-week high $-25\%$. Contraction count $2 \le T \le 5$.
- **Output Contracts:** `VCPSetupResult` containing pattern stage, pivot buy point, volume dry-up ratio ($< 50\%$ 50-day volume), and strict stop-loss anchor (pivot low).
- **Error & Fallback:** Returns `NO_SETUP_DETECTED` with detailed failure stage token if criteria fail.
- **Financial Authority:** Analytical pattern recognition. Requires confluence confirmation and risk governor approval.
- **Production Status:** `DEPLOYED_STABLE`.

### Engine 3: Fundamentals & Valuation Engine
- **Purpose & Scope:** Compute intrinsic valuation models (DCF with terminal fade, Clean Surplus Residual Income, Normalized EV/EBITDA multiples), financial health (Piotroski F-Score, Altman Z-Score), and capital allocation efficiency (ROIC vs. WACC).
- **Authoritative Producer:** `engines/fundamental_engine.py`, `api/routes/fundamentals.py`.
- **Primary Inputs:** SEC 10-K / 10-Q standardized financial statements, share count lineage, free cash flow histories.
- **Input Provenance:** Sourced SEC filings via structured provider feeds. Historical filing dates strictly enforced (Point-in-Time).
- **Output Contracts:** `FundamentalValuationReport` containing base/bull/bear fair values, margin of safety percentage, and quality grade.
- **Freshness Semantics:** Updated on filing publication; cached TTL = 86400s (24 hours).
- **Error & Fallback:** Emits `FUNDAMENTAL_DATA_ABSENT` if filings $> 180$ days stale. Missing balance sheet line items block DCF execution.
- **Financial Authority:** Advisory valuation boundary. Sizing caps based on margin of safety.
- **Production Status:** `DEPLOYED_STABLE`.

### Engine 4: Macro & Market Regime Engine
- **Purpose & Scope:** Determine market regime state (Bull Confirmed, Uptrend Under Pressure, Market in Correction, Stagflation, Deflation) using US Treasury yield curves (10Y-2Y, 10Y-3M), high-yield credit spreads, and index distribution day counts.
- **Authoritative Producer:** `api/routes/macro.py`, `api/routes/regimes.py`.
- **Primary Inputs:** FRED macro series (DGS10, DGS2, BAMLH0A0HYM2), index breadth metrics.
- **Output Contracts:** `MarketRegimeEnvelope` containing regime state enum, risk multiplier clamp ($0.0 \times \text{ to } 1.0 \times$), and distribution count.
- **Freshness Semantics:** EOD macro sync; cache TTL = 14400s (4 hours).
- **Financial Authority:** Hard sizing governor constraint. A regime of `MARKET_IN_CORRECTION` clamps the portfolio risk multiplier to $\le 0.5 \times$.
- **Production Status:** `DEPLOYED_STABLE`.

### Engine 5: Smart Money Engine (`HELD`)
- **Purpose & Scope:** Analysis of SEC Form 4 insider acquisitions, cluster buys, and institutional 13F filings.
- **Authoritative Producer:** `api/routes/smart_money.py`, `docs/domain/smart_money/governance_control_plane/*`.
- **Governing Hold:** `HOLD_SM1_001` (Active). Live production SEC ingestion is **strictly disabled**.
- **Execution Mode:** Bounded offline historical replay and clean-room golden corpus verification.
- **Output Contracts:** `SmartMoneySignalEnvelope` with cluster intensity and disposition flags.
- **Financial Authority:** Purely observational. Zero influence on production risk governor.
- **Production Status:** `QUARANTINED_LOCAL_REPLAY_ONLY`.

### Engine 6: Corporate Events Engine (`HELD`)
- **Purpose & Scope:** Corporate earnings release calendar, dividend ex-dates, splits, and SEC material 8-K disclosures.
- **Authoritative Producer:** `api/routes/market.py`.
- **Governing Hold:** `HOLD_SEC_001` (Active). Live SEC EDGAR egress is blocked in production containers.
- **Execution Mode:** Static fixture playback and mock calendar schedules.
- **Output Contracts:** `CorporateEventRiskEnvelope` emitting `EARNINGS_IMMINENT` warning if earnings $\le 5$ days away.
- **Financial Authority:** Warning filter. Sizing governor enforces $0.0 \times$ allocation when entering pre-earnings blackout window.
- **Production Status:** `QUARANTINED_FIXTURES_ONLY`.

### Engine 7: Radar & Opportunity Screening Engine
- **Purpose & Scope:** Multi-asset scanning and sorting across technical setups, momentum breakouts, and fundamental value traps.
- **Authoritative Producer:** `api/routes/screener.py`, PR #127 (`fix/d058b-radar-reliability-and-provenance`).
- **Primary Inputs:** Pre-computed technical snapshots, VCP candidate pools, and volume liquidity filters.
- **Output Contracts:** `RadarScreenResponse` with explicit state models: `PENDING`, `AVAILABLE`, `EMPTY_UNIVERSE`, `STALE`, and `ERROR`.
- **Coverage Contract:** In accordance with PR #110 remediation, a failed scan (`ERROR`) strictly emits `coverage_status = ERROR` and `scan_coverage_pct = 0.0`. Never reports 100% complete coverage on failure.
- **Financial Authority:** Presentation surface only.
- **Production Status:** Production backend active on `de5d246`; PR #127 reliability hardening open in Draft PR.

### Engine 8: Confluence Synthesis Engine
- **Purpose & Scope:** Multi-pillar weighted scoring synthesizing Technicals ($35\%$), Fundamentals ($30\%$), Setups/VCP ($20\%$), and Macro ($15\%$).
- **Authoritative Producer:** `docs/scoring-semantics.md`, Confluence scoring pipeline.
- **Scoring Remediation History:** Successfully resolved historical `confluenceScore` vs `confluence_score` attribute mismatch that previously triggered silent fallback to 65.0. Missing evidence is now explicitly surfaced via `ScoreProvenance` tokens (`GENUINE_COMPUTED`, `PENALIZED_MISSING`, `IMPUTED_NEUTRAL`).
- **Output Contracts:** `ConfluenceScore` ($0.0 - 100.0$) with breakdown across all active pillars.
- **Financial Authority:** Research priority ranking only. Minimum threshold of 75.0 required for tactical consideration.
- **Production Status:** Mainline verified.

### Engine 9: Risk Governance & Position Sizing Engine
- **Purpose & Scope:** Compute exact share allocations, dollar risk exposure, and portfolio heat limits based on account size, entry price, stop-loss distance, and Cornish-Fisher VaR.
- **Authoritative Producer:** `engines/risk_engine.py`, `frontend/lib/simulation/governorSizingEngine.ts`, PR #129 (`fix(portfolio)`).
- **Primary Inputs:** Validated entry price, structural stop-loss price, verified account equity from Engine 11, macro regime risk multiplier.
- **Mathematical Invariant:**
  $$\text{Shares} = \min\left( \frac{\text{Verified Equity} \times \text{Risk Pct}}{\text{Entry} - \text{Stop}}, \frac{\text{Verified Equity} \times \text{Max Position Pct}}{\text{Entry}} \right) \times \text{Regime Multiplier}$$
- **Fail-Closed Provenance Gate (PR #129):** Position sizing strictly rejects untrusted or unverified equity states (`NOT_LOADED`, `unverified_paper`, `unconfigured`, `unreachable`). If account equity provenance is not `telemetry_verified`, output allocation is strictly **0 shares / \$0.00**.
- **Financial Authority:** Authoritative mathematical sizing constraint. Enforces fail-closed safety.
- **Production Status:** Merged into mainline (`e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`). Pending production deployment.

### Engine 10: Research Synthesis & Cockpit Presentation Engine
- **Purpose & Scope:** Synthesis E V3 three-mode terminal presentation (Guided, Standard, Quant) uniting chart canvases, setup cards, and valuation metrics.
- **Authoritative Producer:** `api/routes/cockpit.py`, `frontend/app/` (PR #131).
- **Output Contracts:** Structured JSON cockpit payload. Discloses analytical confidence and data source provenance.
- **Financial Authority:** Review-only. Every recommendation card displays the invariant notice: *"For Institutional Research & Evaluation Only — Not an Offer or Financial Execution Order"*.
- **Production Status:** PR #131 in open Draft state.

### Engine 11: Portfolio & Account Equity Engine
- **Purpose & Scope:** Real-time calculation of portfolio net asset value (NAV), verified holdings, cash balances, and equity provenance status.
- **Authoritative Producer:** `api/routes/portfolio.py`, `api/services/portfolio_service.py`, `frontend/lib/portfolio.ts` (PR #129).
- **Provenance Contract (RQ-05):**
  - `telemetry_verified`: Direct cryptographic session or server-authenticated broker read. Sizing permitted.
  - `unverified_paper`: Local simulated trading session. Strictly quarantined from live governor sizing.
  - `unconfigured`: Account credentials not established. Sizing disabled.
  - `unreachable`: Gateway timeout. Sizing disabled.
- **Quarantine Policy:** Browser `localStorage` mutations cannot override or simulate verified equity.
- **Production Status:** Merged into mainline (`e10de3b`).

### Engine 12: Market-Wide Discovery Strategy-Adapter Engine (`POC`)
- **Purpose & Scope:** Broad-universe opportunity screening and cohort classification across thousands of US common equities.
- **Authoritative Architecture:** **Model E+ (Enhanced Security Registry Core) with Derived F1-Style Opportunity Projections**.
- **Authoritative Producer:** `finance-arx-market-wide-top35`, candidate `f154032152710a51b209d80a4a21dfb655d312e4` (Tree `30345ac093a913e6a50c8d0a018d77d52083c010`).
- **Population Reconciliation:**
  - 13,308 SEC directory records
  - 5,792 eligible common stocks
  - 127 stored scannable equities in bounded clean store (2.19% stored coverage)
  - 41 Stage 1 rejections (liquidity, price, trading history)
  - 86 Stage 2 evaluated equities
  - 46 Cohort A qualified ($R:R \ge 2.0$, candle count $\ge 50$, Stage 2 uptrend)
  - 40 Cohort B developing opportunities
  - Top 35 displayed in research UI; 11 qualified overflow.
- **Replay Classification:** Enforced as `HISTORICAL_RECONSTRUCTION` due to lack of point-in-time SEC filing acceptance timestamps. `EXACT_ORIGINAL_REPLAY` is strictly prohibited.
- **Seven Corrective Verifications:** `DEFECT-001`, `DEFECT-002`, `FN-001` through `FN-005` verified on candidate `f154032`.
- **Governance Boundary:** Quarantined under `/poc-market-wide`. Zero live execution authority.
- **Production Status:** `RESEARCH_RATIFIED_LOCAL_CANDIDATE_FROZEN`.

---

## 3. Financial Execution Safeguards & Boundary Invariants

1. **`HOLD_EXEC_001` Hard Interlock:** Zero order transmission code paths exist in the production runtime. All trade actions emit simulated audit logs.
2. **Zero Inferred Authority:** Passing unit tests, passing CI builds, or local candidate freezes never constitute authority for live execution.
3. **Fail-Closed Defaults:** Any network interruption, missing data token, or schema ambiguity causes all dependent engines to halt gracefully and enter a safe, non-allocating error state.
