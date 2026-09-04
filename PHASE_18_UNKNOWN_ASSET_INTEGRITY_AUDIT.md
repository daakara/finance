# PHASE 18 — UNKNOWN / UNSUPPORTED ASSET HALLUCINATION & END-TO-END DATA INTEGRITY AUDIT

**Audit Executed**: 2026-09-04T08:53:30+02:00  
**Operating Roles**: Senior Staff Product Engineer, Data Architect, Quantitative Systems Auditor, UX/Trust Specialist  
**Repository State**: Git d97b6c -> Clean Working Tree  
**Pytest Baseline**: 282/282 Passed (100%)  
**Build Baseline**: 99/99 Pages Prerendered Cleanly  

---

## 1. Executive Summary & Core Security Invariant

Phase 18 addresses the ultimate boundary of financial epistemic safety:
**What happens when a user inputs an asset that is completely unknown, uncataloged, fake, or has insufficient data?**

\text{UNKNOWN} \neq \text{FAVORABLE} \quad | \quad \text{UNKNOWN} \neq \text{NEGATIVE} \quad | \quad \text{UNKNOWN} \neq \text{ACTIONABLE}

\text{PLACEHOLDER} \neq \text{DATA} \quad | \quad \text{ZERO} \neq \text{NO ACTIVITY} \quad | \quad \text{EMPTY} \neq \text{NEGATIVE}

\text{CURATED} \neq \text{LIVE} \quad | \quad \text{SYNTHETIC} \neq \text{MARKET DATA}

### The Core Security Invariant
> **IF asset identity or market data cannot be positively verified, THEN downstream analytics MUST NOT invent evidence.**

An unsupported asset must never inherit:
1. MASTER_ASSET_CATALOG fundamentals
2. Another asset's price or a fallback \.00 price
3. Default or synthetic technical indicators (RSI 56.4, ATR, VWAP)
4. Synthetic Brownian motion / drift candles disguised as real historical tape
5. Fabricated multi-year growth forecasts or moat narratives
6. Generic Stage 2 or Bullish verdicts
7. Fabricated optimal entry, stop loss, or take profit levels
8. Position sizing units or cash buying power calculations
9. Positive or negative confluence score bias

---

## 2. Forensic Discovery: Identified Vulnerabilities

Our adversarial investigation uncovered **5 critical fallback leaks** across the frontend and backend architectures:

### Vulnerability 1: Frontend Client-Side Synthetic Fallback Leak
- **Location**: rontend/lib/api.ts (etchAssetAnalytics & generateFallbackAnalytics)
- **Mechanism**: When an unknown asset (e.g. FAKEETF123 or NONEXISTENT_TICKER_999) was queried and both backend and Yahoo Finance returned 404, etchAssetAnalytics fell back to generateFallbackAnalytics.
- **Hallucination**:
  - Invented a deterministic price based on symbol hash (e.g. \.23).
  - Synthesized 252 fake OHLCV candles.
  - Assigned SHARED_FACTOR_SCORES defaults: growthScore: 80, qualityScore: 78, verdict: "Bullish Stage 2 Alignment", piotroskiFScore: 7.
  - Fabricated an upcoming catalyst readout and multi-year growth forecast.

### Vulnerability 2: Master Baseline Default Price Leak (\.00)
- **Location**: rontend/lib/masterCatalog.ts (getMasterBaselinePrice & getMasterBaselineQuote)
- **Mechanism**: getMasterBaselinePrice(symbol, fallback = 100.0) defaulted to 100.0 when a symbol was not in CATALOG_BASELINE_PRICES.
- **Downstream Contamination**:
  - /stock/[ticker]/page.tsx rendered $100.00 spot price for GLX or unknown tickers, calculating a $93.00 stop loss and $100.00 entry zone.
  - /compare/[pair]/page.tsx rendered $100.00 for unknown comparison tickers.

### Vulnerability 3: Screener Custom Query Fallback Leak
- **Location**: rontend/app/screener/page.tsx (generateBuiltinGems)
- **Mechanism**: When a user entered an unknown ticker in the screener search box, CATALOG_BASELINE_PRICES[sym] ?? 100.0 assigned \.00.
- **Downstream Contamination**:
  - Fabricated entry zones (\.50 - \.50), stop losses (\.50), take profits (\.50), and R:R ratios (2.33:1) for completely non-existent assets.
  - Position sizer modal accepted the \ fake price.

### Vulnerability 4: Execution Ladder Generation on Insufficient History (< 50 sessions)
- **Location**: nalyst_dashboard/analyzers/optimal_execution.py (calculate_trade_levels)
- **Mechanism**: When len(price_df) < 50 (such as GLX which has only 47 historical daily candles), the engine flagged setup_pattern = "Trend Evidence Incomplete (< 50 Sessions)", but still calculated numerical entry zones, stop losses, take profits, and a 2.17:1 R:R ratio.
- **Epistemic Invariant Breach**: Unverifiable trend setups cannot be given actionable numerical trade parameters.

### Vulnerability 5: Backend Multi-Year Forecast & Moat Synthesis Leak
- **Location**: nalyst_dashboard/analyzers/catalysts.py & rontend/lib/assetRegistry.ts
- **Mechanism**: When company info was empty, the catalyst engine synthesized a generic multi-year forecast projecting 2.5x price expansion and fabricated moat descriptions.

---

## 3. Unknown-Asset State Machine & Remediation Strategy

| State | Definition | Market Price | Candles | Fundamentals | Execution Levels | Position Sizing | UI Posture |
|---|---|---|---|---|---|---|---|
| **UNSUPPORTED / UNKNOWN** | Ticker does not resolve on upstream exchange | None | [] | None | None | **DISABLED** | ⚠️ Asset Unavailable |
| **INSUFFICIENT_HISTORY** | Real asset with < 50 historical daily bars | Live spot | Live bars | Verified (if filed) | None | **DISABLED** | 🔍 Research Required (< 50 Sessions) |
| **MISSING_FUNDAMENTALS** | Real asset with price action but no SEC filings | Live spot | Live bars | None (0 F-Score) | Constrained | Active with warning | ⏳ Fundamentals Unverified |
| **SUPPORTED_CATALOG** | Verified asset in MASTER_ASSET_CATALOG | Live spot | Live bars | Verified 10-K/Q | Active | Full active | 🟢 Actionable / ⏳ Stage 4 Wait |

---

## 4. Remediation Plan

1. **P0 — Client-Side Fallback Quarantine**: In rontend/lib/api.ts, if an asset is not in MASTER_ASSET_CATALOG and fails upstream resolution, etchAssetAnalytics must throw or return _dataSource: "unavailable" without generating synthetic candles, prices, or factor scores.
2. **P0 — Eliminate \.00 Catalog Baseline**: In rontend/lib/masterCatalog.ts, getMasterBaselinePrice must return undefined for uncataloged assets.
3. **P0 — Execution Engine Insufficient History Clamp**: In optimal_execution.py, when len(price_df) < 50, optimal_entry_min, optimal_entry_max, stop_loss, 	ake_profit_1/2, and isk_reward_ratio must be None with execution_status: "INSUFFICIENT_HISTORY".
4. **P1 — Stock Detail & Compare Defense**: In /stock/[ticker] and /compare/[pair], if price is unavailable, render "Price Unavailable" and suppress execution ladders.
5. **P1 — Screener Defense**: In generateBuiltinGems, uncataloged tickers without live prices must not fabricate \ prices or trade setups.
6. **P2 — Catalyst & Moat Honesty**: For uncataloged assets without SEC disclosures, return empty forecasts and explicit awaiting-filing notices.
