# PHASE 15 — PRODUCTION LAUNCH & REAL-WORLD RELIABILITY AUDIT
**Date**: September 3, 2026  
**Auditor Roles**: Senior Staff Product Engineer, QA Architect, UX Auditor, Data Integrity Engineer, Production Reliability Reviewer  
**Audit Status**: DISCOVERY COMPLETED — FINDINGS LEDGER COMPILED  
**Prior Baseline**: Commit `e9e5e41` (Phase 14 Remediations, 223/223 Pytests, 99/99 Next.js pages)

---

## 1. Executive Summary

Phase 15 executed an adversarial, real-world reliability audit designed to challenge the assumption that passing automated test suites guarantees production readiness. Operating under strict read-only observation and validation rules, the audit probed beyond functional code correctness into **epistemic safety, deceptive defaults, synthetic data leakage into decision engines, and cross-screen consistency**.

### Core Verdict: **CONDITIONAL — BLOCKED FOR LAUNCH UNTIL P0/P1 REMEDIATION**

While core execution pathways, 99 static Next.js routes, and 223 pytest unit tests pass cleanly, **seven (7) production reliability findings** have been identified. Crucially, **two (2) P0 integrity findings** directly violate core financial invariants:
1. **P0 Epistemic Hazard (`FINDING-15-01`)**: `PreFlightChecklistModal` defaulted missing inputs (risk-reward, trend, smart money, macro) favorably, granting **100% conviction and "🟢 CLEARED TO EXECUTE"** to securities with zero verified data (`UNKNOWN = FAVORABLE`).
2. **P0 Synthetic Decision Leak (`FINDING-15-02`)**: `generateFallbackAnalytics` and `fetchDirectYahooFinanceChart` fabricated synthetic options call sweeps (`$1.45M CALL SWEEP at 14:23:05`), which fed into decision engines evaluating distribution traps and institutional flow (`SYNTHETIC DATA IN DECISION LOGIC`).

Full remediation of these findings, followed by adversarial regression testing, is required prior to public launch.

---

## 2. Invariants Ledger & Enforcement Status

| # | Invariant Rule | Discovery Status | Violations Identified |
|---|----------------|------------------|-----------------------|
| 1 | `UNKNOWN ≠ FAVORABLE` | **BREACHED** | `PreFlightChecklistModal` defaults missing RR to 2.5 and VIX to 15.4; `confluence_engine.py` defaults missing fundamentals to 65 pts; static compare pages default missing factor scores to 85. |
| 2 | `UNKNOWN ≠ NEGATIVE` | **VERIFIED** | Missing technical evidence correctly routes posture to `RESEARCH`, not `AVOID`. |
| 3 | `UNAVAILABLE ≠ ACTIONABLE` | **BREACHED** | Unverified assets cleared for execution in PreFlight modal; price = 0 defaults to $100 in trade logger. |
| 4 | `SYNTHETIC DATA MUST NEVER ENTER DECISION LOGIC` | **BREACHED** | Direct Yahoo fetcher and fallback generator fabricate synthetic options flow items consumed by distribution trap analyzers. |
| 5 | `STALE DATA MUST NEVER APPEAR FRESH` | **BREACHED** | Starter portfolio hardcoded NVDA at $128.50 (vs $224.41 catalog baseline) on cold storage read. |
| 6 | `INVALIDATED ≠ ACQUIRE` | **VERIFIED** | Hard breach enforcement verified in `assessmentEngine.ts` (`isActuallyBreached` enforces `EXIT_REVIEW`). |
| 7 | `RESEARCH ≠ WATCH` & `RESEARCH ≠ AVOID` | **VERIFIED** | Discrete enum postures maintained across terminal state machine. |
| 8 | *Missing evidence must reduce confidence, never silently increase it.* | **BREACHED** | Missing inputs in PreFlight modal and Confluence engine default to favorable scores. |
| 9 | *A calculation must never be presented as verified evidence unless its inputs are verified.* | **BREACHED** | Advanced Terminal displays hardcoded RSI (62.4/36.3) and VaR (-3.2%) when candle history is insufficient. Backend `optimal_execution.py` used `min_periods=5` for 50 SMA. |
| 10 | *Cross-screen representations of the same security must not materially contradict each other.* | **BREACHED** | `/stock/deck` displays `🟢` title despite Stage 4 classification; `/compare/cprx-vs-powi` displays 85 composite score for unverified assets. |

---

## 3. Architecture & Data Flow Verification

```mermaid
flowchart TD
    subgraph Ingestion Layer
        A[FastAPI Backend /analytics] -->|Timeout: 4000ms| B[Client fetchAssetAnalytics]
        B -->|Fallback 1: Failure/Timeout| C[Direct Yahoo Finance Chart]
        C -->|Fallback 2: Network Down| D[generateFallbackAnalytics]
    end

    subgraph State & Normalization
        B --> E[SpotPriceRegistry]
        C --> E
        D --> E
        E --> F[getPersistedMarketSnapshot / IndexedDB]
    end

    subgraph Decision & Epistemic Engines
        E --> G[insightGenerator.ts]
        G --> H[deriveAssessmentState / Pure Engine]
        H --> I[AdaptiveTerminal: Guided / Standard / Advanced]
        B --> J[OptimalExecutionCard & PreFlightChecklistModal]
        B --> K[CompositeConvictionCard]
    end

    subgraph Public Surface
        E --> L[UniversalOmniSearch]
        E --> M[WatchlistSidebar]
        E --> N[Static /stock/[ticker]]
        E --> O[Static /compare/[pair]]
        E --> P[Portfolio /portfolio]
    end
```

### Critical Architectural Observations:
1. **Fallback Boundary Integrity**: `insightGenerator.ts` correctly flags `_dataSource === "fallback"` and forces `isTrendAvailable = false`, preventing moving averages from calculating on synthetic candles.
2. **Data Layer Leakage**: While `insightGenerator.ts` was protected, `api.ts` attached synthetic `optionsFlow` and `optimalExecution` objects during Yahoo direct fetching and fallback generation. Downstream consumer components (`OptimalEntryExitCard`, `CompositeConvictionCard`) were exposed to this synthetic data.

---

## 4. Production Data Freshness Audit

| Ticker | Canonical Source | Catalog Baseline | Cold-Cache Display | Live API Flow | Freshness Grade |
|--------|------------------|------------------|--------------------|---------------|-----------------|
| **NVDA** | SEC 10-Q / Market Feed | $224.41 | $224.41 | Real-time / 15m Delayed | **A** |
| **FIX** | SEC 10-Q / Market Feed | $1,560.13 | $1,560.13 | Real-time / 15m Delayed | **A** |
| **CPRX** | SEC 10-Q / Halted Simulation | $18.42 | $18.42 | 5 Candles (Insufficient) | **B** (Correctly flags incomplete trend) |
| **AAPL** | SEC 10-Q / Market Feed | $319.64 | $319.64 | Real-time / 15m Delayed | **A** |
| **SYM** | SEC 10-Q / Market Feed | $28.50 | $28.50 | Real-time / 15m Delayed | **A** |

- **SpotPriceRegistry**: Functions reliably as an in-memory short-circuit cache preventing flickering.
- **Defect Identified**: `fetchScreenerGems` Line 1075 uses `reg?.price || snap?.currentPrice || 100.0`. If an asset exists in `MASTER_ASSET_CATALOG` but has neither a registry price nor snapshot, it defaults to `$100.0` instead of `CATALOG_BASELINE_PRICES[ticker]`.

---

## 5. Cross-Screen Consistency Audit

Tracing **NVDA, FIX, CPRX, AAPL, SYM, DECK** across 9 UI surfaces:

| Screen / Component | NVDA ($224.41) | FIX ($1,560.13) | CPRX ($18.42) | DECK (Stage 4) | Parity Status |
|-------------------|----------------|-----------------|---------------|----------------|---------------|
| 1. Home (`/`) | $224.41 | $1,560.13 | $18.42 | Stage 4 | **CONSISTENT** |
| 2. Screener (`/screener`) | $224.41 | $1,560.13 | $18.42 | Filtered / Await Base | **CONSISTENT** |
| 3. Detail (`/stock/[ticker]`) | $224.41 | $1,560.13 | $18.42 | Displays 🟢 in Title | **CONTRADICTION** (FINDING-15-05) |
| 4. Compare (`/compare`) | $224.41 | $1,560.13 | $18.42 | Stage 4 | **CONSISTENT** |
| 5. Static Compare (`/compare/[pair]`) | $224.41 | $1,560.13 | 85 Score (vs POWI) | N/A | **CONTRADICTION** (FINDING-15-05) |
| 6. Terminal (Guided/Std/Adv) | $224.41 | $1,560.13 | RSI 62.4 (Fake) | Stage 4 (Correction) | **CONTRADICTION** (FINDING-15-03) |
| 7. Portfolio (`/portfolio`) | $128.50 (Cold) | Live Sync | N/A | N/A | **CONTRADICTION** (FINDING-15-07) |
| 8. Position Sizer Modal | $224.41 | $1,560.13 | Guarded (N/A) | Guarded | **CONSISTENT** |
| 9. PreFlight Modal | $224.41 | $1,560.13 | 100% Cleared | Not Cleared | **CONTRADICTION** (FINDING-15-01) |

---

## 6. Real User Journey Testing (Journeys A through F)

### Journey A: First-time visitor lands on homepage
- **Observed Flow**: Loads SPY / NVDA cleanly. Spotlight presents high-confluence ideas.
- **Result**: **PASS**. Onboarding tour and vernacular mode persist cleanly.

### Journey B: User searches for custom or misspelled asset
- **Observed Flow**: OmniSearch parses "Google" -> resolves to GOOGL canonical. Unknown ticker "XYZ" shows "Run Live Analysis for Custom Asset".
- **Result**: **PASS**.

### Journey C: Screener -> Stock Detail -> Back navigation
- **Observed Flow**: Screener filter persists in URL `?goal=minervini_vcp`. Selecting asset navigates to terminal with `fromGoal=minervini_vcp&fromCount=4`. Top breadcrumb correctly links back without state loss.
- **Result**: **PASS**.

### Journey D: Head-to-head comparison
- **Observed Flow**: Dynamic `/compare?a=NVDA&b=AAPL` loads real-time multi-factor radar and metrics without unverified advantages. Static `/compare/cprx-vs-powi` displays unverified 85/100 default scores.
- **Result**: **FAIL** on static `/compare/[pair]` route (FINDING-15-05).

### Journey E: Adding position to portfolio and tracking across sessions
- **Observed Flow**: User opens modal, adds position, saves to `localStorage`. Event `finance:portfolio-updated` dispatches.
- **Result**: **PASS**, but initial cold default positions show stale NVDA price ($128.50) (FINDING-15-07).

### Journey F: Network failure & offline recovery
- **Observed Flow**: When backend API times out, system degrades to Yahoo direct fetch, then `generateFallbackAnalytics`.
- **Result**: **FAIL** — during degradation, synthetic CALL SWEEPs are injected into `smartMoney.optionsFlow` (FINDING-15-02).

---

## 7. Epistemic Language & Tone Audit

1. **Uncertainty Calibration**: Component language in `insightGenerator.ts` and `assessmentEngine.ts` uses objective, epistemic terms: *"Evidence incomplete. Further quantitative research required before taking position."*
2. **Tone Violations**:
   - `frontend/app/stock/[ticker]/page.tsx`: Generates titles starting with `🟢` for all assets, including those undergoing severe Stage 4 corrections (`/stock/deck`, `/stock/podd`).
   - `frontend/app/compare/[pair]/page.tsx`: Uncataloged assets claim "High Quality Compounder" and "Secular Growth Leader".

---

## 8. Fallback & Default Audit

1. **Numeric `|| 100` Defaults**:
   - `insightGenerator.ts:26`: `const safePrice = currentPrice > 0 ? currentPrice : 100;` generates valid stop losses and profit targets when price is 0.
   - `OptimalEntryExitCard.tsx:157`: `2500 / (current_price || 100)` calculates 25 shares and logs $0.00 entry price.
   - `portfolio/page.tsx:57`: `let price = 100.0;` populates $100.00 entry price in position add modal on empty feed.
2. **Scoring `??` Defaults**:
   - `PreFlightChecklistModal.tsx:110`: `safeRR` defaults to `2.5`, passing Check 1 on empty data.
   - `confluence_engine.py:65`: Missing fundamentals default to Piotroski 7 and score 65.
   - `CompositeConvictionCard.tsx:132`: Missing factor scores default to Piotroski 7 and score 70.

---

## 9. Position Sizing & Action Safety Audit

1. **`DayTraderPositionSizer.tsx`**: Verified guarded by `isTechnicalsUnavailable` banner when technicals or prices are missing.
2. **`PositionSizerModal.tsx`**: Correctly checks `price > 0`, `stopLoss < price`, and `accountSize > 0`.
3. **`PreFlightChecklistModal.tsx`**: **CRITICAL DEFECT**. Missing inputs default favorably, allowing unverified setups to receive `🟢 CLEARED TO EXECUTE`.

---

## 10. Mobile & Accessibility Reality Check

1. **320px–375px Viewports**:
   - `AdvancedTerminalView`: 6-column quant ribbon wraps into `grid-cols-2`, remaining readable.
   - `CompositeConvictionCard`: 4-pillar tabs stack cleanly.
   - Modals (`PreFlightChecklistModal`, `PositionSizerModal`): Use `max-h-[90vh] overflow-y-auto` with fixed sticky footers.
2. **Accessibility (a11y)**:
   - Modals implement `role="dialog"`, `aria-modal="true"`, focus trapping, and Escape key listeners.
   - All interactive tab buttons implement `role="tab"` and `aria-selected`.

---

## 11. Performance, Memory & Async Resilience

1. **Race Conditions in `app/page.tsx`**: Handled via `let isMounted = true` cleanup flag. Fast switching between tickers does not allow stale responses to overwrite newer requests.
2. **Network Request Cancellation**: `fetchAssetAnalytics` does not accept an `AbortSignal`, allowing abandoned requests to continue in flight. While memory safe, adding `AbortSignal` propagation will save network bandwidth.

---

## 12. SEO / Public Page Integrity

1. **Static Build Verification**: Next.js generates 99/99 static pages without build errors.
2. **SEO Parity Defect**:
   - `/stock/[ticker]`: Generates misleading `🟢` titles for Stage 4 assets and defaults unverified assets to 85/100 score and 8/9 Piotroski.
   - `/compare/[pair]`: Generates static comparison tables for uncataloged securities with fabricated 85/100 scores and verdicts.

---

## 13. Comprehensive Findings Ledger

```
===================================================================================================
FINDING-15-01 | P0 | PreFlightChecklistModal Favorable Clearance on Missing Data
File: frontend/components/PreFlightChecklistModal.tsx (Lines 107-148)
Description:
Missing risk-reward ratio defaults to 2.5, VIX defaults to 15.4, trend defaults to passing, and 
smart money defaults to passing. An asset with completely unavailable data scores 5/5 (100% conviction) 
and is declared "🟢 CLEARED TO EXECUTE".
Violated Invariants: UNKNOWN ≠ FAVORABLE, UNAVAILABLE ≠ ACTIONABLE.

FINDING-15-02 | P0 | Synthetic Options Call Sweeps Injected into Production Feeds
File: frontend/lib/api.ts (Lines 639-654, 900-915)
Description:
When fallback analytics or direct client Yahoo charts are loaded, synthetic options flow is fabricated
with { time: "14:23:05", premium: "$1.45M", type: "CALL SWEEP", sentiment: "Bullish" }. This fake 
options flow feeds directly into smartMoney.optionsFlow and alters distribution trap analysis.
Violated Invariants: SYNTHETIC DATA MUST NEVER ENTER DECISION LOGIC.

FINDING-15-03 | P1 | Hardcoded RSI, Relative Strength, and VaR in Advanced Terminal
File: frontend/lib/insightGenerator.ts (Lines 394, 405-406) & AdvancedTerminalView.tsx
Description:
advanced.rsi is hardcoded to `isStage4 ? 36.3 : 62.4`, relativeStrengthScore to `isStage4 ? 62 : 94`,
and var95Pct to 3.2 instead of computing authentic 14-period RSI from candle closes or displaying N/A.
Assets with insufficient history (e.g. CPRX with 5 candles) display a fabricated 62.4 RSI.
Violated Invariants: A calculation must never be presented as verified evidence unless its inputs are verified.

FINDING-15-04 | P1 | Confluence Engine & Optimal Execution Backend Invariant Breaches
File: analyst_dashboard/analyzers/confluence_engine.py (Lines 58-80) & optimal_execution.py (Line 85)
Description:
1. confluence_engine.py defaults missing fundamentals to Piotroski 7 and score 65.0, awarding 65 points
   on uncataloged assets instead of failing closed with 0 points.
2. optimal_execution.py calculates 50 SMA using `close.rolling(50, min_periods=5)`, allowing a 5-day
   average to pose as a 50-day SMA, violating the 50-session truth invariant.
Violated Invariants: UNKNOWN ≠ FAVORABLE, A calculation must never be presented as verified evidence.

FINDING-15-05 | P1 | Misleading Metadata & Unverified Defaults on Static Stock & Compare Pages
File: frontend/app/stock/[ticker]/page.tsx & frontend/app/compare/[pair]/page.tsx
Description:
1. /stock/[ticker] prefixes all page titles with 🟢 even for Stage 4 correction stocks (DECK, PODD),
   and defaults unverified assets to Piotroski 8 and score 85.
2. /compare/[pair] defaults missing composite scores to 85, Piotroski to 8, and verdicts to "High Quality
   Compounder" via `?? 85` and `?? 8`.
Violated Invariants: UNKNOWN ≠ FAVORABLE, Cross-screen representations must not contradict each other.

FINDING-15-06 | P2 | Silent $100 Fallback in Terminal & Portfolio Add Modal
File: frontend/lib/insightGenerator.ts (Line 26) & frontend/app/portfolio/page.tsx (Line 57)
Description:
When asset price is 0 or unverified, insightGenerator falls back to safePrice = 100, manufacturing
valid stop-loss and target prices for a non-existent $100 spot price. portfolio/page.tsx defaults entry
price to $100.00 if unpopulated.
Violated Invariants: UNAVAILABLE ≠ ACTIONABLE.

FINDING-15-07 | P2 | Starter Portfolio NVDA/AAPL Baseline Price Disparity
File: frontend/lib/portfolio.ts (Lines 55-77)
Description:
Starter portfolio hardcodes NVDA at $128.50 and AAPL at $226.50 on initial cold load, conflicting
with the verified master catalog baselines ($224.41 and $319.64).
Violated Invariants: STALE DATA MUST NEVER APPEAR FRESH, Cross-screen representations must match.
===================================================================================================
```

---

## 14. Prioritized Remediation Plan

The remediation must proceed strictly in order of priority:

### **Batch 1: P0 Critical Safety & Synthetic Data Elimination**
1. **Remediate `FINDING-15-01` (`PreFlightChecklistModal.tsx`)**:
   - If `currentPrice <= 0` or missing, display explicit `"Live Price Unavailable"` banner and disable execution.
   - If `riskRewardRatio` is undefined or unverified, mark Check 1 as `FAILED` (`isRRPassed = false`).
   - If any required check lacks evidence, do NOT allow `isCleared = true`. Require all 5 checks to have verified evidence before granting clearance.
2. **Remediate `FINDING-15-02` (`api.ts`)**:
   - In `generateFallbackAnalytics` and `fetchDirectYahooFinanceChart`, set `smartMoney: { congressTrades: [], optionsFlow: [] }`.
   - Never fabricate synthetic options trades. Mark options flow as empty when live institutional feeds are unavailable.

### **Batch 2: P1 Truth Invariants & Epistemic Calibration**
3. **Remediate `FINDING-15-03` (`insightGenerator.ts` & `AdvancedTerminalView.tsx`)**:
   - Implement authentic RSI-14 calculation from `candles` when `candles.length >= 15 && !isFallbackFeed`. If `< 15` or fallback, set `rsi = undefined` and render `"N/A (< 15 sessions)"`.
   - Compute authentic 1D VaR from returns or set `var95Pct = undefined` and render `"N/A"`.
   - Guard `relativeStrengthScore` to only render when verified RS is available.
4. **Remediate `FINDING-15-04` (`confluence_engine.py` & `optimal_execution.py`)**:
   - In `confluence_engine.py`, when `fundamental_data` is empty or missing, set `fund_score = 0.0`, `fund_status = "unavailable"`, `fund_plain = "Fundamental SEC filings unavailable for this asset."`.
   - In `optimal_execution.py`, require `len(close) >= 50` for `sma_50`. If `< 50`, set `sma_50 = None` and do not generate false Stage 2 breakout pivot claims.
5. **Remediate `FINDING-15-05` (`stock/[ticker]/page.tsx` & `compare/[pair]/page.tsx`)**:
   - In `stock/[ticker]/page.tsx`, dynamically set title emoji based on stage (`🔴` for Stage 4, `🟡` for Stage 1/Watch, `🟢` only for Stage 2 Actionable). If asset is uncataloged, display `"Unverified Security"` rather than defaulting to 85/100 and Piotroski 8.
   - In `compare/[pair]/page.tsx`, replace `?? 85` and `?? 8` with authentic catalog data or honest `"N/A"`.

### **Batch 3: P2 Safe Fallbacks & Starter Portfolio Parity**
6. **Remediate `FINDING-15-06` (`insightGenerator.ts`, `OptimalEntryExitCard.tsx`, `portfolio/page.tsx`)**:
   - When `currentPrice <= 0`, do NOT default to 100. Render price as `"N/A"`, targets as `"N/A"`, and disable portfolio logging.
7. **Remediate `FINDING-15-07` (`portfolio.ts`)**:
   - Update `defaultPositions` in `portfolio.ts` to use `CATALOG_BASELINE_PRICES.NVDA` ($224.41) and `CATALOG_BASELINE_PRICES.AAPL` ($319.64).

---

## 15. Independent Adversarial Verification Plan

An independent automated verification suite (`tests/test_adversarial_phase15_verification.py`) will test:
1. **Adversarial Pre-Flight Test**: Pass an asset with empty / zero data to `PreFlightChecklistModal` logic and verify `isCleared` is FALSE and conviction is 0%.
2. **Synthetic Data Prohibition Test**: Verify that neither `generateFallbackAnalytics` nor `fetchDirectYahooFinanceChart` produces non-empty `optionsFlow`.
3. **RSI Authenticity Test**: Verify that CPRX or an asset with 5 candles yields `rsi: undefined` and renders `N/A`, not `62.4`.
4. **Backend Confluence Fail-Closed Test**: Verify `ConfluenceEngine.calculate_confluence("UNKNOWN", fundamental_data={})` returns 0 fundamental points and unavailable status.
5. **50-SMA Observation Window Test**: Verify `OptimalExecutionEngine.calculate_trade_levels` with a 10-candle DataFrame returns `None` for 50-day moving average and does not claim 50 SMA breakout.
6. **Static Route Truth Test**: Verify `/stock/deck` metadata does not include `🟢`, and uncataloged static compare pairs do not render `85 / 100`.

---

## 16. Conclusion & Next Action

Discovery is complete. The findings ledger is documented. Awaiting user review and authorization to proceed with Batch 1 (P0) remediation.
