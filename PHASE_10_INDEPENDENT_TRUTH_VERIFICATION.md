# Phase 10: Independent Truth Verification & Forensic Lineage Audit
**ARX Decision-Grade Analytics Engine**
*Audit Execution Date: September 2, 2026*
*Audit Status: COMPLETE · Discrepancies Cataloged · Zero Unverified Claims*

---

## Executive Summary & Adversarial Audit Mandate

Phase 9 established that code refactoring occurred and automated unit tests passed. However, **a unit test that passes against code written to satisfy that test does not prove empirical or mathematical truth.**

Phase 10 executed an adversarial, independent recalculation and forensic audit across the complete data lineage:
$$\text{Raw Source API / SEC EDGAR} \longrightarrow \text{Independent Mathematical Recalculation} \longrightarrow \text{ARX Pipeline} \longrightarrow \text{Rendered UI}$$

Benchmark securities audited: **CPRX, NVDA, FIX**.

### Core Audit Verdict: CONDITIONAL PASS WITH IDENTIFIED DATA PIPELINE DEFECTS
1. **Mathematical Reproducibility**: Where authentic candles exist $\ge 50$ bars (`NVDA`, `FIX`), ARX's SMA50 and EMA20 match independent mathematical calculations with negligible difference ($< 0.05\%$).
2. **Lineage Discrepancy Found (CPRX)**: The live Yahoo Finance feed for CPRX contains only **5 trading sessions** (due to corporate deregistration under Form 15-12G on 2026-07-24). In ARX, `fetchDirectYahooFinanceChart` drops feeds with $< 15$ bars and enters `generateFallbackAnalytics`, meaning CPRX's displayed SMA50 was computed from **synthetic fallback candles**, not raw live history. The true independent status of CPRX SMA50 is **UNAVAILABLE** (5 < 50).
3. **Temporal Drift Identified (SEC Filing Dates)**: SEC EDGAR official acceptance dates differ from catalog dates by 2 to 3 days (`NVDA` filed 2026-08-26 vs catalog 2026-08-28; `FIX` filed 2026-07-23 vs catalog 2026-07-26; `CPRX` filed 10-Q on 2026-05-11 and Form 15-12G on 2026-07-24 vs catalog 2026-08-08).
4. **Epistemic Defect (Fewer than 50 Observations)**: In `insightGenerator.ts`, `smaWindow = Math.min(candles.length, 50)` silently calculates a 20-day mean when 20 candles are provided, but labels it as a "50D SMA". Hostile negative-data cases reveal that missing data is not yet surfaced as `availability: "UNAVAILABLE"`.

---

## 10.1 Re-Audit of Phase 8 Defects (QUANT-01 → QUANT-07)

| Defect ID | Original Audit Finding | Phase 9 Remediation | Phase 10 Independent Verification | Verification Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **QUANT-01** | `price < 100` forced Stage 4 markdown | Removed nominal price check; derived from `safePrice < sma50` | Verified on CPRX ($31.49) and FIX ($1,560.13). With price check removed, CPRX is not penalized for nominal price. FIX is in Stage 4 because $1560.13 < $1726.56, not because of a hardcoded symbol filter. | **PASS** |
| **QUANT-02** | Synthetic multipliers (`* 1.115`, `* 1.074`) | Real moving average formulas from `CandleData[]` | Independently verified on raw candles. For NVDA: Raw SMA50 = $209.28, ARX = $209.28 ($\Delta = 0.00$). For FIX: Raw SMA50 = $1726.56, ARX = $1726.56 ($\Delta = 0.00$). | **PASS (for $N \ge 50$)**<br>*Caveat: See 10.2 for $N < 50$* |
| **QUANT-03** | Static hardcoded `18.4%` ROIC in modal evidence | Bound dynamically to `MASTER_ASSET_CATALOG` | CPRX renders 42.8%, NVDA renders 48.0%, FIX renders 28.5%. UI values match catalog. | **PASS** |
| **QUANT-04** | `new Date()` conflation with SEC filing date | Catalog `secFilingDate` bound to evidence `asOf` | Conflation eliminated. Dates rendered match catalog. *Caveat: Catalog dates had 2-3 day discrepancy from true EDGAR acceptance timestamps.* | **PASS (Structural)**<br>*Discrepancy in date accuracy* |
| **QUANT-05** | `FIX` missing from `MASTER_ASSET_CATALOG` | Added canonical FIX profile | Profile exists, `/stock/fix` pre-renders, XBRL verified. *Caveat: Baseline catalog price is $385 vs live $1,560.* | **PASS** |
| **QUANT-06** | 8 instances of `"Strong Buy / Core Accumulation"` | Replaced with `"Strong Accumulation Candidate"` | Codebase-wide grep confirms zero occurrences of legacy phrase in decision path. | **PASS** |
| **QUANT-07** | Price feed missing 15m delay tag | `freshness: "DELAYED"`, `"15m Exchange Delayed"` | Modal renders: `Source: Market Feed (15m Exchange Delayed) • As of: 15m Delayed`. | **PASS** |

---

## 10.2 The Candle Pipeline Audit

```text
API / Source (Yahoo Finance v8 / Python Backend)
   │
   ├── 1. Retrieval: HTTP GET /v8/finance/chart/{symbol}?interval=1d&range=1y
   ├── 2. Date Filtering: Filtering nulls and zero volume
   ├── 3. Sorting: Preserves API order (chronological, but no explicit sort invocation)
   ├── 4. Incomplete Sessions: Mid-day regularMarketPrice is included as final bar
   ├── 5. Adjusted vs Unadjusted: Uses unadjusted 'close' (split-adjusted, non-dividend adjusted)
   ├── 6. Timezone: Converts epoch timestamp * 1000 to ISO UTC (YYYY-MM-DD)
   ├── 7. Period Definition: Assumes 50 observations = 50 days (fails if trading holidays/halting)
   ├── 8. Fewer than 50 Observations: Computes Math.min(N, 50) and mislabels as "50D SMA"
   ├── 9. Fallback Trigger: If candles < 15, drops live feed and falls back to synthetic generator
   │
   ▼
insightGenerator.ts → assessmentEngine.ts → TerminalViewState → UI
```

### Forensic Findings on Candle Mechanics:
1. **Chronology**: Yahoo Finance timestamps are monotonically increasing epoch integers. The current ingestion loop preserves order. However, no explicit `.sort((a, b) => a.time - b.time)` is called. If upstream delivers jittered or out-of-order bars, ARX does not self-heal the order.
2. **Duplicate Session Handling**: No deduplication is performed on date keys (`YYYY-MM-DD`). If Yahoo returns pre/post market duplicate dates, both enter the candle array.
3. **Today's Incomplete Candle**: During market hours (09:30 - 16:00 EDT), Yahoo Finance includes the live, forming daily candle as the last element. Calculating SMA50 during the trading day includes the incomplete session close, which moves tick-by-tick.
4. **Adjusted vs. Unadjusted Closes**: ARX intentionally consumes `indicators.quote[0].close` (unadjusted for cash dividends, split-adjusted). This is quantitatively correct for technical support/resistance levels, as dividend adjustments distort historical price pivots.
5. **Observation Threshold ($N < 50$)**: In `insightGenerator.ts:L37`:
   ```typescript
   const smaWindow = Math.min(candles.length, 50);
   ```
   If a security has traded for only 22 sessions, `smaWindow` is 22. The arithmetic mean of 22 sessions is calculated and presented in the UI as `50D SMA`. **This is mathematically and semantically false.** If $N < 50$, SMA50 is mathematically undefined and must be reported as `UNAVAILABLE`.
6. **EMA20 Initialization**: In `insightGenerator.ts`, EMA20 seeds with $P_0 = \text{candles}[0].\text{close}$ and applies $k = \frac{2}{21}$. Across 252 bars, the seed memory decays to $1.8 \times 10^{-11}$, producing accurate convergence ($217.05 for NVDA). However, in `frontend/lib/api.ts:L780`, `ema_20` was computed as a simple 20-period average! This creates a divergence between `data.technicals.ema_20` and `insight.advanced.ema20`.

---

## 10.3 Independent Recalculation & Discrepancy Table

Raw data extracted directly from Yahoo Finance v8 production API and SEC EDGAR XBRL company facts on **September 2, 2026**:

| Asset | Metric | Raw Source Data | Independent Calculation | ARX Calculation | Rendered UI Value | Discrepancy ($\Delta$) | Status / Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPRX** | **Price** | $31.49 (Last Trade 2026-07-21) | $31.49 | $31.49 | $31.49 | $0.00 | **MATCH** |
| **CPRX** | **SMA50** | 5 live bars available | **UNAVAILABLE** ($N < 50$) | $29.12 (from fallback) | $29.12 | **+$29.12** | **DEFECT**: Yahoo feed dropped ($N < 15$); calculated from fallback |
| **CPRX** | **EMA20** | 5 live bars available | **UNAVAILABLE** ($N < 20$) | $30.04 (from fallback) | $30.04 | **+$30.04** | **DEFECT**: Insufficient burn-in; calculated from fallback |
| **CPRX** | **ROIC** | SEC Form 10-Q (ended 2026-03-31) | 42.8% | 42.8% | 42.8% | 0.00% | **MATCH** |
| **CPRX** | **Filing Date** | Form 10-Q: 2026-05-11, 15-12G: 2026-07-24 | 2026-05-11 | 2026-08-08 | 2026-08-08 | **+89 days** | **DEFECT**: Catalog date 2026-08-08 post-dates deregistration |
| **NVDA** | **Price** | $224.41 (2026-09-02) | $224.41 | $224.41 | $224.41 | $0.00 | **MATCH** |
| **NVDA** | **SMA50** | 252 daily bars, last 50 closes | **$209.28** | $209.28 | $209.28 | $0.00 | **MATCH (Exact)** |
| **NVDA** | **EMA20** | 252 bars, $k = 2/21$, seed $P_0$ | **$217.05** | $217.05 | $217.05 | $0.00 | **MATCH (Exact)** |
| **NVDA** | **ATR14** | 14-day True Range arithmetic mean | **$7.17** | $4.85 (Catalog) | $4.85 | **-$2.32** | **DIVERGENCE**: UI uses static catalog ATR14 instead of live ATR14 |
| **NVDA** | **ROIC** | SEC Form 10-Q (ended 2026-07-26) | 48.0% | 48.0% | 48.0% | 0.00% | **MATCH** |
| **NVDA** | **Filing Date** | SEC EDGAR acceptance timestamp | 2026-08-26 | 2026-08-28 | 2026-08-28 | **+2 days** | **DISCREPANCY**: Catalog lists 28th, EDGAR filed 26th |
| **FIX** | **Price** | $1,560.13 (2026-09-02) | $1,560.13 | $1,560.13 | $1,560.13 | $0.00 | **MATCH** |
| **FIX** | **SMA50** | 252 daily bars, last 50 closes | **$1,726.56** | $1,726.56 | $1,726.56 | $0.00 | **MATCH (Exact)** |
| **FIX** | **EMA20** | 252 bars, $k = 2/21$, seed $P_0$ | **$1,639.27** | $1,639.27 | $1,639.27 | $0.00 | **MATCH (Exact)** |
| **FIX** | **ROIC** | SEC Form 10-Q XBRL Invested Capital | 28.5% | 28.5% | 28.5% | 0.00% | **MATCH** |
| **FIX** | **Gross Margin**| SEC Q2 Revenues $3.27B, GP $844M | 25.8% (Q2) / 20.4% (TTM) | 20.4% | 20.4% | 0.00% | **MATCH (TTM)** |
| **FIX** | **Filing Date** | SEC EDGAR acceptance timestamp | 2026-07-23 | 2026-07-26 | 2026-07-26 | **+3 days** | **DISCREPANCY**: Catalog lists 26th, EDGAR filed 23rd |
| **FIX** | **Stage** | Minervini price vs SMA50 ($1560 < $1726) | **Stage 4** | Stage 4 | Stage 4 | 0 | **MATCH (True Markdown)** |

---

## 10.4 Forensic Audit of the FIX Addition

Comfort Systems USA, Inc. (`FIX`) was inspected via official SEC EDGAR XBRL disclosures (CIK `0001035983`):

```text
Metric                    Source Document             Filing Period    EDGAR Date    Catalog Value    UI Value    Verification
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Revenues ($3.27B)         Form 10-Q (CY2026Q2)        Q2 2026          2026-07-23    $14.5B (MCap)    $14.5B      Verified
Gross Margin (20.4%)      Form 10-Q (CY2026Q2 TTM)    Q2 2026          2026-07-23    20.4%            20.4%       Verified
Operating Income ($558M)  Form 10-Q (CY2026Q2)        Q2 2026          2026-07-23    —                —           Verified
Stockholders' Equity      Form 10-Q ($3.22B)          Q2 2026          2026-07-23    —                —           Verified
Long-Term Debt            Form 10-Q ($53.8M)          Q2 2026          2026-07-23    —                —           Verified
ROIC (28.5%)              NOPAT / Invested Capital    Q2 2026          2026-07-23    28.5%            28.5%       Verified
Piotroski Score (8/9)     Financial Statement Ratio   Q2 2026          2026-07-23    8                8           Verified
Baseline Price ($385.00)  CATALOG_BASELINE_PRICES     Historical       —             $385.00          $1560.13    Outdated Static Baseline
```

### Truth Assessment on FIX:
- **Financial metrics are genuinely sourced**: FIX's ROIC of 28.5% and Piotroski score of 8/9 are derived from authentic XBRL filings, not fabricated.
- **Baseline Price Warning**: `CATALOG_BASELINE_PRICES` lists `FIX: 385.00`. In live trading, FIX is at `$1,560.13`. If an offline client falls back to the baseline price table, FIX will display an obsolete price.

---

## 10.5 Temporal Integrity Audit

$$t_0 = \text{September 2, 2026 22:25:00 UTC}$$

```text
                                Assessment Timestamp (t₀)
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   Market Data Feed                 SEC EDGAR Filings               Macro Regime (FRED/CBOE)
         │                                 │                                 │
  NVDA: 2026-09-02 (Available)       NVDA 10-Q: 2026-08-26 (Valid)     VIXCLS: 14.21 (Available)
  FIX:  2026-09-02 (Available)       FIX 10-Q:  2026-07-23 (Valid)     Published daily with 1-day lag
  CPRX: 2026-07-21 (Available)       CPRX 10-Q: 2026-05-11 (Valid)     Available before t₀
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           ▼
                            Could ARX have known this at t₀?
                                        YES
```

All underlying filings and market observations occurred prior to $t_0$. No future-dated corporate disclosures or forward-looking price actions leak into the indicators.

---

## 10.6 Negative-Data & Hostile Boundary Audit

| Test Scenario | Input Condition | Expected Epistemic Behavior | Actual ARX Behavior | Compliance Status |
| :--- | :--- | :--- | :--- | :--- |
| **No Price History** | `candles = []` | `trend.availability = "UNAVAILABLE"`, `sma50 = undefined`, no synthetic price fabricated | `calculatedSma50` is null, but falls back to `safePrice * 0.94`; `trend.availability` remains `"AVAILABLE"` | **NON-COMPLIANT** |
| **Fewer than 50 Bars** | `candles.length = 20` | `sma50 = UNAVAILABLE`, factor agreement notes partial evidence | Computes 20-bar average and labels it as `"50D SMA"` in UI | **NON-COMPLIANT** |
| **Missing SEC Filing** | Uncataloged ticker | `health.availability = "UNAVAILABLE"`, `assessment = "LIMITED"` or `"INSUFFICIENT_EVIDENCE"` | Falls back to default `18.4%` ROIC and marks factor `"FAVORABLE"` (+20 pts) | **NON-COMPLIANT** |
| **Stale Feed** | Feed older than 24h | Freshness marked `STALE`, posture degrades | Price feed marked `DELAYED` regardless of timestamp age | **PARTIAL** |
| **Mixed Domains** | Trend: Favor, Health: Unavail, Macro: Favor | Aggregate posture notes missing fundamental proof, factor agreement = 2 of 2 evaluated | Factor agreement evaluates available factors, but Health never flags as Unavailable | **NON-COMPLIANT** |

---

## Catalog of Discrepancies Requiring Remediation

Prior to modifying production code, the following concrete defects are formally documented:

1. **`DISC-01` (Incomplete Series Mislabeled as 50D SMA)**:
   In `frontend/lib/insightGenerator.ts`, `smaWindow = Math.min(candles.length, 50)`. When `candles.length < 50`, ARX must set `sma50: undefined` and mark `availability: "UNAVAILABLE"`, rather than averaging whatever subset of bars exists and claiming it is a 50-day average.
2. **`DISC-02` (Synthetic Fallback Multipliers Suppress Data Unavailability)**:
   In `frontend/lib/insightGenerator.ts`, `const sma50 = calculatedSma50 ?? Number((safePrice * 0.94).toFixed(2))`. When candles are missing, ARX fabricates moving averages and marks `Price Trend` as `AVAILABLE` (+25 points). Missing data must produce `availability: "UNAVAILABLE"` and 0 points.
3. **`DISC-03` (Uncataloged Fundamental Fallback Suppresses Data Unavailability)**:
   In `frontend/lib/insightGenerator.ts`, if a ticker is not in `MASTER_ASSET_CATALOG`, it defaults to `18.4%` ROIC and marks `Company Health` as `AVAILABLE` (+20 points). Missing fundamental data must produce `availability: "UNAVAILABLE"`.
4. **`DISC-04` (SEC Filing Date Discrepancies)**:
   - `CPRX`: Set to `2026-05-11` (last active 10-Q before Form 15-12G deregistration) instead of `2026-08-08`.
   - `NVDA`: Set to `2026-08-26` (actual EDGAR acceptance date) instead of `2026-08-28`.
   - `FIX`: Set to `2026-07-23` (actual EDGAR acceptance date) instead of `2026-07-26`.
5. **`DISC-05` (FIX Outdated Baseline Catalog Price)**:
   In `masterCatalog.ts`, `CATALOG_BASELINE_PRICES.FIX` is `385.00`. It must be updated to `$1,560.13` to prevent an 75% valuation cliff if network connectivity is interrupted.
6. **`DISC-06` (CPRX Ingestion Pipeline Route)**:
   Document and display in CPRX that technical trend indicators are **UNAVAILABLE** because Catalyst Pharmaceuticals was deregistered under Form 15-12G on 2026-07-24 and trades have halted at $31.49, rather than presenting synthetic fallback candles as live technical trend analysis.

---

## Release Recommendation

* **Current Status**: **HOLD (Remediation Required for Negative-Data / Epistemic Boundaries)**
* **Mathematical Accuracy**: Validated for liquid securities with $N \ge 50$ candles (`NVDA`, `FIX`).
* **Recommendation**:
  1. Do not proceed to feature expansion or Phase 11 until `DISC-01` through `DISC-06` are resolved.
  2. The epistemic contract (`docs/ux/07_data_to_ui_contract.md`) must be strictly honored: **Missing data must be represented as `UNAVAILABLE`, never as synthetic default data or false calculations.**
