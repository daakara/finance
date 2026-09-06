# ArxTerminal: Forensic Data Integrity Verification & Evaluation Audit

**Audit Date**: September 6, 2026
**Auditor**: Quantitative Risk Guardian (`quant-guardian`) & Methods Auditor (`critique-methods-auditor`)
**Mandate**: Independent forensic verification of methodological problems, look-ahead bias, strategy provenance, clustering, and calculation integrity.
**Operational Status**: Production Engine Frozen at Commit `4e36862` (`v2.4.0-phase24-freeze`). Zero production trading code modifications.

---

## 1. Fundamental Data Look-Ahead Verification

### The Complete Data Path
We traced the complete execution path from historical cutoff to actionable setup in `scratch/phase24_out_of_sample_validation.py` and `analyst_dashboard/data/market_db.py`:

```
Historical Cutoff Bar (e.g. T-160: 2026-01-14)
      │
      ▼
Price Candles Sliced at Cutoff (hist_slice = df.iloc[:cutoff_idx])  [Clean Point-in-Time]
      │
      ▼
Fundamental Data Lookup: cur.execute("SELECT * FROM asset_factor_snapshots;")
      │
      ▼
SQLite Table: asset_factor_snapshots (73 rows, updated_at: Sept 1-6, 2026) [CONTAMINATED]
      │
      ▼
Factor Assignment: fund_data = {qualityScore: 88, growthScore: 85, piotroskiF: 8}
      │
      ▼
ConfluenceEngine: Computes score combining T-160 price + Sept 2026 fundamentals
      │
      ▼
DecisionHierarchyEngine: Evaluates Precedence 4 (Audited Fundamentals Gate)
      │
      ▼
Actionable Setup Emitted for Historical Evaluation (January 2026)
```

### Forensic Examination of Fundamental Fields

| Field Name | Source Table | Schema Data Type | Database Timestamp | Semantic Meaning of Timestamp | Point-in-Time Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `quality_score` | `asset_factor_snapshots` | `INTEGER` | `updated_at` (Sept 2026) | Time Python script executed `INSERT` | **STATIC / FUTURE LOOK-AHEAD** |
| `growth_score` | `asset_factor_snapshots` | `INTEGER` | `updated_at` (Sept 2026) | Time Python script executed `INSERT` | **STATIC / FUTURE LOOK-AHEAD** |
| `piotroski_f` | `asset_factor_snapshots` | `INTEGER` | `updated_at` (Sept 2026) | Time Python script executed `INSERT` | **STATIC / FUTURE LOOK-AHEAD** |
| `valuation_score` | `asset_factor_snapshots` | `INTEGER` | `updated_at` (Sept 2026) | Time Python script executed `INSERT` | **STATIC / FUTURE LOOK-AHEAD** |
| `composite_score` | `asset_factor_snapshots` | `INTEGER` | `updated_at` (Sept 2026) | Time Python script executed `INSERT` | **STATIC / FUTURE LOOK-AHEAD** |
| `current_price` | `asset_factor_snapshots` | `REAL` | `updated_at` (Sept 2026) | Spot price on Sept 1–6, 2026 | **STATIC / FUTURE LOOK-AHEAD** |

### Key Provenance Findings
1. **No Publication or Period End Dates**: The table `asset_factor_snapshots` has no `period_end_date`, `filing_date`, or `publication_date`. It only contains an `updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`.
2. **Hardcoded Seed Ingestion**: In `scripts/sync_market_db.py` (lines 56–68), factor scores were hardcoded at ingestion:
   ```python
   db.save_factor_snapshot(symbol, {
       "currentPrice": latest_close,
       "growthScore": 85,
       "qualityScore": 88,
       "valuationScore": 75,
       "piotroskiFScore": 8,
       "verdict": "Strong Buy / Core Accumulation"
   })
   ```
   Over **60% of all symbols (45 of 73)** share these exact seeded constants.
3. **Availability at Cutoff**: On **January 14, 2026 ($T-160$)**, an investor could not possess factor snapshots written to SQLite in **September 2026**.
4. **Survivorship & Quality Bias**: Slicing price candles historically while holding fundamental quality constant at September 2026 levels meant the strategy only selected assets that were *future survivors* with strong September 2026 financial metrics.

---

## 2. SEC & Regulatory Filing Provenance

We inspected `analyst_dashboard/data/sec_edgar_fetcher.py` and `api/routes/smart_money.py`:

```
Actual SEC Pipeline in Codebase:
SEC EDGAR API (data.sec.gov)
      ↓
SecEdgarFetcher.get_recent_filings() (Retrieves raw JSON array: form, filingDate, accessionNumber)
      ↓
Returns Top 25 Recent Submissions
      ↓
[DISCONNECTED] No XBRL parser. No balance sheet extractor. No historical factor calculator.
```

- **Missing XBRL Facts**: The system does not query or store `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`.
- **No Point-in-Time Fundamental Store**: The system does not maintain an `as_of_date` or `filing_date` history of balance sheets, income statements, or cash flows.
- **Definitive Conclusion**: **The system currently lacks the data infrastructure required to perform a genuine point-in-time historical fundamental backtest.**

---

## 3. Exact Contamination Scope (Table of All 28 Setups)

Every single one of the 28 historical actionable setups from Phase 24 was independently audited for fundamental contamination:

```
┌─────┬─────────┬────────────┬───────────────────┬─────────────────────┬───────────────────┬──────────────┬───────────────┬────────────────────────────┐
│ Idx │ Symbol  │ Cutoff Date│ Factor Used       │ Factor Timestamp    │ Publication Date  │ Avail at T?  │ Contaminated? │ Contamination Category     │
├─────┼─────────┼────────────┼───────────────────┼─────────────────────┼───────────────────┼──────────────┼───────────────┼────────────────────────────┤
│  1  │ ACLS    │ 2026-02-12 │ Q:66, G:91, F:6   │ 2026-09-06 13:37:02 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  2  │ ACLS    │ 2026-03-13 │ Q:66, G:91, F:6   │ 2026-09-06 13:37:02 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  3  │ ANET    │ 2026-02-12 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:39 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  4  │ BTC-USD │ 2026-05-24 │ Q:70, G:99, F:3   │ 2026-09-01 14:35:17 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  5  │ CDNS    │ 2026-04-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:40 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  6  │ CIEN    │ 2026-01-14 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:36 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  7  │ CRSP    │ 2026-01-12 │ Q:70, G:99, F:4   │ 2026-09-01 14:35:25 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  8  │ DDOG    │ 2026-03-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:38 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│  9  │ DECK    │ 2026-01-14 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:48 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 10  │ DHLGY   │ 2026-05-07 │ Q:70, G:97, F:6   │ 2026-09-01 19:25:57 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 11  │ DUOL    │ 2026-05-11 │ Q:66, G:91, F:6   │ 2026-09-06 13:37:03 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 12  │ ETN     │ 2026-03-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:47 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 13  │ FIX     │ 2026-03-13 │ Q:66, G:52, F:6   │ 2026-09-04 11:15:12 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 14  │ GEV     │ 2026-05-11 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:47 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 15  │ IONQ    │ 2026-01-14 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:42 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 16  │ KLAC    │ 2026-02-12 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:45 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 17  │ KLAC    │ 2026-03-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:45 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 18  │ MPWR    │ 2026-02-12 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:45 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 19  │ MSTR    │ 2026-04-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:40 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 20  │ NVDA    │ 2026-03-16 │ Q:99, G:99, F:9   │ 2026-09-06 13:37:41 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 21  │ PANW    │ 2026-04-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:38 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 22  │ POWI    │ 2026-03-13 │ Q:66, G:91, F:6   │ 2026-09-06 13:37:03 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 23  │ PWR     │ 2026-01-14 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:47 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 24  │ PWR     │ 2026-03-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:47 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 25  │ RKLB    │ 2026-04-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:42 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 26  │ SMCI    │ 2026-03-13 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:37 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 27  │ ULTA    │ 2026-02-12 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:50 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
│ 28  │ VRTX    │ 2026-02-12 │ Q:88, G:85, F:8   │ 2026-09-04 08:27:43 │ UNAVAILABLE_IN_DB │ NO           │ YES ❌        │ Look-Ahead Static Snapshot │
└─────┴─────────┴────────────┴───────────────────┴─────────────────────┴───────────────────┴──────────────┴───────────────┴────────────────────────────┘
```

### Contamination Summary Statistics
- **Total Historical Actionable Setups Evaluated**: 28
- **Number Contaminated by Future / Static Factor Snapshots**: **28 (100.0%)**
- **Number Genuinely Point-in-Time**: **0 (0.0%)**
- **Number Uncertain**: **0 (0.0%)**
- **Confidence in Classification**: **HIGH (100%)**

---

## 4. Strategy Provenance & In-Sample Tuning Chronology

From git logs (`git log --oneline`), commit diffs, and phase audit reports, we established the chronological provenance of strategy rules:

```
                               STRATEGY EVOLUTION CHRONOLOGY
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │ Phase 22 (Commit 5da)│ ---> │ Phase 23 Audit Findings │ ---> │ Commit 4e36862 Deploy   │
 │ Fri Sep 4, ~14:00    │      │ Fri Sep 4, ~15:00       │      │ Fri Sep 4, 16:26 CEST   │
 ├──────────────────────┤      ├─────────────────────────┤      ├─────────────────────────┤
 │ Screener tested on   │      │ Discovered: 76.2% stop  │      │ Introduced Pullback     │
 │ 81 assets in SQLite. │      │ rate on falling knives  │      │ Confirmation Candle     │
 │ Found false buys.    │      │ at T-60, T-40, T-20.    │      │ (is_stabilized gate)    │
 └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                             │
                                                                             ▼
                                                                ┌─────────────────────────┐
                                                                │ Phase 24 "OOS" Backtest │
                                                                │ Fri Sep 4, ~17:00 CEST  │
                                                                ├─────────────────────────┤
                                                                │ Tested 4e36862 on T-160 │
                                                                │ to T-80 in same DB.     │
                                                                │ Reported +4.29% exp.    │
                                                                └─────────────────────────┘
```

### Rule Classification Matrix

| Strategy Rule | Introduction Commit / Date | Direct Rationale | Historical Data Inspected Beforehand? | Influenced by Evaluated Period? | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pullback Confirmation Candle (`is_stabilized`)** | Commit `4e36862` (Sep 4, 2026) | Eliminate falling-knife false positives found in Phase 23 audit | **YES** (Evaluated on $T-60$ to $T-20$ in same DB) | **YES** (Direct empirical response to losses) | **CLEARLY TUNED ON PERIOD** |
| **ATR-Scaled Corridor Geometry** | Commit `4e36862` (Sep 4, 2026) | Prevent stop inflation beyond 6.5% | **YES** (Inspected across 81 assets) | **YES** | **CLEARLY TUNED ON PERIOD** |
| **Confluence Score Hurdle ($\ge 75.0$)** | Commit `5da11f9` (Sep 4, 2026) | Remove step-function cliffs | **YES** (Inspected score distribution) | **POTENTIALLY** | **POTENTIALLY TUNED ON PERIOD** |
| **Minervini Stage 2 Filter** | Commit `fef4bc1` (Phase 20) | Trend-following academic / book literature | **NO** (Standard literature) | **NO** | **PRE-EXISTING** |
| **Decision Hierarchy Precedence** | Commit `fef4bc1` (Phase 20) | Architectural exclusivity contract | **NO** (Software engineering invariant) | **NO** | **PRE-EXISTING** |

**Conclusion**: The core rule that elevated strategy performance from negative returns ($-4.07\%$) in Phase 23 to $+4.29\%$ in Phase 24 was the **pullback confirmation candle gate**. That rule was directly engineered by studying trade failures in this exact SQLite dataset hours earlier.

---

## 5. Clustering Analysis & Effective Sample Size ($N_{eff}$)

### Temporal Clustering Verification
Independently verified from database timestamps: **22 of the 28 setups (78.6%) occurred on just 4 calendar dates**:
- **2026-03-13**: 8 setups (28.6% of entire sample)
- **2026-02-12**: 6 setups (21.4% of entire sample)
- **2026-04-13**: 4 setups (14.3% of entire sample)
- **2026-01-14**: 4 setups (14.3% of entire sample)

### Multi-Level Correlation Analysis

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CORRELATION DECOMPOSITION (N=28 SETUPS)                         │
├─────────────────┬─────────────────────────────────────────────────┬────────────────────┤
│ Dimension       │ Empirical Evidence                              │ Correlation Impact │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 1. Time         │ 78.6% of trades clustered on 4 dates            │ HIGH DEPENDENCE    │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 2. Sector       │ 8 of 28 setups are Semiconductors/EDA; 4 Power   │ HIGH COUPLING      │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 3. Market Move  │ April 13 (SPY +7.75% 20D): 4 of 4 won (100%)     │ TOTAL MACRO DRAG   │
│                 │ Feb 12 (SPY -2.79% 20D): 5 of 6 stopped (83.3%) │                    │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 4. Factor       │ 100% of setups shared static high Quality/Growth│ TOTAL COLLINEARITY │
└─────────────────┴─────────────────────────────────────────────────┴────────────────────┘
```

### Methodological Assessment of Effective Sample Size ($N_{eff}$) & Material Dependence
Treating correlated observations as independent artificially shrinks standard errors by $\sqrt{N}$. Under an intraclass correlation framework ($\rho$):
$$N_{eff} = \frac{N}{1 + (m - 1)\rho}$$
where $m$ is the average cluster size ($m = 28 / 9 \approx 3.11$) and $\rho$ is the average pairwise return correlation within clusters.
- For high-beta growth equities during market pullbacks/rallies ($\rho \approx 0.65\text{–}0.75$):
  $$N_{eff} \approx \frac{28}{1 + (3.11 - 1) \times 0.70} \approx \frac{28}{2.48} \approx 11.3$$
- In scenarios of total macro date-level dominance (where broad market beta dictates nearly 100% of cluster outcome, as observed on April 13 and Feb 12), sensitivity models suggest effective independent decisions could contract as low as $N_{eff} \approx 4\text{ to }6$.
- **Methodological Correction & Defensible Finding**: We do **not** state $N_{eff} \approx 4\text{–}6$ as a measured fact; deriving a single exact $N_{eff}$ requires an explicit cluster definition and empirical correlation estimator. The rigorous and defensible scientific statement is: **The 28 historical observations are materially dependent**, heavily dominated by market beta and calendar date clustering, and cannot be treated as $i.i.d.$ observations.

---

## 6. First-Principles Recalculation of Historical Performance

We audited all 28 individual trade records and recalculated aggregate metrics from first principles:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CALCULATION RECONCILIATION TABLE                                │
├────────────────────────────┬──────────────────┬──────────────────┬─────────────────────┤
│ Metric                     │ Phase 24 Report  │ First Principles │ Discrepancy Cause   │
├────────────────────────────┼──────────────────┼──────────────────┼─────────────────────┤
│ Total Setups               │ 28               │ 28               │ None (Reconciled)   │
│ Winning Setups (TP1)       │ 13 (46.4%)       │ 13 (46.43%)      │ None (Reconciled)   │
│ Stopped Setups             │ 12 (42.9%)       │ 12 (42.86%)      │ None (Reconciled)   │
│ Expired Setups             │ 3 (10.7%)        │ 3 (10.71%)       │ None (Reconciled)   │
│ Average Win                │ +15.33%          │ +15.33%          │ None (Reconciled)   │
│ Average Loss               │ -6.61%           │ -6.61%           │ None (Reconciled)   │
│ Average Expired            │ Not Reported     │ +11.45%          │ DHLGY, FIX, PWR ret │
│ Reported Mean Return       │ +7.78%           │ +5.51% (Trade)   │ 7.78% was B&H 20D!  │
│ Reported Expectancy        │ +4.29%           │ +4.29% (2-state) │ 4.29% ignored exp.! │
│ Full-State Expectancy      │ Not Reported     │ +5.51%           │ Includes expired    │
│ Profit Factor              │ 2.51 : 1         │ 2.95 : 1         │ 2.51 omitted exp.   │
│ Execution Friction (25bps) │ 0 bps (Zero Fee) │ -0.25% per trade │ Frictions omitted   │
│ Net Mean Return (25bps)    │ Not Reported     │ +5.26%           │ After slippage/fees │
└────────────────────────────┴──────────────────┴──────────────────┴─────────────────────┘
```

### Critical Calculation Revelations
1. **The "+7.78% Mean Return" was NOT the Strategy's Trade Return**:
   In Phase 24, $+7.78\%$ was computed by taking the 20-day close price return of the stocks ($Close_{20}/P_0 - 1$). The actual trade-execution return (exiting at $TP_1$ or Stop) was **$+5.51\%$**.
2. **The "+4.29% Expectancy" Used a 2-State Formula that Excluded Expired Trades**:
   $$\text{Expectancy}_{\text{2-state}} = (0.4643 \times 15.33\%) - (0.4286 \times 6.61\%) = +4.284\%$$
   The 3 expired trades (`DHLGY`: $+8.52\%$, `FIX`: $+19.22\%$, `PWR`: $+6.61\%$) were omitted from the formula.
3. **Execution Friction Was 100% Omitted**:
   All returns assumed execution at the exact high tick ($TP_1$) or low tick (Stop) with zero bid-ask spread, zero slippage, and zero commissions.

---

## 7. Tri-Partite Performance Decomposition: Signal Quality vs. Trade Construction vs. Economic Feasibility

To prevent headline figures from masking critical structural vulnerabilities, we formally decouple evaluation into three distinct layers:

```
                      TRI-PARTITE PERFORMANCE SEPARATION
 ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
 │   A. Signal Quality    │   │  B. Trade Construction │   │ C. Economic Feasibility│
 ├────────────────────────┤   ├────────────────────────┤   ├────────────────────────┤
 │ Unconstrained Direction│   │ Execution Geometry     │   │ Frictions & Portfolio  │
 │ • Fixed Horizon Returns│   │ • Stop / TP Efficiency │   │ • 25 bps Round-Trip Fee│
 │   (T+1, T+5, T+10, T+20│   │ • Premature Stop-Outs  │   │ • Capital Concentration│
 │ • Mean MFE: +17.89%    │   │ • Capture Ratio (R/MFE)│   │ • Margin / Liquidity   │
 │ • Mean MAE: -5.82%     │   │ • Realized R:R Ratio   │   │   Spikes (78.6% on 4D) │
 └────────────────────────┘   └────────────────────────┘   └────────────────────────┘
```

### Critical Separation Principle
**A good signal with a bad stop is a fundamentally different failure mode from a bad signal.**
1. **Signal Quality (Raw Predictive Power)**:
   - Does the underlying asset appreciate over fixed time intervals ($T+1, T+5, T+10, T+20$) regardless of stops?
   - In the historical sample, raw directional bias was positive: **Mean MFE was $+17.89\%$**, and even stopped trades rallied an average of $+7.13\%$ before reversal.
2. **Trade Construction Quality (Execution Geometry)**:
   - Did the stop-loss get hit before the profit target?
   - In the historical sample, **75% of stopped trades suffered premature stop-outs** (reaching $>+5.0\%$ gain before stopping out). A rigid $-6.5\%$ stop placed under normal market volatility clipped trades that subsequently reached their targets.
3. **Economic & Portfolio Feasibility**:
   - Factoring in 25–30 bps round-trip friction and capital constraints: 22 of 28 trades fired across just 4 dates, creating massive portfolio leverage demands and execution liquidity stress.

---

## 8. Confluence Score Reassessment: Ordinal Ranking vs. Probabilistic Calibration

Inspection of `ConfluenceEngine.calculate_confluence`:
- **Formula**: Linear weighted sum of 5 heuristic sub-scores:
  $$\text{Confluence} = 0.35 \times Tech + 0.30 \times Fund + 0.15 \times Smart + 0.10 \times Macro + 0.10 \times Cat$$
- **Ordinal Setup Quality, Not Probability**:
  - The score is an **ordinal ranking heuristic** for filtering setup quality, not an estimated probability.
  - Testing calibration via **Brier scores or probability calibration curves is methodologically premature and inappropriate**, as the engine does not claim to output posterior probabilities $\hat{p} \in [0, 1]$.
- **The Correct Test: Monotonicity Across Score Bands**:
  - The valid scientific test for an ordinal quality index is **Monotonicity**: do higher-confluence setups produce higher win rates, higher Sharpe, and higher mean returns than lower-confluence ones?
  - **Empirical Historical Check**:
    - Confluence $75.0 - 79.9$: $42.9\%$ Win Rate, $+3.12\%$ Mean Return
    - Confluence $80.0 - 84.9$: $50.0\%$ Win Rate, $+5.81\%$ Mean Return
    - Confluence $\ge 85.0$: $50.0\%$ Win Rate, $+6.14\%$ Mean Return
    - While return increased, win rate plateaued at $50.0\%$ above 80, failing strict monotonicity.
- **Formal Recommendation**:
  - Remove probabilistic terminology ("Confidence") from all UI and API contracts.
  - Mandate **Confluence Monotonicity Tracking** in the forward evaluation scorecard across four score tiers ($<75, 75-79.9, 80-84.9, 85+$).

---

## 9. Independent Live Cohort Verification ($N=9$)

All 9 live recommendations in `paper_trading_ledger.json` were audited:

| Field | Audit Finding | Verification Status |
| :--- | :--- | :--- |
| **Prediction Timestamp** | `2026-09-04T17:32:00Z` | **VERIFIED** |
| **Production Commit** | `4e36862` | **VERIFIED** |
| **Engine Tag** | `v2.4.0-phase24-freeze` | **VERIFIED** |
| **Decision Snapshot Hash** | SHA-256 verified for all 9 | **VERIFIED** |
| **Inputs Snapshot Hash** | SHA-256 verified for all 9 | **VERIFIED** |
| **Current Prices & Excursions** | All prices identical to entry; MFE/MAE = 0.00% | **VERIFIED** |
| **Completed Sessions Traded** | **0 sessions** (Weekend close) | **VERIFIED** |
| **Outcome Modifications** | Zero modifications since creation | **VERIFIED** |

---

## 10. Summary of Methodological Disproofs

1. **The Historical Benchmark Cannot Be Used as Proof of Edge**:
   - 100% of setups used static September 2026 factor data.
   - 78.6% of setups were clustered on 4 calendar dates.
   - The confirmation candle rule was tuned on this exact dataset.
2. **The System Does Not Learn**: It is an iterative engineering artifact.
3. **The Live Baseline is Intact**: 9 open signals are locked and awaiting market open.
