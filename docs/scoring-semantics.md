# ARX Canonical Scoring Semantics Specification (Phase 19)

This specification formally establishes the mathematical foundations, data requirements, bounding constraints, and fail-closed behaviors for all 7 analytical scores across the ARX platform.

---

## Non-Negotiable Epistemic Principles

1. **`UNKNOWN ≠ FAVORABLE`**: Missing data must never default to an optimistic score or actionable rating.
2. **`UNKNOWN ≠ NEGATIVE`**: Missing disclosures must be categorized as unassessed or insufficient evidence, never penalized as structural distress.
3. **`UNKNOWN ≠ ACTIONABLE`**: No buy target, stop loss, or execution order can be generated without verified data.
4. **`SYNTHETIC ≠ MARKET DATA`**: Never synthesize prices, moving averages, or stop levels.

---

## Canonical Scores Hierarchy

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. Confluence Score (0–100)                     │
│       Multi-Factor Conviction SSOT (Technicals + Moat + Flow + Macro)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│  2. Composite Factor Score      │           │      3. Gem Score (0–100)       │
│      Fundamental DNA Mean       │           │   Screener Quality + Readiness  │
└────────────────┬────────────────┘           └─────────────────────────────────┘
                 │
  ┌──────────────┴──────────────┐
  ▼                             ▼
┌──────────────────┐   ┌──────────────────┐
│ 5. Piotroski-F   │   │ 7. Tail Risk     │
│   (0–9 Points)   │   │     (0–100)      │
└──────────────────┘   └──────────────────┘
```

---

## 1. Confluence Score (`confluenceScore`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | Single Source of Truth (SSOT) for total investment conviction. |
| **Output Range** | `[0.0, 100.0]` float. |
| **Pillar Weights** | Technical Structure: **25%**, Fundamental Moat: **25%**, Corporate Insiders & Flow: **25%**, Macro Safety Floor: **25%**. |
| **Penalty Factors** | Catalyst dilution risk (−15.0), Phase 1/2 clinical uncertainty (−10.0), regulatory filing delinquency (−12.0). |
| **Input Streams** | 14-day RSI, ATR(14), 20 EMA, 50 SMA, Piotroski-F, SEC Form 4 insider transactions, Congressional stock transactions, FRED yield curve (10Y–2Y), ICE BofA High Yield Credit Spread. |
| **Interpretation** | • `80.0–100.0`: **🟢 Green Light (High Conviction Alignment)**<br>• `65.0–79.9`: **💡 Solid Accumulation Setup**<br>• `49.0–64.9`: **🟡 Selective / Mixed Signals**<br>• `0.0–48.9`: **🔴 Red Light (Defensive / Capital Preservation)** |
| **Minimum Data** | Requires minimum **50 daily bars** and verified market tape. |
| **Missing-Data Behavior** | Fail-closed: unassessed pillars default to neutral baseline (50.0) with explicit warning flags (`warningsCount`). Does NOT award full score. |

---

## 2. Composite Factor Score (`compositeFactorScore`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | Unweighted composite of core corporate fundamental factors. |
| **Output Range** | `[0, 100]` integer. |
| **Mathematical Formula** | \(\text{Composite} = \frac{1}{N} \sum_{i=1}^{N} \text{Factor}_i\), where available factors include `growthScore`, `qualityScore`, `valuationScore`, `momentumScore`, `tailRiskScore`. |
| **Interpretation** | • `80–100`: Core Institutional Compounder<br>• `60–79`: Moderate Growth Compounder<br>• `< 60`: High Volatility / Speculative |
| **Minimum Data** | At least 2 verified fundamental statement disclosures. |
| **Missing-Data Behavior** | If fundamental statements (10-K/10-Q) are missing, `compositeFactorScore` is `None` / `null`, and verdict reports `"Awaiting Verified Fundamental Filing"`. |

---

## 3. Gem Score (`gemScore`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | Screener ranking metric combining fundamental quality with technical breakout readiness. |
| **Output Range** | `[0, 100]` integer. |
| **Mathematical Formula** | \(\text{GemScore} = \text{round}(0.40 \times \text{Quality} + 0.35 \times \text{Momentum} + 0.25 \times \text{Growth})\). |
| **Interpretation** | Ranks candidates within the institutional screener universe. |
| **Minimum Data** | Requires valid historical price series and verified fundamental snapshot. |
| **Missing-Data Behavior** | Unverified or unseasoned assets are excluded from actionable high-R:R filters; their trade levels are strictly suppressed (`null`). |

---

## 4. Setup Score (`setupScore`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | Terminal view score representing trade readiness. |
| **Semantic Invariant** | **Unified with `confluenceScore`**. In Phase 18/19, separate hardcoded setup scores (e.g. `60`) were eradicated. `setupScore` is derived directly as \(\text{round}(\text{confluenceScore})\). |
| **Missing-Data Behavior** | Reports `confluenceScore` or `undefined` when confluence is uncalculated. |

---

## 5. Piotroski F-Score (`piotroskiFScore` / `piotroski_f`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | 9-point fundamental accounting health rubric based on Joseph Piotroski's framework. |
| **Output Range** | `[0, 9]` discrete integer. |
| **Rubric Components** | • **Profitability (4 pts)**: Positive Net Income, Positive ROA, Positive Operating Cash Flow, Cash Flow > Net Income.<br>• **Leverage & Liquidity (3 pts)**: Lower Long-Term Debt Ratio, Higher Current Ratio, No Share Dilution.<br>• **Operating Efficiency (2 pts)**: Higher Gross Margin, Higher Asset Turnover Ratio. |
| **Interpretation** | • `8–9`: Elite Fundamental Health<br>• `5–7`: Stable / Typical Operating Health<br>• `0–4`: Severe Accounting / Solvency Vulnerability |
| **Missing-Data Behavior** | Points are awarded strictly upon verified SEC statement rows. Missing rows earn 0 points; unfiled companies cannot score above 0. |

---

## 6. Altman Z-Score (`altman_z`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | Parametric credit-strength and bankruptcy probability indicator for manufacturing and non-manufacturing firms. |
| **Output Range** | Continuous float. |
| **Standard Formula** | \(Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5\)<br>• \(X_1\): Working Capital / Total Assets<br>• \(X_2\): Retained Earnings / Total Assets<br>• \(X_3\): EBIT / Total Assets<br>• \(X_4\): Market Value of Equity / Total Liabilities<br>• \(X_5\): Sales / Total Assets |
| **Zones** | • \(Z > 2.99\): **Safe Zone** (Low probability of insolvency)<br>• \(1.81 \le Z \le 2.99\): **Grey Zone** (Moderate vulnerability)<br>• \(Z < 1.81\): **Distress Zone** (Elevated probability of bankruptcy within 24 months) |
| **Missing-Data Behavior** | When balance sheet items are absent, `altman_z` returns `None` and is omitted from credit reports. |

---

## 7. Tail Risk Score (`tailRiskScore`)

| Attribute | Specification |
| :--- | :--- |
| **System Role** | Downside extreme volatility and fat-tail risk index. |
| **Output Range** | `[0, 100]` integer. Higher = safer (lower tail risk). |
| **Mathematical Formula** | Derived from Cornish-Fisher Modified VaR 95%: \(\text{TailRisk} = \min(99, \max(40, \text{round}(100 - |\text{MVaR}_{95}| \times 7)))\). |
| **Interpretation** | • `80–99`: Mild Downside Tail Risk (Low skewness, low kurtosis)<br>• `60–79`: Moderate Volatility Exposure<br>• `40–59`: Severe Fat-Tail Exposure (High negative skew / kurtosis leptokurtic distribution) |
| **Minimum Data** | Minimum 30 closed daily returns. |
| **Missing-Data Behavior** | Defaults to neutral baseline (65) when kurtosis cannot be reliably computed. |

---

## 8. Data Readiness, Modeling Gates & Presentation Decoupling

### 8.1 The 15 Daily Sessions Baseline Modeling Gate
ARX Terminal enforces a strict entry gate for all quantitative evaluations:
- **Mathematical Function:** `validate_and_prune_daily_history` (`api/routes/analytics.py`).
- **Invariant Requirement:** Quantitative evaluation algorithms cannot commence with fewer than **15 complete daily sessions**.
- **Fail-Closed Behavior:** Any recognized equity with $<15$ sessions is rejected with `HTTP 422 Unprocessable Entity` (`status: "INSUFFICIENT_HISTORY"`).
- **Epistemic Rationale:** Any moving averages, volatility metrics, or factor weights computed on $<15$ sessions produce uncalibrated noise and violate `ARX_INV_011` (Strict Burn-In).

### 8.2 Progression of Indicator History Requirements & Implementation Realities
The 15-session gate is a baseline runtime prerequisite, **not an automatic activation date** for every quantitative indicator or model. Individual indicators, models, and statistical invariants enforce distinct mathematical inputs and burn-in characteristics:

| Indicator / Metric | Horizon Prerequisite | Implementation Reality & Code Reference | Enforcement Type |
| :--- | :--- | :--- | :--- |
| **Baseline Modeling Gate** | $\ge 15$ closed daily bars | `api/routes/analytics.py:1121` (`validate_and_prune_daily_history` checks `len(work_df) < 15`). Fails closed with HTTP 422 `INSUFFICIENT_HISTORY`. Also `optimal_execution.py:166` (`len(close) < 15` for Day Trader). | **Enforced Runtime Gate** |
| **20 EMA / 20-Day ADV** | $\ge 20$ closed daily bars (mathematical burn-in) | `api/routes/analytics.py:143` and `optimal_execution.py:194` use `close.ewm(span=20, adjust=False).mean()`. Note: `pandas.ewm` initializes from observation 1 ($N \ge 1$); smoothing stability requires ~20 bars, but code does not fail closed on $<20$ bars. | **Smoothing Heuristic** (Unenforced runtime gate) |
| **Tail Risk (MVaR 95%)** | $\ge 30$ closed returns (sample confidence) | `analyst_dashboard/analyzers/advanced_risk_analyzer.py:50-120` computes skew, kurtosis, and MVaR on whatever returns pass the 15-day gate. $N \ge 30$ is an invariant standard (`ARX_INV_013`), but runtime code lacks an explicit `len(returns) < 30` cutoff. | **Statistical Standard** (`ARX_INV_013`; runtime gate unresolved) |
| **50 SMA & Swing Levels** | $\ge 50$ closed daily bars | `api/routes/analytics.py:156` checks `len(close) >= 50` for `latest_sma_50`. `optimal_execution.py:137` checks `len(close) < 50` to suppress swing trade levels (`Trend Evidence Incomplete (< 50 Sessions)`). `ConfluenceEngine` itself does NOT enforce a 50-session cutoff. | **Enforced Runtime Gate** (for 50 SMA & Swing Levels) |
| **200 SMA & Minervini VCP** | $\ge 200$ bars (200 SMA), 222 bars (slope), up to 252 bars (52w high/low) | `analyst_dashboard/vcp/classifier.py:82-90` and `predicate_registry.py:132,163` (`PRED_HISTORY_200_SESSIONS`, `PRED_MINERVINI_STAGE_2`). Requires 200 bars for 200 SMA, 222 bars for 22-day slope, and 252 bars for 52-week metrics. Fails closed if inputs are None. | **Enforced Mathematical Gate** |

### 8.3 The Cardinal Separation Principle
$$\boxed{\text{Listing Identity} \ne \text{SEC Filing Completeness} \ne \text{Scoring Eligibility} \ne \text{Execution Clearance}}$$

Sufficient historical price bars **never** establish automatic analytical, scoring, or execution readiness:
1. **Listing Identity (`Verified` / `Not verified`):** Attests exclusively that the instrument is an active US exchange-listed equity in the Security Master (`/api/v1/market/instruments/{symbol}`).
2. **SEC Filing Completeness (`Verified` / `Not verified`):** Attests verified Form 4 / Form 13F metadata in PostgreSQL; strictly defaults to `Not verified` in the absence of verified records. An active listing identity **never** implies filing completeness or insider ownership verification.
3. **Scoring Eligibility:** Confluence scoring requires verified multi-pillar evidence (technicals, fundamentals, corporate filings, macro) and remains held under `ARX_CONFLUENCE_INTEGRITY_HOLD`. Price history alone never activates scoring conviction.
4. **Execution Clearance:** All live capital commitment and order routing is permanently disabled under `HOLD_EXEC_001`. Having 15, 50, 200, or 252 sessions **never** clears an asset for live execution; all trade paths route to simulated order models with fail-closed sizing (`INV-28`).

### 8.4 Presentation Decoupling & Zero-Invariance
1. **Listing Identity (`resolveListingIdentity`):** Renders `Listing Identity: Verified` when confirmed as an active US exchange-listed equity.
2. **SEC Filing Coverage (`resolveSecFilingCoverage`):** Renders `SEC Filing Coverage: Verified` only when supported by verified Form 4 / Form 13F metadata; otherwise `Not verified`.
3. **Zero-Invariance (`ARX_INV_001`):** Missing indicators, unassessed factors, or uncalibrated risk metrics must never default to `0.0`, `0`, or `0/100`. Missing metrics must be explicitly presented as `UNAVAILABLE` or `N/A`.

