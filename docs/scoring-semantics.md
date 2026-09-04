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
