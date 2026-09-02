# ARX Data-to-UI Contract Specification

## 1. Overview

This contract establishes the formal data interface between the quantitative engine and the Next.js TypeScript presentation layer (`frontend/`).

---

## 2. Supported vs. Proposed Data Matrix

| Domain | Observed / Derived in Pipeline | Source Integration | Freshness Taxonomy |
| :--- | :--- | :--- | :--- |
| **Price & Candlesticks** | 1D, 1W, 1M, 3M, 1Y, 5Y OHLCV historical candles | `yfinance` / AlphaVantage / Cache | `DELAYED` (15m) / `END_OF_DAY` |
| **Moving Averages (50D SMA, 20 EMA)** | Derived arithmetic/exponential means | Calculated over valid candle window | `END_OF_DAY` |
| **Fundamentals (ROIC, Margins)** | SEC Form 10-Q/10-K reported financials | EDGAR database / `MASTER_ASSET_CATALOG` | `QUARTERLY` |
| **Smart Money (Insiders, STOCK Act)** | SEC Form 4 insider filings, House/Senate disclosures | SEC RSS XML feed / Disclosure API | `DAILY` |
| **Macro Regime (VIX, 10Y Yield)** | CBOE Volatility Index, 10-Year Constant Maturity Yield | St. Louis Fed FRED API (`VIXCLS`, `DGS10`) | `DAILY` |
| **Risk Modeling (VaR 95%, ATR)** | Cornish-Fisher expansion, 14-day True Range | Derived statistical window | `END_OF_DAY` |

---

## 3. The `ARXAssessment` TypeScript Contract

```typescript
export type TimeHorizon = "INTRADAY" | "SWING" | "POSITION" | "LONG_TERM";
export type Assessment = "FAVORABLE" | "MIXED" | "UNFAVORABLE" | "INSUFFICIENT_EVIDENCE";
export type DecisionPosture = "RESEARCH" | "WATCH" | "ACQUIRE" | "HOLD" | "TRIM" | "EXIT_REVIEW" | "AVOID";
export type OwnershipState = "NOT_OWNED" | "OWNED" | "UNKNOWN";
export type OwnershipSource = "USER_DECLARED" | "PORTFOLIO_IMPORT" | "BROKER_CONNECTION" | "UNKNOWN";
export type ExperienceMode = "GUIDED" | "STANDARD" | "ADVANCED";

export type Freshness = "REALTIME" | "DELAYED" | "END_OF_DAY" | "DAILY" | "QUARTERLY" | "STALE" | "UNKNOWN";
export type EvidenceAvailability = "AVAILABLE" | "PARTIAL" | "UNAVAILABLE" | "STALE";

export interface DataProvenance {
  source: string;
  observedAt: string;
  publishedAt?: string;
  effectiveAt?: string;
  freshness: Freshness;
}

export interface ModelProvenance {
  modelId: string;
  modelVersion: string;
  rulesetVersion: string;
  calculatedAt: string;
}

export interface EvidenceItem {
  metricName: string;
  currentValue: string | number;
  benchmarkValue: string | number;
  provenance: DataProvenance;
  significance: "HIGH" | "MEDIUM" | "LOW";
  status: "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "UNAVAILABLE";
}

export interface FactorAgreement {
  favorable: number;
  mixed: number;
  unfavorable: number;
  unavailable: number;
  evaluated: number;
  displayLabel: string; // e.g. "3 of 4 evaluated factors are favorable" (NOT "75% confidence")
}

export interface DomainAssessment {
  domainId: "trend" | "health" | "smart_money" | "macro";
  domainName: "Price Trend" | "Company Health" | "Smart Money Flow" | "Macro Regime";
  availability: EvidenceAvailability;
  status: "FAVORABLE" | "MIXED" | "UNFAVORABLE" | "UNAVAILABLE";
  pointImpact: number; // e.g. -25 (Model weighting choice, not empirical fact)
  importanceLevel: "HIGH" | "MEDIUM" | "LOW";
  observation: string; // Raw empirical fact (e.g. "Price is 10.3% below 50-day average")
  modelRule: string;   // Modeling rule (e.g. "Price below 50-day SMA deducts 25 points")
  evidence: EvidenceItem[];
  whatWouldChangeAssessment: string;
}

export interface ARXAction {
  id: string;
  type: "RESEARCH" | "SIZE_POSITION" | "SET_ALERT" | "ADD_WATCHLIST" | "REVIEW_THESIS" | "COMPARE";
  label: string;
  enabled: boolean;
  reason?: string;
}

export interface ARXAssessment {
  symbol: string;
  companyName: string;
  horizon: TimeHorizon;
  ownership: {
    state: OwnershipState;
    source: OwnershipSource;
  };
  
  // Model & Pipeline Provenance
  modelProvenance: ModelProvenance;
  overallEligibility: "ELIGIBLE" | "LIMITED" | "INELIGIBLE";
  
  // Assessment & Agreement (Not Probability)
  assessment: Assessment;
  factorAgreement: FactorAgreement;
  domains: DomainAssessment[];
  
  // Contextual Posture (Derived deterministically)
  posture: DecisionPosture;
  postureLabel: string; // e.g. "Actionable Setup" | "Wait for Trigger" | "Thesis Intact"
  
  // Explanation & Risk Invariants
  headlineSummary: string;
  primaryReason: string;
  setupInvalidationLevel?: {
    price: number;
    distancePct: number;
    description: string;
  };
  whatWouldChangeAssessment: string;
  
  // Non-Prescriptive Actions
  availableActions: ARXAction[];
}
```

---

## 4. Metric-Specific Data Eligibility Rules

Data eligibility is metric-specific and calculation-dependent (never an arbitrary blanket threshold):

* **Price Trend (50D SMA, 20 EMA)**: Requires $\ge 50$ consecutive valid trading-day close observations.
* **Volatility & Risk (ATR 14, VaR 95%)**: Requires $\ge 14$ sessions of True Range and $\ge 60$ days of returns.
* **Company Health (ROIC, Margins)**: Requires at least one SEC Form 10-Q/10-K filing.
* **Smart Money**: Requires at least one Form 4 insider transaction or quarterly 13F filing.

**Domain Decoupling Invariant**: Missing evidence in one domain (e.g. Fundamentals) does not invalidate an independent domain (e.g. Technical Price Trend). If fundamentals are absent, the system declares:
$$\text{Technical Assessment: FAVORABLE} \quad | \quad \text{Fundamental Evidence: UNAVAILABLE}$$
