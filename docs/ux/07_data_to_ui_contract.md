# ARX Data-to-UI Contract Specification

## 1. Overview

This contract establishes the formal data interface between the Python quantitative engine (`api/`, `engines/`) and the Next.js TypeScript presentation layer (`frontend/`).

---

## 2. Supported vs. Proposed Data Matrix

| Domain | Supported in Live Backend / Registry | Partially Supported / Mocked | Not Currently Available |
| :--- | :--- | :--- | :--- |
| **Price & Candlesticks** | 1D, 1W, 1M, 3M, 1Y, 5Y OHLCV historical candles | Intraday 1m/5m intervals (historical cache) | Live Level 2 orderbook stream |
| **Fundamentals** | ROIC, P/E, PEG, Gross Margins, Piotroski F-Score | Forward consensus revisions | Real-time earnings conference transcripts |
| **Smart Money** | SEC Form 4 insider transactions, Congressional trades | Institutional 13F net quarter changes | Real-time dark pool block prints |
| **Macro Regime** | FRED 10Y Treasury yield, VIX, S&P 500 regime | Sector rotation model | Live Federal Reserve rate probability curve |
| **Risk Modeling** | Cornish-Fisher VaR (95% 1-Day), ATR (14D), Beta | Multi-asset covariance matrix | Monte Carlo 10,000-path simulator |

---

## 3. The `ARXAssessment` TypeScript Contract

```typescript
export type TimeHorizon = "INTRADAY" | "SWING" | "POSITION" | "LONG_TERM";
export type Assessment = "FAVORABLE" | "MIXED" | "UNFAVORABLE" | "INSUFFICIENT_EVIDENCE";
export type DecisionPosture = "RESEARCH" | "WATCH" | "ACQUIRE" | "HOLD" | "TRIM" | "EXIT_REVIEW" | "AVOID";
export type OwnershipState = "NOT_OWNED" | "OWNED" | "UNKNOWN";
export type ExperienceMode = "GUIDED" | "STANDARD" | "ADVANCED";

export interface EvidenceItem {
  metricName: string;
  currentValue: string | number;
  benchmarkValue: string | number;
  source: string;              // e.g. "SEC Form 10-Q" | "FRED API" | "Yahoo Finance"
  asOf: string;                // ISO timestamp or date
  freshness: "REALTIME" | "DAILY_CLOSE" | "QUARTERLY_FILING";
  significance: "HIGH" | "MEDIUM" | "LOW";
  status: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
}

export interface FactorAttributionItem {
  factorId: string;
  factorName: "Company Health" | "Price Trend" | "Smart Money Flow" | "Macro Regime";
  pointImpact: number;         // e.g. -25
  importanceLevel: "HIGH" | "MEDIUM" | "LOW";
  plainEnglishReason: string;
  evidence: EvidenceItem[];
  whatWouldChangeAssessment: string;
}

export interface ARXAction {
  id: string;
  type: "RESEARCH" | "SIZE_POSITION" | "SET_ALERT" | "ADD_WATCHLIST" | "STRESS_TEST" | "REVIEW_THESIS" | "COMPARE";
  label: string;
  enabled: boolean;
  reason?: string;
}

export interface ARXAssessment {
  symbol: string;
  companyName: string;
  timestamp: string;
  horizon: TimeHorizon;
  
  // Core Analytical Output
  assessment: Assessment;
  evidenceQuality: {
    level: "HIGH" | "MEDIUM" | "LOW";
    reason: string;
  };
  
  // Decision Posture (Derived from Context + Evidence)
  posture: DecisionPosture;
  postureLabel: string;        // e.g. "Actionable Setup" or "Wait for Trigger"
  
  // Traceable Narrative
  headlineSummary: string;
  factors: FactorAttributionItem[];
  
  // Downside Floor & Risk
  primaryRiskSummary: string;
  invalidationLevel?: {
    price: number;
    description: string;
  };
  
  // Triggers & Change Conditions
  keyTriggers: {
    label: string;
    targetPrice?: number;
    condition: string;
  }[];
  
  // Available Actions
  availableActions: ARXAction[];
}
```
