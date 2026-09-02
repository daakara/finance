# ARX UI State Contract: Deterministic Mapping & Rendering Invariants

## 1. Overview

This contract specifies the **deterministic mapping rules** from the multi-layer assessment engine to the presentation components. 

UI components must **never calculate business logic or infer financial state**. They strictly consume the normalized `TerminalViewState` and project it through the active cognitive lens.

---

## 2. The Nine-Layer Architecture

```
1. USER CONTEXT (Intent, Ownership, Horizon, Mode)
       ↓
2. DATA LAYER (Price, Fundamentals, Macro, Filings)
       ↓
3. DATA QUALITY / ELIGIBILITY (Freshness, Coverage, Validity, Completeness)
       ↓
4. EVIDENCE LAYER (Raw Metrics + Benchmarks)
       ↓
5. ASSESSMENT ENGINE (Favorable, Mixed, Unfavorable, Insufficient)
       ↓
6. DECISION POSTURE ENGINE (Research, Watch, Acquire, Hold, Trim, Exit Review, Avoid)
       ↓
7. EXPLANATION ENGINE (Observation, Model Rule, Interpretation, Change Condition)
       ↓
8. PRESENTATION LAYER (Guided, Standard, Advanced)
       ↓
9. ACTION & MONITORING (Alerts, Sizing, Watchlist, Thesis Tracking)
```

---

## 3. The `TerminalViewState` Contract

```typescript
export type Freshness = "REAL_TIME" | "DELAYED" | "END_OF_DAY" | "DAILY" | "QUARTERLY" | "STALE" | "UNKNOWN";
export type DataQualityLevel = "VERIFIED_HIGH" | "VERIFIED_MEDIUM" | "DEGRADED" | "INSUFFICIENT";
export type Assessment = "FAVORABLE" | "MIXED" | "UNFAVORABLE" | "INSUFFICIENT_EVIDENCE";
export type DecisionPosture = "RESEARCH" | "WATCH" | "ACQUIRE" | "HOLD" | "TRIM" | "EXIT_REVIEW" | "AVOID";
export type OwnershipState = "NOT_OWNED" | "OWNED" | "UNKNOWN";
export type OwnershipSource = "USER_DECLARED" | "PORTFOLIO_IMPORT" | "BROKER_CONNECTION" | "UNKNOWN";
export type TimeHorizon = "INTRADAY" | "SWING" | "POSITION" | "LONG_TERM";
export type ExperienceMode = "GUIDED" | "STANDARD" | "ADVANCED";

export interface DataProvenanceMetadata {
  priceAsOf: string;
  priceFreshness: Freshness;
  fundamentalsAsOf: string;
  fundamentalsFreshness: Freshness;
  macroAsOf: string;
  macroFreshness: Freshness;
  filingsAsOf: string;
  filingsFreshness: Freshness;
  dataCompletenessPct: number; // 0 - 100%
}

export interface FactorAlignmentInfo {
  agreeingFactors: number;
  totalFactors: number;
  alignmentLabel: string; // e.g. "3 of 4 factors currently align" (NOT "75% confidence")
}

export interface TerminalViewState {
  // 1. Identity & Context
  symbol: string;
  companyName: string;
  horizon: TimeHorizon;
  ownership: {
    state: OwnershipState;
    source: OwnershipSource;
  };
  experienceMode: ExperienceMode;

  // 2. Data Quality & Eligibility
  dataQuality: {
    level: DataQualityLevel;
    isEligibleForAssessment: boolean;
    missingDataReasons?: string[];
    provenance: DataProvenanceMetadata;
  };

  // 3. Core Assessment & Factor Alignment
  assessment: Assessment;
  factorAlignment: FactorAlignmentInfo;

  // 4. Decision Posture & Non-Prescriptive Human Vocabulary
  posture: DecisionPosture;
  uiStateLabel: string; // e.g. "Actionable Setup" | "Wait for Trigger" | "Thesis Intact" | "Thesis Needs Review"
  headlineExplanation: string;

  // 5. Invalidation & Risk Terms (Non-Prescriptive)
  setupInvalidationLevel?: {
    price: number;
    distancePct: number;
    description: string;
  };

  // 6. Actionable Conditions
  whatWouldChangeAssessment: string;

  // 7. Contextual Available Actions
  primaryAction: {
    label: string;
    actionType: "SET_ALERT" | "SIZE_TRADE" | "REVIEW_THESIS" | "RESEARCH_PROFILE";
    enabled: boolean;
  };
  secondaryActions: {
    label: string;
    actionType: string;
    enabled: boolean;
  }[];
}
```

---

## 4. Deterministic Rendering Rules Matrix

| Assessment | Data Quality | Ownership | Horizon | Derived Posture | User-Facing Headline | Primary UI Action | Secondary Actions |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **`INSUFFICIENT`** | `INSUFFICIENT` | Any | Any | `RESEARCH` | **Assessment Unavailable — More Data Required** | `[ Research Profile ]` | `[ Set Volume Alert ]` |
| **`FAVORABLE`** | `VERIFIED_HIGH` | `NOT_OWNED` | `SWING` | `ACQUIRE` | **Actionable Setup (In Buy Zone)** | `[ Calculate Position Size ]` | `[ Set Alert ]` `[ Compare ]` |
| **`MIXED`** | `VERIFIED_HIGH` | `NOT_OWNED` | `SWING` | `WATCH` | **Wait for Trigger (Not Ready)** | `[ Set Alert on 50D SMA ]` | `[ Add to Watchlist ]` |
| **`FAVORABLE`** | `VERIFIED_HIGH` | `OWNED` | `LONG_TERM` | `HOLD` | **Thesis Intact (Continue Holding)** | `[ Monitor Support Floor ]` | `[ Check Portfolio VaR ]` |
| **`UNFAVORABLE`** | `VERIFIED_HIGH` | `OWNED` | `SWING` | `EXIT_REVIEW` | **Thesis Needs Review (Invalidated)** | `[ Review Invalidation Criteria ]` | `[ Rebalance Cash ]` |
| **`UNFAVORABLE`** | `VERIFIED_HIGH` | `NOT_OWNED` | Any | `AVOID` | **Unfavorable Setup (High Risk)** | `[ Explore Alternatives ]` | `[ View Risk Drivers ]` |

---

## 5. Non-Prescriptive Vocabulary Standards

| Forbidden / Misleading Term | Canonical Grounded Replacement | Reason |
| :--- | :--- | :--- |
| ❌ *"87% Confidence"* | ✅ **"3 of 4 factors currently align"** | Eliminates false win-rate certainty. |
| ❌ *"ARX's Empirical Model"* | ✅ **"Traceable Observation + ARX Setup Criteria"** | Distinguishes facts from weighting rules. |
| ❌ *"[ Execute Stop ]"* | ✅ **"[ Review Invalidation Criteria ]"** | ARX is decision support, not an automated broker. |
| ❌ *"Risk Floor"* | ✅ **"Setup Invalidation Level"** (or *"Your Stop Level"*) | Avoids implying personalized risk authority. |
| ❌ *"ARX detects your holding"* | ✅ **"Ownership: User Declared / Portfolio Active"** | Honest about data provenance. |
