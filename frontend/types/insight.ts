// ARX Quantitative Insight & Contract Types (docs/ux/07_data_to_ui_contract.md & 09_ui_state_contract.md)

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
  provenance?: DataProvenance;
  source?: string;
  asOf?: string;
  freshness?: Freshness;
  significance?: "HIGH" | "MEDIUM" | "LOW";
  status: "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "UNAVAILABLE";
}

export interface FactorAgreement {
  favorable: number;
  mixed: number;
  unfavorable: number;
  unavailable: number;
  evaluated: number;
  displayLabel: string;
}

export interface DomainAssessment {
  domainId: "trend" | "health" | "smart_money" | "macro" | string;
  domainName: "Price Trend" | "Company Health" | "Smart Money Flow" | "Macro Regime" | string;
  availability: EvidenceAvailability;
  status: "FAVORABLE" | "MIXED" | "UNFAVORABLE" | "UNAVAILABLE";
  pointImpact: number; // e.g. -25
  importanceLevel: "HIGH" | "MEDIUM" | "LOW";
  observation: string; // Raw empirical fact
  modelRule: string;   // Model weighting rule
  evidence: EvidenceItem[];
  whatWouldChangeAssessment: string;
}

// Backward-compatible factor item alias
export interface FactorAttributionItem {
  factorId?: string;
  factorName: string;
  category?: string;
  impact: number;
  importanceLevel?: "HIGH" | "MEDIUM" | "LOW";
  plainEnglishReason: string;
  reason?: string;
  sentiment: "positive" | "negative" | "neutral";
  evidence?: EvidenceItem[];
  whatWouldChangeAssessment: string;
}

export type ScoreAttributionItem = FactorAttributionItem;

export interface ARXAction {
  id: string;
  type: "RESEARCH" | "SIZE_POSITION" | "SET_ALERT" | "ADD_WATCHLIST" | "REVIEW_THESIS" | "COMPARE";
  label: string;
  enabled: boolean;
  reason?: string;
}

export interface TerminalViewState {
  symbol: string;
  companyName: string;
  currentPrice: number;
  changePct: number;
  horizon: TimeHorizon;
  ownership: {
    state: OwnershipState;
    source: OwnershipSource;
  };
  modelProvenance: ModelProvenance;
  overallEligibility: "ELIGIBLE" | "LIMITED" | "INELIGIBLE";
  assessment: Assessment;
  factorAgreement: FactorAgreement;
  domains: DomainAssessment[];
  posture: DecisionPosture;
  uiStateLabel: string;
  headlineExplanation: string;
  setupInvalidationLevel?: {
    price: number;
    distancePct: number;
    description: string;
  };
  whatWouldChangeAssessment: string;
  primaryAction: {
    label: string;
    actionType: "SET_ALERT" | "SIZE_TRADE" | "REVIEW_THESIS" | "RESEARCH_PROFILE";
    enabled: boolean;
  };
  availableActions: ARXAction[];
}

export interface QuantitativeInsight {
  id: string;
  symbol: string;
  companyName: string;
  price: number;
  changePct: number;
  setupScore: number;
  
  // Horizon & Posture State
  horizon: TimeHorizon;
  assessment: Assessment;
  posture: DecisionPosture;
  postureLabel: string;
  ownership: OwnershipState;
  
  // Normalized Terminal State
  terminalState: TerminalViewState;
  
  // Legacy verdict for existing components
  verdict: "WAIT_FOR_TRIGGER" | "STRONG_BUY_ZONE" | "ACTIONABLE_BUY_ZONE" | "PILOT_BUY" | "AVOID_STAGE_4" | "TAKE_PROFIT";
  verdictLabel: string;
  
  // Tier 1: Human Language (Guided)
  human: {
    assessmentHeadline: string;
    assessmentDescription: string;
    whyPills: {
      category: "Company Health" | "Price Trend" | "Smart Money" | "Market Outlook";
      status: "Healthy" | "Weak" | "Neutral" | "Supportive" | "Bearish" | "Caution";
      description: string;
      sentiment: "positive" | "negative" | "neutral" | "warning";
    }[];
    reclaimMilestone: string;
    watchLevels: {
      watchZone: string;
      keyLevel: string;
      riskStop: string;
    };
    actionCallout: {
      action: "WATCH" | "ENTER" | "SCALE_IN" | "TRIM";
      guidance: string;
    };
  };

  // Tier 2: Explanation (Standard)
  standard: {
    bottomLine: string;
    signalsRatio: string;
    confluenceBreakdown: {
      dimension: string;
      score: number;
      label?: string;
    }[];
    keyLevels: {
      currentPrice: number;
      watchZone: string;
      sma50?: number;
      stopLoss: number;
      stopLossPct: number;
      target1: number;
      target1Pct: number;
      target2: number;
      target2Pct: number;
      profitRiskRatio: number;
    };
    setupSummary: string;
  };

  // Tier 3: Quantitative Data (Advanced)
  advanced: {
    rsi: number;
    ema20?: number;
    sma50?: number;
    atr?: number;
    rvol?: number;
    beta?: number;
    marketCap: string;
    peRatio?: number;
    roic?: number;
    debtToEquity?: number;
    nextEarningsDate?: string;
    vcpStage?: number;
    relativeStrengthScore?: number;
    var95Pct?: number;
  };

  // Why Score Attribution
  scoreAttribution: {
    finalScore: number;
    items: FactorAttributionItem[];
    catalystToIncreaseScore: string;
  };

  primaryRiskSummary: string;
  whatWouldChangeAssessment: string;
  availableActions: ARXAction[];
}
