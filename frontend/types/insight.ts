// ARX Quantitative Insight & Contract Types (docs/ux/07_data_to_ui_contract.md)

export type TimeHorizon = "INTRADAY" | "SWING" | "POSITION" | "LONG_TERM";
export type Assessment = "FAVORABLE" | "MIXED" | "UNFAVORABLE" | "INSUFFICIENT_EVIDENCE";
export type DecisionPosture = "RESEARCH" | "WATCH" | "ACQUIRE" | "HOLD" | "TRIM" | "EXIT_REVIEW" | "AVOID";
export type OwnershipState = "NOT_OWNED" | "OWNED" | "UNKNOWN";
export type ExperienceMode = "GUIDED" | "STANDARD" | "ADVANCED";

export interface EvidenceItem {
  metricName: string;
  currentValue: string | number;
  benchmarkValue: string | number;
  source?: string;
  asOf?: string;
  freshness?: "REALTIME" | "DAILY_CLOSE" | "QUARTERLY_FILING";
  significance?: "HIGH" | "MEDIUM" | "LOW";
  status: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
}

export interface FactorAttributionItem {
  factorId?: string;
  factorName: "Company Health" | "Price Trend" | "Smart Money Flow" | "Macro Regime" | string;
  category?: string; // Backward-compatible alias
  impact: number;    // e.g. +20, -25
  importanceLevel?: "HIGH" | "MEDIUM" | "LOW";
  plainEnglishReason: string;
  reason?: string;   // Backward-compatible alias
  sentiment: "positive" | "negative" | "neutral";
  evidence?: EvidenceItem[];
  whatWouldChangeAssessment: string;
}

// Backward-compatible alias for existing modal props
export type ScoreAttributionItem = FactorAttributionItem;

export interface ARXAction {
  id: string;
  type: "RESEARCH" | "SIZE_POSITION" | "SET_ALERT" | "ADD_WATCHLIST" | "STRESS_TEST" | "REVIEW_THESIS" | "COMPARE";
  label: string;
  enabled: boolean;
  reason?: string;
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
  
  // Legacy verdict for existing components
  verdict: "WAIT_FOR_TRIGGER" | "STRONG_BUY_ZONE" | "PILOT_BUY" | "AVOID_STAGE_4" | "TAKE_PROFIT";
  verdictLabel: string;
  
  // Tier 1: Human Language (Guided: "Help me understand")
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

  // Tier 2: Explanation (Standard: "Help me decide")
  standard: {
    bottomLine: string;
    signalsRatio: string; // e.g. "1 of 4 Positive Signals"
    confluenceBreakdown: {
      dimension: string;
      score: number;
      label?: string;
    }[];
    keyLevels: {
      currentPrice: number;
      watchZone: string;
      sma50: number;
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

  // Tier 3: Quantitative Data (Advanced: "Give me control")
  advanced: {
    rsi: number;
    ema20: number;
    sma50: number;
    atr: number;
    rvol: number;
    beta: number;
    marketCap: string;
    peRatio: number;
    roic: number;
    debtToEquity: number;
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

  // Downside Risk & Invalidation
  primaryRiskSummary: string;
  whatWouldChangeAssessment: string;
  availableActions: ARXAction[];
}
