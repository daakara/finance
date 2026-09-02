// Quantitative Insight Schema & Adaptive Experience Types
export type ExperienceMode = "GUIDED" | "STANDARD" | "ADVANCED";

export interface InsightMetric {
  label: string;
  value: string | number;
  sentiment?: "positive" | "negative" | "neutral" | "warning";
  subtext?: string;
}

export interface ScoreAttributionItem {
  category: string;
  impact: number; // e.g. +20, -25
  reason: string;
  sentiment: "positive" | "negative" | "neutral";
}

export interface QuantitativeInsight {
  id: string;
  symbol: string;
  companyName: string;
  price: number;
  changePct: number;
  setupScore: number;
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
    items: ScoreAttributionItem[];
    catalystToIncreaseScore: string;
  };
}
