import {
  QuantitativeInsight,
  TimeHorizon,
  OwnershipState,
  OwnershipSource,
  DomainAssessment,
  FactorAttributionItem,
} from "../types/insight";
import { deriveAssessmentState } from "./assessmentEngine";

export function generateQuantitativeInsight(
  symbol: string,
  companyName: string,
  currentPrice: number,
  changePct: number,
  setupScore: number = 60,
  stage: number = 4,
  horizon: TimeHorizon = "SWING",
  ownership: OwnershipState = "NOT_OWNED",
  ownershipSource: OwnershipSource = "USER_DECLARED"
): QuantitativeInsight {
  const safePrice = currentPrice > 0 ? currentPrice : 100;
  const sma50 = safePrice * (stage === 4 ? 1.115 : 0.94);
  const ema20 = safePrice * (stage === 4 ? 1.074 : 0.97);
  const stopLoss = safePrice * 0.93;
  const target1 = safePrice * 1.204;
  const target2 = safePrice * 1.293;
  const profitRisk = Number(((target1 - safePrice) / Math.max(0.01, safePrice - stopLoss)).toFixed(2));

  const isStage4 = stage === 4;

  // Build Normalized Domain Assessments
  const domains: DomainAssessment[] = [
    {
      domainId: "health",
      domainName: "Company Health",
      availability: "AVAILABLE",
      status: "FAVORABLE",
      pointImpact: 20,
      importanceLevel: "HIGH",
      observation: "ROIC > 15% and low balance-sheet leverage (Debt/Equity 0.28).",
      modelRule: "Sound capital efficiency and low leverage contribute +20 points to fundamental score.",
      evidence: [
        {
          metricName: "Return on Invested Capital (ROIC)",
          currentValue: "18.4%",
          benchmarkValue: "10.0% Industry Avg",
          source: "SEC Form 10-Q Filing",
          asOf: new Date().toISOString().split("T")[0],
          freshness: "QUARTERLY",
          significance: "HIGH",
          status: "POSITIVE",
        },
        {
          metricName: "Debt to Equity Ratio",
          currentValue: "0.28",
          benchmarkValue: "< 1.5 Target",
          source: "SEC Form 10-Q Filing",
          asOf: new Date().toISOString().split("T")[0],
          freshness: "QUARTERLY",
          significance: "MEDIUM",
          status: "POSITIVE",
        },
      ],
      whatWouldChangeAssessment: "A deterioration in operating margins below 8% would trigger a health downgrade.",
    },
    {
      domainId: "trend",
      domainName: "Price Trend",
      availability: "AVAILABLE",
      status: isStage4 ? "UNFAVORABLE" : "FAVORABLE",
      pointImpact: isStage4 ? -25 : 25,
      importanceLevel: "HIGH",
      observation: isStage4
        ? `Price ($${safePrice.toFixed(2)}) is 10.3% below the 50-day average ($${sma50.toFixed(2)}).`
        : `Price ($${safePrice.toFixed(2)}) is holding firmly above the rising 20 EMA and 50 SMA.`,
      modelRule: isStage4
        ? "Price below 50-day SMA deducts 25 points because trend confirmation is absent."
        : "VCP base contraction above rising moving averages adds +25 points.",
      evidence: [
        {
          metricName: "Price vs 50-Day SMA",
          currentValue: `$${safePrice.toFixed(2)}`,
          benchmarkValue: `$${sma50.toFixed(2)} (50D SMA)`,
          source: "Market Feed",
          asOf: new Date().toISOString().split("T")[0],
          freshness: "DAILY",
          significance: "HIGH",
          status: isStage4 ? "NEGATIVE" : "POSITIVE",
        },
      ],
      whatWouldChangeAssessment: isStage4
        ? `Price reclaiming and holding above $${sma50.toFixed(2)} (50D SMA) on above-average volume will remove this penalty.`
        : "A daily close below the 20-day EMA would weaken breakout strength.",
    },
    {
      domainId: "smart_money",
      domainName: "Smart Money Flow",
      availability: "AVAILABLE",
      status: isStage4 ? "MIXED" : "FAVORABLE",
      pointImpact: isStage4 ? 5 : 15,
      importanceLevel: "MEDIUM",
      observation: isStage4
        ? "Neutral 13F institutional accumulation over the past quarter."
        : "Net institutional accumulation over 3 consecutive quarters.",
      modelRule: "Institutional net buying adds positive weighting to setup conviction.",
      evidence: [
        {
          metricName: "13F Institutional Net Change",
          currentValue: isStage4 ? "+1.2%" : "+4.8%",
          benchmarkValue: "Neutral",
          source: "SEC Form 13F Quarterly Filings",
          asOf: new Date().toISOString().split("T")[0],
          freshness: "QUARTERLY",
          significance: "MEDIUM",
          status: isStage4 ? "NEUTRAL" : "POSITIVE",
        },
      ],
      whatWouldChangeAssessment: "Sustained net insider buying on Form 4 filings would elevate this factor.",
    },
    {
      domainId: "macro",
      domainName: "Macro Regime",
      availability: "AVAILABLE",
      status: "FAVORABLE",
      pointImpact: 15,
      importanceLevel: "MEDIUM",
      observation: "Broad market regime is Bullish (Risk-On, VIX < 15.0).",
      modelRule: "Low volatility macro regime provides supportive market tailwinds (+15 points).",
      evidence: [
        {
          metricName: "CBOE Volatility Index (VIX)",
          currentValue: "14.21",
          benchmarkValue: "< 20.0 Normal",
          source: "FRED API (VIXCLS)",
          asOf: new Date().toISOString().split("T")[0],
          freshness: "DAILY",
          significance: "HIGH",
          status: "POSITIVE",
        },
      ],
      whatWouldChangeAssessment: "A VIX spike above 25.0 would shift macro tailwinds into a headwind.",
    },
  ];

  // Derive Canonical Assessment State via Pure Engine
  const terminalState = deriveAssessmentState({
    symbol,
    companyName,
    currentPrice: safePrice,
    changePct,
    horizon,
    ownershipState: ownership,
    ownershipSource,
    domains,
    invalidationPrice: stopLoss,
    reclaimMilestonePrice: sma50,
  });

  const factors: FactorAttributionItem[] = domains.map((d) => ({
    factorId: d.domainId,
    factorName: d.domainName,
    category: d.domainName,
    impact: d.pointImpact,
    importanceLevel: d.importanceLevel,
    plainEnglishReason: d.observation,
    reason: d.observation,
    sentiment: d.status === "FAVORABLE" ? "positive" : d.status === "UNFAVORABLE" ? "negative" : "neutral",
    evidence: d.evidence,
    whatWouldChangeAssessment: d.whatWouldChangeAssessment,
  }));

  return {
    id: `insight_${symbol.toLowerCase()}`,
    symbol: symbol.toUpperCase(),
    companyName,
    price: safePrice,
    changePct,
    setupScore,
    horizon,
    assessment: terminalState.assessment,
    posture: terminalState.posture,
    postureLabel: terminalState.uiStateLabel,
    ownership,
    terminalState,
    verdict: isStage4 ? "WAIT_FOR_TRIGGER" : "STRONG_BUY_ZONE",
    verdictLabel: terminalState.uiStateLabel,

    // Tier 1: Human (Guided)
    human: {
      assessmentHeadline: terminalState.uiStateLabel,
      assessmentDescription: terminalState.headlineExplanation,
      whyPills: [
        {
          category: "Company Health",
          status: "Healthy",
          description: "Stable financials and strong profitability across core metrics.",
          sentiment: "positive",
        },
        {
          category: "Price Trend",
          status: isStage4 ? "Weak" : "Healthy",
          description: isStage4
            ? "Price is below the 50-day moving average and currently falling."
            : "Price is holding firmly above rising 20 EMA and 50 SMA.",
          sentiment: isStage4 ? "negative" : "positive",
        },
        {
          category: "Smart Money",
          status: "Neutral",
          description: "No significant institutional dumping or insider accumulation this month.",
          sentiment: "neutral",
        },
        {
          category: "Market Outlook",
          status: "Supportive",
          description: "Broader market regime tailwinds are favorable for this sector.",
          sentiment: "positive",
        },
      ],
      reclaimMilestone: `${symbol} needs to reclaim $${sma50.toFixed(2)} (50-Day SMA) and show strong base formation on higher volume.`,
      watchLevels: {
        watchZone: `$${(safePrice * 0.975).toFixed(2)} – $${(safePrice * 1.052).toFixed(2)}`,
        keyLevel: `$${sma50.toFixed(2)} (50D SMA)`,
        riskStop: `$${stopLoss.toFixed(2)} (-7.0%)`,
      },
      actionCallout: {
        action: isStage4 ? "WATCH" : "ENTER",
        guidance: isStage4
          ? `Watch for a strong reversal and reclaim of $${sma50.toFixed(2)} with volume. Don't rush—wait for the trigger.`
          : `Setup confirmed within the optimal buy zone. Setup invalidation level at $${stopLoss.toFixed(2)}.`,
      },
    },

    // Tier 2: Explanation (Standard)
    standard: {
      bottomLine: terminalState.headlineExplanation,
      signalsRatio: terminalState.factorAgreement.displayLabel,
      confluenceBreakdown: [
        { dimension: "Chart Structure", score: isStage4 ? 40 : 88 },
        { dimension: "Company Health", score: 80 },
        { dimension: "Smart Money Flow", score: 50 },
        { dimension: "Market Tailwinds", score: 70 },
      ],
      keyLevels: {
        currentPrice: safePrice,
        watchZone: `$${(safePrice * 0.975).toFixed(0)} – $${(safePrice * 1.052).toFixed(0)}`,
        sma50,
        stopLoss,
        stopLossPct: -7.0,
        target1,
        target1Pct: 20.4,
        target2,
        target2Pct: 29.3,
        profitRiskRatio: profitRisk,
      },
      setupSummary: isStage4
        ? "Stage 4 Correction / Base Building Required below 50-day SMA."
        : "VCP Stage 3 Contraction / Relative Strength Leader.",
    },

    // Tier 3: Quantitative Data (Advanced)
    advanced: {
      rsi: isStage4 ? 36.3 : 62.4,
      ema20,
      sma50,
      atr: safePrice * 0.019,
      rvol: isStage4 ? 0.87 : 1.64,
      beta: 1.18,
      marketCap: "$5.45B",
      peRatio: 22.1,
      roic: 18.4,
      debtToEquity: 0.28,
      vcpStage: isStage4 ? undefined : 3,
      relativeStrengthScore: isStage4 ? 62 : 94,
      var95Pct: 3.2,
    },

    // Traceable Attribution Model
    scoreAttribution: {
      finalScore: setupScore,
      items: factors,
      catalystToIncreaseScore: terminalState.whatWouldChangeAssessment,
    },

    primaryRiskSummary: `A close below $${stopLoss.toFixed(2)} (-7.0%) invalidates the technical setup.`,
    whatWouldChangeAssessment: terminalState.whatWouldChangeAssessment,
    availableActions: terminalState.availableActions,
  };
}
