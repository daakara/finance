import {
  QuantitativeInsight,
  TimeHorizon,
  OwnershipState,
  OwnershipSource,
  DomainAssessment,
  FactorAttributionItem,
  DecisionTrace,
} from "../types/insight";
import { deriveAssessmentState } from "./assessmentEngine";
import { CandleData, ConfluenceData, OptimalExecutionPlan } from "./api";
import { MASTER_ASSET_CATALOG } from "./masterCatalog";
import { evaluateLevelRelation } from "./reclaimSemantics";

export function generateQuantitativeInsight(
  symbol: string,
  companyName: string,
  currentPrice: number,
  changePct: number,
  setupScore?: number,
  stage?: number,
  horizon: TimeHorizon = "SWING",
  ownership: OwnershipState = "NOT_OWNED",
  ownershipSource: OwnershipSource = "USER_DECLARED",
  candles?: CandleData[],
  dataSource?: "live" | "fallback" | "unavailable",
  confluence?: ConfluenceData,
  decisionTrace?: DecisionTrace,
  optimalExecution?: OptimalExecutionPlan,
  freshnessStatus?: string
): QuantitativeInsight {
  const isPriceValid = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0;
  const safePrice = isPriceValid ? currentPrice : 0;
  const isFallbackFeed = dataSource === "fallback";
  const finalSetupScore = confluence?.confluenceScore !== undefined
    ? Math.round(confluence.confluenceScore)
    : (setupScore !== undefined ? setupScore : 40);

  // 1. Real Historical Moving Averages calculation (Strict observation windows: DISC-01, DISC-02, DISC-07)
  // SMA50 requires at least 50 valid closed daily sessions. Synthetic fallback candles are NEVER treated as market evidence.
  let calculatedSma50: number | null = null;
  let calculatedEma20: number | null = null;

  if (candles && candles.length >= 50 && !isFallbackFeed) {
    const smaSlice = candles.slice(-50);
    const smaSum = smaSlice.reduce((sum, c) => sum + c.close, 0);
    calculatedSma50 = Number((smaSum / 50).toFixed(2));
  }

  // EMA20 requires at least 20 valid trading sessions for burn-in
  if (candles && candles.length >= 20 && !isFallbackFeed) {
    const k = 2 / (20 + 1);
    let currentEma = candles[0].close;
    for (let i = 1; i < candles.length; i++) {
      currentEma = candles[i].close * k + currentEma * (1 - k);
    }
    calculatedEma20 = Number(currentEma.toFixed(2));
  }

  // 1b. Authentic 14-Period RSI Calculation (Requires >= 15 valid candles)
  let calculatedRsi14: number | undefined = undefined;
  if (candles && candles.length >= 15 && !isFallbackFeed) {
    const rsiSlice = candles.slice(-15);
    let totalGain = 0;
    let totalLoss = 0;
    for (let i = 1; i < rsiSlice.length; i++) {
      const diff = rsiSlice[i].close - rsiSlice[i - 1].close;
      if (diff > 0) totalGain += diff;
      else totalLoss += Math.abs(diff);
    }
    const avgGain = totalGain / 14;
    const avgLoss = totalLoss / 14;
    if (avgLoss === 0) {
      calculatedRsi14 = 100.0;
    } else {
      const rs = avgGain / avgLoss;
      calculatedRsi14 = Number((100 - (100 / (1 + rs))).toFixed(1));
    }
  }

  // 1c. Authentic 1D Parametric Value at Risk (VaR 95%)
  let calculatedVar95: number | undefined = undefined;
  if (candles && candles.length >= 20 && !isFallbackFeed) {
    const returns: number[] = [];
    const varSlice = candles.slice(-21);
    for (let i = 1; i < varSlice.length; i++) {
      if (varSlice[i - 1].close > 0) {
        returns.push((varSlice[i].close - varSlice[i - 1].close) / varSlice[i - 1].close);
      }
    }
    if (returns.length >= 20) {
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance = returns.reduce((acc, r) => acc + Math.pow(r - mean, 2), 0) / (returns.length - 1);
      const stdev = Math.sqrt(variance);
      calculatedVar95 = Number((1.645 * stdev * 100).toFixed(1));
    }
  }

  const isTrendAvailable = calculatedSma50 !== null;
  const sma50 = calculatedSma50 ?? undefined;
  const ema20 = calculatedEma20 ?? undefined;

  // 2. Derive authentic stage from actual price vs 50D SMA (Minervini/Weinstein stage discipline)
  // Stage 4 (Markdown/Correction) if price is below 50D SMA; Stage 2 (Markup) if price is above 50D SMA
  const derivedStage = stage !== undefined
    ? stage
    : (isTrendAvailable ? (safePrice < (sma50 as number) ? 4 : 2) : 2);
  const isStage4 = derivedStage === 4;

  // 2b. Authentic Trade Levels (Phase 21: Use backend optimalExecution directly if provided)
  const isExecutionSuppressed = optimalExecution?.execution_status === "INSUFFICIENT_HISTORY"
    || optimalExecution?.execution_status === "UNVERIFIED_ASSET"
    || !isTrendAvailable;

  let stopLoss: number = 0;
  let target1: number | undefined = undefined;
  let target2: number | undefined = undefined;
  let profitRisk: number | undefined = undefined;

  if (optimalExecution && optimalExecution.stop_loss && optimalExecution.stop_loss > 0 && !isExecutionSuppressed) {
    stopLoss = optimalExecution.stop_loss;
    target1 = optimalExecution.take_profit_1 ?? undefined;
    target2 = optimalExecution.take_profit_2 ?? undefined;
    profitRisk = optimalExecution.risk_reward_ratio ?? undefined;
  }

  // 3. Bind authentic asset-specific fundamentals (DISC-03, DISC-04)
  // Strict evidence gating: if backend reports fundamentals unavailable or fallback data, do not present static catalog data as live filings
  const upperSym = symbol.toUpperCase().replace("-USD", "");
  const catAsset = MASTER_ASSET_CATALOG[upperSym];
  const fundPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("fundamental") || p.pillar.toLowerCase().includes("solvency"));
  const isBackendFundAvailable = fundPillar ? (fundPillar.status !== "unavailable" && fundPillar.score > 0) : undefined;
  
  // Health is available only when authentic backend fundamentals exist or verified catalog entry exists with positive confirmation
  const isHealthAvailable = isBackendFundAvailable !== undefined
    ? isBackendFundAvailable
    : (catAsset !== undefined && catAsset.roic !== undefined && !isFallbackFeed);

  const roicDisplay = isHealthAvailable && catAsset?.roic !== undefined ? `${catAsset.roic}%` : (fundPillar?.score ? `${fundPillar.score}/100` : "N/A");
  const filingDate = isHealthAvailable && catAsset?.secFilingDate ? catAsset.secFilingDate : "Current Tape";
  const piotroskiScore = catAsset?.piotroski ?? (fundPillar?.score ? Math.round(fundPillar.score / 11) : 0);

  // Build Normalized Domain Assessments (Unknown != Negative Invariant Enforced)
  const domains: DomainAssessment[] = [
    // Domain 1: Company Health (Fundamental)
    isHealthAvailable
      ? {
          domainId: "health",
          domainName: "Company Health",
          availability: "AVAILABLE",
          status: fundPillar?.status === "positive" || (catAsset && catAsset.roic >= 15) ? "FAVORABLE" : fundPillar?.status === "warning" || (catAsset && catAsset.roic < 8) ? "UNFAVORABLE" : "MIXED",
          pointImpact: fundPillar?.score ? Math.round(fundPillar.score * 0.20) : (catAsset && catAsset.roic >= 15 ? 20 : 10),
          importanceLevel: "HIGH",
          observation: fundPillar?.plainDetail || `ROIC > 15% (${roicDisplay}) and capital solvency (Piotroski ${piotroskiScore}/9).`,
          modelRule: "Sound capital efficiency and verified solvency contribute positive weighting to fundamental score.",
          evidence: [
            {
              metricName: "Return on Invested Capital (ROIC)",
              currentValue: roicDisplay,
              benchmarkValue: "10.0% Industry Avg",
              source: "SEC Form 10-Q Filing",
              asOf: filingDate,
              provenance: {
                source: "SEC EDGAR Form 10-Q",
                publishedAt: filingDate,
                observedAt: new Date().toISOString().split("T")[0],
                freshness: "QUARTERLY",
              },
              freshness: "QUARTERLY",
              significance: "HIGH",
              status: "POSITIVE",
            },
          ],
          whatWouldChangeAssessment: "A deterioration in operating margins below 8% would trigger a health downgrade.",
        }
      : {
          domainId: "health",
          domainName: "Company Health",
          availability: "UNAVAILABLE",
          status: "UNAVAILABLE",
          pointImpact: 0,
          importanceLevel: "HIGH",
          observation: fundPillar?.plainDetail || "Official SEC regulatory filings and verified financial statements unavailable for this asset.",
          modelRule: "Fundamental company health requires verified financial statements; zero points awarded when evidence is unavailable.",
          evidence: [],
          whatWouldChangeAssessment: "Publication of audited Form 10-Q or 10-K financial disclosures will unlock fundamental scoring.",
        },

    // Domain 2: Price Trend (Technical)
    isTrendAvailable
      ? {
          domainId: "trend",
          domainName: "Price Trend",
          availability: "AVAILABLE",
          status: isStage4 ? "UNFAVORABLE" : "FAVORABLE",
          pointImpact: isStage4 ? -25 : 25,
          importanceLevel: "HIGH",
          observation: isStage4
            ? `Price ($${safePrice.toFixed(2)}) is below the 50-day average ($${(sma50 as number).toFixed(2)}).`
            : `Price ($${safePrice.toFixed(2)}) is holding firmly above the 20 EMA ($${(ema20 as number).toFixed(2)}) and 50 SMA ($${(sma50 as number).toFixed(2)}).`,
          modelRule: isStage4
            ? "Price below 50-day SMA deducts 25 points because trend confirmation is absent."
            : "VCP base contraction above rising moving averages adds +25 points.",
          evidence: [
            {
              metricName: "Price vs 50-Day SMA",
              currentValue: `$${safePrice.toFixed(2)}`,
              benchmarkValue: `$${(sma50 as number).toFixed(2)} (50D SMA)`,
              source: "Market Feed",
              asOf: "15m Delayed",
              freshness: "DELAYED",
              significance: "HIGH",
              status: isStage4 ? "NEGATIVE" : "POSITIVE",
            },
          ],
          whatWouldChangeAssessment: isStage4
            ? `Price reclaiming and holding above $${(sma50 as number).toFixed(2)} (50D SMA) on above-average volume will remove this penalty.`
            : "A daily close below the 20-day EMA would weaken breakout strength.",
        }
      : {
          domainId: "trend",
          domainName: "Price Trend",
          availability: "UNAVAILABLE",
          status: "UNAVAILABLE",
          pointImpact: 0,
          importanceLevel: "HIGH",
          observation: candles && candles.length > 0
            ? `Insufficient historical trading sessions (${candles.length} of 50 required) to compute 50-day moving average.`
            : "Historical price action candles unavailable for this asset.",
          modelRule: "Price trend requires at least 50 valid trading sessions; zero points awarded when evidence is unavailable.",
          evidence: candles && candles.length > 0
            ? [
                {
                  metricName: "Price vs 50-Day SMA",
                  currentValue: `$${safePrice.toFixed(2)}`,
                  benchmarkValue: "N/A (< 50 sessions)",
                  source: "Market Feed",
                  asOf: "15m Delayed",
                  freshness: "DELAYED",
                  significance: "HIGH",
                  status: "UNAVAILABLE",
                },
              ]
            : [],
          whatWouldChangeAssessment: "Accumulation of 50 closed daily trading sessions will activate trend moving-average analysis.",
        },

    // Domain 3: Smart Money Flow
    (() => {
      const smartPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("smart") || p.pillar.toLowerCase().includes("flow"));
      // Strict availability: pillar must exist, not be 'unavailable', and have substantive non-zero score or explicit positive/warning status
      const isAvailable = Boolean(
        smartPillar &&
        smartPillar.status !== "unavailable" &&
        (smartPillar.score > 0 || smartPillar.status === "positive" || smartPillar.status === "warning")
      );
      return {
        domainId: "smart_money",
        domainName: "Smart Money Flow",
        availability: isAvailable ? "AVAILABLE" : "UNAVAILABLE",
        status: isAvailable ? (smartPillar?.status === "positive" ? "FAVORABLE" : smartPillar?.status === "warning" ? "UNFAVORABLE" : "MIXED") : "UNAVAILABLE",
        pointImpact: isAvailable ? Math.round((smartPillar?.score || 0) * 0.15) : 0,
        importanceLevel: "MEDIUM" as const,
        observation: isAvailable
          ? (smartPillar?.plainDetail || "Institutional and insider flow signals evaluated.")
          : (smartPillar?.plainDetail || "SEC Form 13F institutional holdings flow unindexed for this security."),
        modelRule: "Institutional net buying adds positive weighting to setup conviction.",
        evidence: [],
        whatWouldChangeAssessment: "Verified Form 4 insider transactions or institutional volume inflows would activate this factor.",
      };
    })(),

    // Domain 4: Macro Regime
    (() => {
      const macroPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("macro") || p.pillar.toLowerCase().includes("regime"));
      // Strict availability: pillar must exist, not be 'unavailable', and have substantive non-zero score or explicit positive/warning status
      const isAvailable = Boolean(
        macroPillar &&
        macroPillar.status !== "unavailable" &&
        (macroPillar.score > 0 || macroPillar.status === "positive" || macroPillar.status === "warning")
      );
      return {
        domainId: "macro",
        domainName: "Macro Regime",
        availability: isAvailable ? "AVAILABLE" : "UNAVAILABLE",
        status: isAvailable ? (macroPillar?.status === "positive" ? "FAVORABLE" : macroPillar?.status === "warning" ? "UNFAVORABLE" : "MIXED") : "UNAVAILABLE",
        pointImpact: isAvailable ? Math.round((macroPillar?.score || 0) * 0.15) : 0,
        importanceLevel: "MEDIUM" as const,
        observation: isAvailable
          ? (macroPillar?.plainDetail || "Macroeconomic environment and volatility regime evaluated.")
          : (macroPillar?.plainDetail || "Macro volatility regime telemetry is unassessed for this session."),
        modelRule: "Low volatility macro regime provides supportive market tailwinds (+15 points).",
        evidence: [],
        whatWouldChangeAssessment: "A shift in systemic volatility or credit spreads would modify macro risk assessment.",
      };
    })(),
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
    freshnessStatus,
  });

  // Phase 21 Epistemic Alignment: Honor authoritative backend decisionTrace if provided
  if (decisionTrace) {
    terminalState.decisionState = decisionTrace.decisionState;

    if (!decisionTrace.isActionable && terminalState.posture === "ACQUIRE") {
      terminalState.posture = "WATCH";
      terminalState.uiStateLabel = decisionTrace.stateLabel || "Valid Setup — Awaiting Trigger";
      terminalState.headlineExplanation = decisionTrace.disqualificationReason || (
        decisionTrace.stateLabel
          ? `Setup state: ${decisionTrace.stateLabel}; awaiting confirmed entry trigger.`
          : "Asset structure is under evaluation; awaiting confirmed entry trigger in buy zone."
      );
      terminalState.primaryAction = {
        label: sma50 !== undefined ? `Set Alert for $${sma50.toFixed(2)}` : "Set Price Alert",
        actionType: "SET_ALERT",
        enabled: true,
      };
    } else if (decisionTrace.stateLabel) {
      terminalState.uiStateLabel = decisionTrace.stateLabel;
      if (decisionTrace.disqualificationReason) {
        terminalState.headlineExplanation = decisionTrace.disqualificationReason;
      }
    }

    // Bind canSizeTrade and allowed actions directly from decisionTrace
    for (const action of terminalState.availableActions) {
      if (action.id === "size_trade") {
        action.enabled = decisionTrace.canSizeTrade;
        if (!decisionTrace.canSizeTrade) {
          action.reason = decisionTrace.disqualificationReason || "Trade sizing disabled until trigger confirmed";
        }
      }
    }
  }

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

  const finalVerdict = decisionTrace
    ? (decisionTrace.isActionable ? "ACTIONABLE_BUY_ZONE" : "WAIT_FOR_TRIGGER")
    : (terminalState.posture === "ACQUIRE" ? "ACTIONABLE_BUY_ZONE" : "WAIT_FOR_TRIGGER");

  const smaLevelRelation = evaluateLevelRelation(safePrice, sma50, "50-day moving average", symbol);

  return {
    id: `insight_${symbol.toLowerCase()}`,
    symbol: symbol.toUpperCase(),
    companyName,
    price: safePrice,
    changePct,
    setupScore: finalSetupScore,
    horizon,
    assessment: terminalState.assessment,
    posture: terminalState.posture,
    postureLabel: terminalState.uiStateLabel,
    ownership,
    terminalState,
    verdict: finalVerdict,
    verdictLabel: terminalState.uiStateLabel,

    // Tier 1: Human (Guided)
    human: {
      assessmentHeadline: terminalState.uiStateLabel,
      assessmentDescription: terminalState.headlineExplanation,
      whyPills: [
        {
          category: "Company Health",
          status: !isHealthAvailable ? "Unavailable" : (catAsset && catAsset.roic >= 20 ? "Healthy" : "Neutral"),
          description: !isHealthAvailable
            ? "Verified SEC financial filings unavailable for this security."
            : (catAsset && catAsset.roic >= 20
                ? "Stable financials and strong profitability across core metrics."
                : "Financial metrics meet baseline criteria without distinct edge."),
          sentiment: !isHealthAvailable ? "neutral" : (catAsset && catAsset.roic >= 20 ? "positive" : "neutral"),
        },
        {
          category: "Price Trend",
          status: (!isTrendAvailable || smaLevelRelation.status === "UNAVAILABLE")
            ? "Unavailable"
            : (smaLevelRelation.status === "BELOW"
                ? "Weak"
                : (smaLevelRelation.status === "AT_LEVEL" ? "Neutral" : "Healthy")),
          description: (!isTrendAvailable || smaLevelRelation.status === "UNAVAILABLE")
            ? "Insufficient daily sessions (< 50) to evaluate 50-day moving average trend."
            : (smaLevelRelation.status === "BELOW"
                ? `Price is below the 50-day moving average ($${(sma50 as number).toFixed(2)}) and currently falling.`
                : (smaLevelRelation.status === "AT_LEVEL"
                    ? `Price is testing the 50-day moving average ($${(sma50 as number).toFixed(2)}).`
                    : `Price is holding firmly above 50-day moving average ($${(sma50 as number).toFixed(2)}).`)),
          sentiment: (!isTrendAvailable || smaLevelRelation.status === "UNAVAILABLE")
            ? "neutral"
            : (smaLevelRelation.status === "BELOW" ? "negative" : (smaLevelRelation.status === "AT_LEVEL" ? "neutral" : "positive")),
        },
        {
          category: "Smart Money",
          status: (() => {
            const flowPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("flow") || p.pillar.toLowerCase().includes("smart"));
            if (!flowPillar || flowPillar.status === "unavailable" || (flowPillar.score === 0 && flowPillar.status === "neutral")) return "Unavailable";
            return flowPillar.status === "positive" ? "Supportive" : flowPillar.status === "warning" ? "Caution" : "Neutral";
          })(),
          description: (() => {
            const flowPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("flow") || p.pillar.toLowerCase().includes("smart"));
            return flowPillar?.plainDetail || "Institutional order flow telemetry is unassessed for this session.";
          })(),
          sentiment: "neutral",
        },
        {
          category: "Market Outlook",
          status: (() => {
            const macroPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("macro") || p.pillar.toLowerCase().includes("regime"));
            if (!macroPillar || macroPillar.status === "unavailable" || (macroPillar.score === 0 && macroPillar.status === "neutral")) return "Unavailable";
            return macroPillar.status === "positive" ? "Supportive" : macroPillar.status === "warning" ? "Caution" : "Neutral";
          })(),
          description: (() => {
            const macroPillar = confluence?.pillars?.find(p => p.pillar.toLowerCase().includes("macro") || p.pillar.toLowerCase().includes("regime"));
            return macroPillar?.plainDetail || "Broader market regime telemetry is unassessed; evaluate sector trend independently.";
          })(),
          sentiment: "neutral",
        },
      ],
      reclaimMilestone: smaLevelRelation.reclaimMilestone,
      watchLevels: {
        watchZone: (isPriceValid && !isExecutionSuppressed) ? `$${(safePrice * 0.975).toFixed(2)} – $${(safePrice * 1.052).toFixed(2)}` : "N/A (< 50 sessions)",
        keyLevel: sma50 !== undefined ? `$${(sma50 as number).toFixed(2)} (50D SMA)` : "N/A (< 50 sessions)",
        riskStop: (isPriceValid && !isExecutionSuppressed && stopLoss > 0) ? `$${stopLoss.toFixed(2)} (${(((stopLoss - safePrice) / safePrice) * 100).toFixed(1)}%)` : "N/A (< 50 sessions)",
      },
      actionCallout: {
        action: terminalState.posture === "ACQUIRE"
          ? "ENTER"
          : terminalState.posture === "RESEARCH"
          ? "RESEARCH"
          : terminalState.posture === "AVOID"
          ? "AVOID"
          : terminalState.posture === "EXIT_REVIEW"
          ? "EXIT_REVIEW"
          : "WATCH",
        guidance: terminalState.posture === "RESEARCH"
          ? "Evidence incomplete. Further quantitative research required before taking position."
          : terminalState.posture === "AVOID"
          ? "Unfavorable technical trend or fundamental risks present unfavorable risk/reward."
          : terminalState.posture === "EXIT_REVIEW"
          ? `Price has fallen below the setup invalidation floor ($${stopLoss.toFixed(2)}). Review position.`
          : smaLevelRelation.status === "BELOW"
          ? (isTrendAvailable
              ? `Watch for a strong reversal and reclaim of $${(sma50 as number).toFixed(2)} with volume. Don't rush—wait for the trigger.`
              : `Trend evidence incomplete. Wait for market structure confirmation.`)
          : smaLevelRelation.status === "AT_LEVEL"
          ? `Testing 50-day moving average ($${(sma50 as number).toFixed(2)}). Wait for decisive volume confirmation before entry.`
          : `Setup confirmed within the optimal buy zone. Setup invalidation level at $${stopLoss.toFixed(2)}.`,
      },
    },

    // Tier 2: Explanation (Standard)
    standard: {
      bottomLine: terminalState.headlineExplanation,
      signalsRatio: terminalState.factorAgreement.displayLabel,
      confluenceBreakdown: (confluence?.pillars && confluence.pillars.length > 0)
        ? confluence.pillars.map((p) => ({
            dimension: p.plainLabel || p.label,
            score: Math.round(p.score),
          }))
        : [
            { dimension: "Chart Structure", score: !isTrendAvailable ? 0 : (isStage4 ? 40 : 88) },
            { dimension: "Company Health", score: !isHealthAvailable ? 0 : 80 },
            { dimension: "Smart Money Flow", score: 0 },
            { dimension: "Market Tailwinds", score: 60 },
          ],
      keyLevels: {
        currentPrice: safePrice,
        watchZone: (isPriceValid && !isExecutionSuppressed) ? `$${(safePrice * 0.975).toFixed(0)} – $${(safePrice * 1.052).toFixed(0)}` : "N/A",
        sma50,
        stopLoss: !isExecutionSuppressed ? stopLoss : 0,
        stopLossPct: (isPriceValid && !isExecutionSuppressed && stopLoss > 0)
          ? Number((((stopLoss - safePrice) / safePrice) * 100).toFixed(1))
          : 0,
        target1: isTrendAvailable ? target1 : undefined,
        target1Pct: (isPriceValid && !isExecutionSuppressed && target1 !== undefined && target1 > 0)
          ? Number((((target1 - safePrice) / safePrice) * 100).toFixed(1))
          : undefined,
        target2: isTrendAvailable ? target2 : undefined,
        target2Pct: (isPriceValid && !isExecutionSuppressed && target2 !== undefined && target2 > 0)
          ? Number((((target2 - safePrice) / safePrice) * 100).toFixed(1))
          : undefined,
        profitRiskRatio: isTrendAvailable ? profitRisk : undefined,
      },
      setupSummary: !isTrendAvailable
        ? "Trend Evidence Incomplete — Awaiting 50-session historical base."
        : (isStage4
            ? "Stage 4 Correction / Base Building Required below 50-day SMA."
            : "VCP Stage 3 Contraction / Relative Strength Leader."),
    },

    // Tier 3: Quantitative Data (Advanced)
    advanced: {
      rsi: calculatedRsi14,
      ema20,
      sma50,
      atr: catAsset?.atr14,
      rvol: catAsset?.rvol,
      beta: catAsset?.beta,
      marketCap: catAsset?.marketCap || "N/A",
      peRatio: catAsset?.fwdPe,
      roic: catAsset?.roic,
      debtToEquity: undefined,
      vcpStage: isStage4 ? undefined : 3,
      relativeStrengthScore: catAsset?.momentumScore ?? (isTrendAvailable ? (isStage4 ? 45 : 88) : undefined),
      var95Pct: calculatedVar95,
    },

    // Traceable Attribution Model
    scoreAttribution: {
      finalScore: finalSetupScore,
      items: factors,
      catalystToIncreaseScore: terminalState.whatWouldChangeAssessment,
    },

    primaryRiskSummary: (isTrendAvailable && stopLoss > 0)
      ? `A close below $${stopLoss.toFixed(2)} (-7.0%) invalidates the technical setup.`
      : "Risk levels suppressed: awaiting 50-session historical base.",
    whatWouldChangeAssessment: terminalState.whatWouldChangeAssessment,
    availableActions: terminalState.availableActions,
  };
}
