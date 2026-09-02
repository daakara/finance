import { QuantitativeInsight } from "../types/insight";

export function generateQuantitativeInsight(
  symbol: string,
  companyName: string,
  currentPrice: number,
  changePct: number,
  setupScore: number = 60,
  stage: number = 4
): QuantitativeInsight {
  const safePrice = currentPrice > 0 ? currentPrice : 100;
  const sma50 = safePrice * (stage === 4 ? 1.115 : 0.94);
  const ema20 = safePrice * (stage === 4 ? 1.074 : 0.97);
  const stopLoss = safePrice * 0.93;
  const target1 = safePrice * 1.204;
  const target2 = safePrice * 1.293;
  const profitRisk = Number(((target1 - safePrice) / Math.max(0.01, safePrice - stopLoss)).toFixed(2));

  const isStage4 = stage === 4;

  return {
    id: `insight_${symbol.toLowerCase()}`,
    symbol: symbol.toUpperCase(),
    companyName,
    price: safePrice,
    changePct,
    setupScore,
    verdict: isStage4 ? "WAIT_FOR_TRIGGER" : "STRONG_BUY_ZONE",
    verdictLabel: isStage4 ? "Selective Entry: Wait for Trigger" : "High-Conviction Breakout Setup",

    // Tier 1: Human (Guided)
    human: {
      assessmentHeadline: isStage4
        ? "Interesting, but not ready yet."
        : "Strong momentum setup confirmed.",
      assessmentDescription: isStage4
        ? "The company looks financially healthy, but the stock is in a downtrend and needs to show stronger signs of recovery before we consider an entry."
        : "The stock is consolidating near highs with volatility contracting. Multiple technical and fundamental models agree.",
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
      reclaimMilestone: `${symbol} needs to reclaim $${sma50.toFixed(2)} (50-Day SMA) and show strong price base formation on higher volume.`,
      watchLevels: {
        watchZone: `$${(safePrice * 0.975).toFixed(2)} – $${(safePrice * 1.052).toFixed(2)}`,
        keyLevel: `$${sma50.toFixed(2)} (50D SMA)`,
        riskStop: `$${stopLoss.toFixed(2)} (-7.0%)`,
      },
      actionCallout: {
        action: isStage4 ? "WATCH" : "ENTER",
        guidance: isStage4
          ? `Watch for a strong reversal and reclaim of $${sma50.toFixed(2)} with volume. Don't rush—wait for the trigger.`
          : `Setup confirmed within the optimal buy zone. Place GTC stop at $${stopLoss.toFixed(2)}.`,
      },
    },

    // Tier 2: Explanation (Standard)
    standard: {
      bottomLine: isStage4
        ? "Mixed signal environment (1 positive, 1 warning). Take half-position sizing and honor stops tightly."
        : "Confluence confirmed across 3 of 4 core models. Standard position sizing recommended.",
      signalsRatio: isStage4 ? "1 of 4 Positive Signals" : "3 of 4 Positive Signals",
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

    // Transparent Additive Attribution Model
    scoreAttribution: {
      finalScore: setupScore,
      items: isStage4
        ? [
            { category: "Company Financial Health", impact: 20, reason: "ROIC > 15% and low leverage", sentiment: "positive" },
            { category: "Macro Regime Tailwinds", impact: 15, reason: "Broad market regime is bullish", sentiment: "positive" },
            { category: "Smart Money & Insiders", impact: 5, reason: "Neutral 13F institutional flow", sentiment: "neutral" },
            { category: "Price Structure & Moving Averages", impact: -25, reason: "Price below declining 50-day SMA", sentiment: "negative" },
            { category: "Momentum & Relative Strength", impact: -10, reason: "RVOL below 1.0 on down days", sentiment: "negative" },
          ]
        : [
            { category: "Company Financial Health", impact: 25, reason: "Top decile profitability & margin expansion", sentiment: "positive" },
            { category: "Macro Regime Tailwinds", impact: 20, reason: "Sector momentum leader", sentiment: "positive" },
            { category: "VCP Volatility Contraction", impact: 25, reason: "3-stage base contraction near 52W high", sentiment: "positive" },
            { category: "Smart Money Accumulation", impact: 15, reason: "Net institutional net buy over 3 quarters", sentiment: "positive" },
            { category: "Overbought / Beta Caution", impact: -5, reason: "RSI approaching 65+", sentiment: "neutral" },
          ],
      catalystToIncreaseScore: `Reclaiming $${sma50.toFixed(2)} (50-Day SMA) on above-average volume (+30% RVOL) would immediately upgrade the score to 80+.`,
    },
  };
}
