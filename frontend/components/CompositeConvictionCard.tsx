"use client";

import { useMemo } from "react";
import { AnalyticsResponse } from "../lib/api";
import { FredMacroData, SecForm4Trade } from "../lib/institutionalFeeds";

interface CompositeConvictionCardProps {
  symbol: string;
  data: AnalyticsResponse | null;
  macro: FredMacroData | null;
  insiders: SecForm4Trade[];
  userRole: "DAY_TRADER" | "LONG_TERM";
}

export default function CompositeConvictionCard({
  symbol,
  data,
  macro,
  insiders,
  userRole,
}: CompositeConvictionCardProps) {
  const cleanSym = symbol.toUpperCase().replace("-USD", "");
  const matchedInsider = insiders.find((i) => i.ticker === cleanSym);

  // Compute 4-Factor Institutional Synthesis Score (0 to 100)
  const synthesis = useMemo(() => {
    let score = 50;
    const reasons: { label: string; detail: string; status: "positive" | "neutral" | "warning"; icon: string }[] = [];

    // 1. Technical Price Momentum & Volatility
    const exec = data?.optimalExecution;
    const currentPrice = data?.currentPrice || exec?.current_price || 0;
    const stopLoss = exec?.stop_loss || 0;
    const pattern = exec?.setup_pattern || "";

    if (pattern.includes("Correction") || pattern.includes("Stage 4") || exec?.stage_phase?.includes("Stage 4")) {
      score -= 10;
      reasons.push({
        label: "Technical Setup",
        detail: "Stage 4 Correction / Markdown Phase. Require constructive base consolidation above support before initiating entries.",
        status: "warning",
        icon: "⚠️",
      });
    } else if (pattern.includes("Breakout") || pattern.includes("Stage 2") || pattern.includes("Pullback") || pattern.includes("VCP")) {
      score += 15;
      reasons.push({
        label: "Technical Setup",
        detail: `${exec?.stage_phase || "Institutional Accumulation"} consolidating near key support levels.`,
        status: "positive",
        icon: "📈",
      });
    } else {
      score += 5;
      reasons.push({
        label: "Technical Setup",
        detail: "Neutral momentum consolidation range. Awaiting directional breakout.",
        status: "neutral",
        icon: "📊",
      });
    }

    // 2. Corporate Insiders (SEC Form 4)
    if (matchedInsider && matchedInsider.isSignificantBuy) {
      score += 18;
      reasons.push({
        label: "Corporate Insiders",
        detail: `${matchedInsider.insiderName} (${matchedInsider.insiderRole}) purchased $${(matchedInsider.totalValueUsd / 1000000).toFixed(1)}M USD on open market.`,
        status: "positive",
        icon: "🏢",
      });
    } else {
      reasons.push({
        label: "Corporate Insiders",
        detail: "No high-conviction C-Suite open-market purchases filed on SEC EDGAR in last 30 days.",
        status: "neutral",
        icon: "🏢",
      });
    }

    // 3. Macroeconomic Regime (FRED)
    if (macro && macro.macroRiskMultiplier >= 1.0) {
      score += 12;
      reasons.push({
        label: "Macro Regime",
        detail: `FRED 10Y-2Y Yield Curve is positive (+${macro.yieldCurve10Y2Y.toFixed(2)}%) and high-yield spreads are tight (${macro.highYieldCreditSpread.toFixed(2)}%). Supportive liquidity.`,
        status: "positive",
        icon: "🏛️",
      });
    } else {
      score -= 8;
      reasons.push({
        label: "Macro Regime",
        detail: "Yield curve inversion or elevated credit spreads warrant defensive position sizing.",
        status: "warning",
        icon: "⚠️",
      });
    }

    // 4. Downside Protection Floor
    if (stopLoss > 0 && currentPrice > 0) {
      const riskPct = Math.abs(((currentPrice - stopLoss) / currentPrice) * 100);
      reasons.push({
        label: "Risk Boundary",
        detail: `Stop-loss protection anchored at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% max risk budget).`,
        status: "positive",
        icon: "🛡️",
      });
    }

    const finalScore = Math.min(96, Math.max(25, score));
    let verdictTitle = "MODERATE ACCUMULATION CONVICTION";
    let verdictColor = "text-cyan-400";

    if (finalScore >= 80) {
      verdictTitle = "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT";
      verdictColor = "text-emerald-400";
    } else if (finalScore < 50) {
      verdictTitle = "DEFENSIVE / RISK-OFF SIZING REQUIRED";
      verdictColor = "text-rose-400";
    }

    return {
      score: finalScore,
      verdictTitle,
      verdictColor,
      reasons,
    };
  }, [data, macro, matchedInsider]);

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-2xl p-4 sm:p-5 shadow-2xl space-y-4 font-mono">
      {/* Top Banner: Composite Score & Actionable Consensus */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
            <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400">
              Multi-Source Synthesis • {cleanSym}
            </span>
          </div>
          <h2 className={`text-base sm:text-lg font-black tracking-tight ${synthesis.verdictColor}`}>
            {synthesis.verdictTitle}
          </h2>
        </div>

        {/* Big Circular/Pill Score Gauge */}
        <div className="flex items-center space-x-3 bg-[#090d14] px-3.5 py-2 rounded-xl border border-[#243044]">
          <div className="text-right">
            <span className="text-[9px] uppercase font-bold text-slate-400 block">Synthesis Score</span>
            <span className="text-xs text-slate-300 font-medium">4-Feed Confluence</span>
          </div>
          <div className={`text-2xl sm:text-3xl font-black tabular-nums ${synthesis.verdictColor}`}>
            {synthesis.score}<span className="text-xs text-slate-500 font-normal">/100</span>
          </div>
        </div>
      </div>

      {/* 4 Multi-Source Evidence Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {synthesis.reasons.map((r, idx) => (
          <div
            key={idx}
            className="bg-[#090d14] p-3 rounded-xl border border-[#1e293b] space-y-1.5 flex flex-col justify-between"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-1.5">
                <span className="text-sm">{r.icon}</span>
                <strong className="text-xs font-bold text-slate-200">{r.label}</strong>
              </div>
              <span
                className={`w-2 h-2 rounded-full ${
                  r.status === "positive"
                    ? "bg-emerald-400"
                    : r.status === "warning"
                    ? "bg-rose-400"
                    : "bg-cyan-400"
                }`}
              ></span>
            </div>
            <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
              {r.detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}