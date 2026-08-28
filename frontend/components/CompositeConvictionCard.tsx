"use client";

import { useMemo, useState, useEffect } from "react";
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
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };

    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  // Compute 4-Factor Institutional Synthesis Score (0 to 100)
  const synthesis = useMemo(() => {
    let score = 50;
    const reasons: {
      label: string;
      plainLabel: string;
      detail: string;
      plainDetail: string;
      status: "positive" | "neutral" | "warning";
      icon: string;
    }[] = [];

    // 1. Technical Price Momentum & Volatility
    const exec = data?.optimalExecution;
    const currentPrice = data?.currentPrice || exec?.current_price || 0;
    const stopLoss = exec?.stop_loss || 0;
    const pattern = exec?.setup_pattern || "";

    if (pattern.includes("Correction") || pattern.includes("Stage 4") || exec?.stage_phase?.includes("Stage 4")) {
      score -= 10;
      reasons.push({
        label: "Technical Setup",
        plainLabel: "Chart Structure",
        detail: "Stage 4 Correction / Markdown Phase. Require constructive base consolidation above support before initiating entries.",
        plainDetail: "Price is falling knife mode. Wait for it to build a floor before putting money in.",
        status: "warning",
        icon: "⚠️",
      });
    } else if (pattern.includes("Breakout") || pattern.includes("Stage 2") || pattern.includes("Pullback") || pattern.includes("VCP")) {
      score += 15;
      reasons.push({
        label: "Technical Setup",
        plainLabel: "Chart Structure",
        detail: `${exec?.stage_phase || "Institutional Accumulation"} consolidating near key support levels.`,
        plainDetail: "Coiled spring setup. Buyers are stepping in at key moving averages with tightening volume.",
        status: "positive",
        icon: "📈",
      });
    } else {
      score += 5;
      reasons.push({
        label: "Technical Setup",
        plainLabel: "Chart Structure",
        detail: "Neutral momentum consolidation range. Awaiting directional breakout.",
        plainDetail: "Trading in a sideways tunnel. No clear breakout yet — let the market pick a direction.",
        status: "neutral",
        icon: "📊",
      });
    }

    // 2. Corporate Insiders (SEC Form 4)
    if (matchedInsider && matchedInsider.isSignificantBuy) {
      score += 18;
      reasons.push({
        label: "Corporate Insiders",
        plainLabel: "Insider Buying",
        detail: `${matchedInsider.insiderName} (${matchedInsider.insiderRole}) purchased $${(matchedInsider.totalValueUsd / 1000000).toFixed(1)}M USD on open market.`,
        plainDetail: `Company insiders put real skin in the game: $${(matchedInsider.totalValueUsd / 1000000).toFixed(1)}M bought with their own money.`,
        status: "positive",
        icon: "🏢",
      });
    } else {
      reasons.push({
        label: "Corporate Insiders",
        plainLabel: "Insider Buying",
        detail: "No high-conviction C-Suite open-market purchases filed on SEC EDGAR in last 30 days.",
        plainDetail: "No recent big boss insider purchases filed with the SEC this month.",
        status: "neutral",
        icon: "🏢",
      });
    }

    // 3. Macroeconomic Regime (FRED)
    if (macro && macro.macroRiskMultiplier >= 1.0) {
      score += 12;
      reasons.push({
        label: "Macro Regime",
        plainLabel: "Market Tailwinds",
        detail: `FRED 10Y-2Y Yield Curve is positive (+${macro.yieldCurve10Y2Y.toFixed(2)}%) and high-yield spreads are tight (${macro.highYieldCreditSpread.toFixed(2)}%). Supportive liquidity.`,
        plainDetail: "Interest rate curves and credit markets look healthy. Green light for broader market liquidity.",
        status: "positive",
        icon: "🏛️",
      });
    } else {
      score -= 8;
      reasons.push({
        label: "Macro Regime",
        plainLabel: "Market Tailwinds",
        detail: "Yield curve inversion or elevated credit spreads warrant defensive position sizing.",
        plainDetail: "Macro warning flags waving. Keep position sizes reasonable and don't over-leverage.",
        status: "warning",
        icon: "⚠️",
      });
    }

    // 4. Downside Protection Floor
    if (stopLoss > 0 && currentPrice > 0) {
      const riskPct = Math.abs(((currentPrice - stopLoss) / currentPrice) * 100);
      reasons.push({
        label: "Risk Boundary",
        plainLabel: "Safety Trapdoor",
        detail: `Stop-loss protection anchored at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% max risk budget).`,
        plainDetail: `Exit rule clearly defined at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% risk). If it breaks this floor, we fold immediately.`,
        status: "positive",
        icon: "🛡️",
      });
    }

    const finalScore = Math.min(96, Math.max(25, score));
    let verdictTitle = isPlain ? "💡 SOLID ACCUMULATION SETUP" : "MODERATE ACCUMULATION CONVICTION";
    let verdictColor = "text-cyan-400";
    let bottomLineText = `Solid core business with orderly price action. Don't chase green candles; accumulate near support at $${(currentPrice * 0.98).toFixed(2)} with a disciplined stop loss.`;

    if (finalScore >= 80) {
      verdictTitle = isPlain ? "🟢 GREEN LIGHT: HIGH CONVICTION" : "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT";
      verdictColor = "text-emerald-400";
      bottomLineText = `Smart money, solid technicals, and macro tailwinds are aligned. Favorable risk-to-reward ratio for initiating or compounding positions.`;
    } else if (finalScore < 50) {
      verdictTitle = isPlain ? "🔴 RED LIGHT: WAIT FOR DUST TO SETTLE" : "DEFENSIVE / RISK-OFF SIZING REQUIRED";
      verdictColor = "text-rose-400";
      bottomLineText = `Too much technical turbulence or lack of insider backing. Sit on your hands or size down significantly until a clear base forms.`;
    }

    return {
      score: finalScore,
      verdictTitle,
      verdictColor,
      bottomLineText,
      reasons,
    };
  }, [data, macro, matchedInsider, isPlain]);

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-2xl p-4 sm:p-5 shadow-2xl space-y-4 font-sans">
      {/* Top Banner: Composite Score & Actionable Consensus */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
            <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400 font-mono">
              {isPlain ? `No-BS Summary • ${cleanSym}` : `Multi-Source Synthesis • ${cleanSym}`}
            </span>
          </div>
          <h2 className={`text-base sm:text-lg font-black tracking-tight ${synthesis.verdictColor}`}>
            {synthesis.verdictTitle}
          </h2>
        </div>

        {/* Big Circular/Pill Score Gauge */}
        <div className="flex items-center space-x-3 bg-[#090d14] px-3.5 py-2 rounded-xl border border-[#243044]">
          <div className="text-right">
            <span className="text-[9px] uppercase font-bold text-slate-400 block font-mono">
              {isPlain ? "Setup Score" : "Synthesis Score"}
            </span>
            <span className="text-xs text-slate-300 font-medium">
              {isPlain ? "4-Signal Check" : "4-Feed Confluence"}
            </span>
          </div>
          <div className={`text-2xl sm:text-3xl font-black font-mono tabular-nums ${synthesis.verdictColor}`}>
            {synthesis.score}<span className="text-xs text-slate-500 font-normal">/100</span>
          </div>
        </div>
      </div>

      {/* 💡 The Bottom Line Callout Box */}
      <div className="bg-[#090d14] border-l-4 border-cyan-500 p-3 sm:p-3.5 rounded-r-xl border-t border-b border-r border-[#243044] flex items-start gap-2.5">
        <span className="text-base sm:text-lg shrink-0 select-none">🎯</span>
        <div className="space-y-0.5 min-w-0">
          <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider block font-mono">
            The Bottom Line (No Wall Street Fluff)
          </span>
          <p className="text-xs sm:text-sm text-slate-200 leading-relaxed">
            {synthesis.bottomLineText}
          </p>
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
                <strong className="text-xs font-bold text-slate-200">
                  {isPlain ? r.plainLabel : r.label}
                </strong>
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
              {isPlain ? r.plainDetail : r.detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}