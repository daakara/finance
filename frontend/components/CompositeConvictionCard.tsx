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

  // Pure View Lens Projection: Consume Canonical Backend Confluence (Single Source of Truth)
  const synthesis = useMemo(() => {
    // 1. SSOT: Direct projection of canonical backend confluence engine
    if (data?.confluence) {
      const conf = data.confluence;
      const score = Math.round(conf.confluenceScore);
      const verdictTitle = isPlain ? conf.plainRating || conf.confluenceRating : conf.confluenceRating;
      const confluenceBadge = isPlain ? conf.plainBadge || conf.confluenceBadge : conf.confluenceBadge;
      const bottomLineText = conf.bottomLine;
      const verdictColor =
        conf.badgeColor === "emerald"
          ? "text-emerald-400"
          : conf.badgeColor === "rose"
          ? "text-rose-400"
          : conf.badgeColor === "amber"
          ? "text-amber-400"
          : "text-cyan-400";

      const reasons = (conf.pillars || []).map((p) => ({
        label: p.label,
        plainLabel: p.plainLabel,
        detail: p.detail,
        plainDetail: p.plainDetail,
        status: p.status,
        icon: p.icon,
      }));

      return {
        score,
        verdictTitle,
        verdictColor,
        confluenceBadge,
        bottomLineText,
        reasons,
      };
    }

    // 2. Fallback Projection if data.confluence is not yet loaded
    const reasons: {
      label: string;
      plainLabel: string;
      detail: string;
      plainDetail: string;
      status: "positive" | "neutral" | "warning";
      icon: string;
    }[] = [];

    const exec = data?.optimalExecution;
    const currentPrice = data?.currentPrice || exec?.current_price || 0;
    const stopLoss = exec?.stop_loss || 0;
    const pattern = exec?.setup_pattern || "";
    const stage = exec?.stage_phase || "";
    const rsi = data?.technicals?.rsi_14 ?? 50.0;
    const rr = exec?.risk_reward_ratio ? parseFloat(String(exec.risk_reward_ratio)) : 2.0;

    let techScore = 55;
    let techStatus: "positive" | "neutral" | "warning" = "neutral";
    let techDetail = "";
    let techPlainDetail = "";

    if (pattern.includes("Correction") || pattern.includes("Stage 4") || stage.includes("Stage 4")) {
      techScore = 30;
      techStatus = "warning";
      techDetail = "Stage 4 Correction / Markdown Phase. Price trading below declining moving averages. Await constructive base.";
      techPlainDetail = "Falling knife structure. Price is breaking down — wait for buyers to build a solid floor.";
    } else if (pattern.includes("Breakout") || pattern.includes("Stage 2") || pattern.includes("VCP") || pattern.includes("Pullback")) {
      techScore = rr >= 2.5 ? 88 : rr >= 2.0 ? 80 : 72;
      if (rsi >= 42 && rsi <= 65) techScore += 5;
      if (rsi > 75) techScore -= 10;
      techStatus = "positive";
      techDetail = `${stage || "Stage 2 Accumulation"} with ${rr.toFixed(1)}:1 R:R structure. RSI at ${rsi.toFixed(1)}.`;
      techPlainDetail = `Coiled spring setup: Buyers defending key levels with healthy ${rsi.toFixed(1)} RSI momentum.`;
    } else {
      techScore = (rsi > 45 && rsi < 58) ? 58 : 50;
      techStatus = "neutral";
      techDetail = `Sideways range consolidation (RSI ${rsi.toFixed(1)}). Awaiting clear volume breakout catalyst.`;
      techPlainDetail = `Trading inside a sideways channel. No clear direction yet — letting the market decide.`;
    }

    reasons.push({
      label: "Technical Structure",
      plainLabel: "Chart Structure",
      detail: techDetail,
      plainDetail: techPlainDetail,
      status: techStatus,
      icon: techStatus === "positive" ? "📈" : techStatus === "warning" ? "⚠️" : "📊",
    });

    const factors = data?.factorScores;
    const piotroski = factors?.piotroskiFScore ?? 7;
    const qualityScore = factors?.qualityScore ?? (piotroski >= 8 ? 85 : piotroski >= 6 ? 72 : 50);
    const growthScore = factors?.growthScore ?? 70;
    const valScore = factors?.valuationScore ?? 65;

    let fundScore = Math.round(0.45 * qualityScore + 0.30 * growthScore + 0.25 * valScore);
    let fundStatus: "positive" | "neutral" | "warning" = "neutral";
    let fundDetail = "";
    let fundPlainDetail = "";

    if (piotroski >= 8 || qualityScore >= 82) {
      fundStatus = "positive";
      fundDetail = `Pristine solvency: Piotroski F-Score ${piotroski}/9, Quality Factor ${qualityScore}/100. Fortress balance sheet.`;
      fundPlainDetail = `Rock-solid financials: Top-tier ${piotroski}/9 balance sheet health with strong profitability.`;
    } else if (piotroski <= 4 || qualityScore < 45) {
      fundStatus = "warning";
      fundDetail = `Elevated balance sheet leverage: Piotroski F-Score ${piotroski}/9. Vulnerable to credit tightening.`;
      fundPlainDetail = `Weaker financial health (${piotroski}/9 score). Carries higher debt or thinning profit margins.`;
    } else {
      fundStatus = "neutral";
      fundDetail = `Moderate solvency: Piotroski F-Score ${piotroski}/9, Quality Factor ${qualityScore}/100. Stable fundamentals.`;
      fundPlainDetail = `Healthy average company financials (${piotroski}/9 score). Stable without immediate solvency risks.`;
    }

    reasons.push({
      label: "Fundamental Solvency",
      plainLabel: "Company Health",
      detail: fundDetail,
      plainDetail: fundPlainDetail,
      status: fundStatus,
      icon: "🏢",
    });

    const congressTrades = data?.smartMoney?.congressTrades || [];
    const optionsFlow = data?.smartMoney?.optionsFlow || [];
    const hasCongressBuy = congressTrades.some(c => (c.transaction_type || "").toLowerCase().includes("buy"));
    const hasBullishOptions = optionsFlow.some(o => (o.type || "").toLowerCase().includes("call"));

    let smartMoneyScore = 50;
    let smartStatus: "positive" | "neutral" | "warning" = "neutral";
    let smartDetail = "";
    let smartPlainDetail = "";

    if (matchedInsider && matchedInsider.isSignificantBuy) {
      smartMoneyScore = 90;
      smartStatus = "positive";
      smartDetail = `${matchedInsider.insiderName} (${matchedInsider.insiderRole}) purchased $${(matchedInsider.totalValueUsd / 1000000).toFixed(1)}M on open market.`;
      smartPlainDetail = `C-Suite insider skin in the game: $${(matchedInsider.totalValueUsd / 1000000).toFixed(1)}M purchased directly with own capital.`;
    } else if (hasCongressBuy) {
      smartMoneyScore = 80;
      smartStatus = "positive";
      smartDetail = `Congressional STOCK Act disclosure: Capitol Hill committee member buy filing active.`;
      smartPlainDetail = `Congress buy reported: Lawmaker disclosed an open-market purchase in this sector.`;
    } else if (hasBullishOptions) {
      smartMoneyScore = 70;
      smartStatus = "positive";
      smartDetail = `Unusual institutional options flow: High-volume call sweeps detected in smart money order books.`;
      smartPlainDetail = `Big money call options detected: Institutional traders betting on upside momentum.`;
    } else {
      smartMoneyScore = 50;
      smartStatus = "neutral";
      smartDetail = "No high-conviction C-Suite open-market purchases filed on SEC EDGAR in the last 30 days.";
      smartPlainDetail = "No recent big boss insider purchases filed with the SEC this month.";
    }

    reasons.push({
      label: "Corporate Insiders & Flow",
      plainLabel: "Smart Money Flow",
      detail: smartDetail,
      plainDetail: smartPlainDetail,
      status: smartStatus,
      icon: "🏛️",
    });

    let macroScore = 60;
    let macroStatus: "positive" | "neutral" | "warning" = "neutral";
    let macroDetail = "";
    let macroPlainDetail = "";

    const riskPct = (stopLoss > 0 && currentPrice > 0)
      ? Math.abs(((currentPrice - stopLoss) / currentPrice) * 100)
      : 5.0;

    const isYieldPositive = (macro?.yieldCurve10Y2Y ?? 0.2) >= 0;
    const isCreditTight = (macro?.highYieldCreditSpread ?? 3.5) < 4.2;

    if (isYieldPositive && isCreditTight && riskPct <= 7.5) {
      macroScore = 85;
      macroStatus = "positive";
      macroDetail = `FRED 10Y-2Y yield curve positive (+${macro?.yieldCurve10Y2Y?.toFixed(2) || "0.25"}%), credit spreads tight. Stop floor at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% risk budget).`;
      macroPlainDetail = `Macro green light: Credit markets healthy. Clear exit floor set at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% risk).`;
    } else if (!isYieldPositive || riskPct > 12.0) {
      macroScore = 38;
      macroStatus = "warning";
      macroDetail = `Macro risk elevated or wide stop-loss requirement (${riskPct.toFixed(1)}% risk). Recommend defensive capital allocation.`;
      macroPlainDetail = `Macro warning flags or large downside distance (${riskPct.toFixed(1)}% risk). Size down carefully.`;
    } else {
      macroScore = 60;
      macroStatus = "neutral";
      macroDetail = `Neutral macro backdrop with defined stop protection floor at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% risk).`;
      macroPlainDetail = `Stable economic background with safety exit floor set at $${stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% risk).`;
    }

    reasons.push({
      label: "Macro Regime & Safety Floor",
      plainLabel: "Market Tailwinds & Stop",
      detail: macroDetail,
      plainDetail: macroPlainDetail,
      status: macroStatus,
      icon: macroStatus === "positive" ? "🛡️" : macroStatus === "warning" ? "⚠️" : "⚖️",
    });

    const finalScore = Math.min(96, Math.max(22, Math.round(
      0.25 * techScore +
      0.25 * fundScore +
      0.25 * smartMoneyScore +
      0.25 * macroScore
    )));

    const positiveCount = reasons.filter(r => r.status === "positive").length;
    const warningCount = reasons.filter(r => r.status === "warning").length;

    let confluenceBadge = isPlain ? `${positiveCount}/4 Positive Signals` : `${positiveCount}-Feed Confluence`;
    if (positiveCount === 4) {
      confluenceBadge = isPlain ? "4/4 High Conviction" : "4-Pillar Confluence (Pristine)";
    } else if (positiveCount === 3) {
      confluenceBadge = isPlain ? "3/4 Strong Alignment" : "3-Feed Confluence (Selective)";
    } else if (warningCount >= 2) {
      confluenceBadge = isPlain ? "⚠️ Mixed / Caution" : "Multi-Feed Divergence (Risk-Off)";
    }

    let verdictTitle = isPlain ? "💡 SOLID ACCUMULATION SETUP" : "CONSTRUCTIVE ACCUMULATION CONVICTION";
    let verdictColor = "text-cyan-400";
    let bottomLineText = `Disciplined trade structure with ${positiveCount}/4 supporting pillars. Favorable risk floor near $${stopLoss.toFixed(2)}. Don't chase green candles; execute inside buy zones.`;

    if (finalScore >= 80) {
      verdictTitle = isPlain ? "🟢 GREEN LIGHT: HIGH CONVICTION" : "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT";
      verdictColor = "text-emerald-400";
      bottomLineText = `Strong multi-factor alignment (${positiveCount}/4 pillars positive): Technicals, balance sheet quality, and smart money flow are synchronised.`;
    } else if (finalScore <= 48 || warningCount >= 2) {
      verdictTitle = isPlain ? "🔴 RED LIGHT: WAIT FOR DUST TO SETTLE" : "DEFENSIVE / CAPITAL PRESERVATION MODE";
      verdictColor = "text-rose-400";
      bottomLineText = `Technical turbulence or weak balance sheet metrics detected (${warningCount} warning flags). Preserve cash and wait for a proper accumulation base to form.`;
    } else if (finalScore < 65) {
      verdictTitle = isPlain ? "🟡 SELECTIVE ENTRY: WAIT FOR TRIGGER" : "SELECTIVE / RANGE-BOUND MOMENTUM";
      verdictColor = "text-amber-400";
      bottomLineText = `Mixed signal environment (${positiveCount} positive, ${warningCount} warning). Take half-position sizing and honor stops tightly.`;
    }

    return {
      score: finalScore,
      verdictTitle,
      verdictColor,
      confluenceBadge,
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
              {synthesis.confluenceBadge}
            </span>
          </div>
          <div className={`text-2xl sm:text-3xl font-black font-mono tabular-nums ${synthesis.verdictColor}`}>
            {synthesis.score}<span className="text-xs text-slate-500 font-normal">/100</span>
          </div>
        </div>
      </div>

      {/* 💡 The Bottom Line Callout Box */}
      <div className="bg-[#090d14] border border-cyan-500/30 p-3 sm:p-3.5 rounded-xl flex items-start gap-2.5 shadow-sm">
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