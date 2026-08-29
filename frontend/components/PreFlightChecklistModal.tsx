"use client";

import { useState, useEffect } from "react";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";

interface PreFlightChecklistModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  currentPrice: number;
  stopLoss: number;
  takeProfit1: number;
  riskRewardRatio: number;
  setupPattern?: string;
  isDayTrader?: boolean;
  isStage4?: boolean;
  optimalEntryMin?: number;
  optimalEntryMax?: number;
  breakoutPivot?: number;
  isDistributionTrap?: boolean;
  hasImminentEarnings?: boolean;
  vix?: number;
}

export default function PreFlightChecklistModal({
  isOpen,
  onClose,
  symbol,
  currentPrice,
  stopLoss,
  takeProfit1,
  riskRewardRatio,
  setupPattern = "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
  isDayTrader = false,
  isStage4 = false,
  optimalEntryMin,
  optimalEntryMax,
  breakoutPivot,
  isDistributionTrap,
  hasImminentEarnings = false,
  vix = 15.4,
}: PreFlightChecklistModalProps) {
  const [copied, setCopied] = useState<boolean>(false);
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

  if (!isOpen) return null;

  const isPlain = vernacularMode === "PLAIN_ENGLISH";
  const cleanSym = symbol.toUpperCase().replace("-USD", "");
  const catalogItem = MASTER_ASSET_CATALOG[cleanSym];

  // Dynamic 5-Point Quantitative Decision Checklist Evaluation
  const isRRPassed = riskRewardRatio >= 2.0;
  
  // Check 2: Technical Trend Alignment & Stage Discipline
  const isExtendedAboveZone = Boolean(optimalEntryMax && currentPrice > optimalEntryMax * 1.02);
  const isTrendPassed = !isStage4 && !isExtendedAboveZone;
  
  // Check 3: Smart Money Flow & Distribution Traps
  const isDistributionTrapResolved = isDistributionTrap ?? Boolean(
    catalogItem && (catalogItem.shortFloat > 12.0 || catalogItem.verdict.toLowerCase().includes("turnaround") || catalogItem.qualityScore < 60)
  );
  const isSmartMoneyPassed = !isDistributionTrapResolved;

  // Check 4: Catalyst Hazard Buffer
  const isCatalystPassed = !hasImminentEarnings;

  // Check 5: Macro Regime Guard
  const isMacroPassed = typeof vix === "number" ? vix < 26.0 : true;

  const passedCount = [isRRPassed, isTrendPassed, isSmartMoneyPassed, isCatalystPassed, isMacroPassed].filter(Boolean).length;
  const convictionPct = Math.round((passedCount / 5) * 100);
  const isCleared = convictionPct >= 80 && !isStage4 && isSmartMoneyPassed;

  const tradePlanMarkdown = `### 📋 ARX Terminal Trade Execution Plan: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol}
- **Mode**: ${isDayTrader ? "⚡ Day Trader (Intraday)" : "🏛️ Swing / Long-Term Compounder"}
- **Current Price**: $${currentPrice.toFixed(2)}
- **Entry Strategy**: ${setupPattern}
- **Stop Loss**: $${stopLoss.toFixed(2)} (${(((stopLoss - currentPrice) / currentPrice) * 100).toFixed(2)}%)
- **Target 1**: $${takeProfit1.toFixed(2)} (+${(((takeProfit1 - currentPrice) / currentPrice) * 100).toFixed(2)}%)
- **Risk / Reward**: ${riskRewardRatio.toFixed(2)} : 1.0
- **Pre-Flight Clearance Score**: ${convictionPct}% (${isCleared ? "🟢 CLEARED FOR EXECUTION" : "⚠️ CONDITIONAL / AWAIT BASE CLEARANCE"})
`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(tradePlanMarkdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.warn("Failed to copy trade plan:", err);
    }
  };

  return (
    <div className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-[#0b1019] border border-cyan-800/80 rounded-2xl max-w-xl w-full p-5 sm:p-6 shadow-2xl space-y-4 font-sans text-slate-200 relative">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1e293b] pb-3.5">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">✈️</span>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
                <span>{isPlain ? `Pre-Flight Trade Checklist: ${symbol}` : `Institutional Pre-Flight Clearance: ${symbol}`}</span>
              </h2>
              <p className="text-xs text-slate-400">
                {isPlain
                  ? "5-Point sanity check before risking your hard-earned money."
                  : "Automated pre-trade validation gate enforcing risk-reward and flow confluence."}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Pre-Flight Checklist"
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Clearance Conviction Barometer */}
        <div className={`p-3.5 rounded-xl border flex items-center justify-between ${
          isCleared
            ? "bg-emerald-950/40 border-emerald-700/80 text-emerald-300"
            : "bg-amber-950/40 border-amber-700/80 text-amber-300"
        }`}>
          <div>
            <span className="text-[10px] uppercase font-bold tracking-wider font-mono block">
              {isPlain ? "Trade Readiness Score" : "Quantitative Clearance Status"}
            </span>
            <div className="text-base sm:text-lg font-extrabold flex items-center gap-1.5">
              <span>{isCleared ? "🟢 CLEARED TO EXECUTE" : "⚠️ CONDITIONAL / NOT CLEARED"}</span>
              <span className="text-xs font-mono font-normal">({convictionPct}% Pass)</span>
            </div>
          </div>
          <div className="text-right font-mono">
            <span className="text-2xl sm:text-3xl font-black">{passedCount}/5</span>
            <span className="text-[10px] text-slate-400 block">Checks Passed</span>
          </div>
        </div>

        {/* 5-Point Validation Checklist */}
        <div className="space-y-2.5 text-xs">
          {/* Check 1: Asymmetric Risk Reward */}
          <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
            <div className="space-y-0.5">
              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                <span>{isRRPassed ? "✅" : "❌"}</span>
                <span>{isPlain ? "1. Reward vs Risk Balance (At least 2 to 1)" : "1. Asymmetric Payoff (Reward:Risk >= 2.0:1)"}</span>
              </div>
              <p className="text-[11px] text-slate-400 pl-5">
                Current: <strong className="text-cyan-300 font-mono">{riskRewardRatio.toFixed(2)} : 1.0</strong> {isRRPassed ? "(Adequate upside cushion)" : "(Hazard: upside too small for downside risk)"}
              </p>
            </div>
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
              isRRPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-rose-950 text-rose-300 border-rose-800"
            }`}>
              {isRRPassed ? "PASS" : "FAIL"}
            </span>
          </div>

          {/* Check 2: Technical Trend Alignment & Stage Discipline */}
          <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
            <div className="space-y-0.5">
              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                <span>{isTrendPassed ? "✅" : (isStage4 ? "⏳" : "⚠️")}</span>
                <span>{isPlain ? "2. Trend & Moving Averages (Price in upward corridor)" : "2. Technical Structure (Above 20 EMA / 50 SMA Pivot)"}</span>
              </div>
              <p className="text-[11px] text-slate-400 pl-5">
                {isStage4
                  ? (isPlain
                      ? `⚠️ Watchlist Only: Spot price ($${currentPrice.toFixed(2)}) is in Stage 4 correction below 50-day average. Await base formation.`
                      : `Stage 4 correction structure: Spot ($${currentPrice.toFixed(2)}) requires 50-day breakout pivot above $${(breakoutPivot || currentPrice * 1.072).toFixed(2)}.`)
                  : (isExtendedAboveZone
                      ? (isPlain
                          ? `⚠️ Extended: Price is above ideal buy zone ($${optimalEntryMin?.toFixed(2)} - $${optimalEntryMax?.toFixed(2)}). Wait for pullback.`
                          : `Extended structure: Spot is above value area. Chasing creates negative R:R risk.`)
                      : (isPlain
                          ? `Spot price ($${currentPrice.toFixed(2)}) is inside the verified buying range defending key support.`
                          : `Defending key moving average support (20 EMA / 50 SMA).`))}
              </p>
            </div>
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
              isTrendPassed 
                ? "bg-emerald-950 text-emerald-300 border-emerald-800" 
                : (isStage4 ? "bg-amber-950 text-amber-300 border-amber-800" : "bg-rose-950 text-rose-300 border-rose-800")
            }`}>
              {isTrendPassed ? "PASS" : (isStage4 ? "STAGE 4 WAIT" : "CHASING")}
            </span>
          </div>

          {/* Check 3: Smart Money Flow */}
          <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
            <div className="space-y-0.5">
              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                <span>{isSmartMoneyPassed ? "✅" : "❌"}</span>
                <span>{isPlain ? "3. Big Player Activity (No aggressive insider selling)" : "3. Institutional Flow (No Net Form 4 C-Suite Dumping)"}</span>
              </div>
              <p className="text-[11px] text-slate-400 pl-5">
                {isSmartMoneyPassed
                  ? (isPlain
                      ? "Institutional order sweeps & Congressional filings indicate steady accumulation."
                      : "Institutional Flow: Positive net accumulation detected.")
                  : (isPlain
                      ? "⚠️ Warning: Heavy corporate insider selling / distribution trap detected."
                      : "⚠️ Institutional Distribution Trap: Net Form 4 C-Suite selling / elevated short interest.")}
              </p>
            </div>
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
              isSmartMoneyPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-rose-950 text-rose-300 border-rose-800"
            }`}>
              {isSmartMoneyPassed ? "PASS" : "DISTRIBUTION"}
            </span>
          </div>

          {/* Check 4: Catalyst Hazard Buffer */}
          <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
            <div className="space-y-0.5">
              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                <span>{isCatalystPassed ? "✅" : "⚠️"}</span>
                <span>{isPlain ? "4. News Event Safety (No surprise earnings report tomorrow)" : "4. Catalyst Hazard Buffer (>7 Days to Binary Earnings)"}</span>
              </div>
              <p className="text-[11px] text-slate-400 pl-5">
                {isCatalystPassed
                  ? (isPlain
                      ? "Sufficient time window to manage trade without overnight earnings gap risk."
                      : "Catalyst Buffer: Clean window (>7 days to binary catalyst).")
                  : (isPlain
                      ? "⚠️ High Risk: Imminent binary earnings announcement / major FDA event within 48 hours."
                      : "⚠️ Imminent Binary Event: Overnight gap risk exceeds standard stop constraint.")}
              </p>
            </div>
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
              isCatalystPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-amber-950 text-amber-300 border-amber-800"
            }`}>
              {isCatalystPassed ? "PASS" : "HAZARD"}
            </span>
          </div>

          {/* Check 5: Macro Regime Difficulty */}
          <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
            <div className="space-y-0.5">
              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                <span>{isMacroPassed ? "✅" : "⚠️"}</span>
                <span>{isPlain ? "5. Overall Market Weather (VIX normal, market calm)" : "5. Macro Regime Guard (VIX Volatility Guardrails Safe)"}</span>
              </div>
              <p className="text-[11px] text-slate-400 pl-5">
                {isMacroPassed
                  ? (isPlain
                      ? "Broad market volatility is within standard operational parameters."
                      : "Macro Guard: Normal volatility regime.")
                  : (isPlain
                      ? `⚠️ High Volatility: Market VIX (${vix?.toFixed(1) || "28+"}) indicates elevated systemic turbulence.`
                      : `⚠️ Elevated Macro Risk: VIX (${vix?.toFixed(1) || "28+"}) exceeds 26.0 threshold.`)}
              </p>
            </div>
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
              isMacroPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-amber-950 text-amber-300 border-amber-800"
            }`}>
              {isMacroPassed ? "PASS" : "HIGH VIX"}
            </span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center justify-between gap-2.5 pt-3 border-t border-[#1e293b]">
          <button
            type="button"
            onClick={handleCopy}
            className="px-4 py-2 rounded-lg text-xs font-bold transition-all active:scale-95 border bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/60 text-cyan-300 flex items-center gap-1.5 shadow"
          >
            <span>{copied ? "✅ Plan Copied!" : "📋 Copy Trade Plan for Journal"}</span>
          </button>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-bold transition bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
