"use client";

import { useState, useEffect } from "react";
import { PortfolioPosition } from "../lib/portfolio";
import { trackMacroShockSimulation } from "../lib/matomo";

interface MacroStressTestSimulatorProps {
  positions: PortfolioPosition[];
  totalEquity: number;
}

export default function MacroStressTestSimulator({
  positions,
  totalEquity,
}: MacroStressTestSimulatorProps) {
  const [selectedPreset, setSelectedPreset] = useState<"TECH_SELLOFF" | "YIELD_SURGE" | "VIX_SHOCK" | "CUSTOM">("TECH_SELLOFF");
  const [customShockPct, setCustomShockPct] = useState<number>(-5.0);
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

  // Compute portfolio average beta (defaulting to 1.15)
  const portfolioBeta = positions.length > 0 ? 1.25 : 1.0;

  let shockMagnitudePct = customShockPct;
  let shockTitle = "Custom Market Scenario";
  let shockDescription = "Simulated custom market shock on total portfolio equity.";

  if (selectedPreset === "TECH_SELLOFF") {
    shockMagnitudePct = -5.0 * portfolioBeta;
    shockTitle = "Tech Sector Pullback (-5.0% QQQ)";
    shockDescription = "Simulates broad high-beta technology sector contraction with covariance contagion.";
  } else if (selectedPreset === "YIELD_SURGE") {
    shockMagnitudePct = -3.8;
    shockTitle = "10-Yr Treasury Yield Surge (+50 bps)";
    shockDescription = "Simulates interest rate shock compressing valuation multiples on growth assets.";
  } else if (selectedPreset === "VIX_SHOCK") {
    shockMagnitudePct = -8.5;
    shockTitle = "Black-Swan Volatility Spike (VIX -> 35)";
    shockDescription = "Simulates non-normal Cornish-Fisher fat-tail liquidation across institutional equities.";
  }

  useEffect(() => {
    trackMacroShockSimulation(shockTitle, shockMagnitudePct);
  }, [selectedPreset, shockMagnitudePct]);

  const effectiveEquity = totalEquity > 0 ? totalEquity : 25000;
  const dollarImpact = (effectiveEquity * (shockMagnitudePct / 100));
  const postShockEquity = Math.max(0, effectiveEquity + dollarImpact);
  const recommendedCashPct = shockMagnitudePct < -5 ? 25 : shockMagnitudePct < 0 ? 15 : 5;
  const recommendedCashDollar = (effectiveEquity * (recommendedCashPct / 100));

  return (
    <div className="bg-[#0b1019] border border-[#243044] rounded-2xl p-5 sm:p-6 shadow-xl space-y-4 font-sans text-slate-200">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xl">🌪️</span>
            <h2 className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <span>{isPlain ? "Crash Test & What-If Stress Simulator" : "Macro Shock Engine & Scenario Stress-Tester"}</span>
            </h2>
            <span className="text-[10px] bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded font-mono font-bold">
              BETA COVARIANCE SIM
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            {isPlain
              ? "See how bad market drops or interest rate hikes will impact your account balance, and how much cash to keep safe."
              : "Stress-test portfolio equity under simulated macroeconomic shocks and derive defensive liquidity cushions."}
          </p>
        </div>

        {/* Preset Selector */}
        <div className="flex flex-wrap items-center gap-1.5 bg-[#090d14] p-1 rounded-xl border border-[#1e293b] text-xs font-bold font-mono">
          <button
            type="button"
            onClick={() => setSelectedPreset("TECH_SELLOFF")}
            className={`px-2.5 py-1 rounded-lg transition ${
              selectedPreset === "TECH_SELLOFF" ? "bg-cyan-500 text-slate-950 shadow" : "text-slate-400 hover:text-white"
            }`}
          >
            📉 QQQ -5%
          </button>
          <button
            type="button"
            onClick={() => setSelectedPreset("YIELD_SURGE")}
            className={`px-2.5 py-1 rounded-lg transition ${
              selectedPreset === "YIELD_SURGE" ? "bg-amber-500 text-slate-950 shadow font-extrabold" : "text-slate-400 hover:text-amber-300"
            }`}
          >
            📈 +50bps Yield
          </button>
          <button
            type="button"
            onClick={() => setSelectedPreset("VIX_SHOCK")}
            className={`px-2.5 py-1 rounded-lg transition ${
              selectedPreset === "VIX_SHOCK" ? "bg-rose-500 text-slate-950 shadow font-extrabold" : "text-slate-400 hover:text-rose-300"
            }`}
          >
            💥 VIX 35
          </button>
          <button
            type="button"
            onClick={() => setSelectedPreset("CUSTOM")}
            className={`px-2.5 py-1 rounded-lg transition ${
              selectedPreset === "CUSTOM" ? "bg-purple-500 text-slate-950 shadow font-extrabold" : "text-slate-400 hover:text-purple-300"
            }`}
          >
            🎛️ Custom
          </button>
        </div>
      </div>

      {/* Custom Slider if selected */}
      {selectedPreset === "CUSTOM" && (
        <div className="p-3.5 rounded-xl bg-[#111722] border border-[#1e293b] space-y-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-300 font-bold">{isPlain ? "Simulated Market Move:" : "Simulated Benchmark Shock:"}</span>
            <span className={`text-sm font-black ${customShockPct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {customShockPct >= 0 ? "+" : ""}{customShockPct.toFixed(1)}%
            </span>
          </div>
          <input
            type="range"
            min="-20"
            max="15"
            step="0.5"
            value={customShockPct}
            onChange={(e) => setCustomShockPct(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
            <span>-20% (Crash)</span>
            <span>0% (Neutral)</span>
            <span>+15% (Rally)</span>
          </div>
        </div>
      )}

      {/* 3-Card Projected Scenario Impact Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
        <div className="bg-[#111722] p-3.5 rounded-xl border border-[#1e293b] space-y-1">
          <span className="text-[10px] text-slate-400 block uppercase font-bold">
            {isPlain ? "Expected Balance After Shock" : "Projected Stressed Equity"}
          </span>
          <strong className="text-lg font-black text-white tabular-nums">
            ${postShockEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </strong>
          <span className="text-[11px] text-slate-400 block font-sans">
            Base: ${effectiveEquity.toLocaleString()}
          </span>
        </div>

        <div className="bg-[#111722] p-3.5 rounded-xl border border-[#1e293b] space-y-1">
          <span className="text-[10px] text-slate-400 block uppercase font-bold">
            {isPlain ? "Account Gain / Drop" : "Simulated PnL Variance"}
          </span>
          <strong className={`text-lg font-black tabular-nums ${dollarImpact >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {dollarImpact >= 0 ? "+" : ""}${Math.abs(dollarImpact).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{" "}
            <span className="text-xs">({shockMagnitudePct.toFixed(1)}%)</span>
          </strong>
          <span className="text-[11px] text-slate-400 block font-sans truncate" title={shockTitle}>
            {shockTitle}
          </span>
        </div>

        <div className="bg-[#111722] p-3.5 rounded-xl border border-cyan-800/50 space-y-1">
          <span className="text-[10px] text-cyan-400 block uppercase font-bold">
            {isPlain ? "Recommended Cash Safety Cushion" : "Defensive Liquidity Recommendation"}
          </span>
          <strong className="text-lg font-black text-cyan-300 tabular-nums">
            {recommendedCashPct}% (${recommendedCashDollar.toLocaleString()})
          </strong>
          <span className="text-[11px] text-slate-400 block font-sans">
            {isPlain ? "Keep in cash to buy dips" : "Dry powder reserve for volatility floor"}
          </span>
        </div>
      </div>

      {/* Scenario Context Explanation */}
      <div className="p-3 rounded-lg bg-[#090d14] border border-[#1e293b] text-xs flex items-start gap-2.5">
        <span className="text-cyan-400 shrink-0 font-bold">💡 {isPlain ? "Scenario Assessment:" : "Macro Model Insight:"}</span>
        <span className="text-slate-300 leading-relaxed font-sans">{shockDescription}</span>
      </div>
    </div>
  );
}
