"use client";

import { useState, useEffect } from "react";
import { AssetFactorScores, MacroDifficultyRating, ExpectedReturnForecast } from "../lib/api";
import { SHARED_FACTOR_SCORES, DEFAULT_MACRO_DIFFICULTY, DEFAULT_EXPECTED_RETURN } from "../lib/constants";

interface AssetFactorRadarProps {
  symbol: string;
  factorScores?: AssetFactorScores;
  macroDifficulty?: MacroDifficultyRating;
  expectedReturn?: ExpectedReturnForecast;
}

export default function AssetFactorRadar({ symbol, factorScores, macroDifficulty, expectedReturn }: AssetFactorRadarProps) {
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

  const defaultObj = SHARED_FACTOR_SCORES[symbol?.toUpperCase()?.replace("-USD", "")] || SHARED_FACTOR_SCORES["AAPL"];
  const scores = factorScores || defaultObj.scores;
  const mdr = macroDifficulty || DEFAULT_MACRO_DIFFICULTY;
  const er = expectedReturn || DEFAULT_EXPECTED_RETURN;

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono">
      {/* Header with Financial Health Score */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <svg className="w-4 h-4 text-cyan-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
              <span>{isPlain ? `${symbol} Business DNA & BS Detector` : `${symbol} Fundamental & Factor Profile`}</span>
            </h3>
            {scores.piotroskiFScore && (
              <span className="text-[10px] bg-[#1b2434] text-cyan-300 border border-cyan-800/60 px-2 py-0.5 rounded">
                {isPlain ? `BS Detector: ${scores.piotroskiFScore}/9` : `Piotroski: ${scores.piotroskiFScore}/9`}
              </span>
            )}
          </div>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
            {isPlain
              ? "Checking balance sheet honesty, growth speed, pricing power, and crash protection."
              : "5-factor fundamental rating across growth, balance sheet quality, valuation, price momentum, and downside risk"}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 inline-flex items-center gap-1">
              <span>🧬</span> {isPlain ? "Accounting Truth Check & 5-Pillar Scorecard" : "5-Factor Fundamental DNA & Piotroski 9-Point Solvency Matrix"}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className="bg-cyan-950/80 border border-cyan-700/80 px-3 py-1 rounded-lg text-right">
            <span className="text-[9px] sm:text-[10px] text-cyan-300 block uppercase leading-none font-bold">
              {isPlain ? "Business Score" : "Health Score"}
            </span>
            <span className="text-sm sm:text-base font-bold text-cyan-400">{scores.compositeFactorScore} / 100</span>
          </div>
          <span className="text-[11px] sm:text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">
            {scores.verdict}
          </span>
        </div>
      </div>

      {/* 5-Factor Score Breakdown */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {[
          { label: "Growth", plainLabel: "Revenue Engine", score: scores.growthScore, color: "text-emerald-400", bar: "bg-emerald-500" },
          { label: "Quality", plainLabel: "Moat & Margins", score: scores.qualityScore, color: "text-cyan-400", bar: "bg-cyan-500" },
          { label: "Valuation", plainLabel: "Bargain Level", score: scores.valuationScore, color: "text-purple-400", bar: "bg-purple-500" },
          { label: "Momentum", plainLabel: "Trend Force", score: scores.momentumScore, color: "text-amber-400", bar: "bg-amber-500" },
          { label: "Tail Risk", plainLabel: "Crash Armor", score: scores.tailRiskScore, color: "text-rose-400", bar: "bg-rose-500" },
        ].map((f) => (
          <div key={f.label} className="bg-[#090d14] p-3 rounded-lg border border-[#243044] space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400">{isPlain ? f.plainLabel : f.label}</span>
              <span className={`font-bold ${f.color}`}>{f.score}</span>
            </div>
            <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
              <div className={`${f.bar} h-full rounded-full`} style={{ width: `${f.score}%` }}></div>
            </div>
          </div>
        ))}
      </div>

      {/* Macro Regime Intelligence & 90-Day Expected Return Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* FRED Macro Regime Card */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-2">
          <div className="flex items-center justify-between border-b border-[#1b2434] pb-2">
            <span className="text-xs font-bold text-slate-200">
              {isPlain ? "Economic Climate (Federal Reserve)" : "Macro Regime (FRED Data)"}
            </span>
            <span className="text-[10px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded font-semibold">
              MDR: {mdr.rating} / 5 ({mdr.regime})
            </span>
          </div>
          <div className="text-xs text-slate-300 space-y-1.5 leading-relaxed">
            <p><span className="text-slate-400 font-semibold">Interest Rates: </span>{mdr.interestRateImpact}</p>
            <p><span className="text-slate-400 font-semibold">Inflation: </span>{mdr.inflationImpact}</p>
          </div>
        </div>

        {/* 90-Day Expected Return Simulation */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-2">
          <div className="flex items-center justify-between border-b border-[#1b2434] pb-2">
            <span className="text-xs font-bold text-slate-200">
              {isPlain ? "90-Day Price Forecast Odds" : "90-Day Expected Return E[R]"}
            </span>
            <span className="text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/40 px-2 py-0.5 rounded font-semibold">
              Vol: {er.annualizedVolatility}%
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs pt-1">
            <div className="bg-[#111722] p-2 rounded border border-[#1b2434]">
              <span className="text-[10px] text-slate-400 block">{isPlain ? "Bad Case" : "P10 Bear"}</span>
              <span className="font-bold text-rose-400">{er.p10Pessimistic}%</span>
            </div>
            <div className="bg-[#111722] p-2 rounded border border-[#1b2434]">
              <span className="text-[10px] text-slate-400 block">{isPlain ? "Expected" : "P50 Base"}</span>
              <span className="font-bold text-emerald-400">{er.p50Expected > 0 ? `+${er.p50Expected}` : er.p50Expected}%</span>
            </div>
            <div className="bg-[#111722] p-2 rounded border border-[#1b2434]">
              <span className="text-[10px] text-slate-400 block">{isPlain ? "Bull Run" : "P90 Bull"}</span>
              <span className="font-bold text-cyan-400">+{er.p90Optimistic}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

