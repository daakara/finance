"use client";

import { AssetFactorScores, MacroDifficultyRating, ExpectedReturnForecast } from "../lib/api";

interface AssetFactorRadarProps {
  symbol: string;
  factorScores?: AssetFactorScores;
  macroDifficulty?: MacroDifficultyRating;
  expectedReturn?: ExpectedReturnForecast;
}

export default function AssetFactorRadar({ symbol, factorScores, macroDifficulty, expectedReturn }: AssetFactorRadarProps) {
  const scores = factorScores || {
    growthScore: 84,
    qualityScore: 90,
    valuationScore: 72,
    momentumScore: 78,
    tailRiskScore: 82,
    compositeFactorScore: 82,
    verdict: "Strong Buy / Core Hold",
    piotroskiFScore: 8,
  };

  const mdr = macroDifficulty || {
    rating: 1,
    regime: "Optimal Expansionary Goldilocks",
    interestRateImpact: "Steepening curve (+0.47%) and tight credit spreads fuel strong risk-on alpha",
    inflationImpact: "CPI (2.4% YoY) moderation reduces discount rate pressure on valuations",
    yield_curve_spread: 0.47,
    fed_funds_rate: 3.63,
    credit_spread_oas: 2.69,
    cpi_yoy: 2.4,
  };

  const er = expectedReturn || {
    p10Pessimistic: -8.4,
    p50Expected: +18.6,
    p90Optimistic: +38.2,
    annualizedVolatility: 22.4,
    forecastHorizonDays: 90,
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-5">
      {/* Header with Financial Health Score */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h3 className="text-base font-bold text-slate-100 font-mono tracking-tight">
              {symbol} Fundamental & Factor Profile
            </h3>
            {scores.piotroskiFScore && (
              <span className="text-[10px] font-mono bg-[#1b2434] text-cyan-300 border border-cyan-800/60 px-2 py-0.5 rounded">
                Piotroski Health: {scores.piotroskiFScore}/9
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            5-factor fundamental rating across growth, balance sheet quality, valuation, price momentum, and downside risk
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div className="bg-cyan-950/80 border border-cyan-700/80 px-3 py-1 rounded-lg text-right font-mono">
            <span className="text-[10px] text-cyan-300 block uppercase leading-none font-bold">Health Score</span>
            <span className="text-base font-bold text-cyan-400">{scores.compositeFactorScore} / 100</span>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 font-mono">
            {scores.verdict}
          </span>
        </div>
      </div>

      {/* 5-Factor Dimension Bars Grid */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {/* Growth Factor */}
        <div className="bg-[#090d14] border border-[#243044] rounded-lg p-3 space-y-1.5">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">Growth</span>
            <span className="text-cyan-400 font-bold">{scores.growthScore}/100</span>
          </div>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${scores.growthScore}%` }}></div>
          </div>
          <span className="text-[9px] text-slate-500 block">Revenue & User Expansion</span>
        </div>

        {/* Quality / Health Factor */}
        <div className="bg-[#090d14] border border-[#243044] rounded-lg p-3 space-y-1.5">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">Quality & Health</span>
            <span className="text-emerald-400 font-bold">{scores.qualityScore}/100</span>
          </div>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${scores.qualityScore}%` }}></div>
          </div>
          <span className="text-[9px] text-slate-500 block">
            {scores.piotroskiFScore ? `Piotroski ${scores.piotroskiFScore}/9 Health` : "Strong Balance Sheet"}
          </span>
        </div>

        {/* Valuation Factor */}
        <div className="bg-[#090d14] border border-[#243044] rounded-lg p-3 space-y-1.5">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">Valuation</span>
            <span className="text-amber-400 font-bold">{scores.valuationScore}/100</span>
          </div>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-amber-500 h-full rounded-full" style={{ width: `${scores.valuationScore}%` }}></div>
          </div>
          <span className="text-[9px] text-slate-500 block">Price Multiples vs Peers</span>
        </div>

        {/* Momentum Factor */}
        <div className="bg-[#090d14] border border-[#243044] rounded-lg p-3 space-y-1.5">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">Momentum</span>
            <span className="text-purple-400 font-bold">{scores.momentumScore}/100</span>
          </div>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-purple-500 h-full rounded-full" style={{ width: `${scores.momentumScore}%` }}></div>
          </div>
          <span className="text-[9px] text-slate-500 block">Trend Above 50-Day Avg</span>
        </div>

        {/* Tail-Risk Factor */}
        <div className="bg-[#090d14] border border-[#243044] rounded-lg p-3 space-y-1.5">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">Crash Protection</span>
            <span className="text-rose-400 font-bold">{scores.tailRiskScore}/100</span>
          </div>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-500 h-full rounded-full" style={{ width: `${scores.tailRiskScore}%` }}></div>
          </div>
          <span className="text-[9px] text-slate-500 block">Controlled Drawdown Risk</span>
        </div>
      </div>

      {/* Expected Return Forecast & Macro Difficulty Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        {/* Expected Return ($E[R]$) Forecast Band */}
        <div className="bg-[#090d14] border border-[#243044] rounded-xl p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              ?? 90-Day Expected Return ($E[R]$) Band
            </span>
            <span className="text-[10px] text-slate-400 font-mono">Monte Carlo Sim</span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center pt-1">
            <div className="bg-[#111722] p-2 rounded-lg border border-[#243044]">
              <span className="text-[10px] text-slate-400 font-mono block">P10 (Pessimistic)</span>
              <span className="text-sm font-bold font-mono text-rose-400">{er.p10Pessimistic}%</span>
            </div>
            <div className="bg-[#1b2434] p-2 rounded-lg border border-cyan-500/50 shadow-inner">
              <span className="text-[10px] text-cyan-300 font-mono block font-bold">P50 (Median)</span>
              <span className="text-base font-bold font-mono text-cyan-400">+{er.p50Expected}%</span>
            </div>
            <div className="bg-[#111722] p-2 rounded-lg border border-[#243044]">
              <span className="text-[10px] text-slate-400 font-mono block">P90 (Optimistic)</span>
              <span className="text-sm font-bold font-mono text-emerald-400">+{er.p90Optimistic}%</span>
            </div>
          </div>
        </div>

        {/* Macro Difficulty Rating (MDR) Powered by FRED */}
        <div className="bg-[#090d14] border border-[#243044] rounded-xl p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              ?? Macro Difficulty Rating (MDR: {mdr.rating}/5)
            </span>
            <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold ${
              mdr.rating <= 2 ? "bg-emerald-950 text-emerald-400 border border-emerald-800" : "bg-amber-950 text-amber-400 border border-amber-800"
            }`}>
              {mdr.regime}
            </span>
          </div>

          {/* FRED Macro Indicators Grid */}
          <div className="grid grid-cols-4 gap-1.5 py-1 text-center bg-[#111722] p-2 rounded-lg border border-[#243044]">
            <div>
              <span className="text-[9px] text-slate-400 block font-mono">10Y-2Y Curve</span>
              <span className="text-xs font-bold font-mono text-emerald-400">
                {mdr.yield_curve_spread !== undefined ? `+${mdr.yield_curve_spread}%` : "+0.47%"}
              </span>
            </div>
            <div>
              <span className="text-[9px] text-slate-400 block font-mono">Fed Funds</span>
              <span className="text-xs font-bold font-mono text-cyan-400">
                {mdr.fed_funds_rate !== undefined ? `${mdr.fed_funds_rate}%` : "3.63%"}
              </span>
            </div>
            <div>
              <span className="text-[9px] text-slate-400 block font-mono">HY OAS Spread</span>
              <span className="text-xs font-bold font-mono text-purple-400">
                {mdr.credit_spread_oas !== undefined ? `${mdr.credit_spread_oas}%` : "2.69%"}
              </span>
            </div>
            <div>
              <span className="text-[9px] text-slate-400 block font-mono">CPI Inflation</span>
              <span className="text-xs font-bold font-mono text-amber-400">
                {mdr.cpi_yoy !== undefined ? `${mdr.cpi_yoy}%` : "2.4%"}
              </span>
            </div>
          </div>

          <div className="text-xs text-slate-300 space-y-1 font-mono">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Rate Policy:</span>
              <span className="text-emerald-400 font-semibold">{mdr.interestRateImpact}</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Inflation Trend:</span>
              <span className="text-cyan-400 font-semibold">{mdr.inflationImpact}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

