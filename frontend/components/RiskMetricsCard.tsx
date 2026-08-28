"use client";

import { useState, useEffect } from "react";
import { AnalyticsResponse } from "../lib/api";

interface RiskMetricsCardProps {
  analyticsData?: AnalyticsResponse;
  userRole?: "DAY_TRADER" | "LONG_TERM";
}

export default function RiskMetricsCard({ analyticsData, userRole = "LONG_TERM" }: RiskMetricsCardProps) {
  const metrics = analyticsData?.analytics?.advanced_metrics;
  const technicals = analyticsData?.technicals;
  const isDayTrader = userRole === "DAY_TRADER";
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

  const var95 = metrics?.Modified_VaR_95 ?? -3.42;
  const var99 = metrics?.Modified_VaR_99 ?? -5.18;
  const sortino = metrics?.Sortino_Ratio ?? 1.84;
  const calmar = metrics?.Calmar_Ratio ?? 2.15;
  const atr14 = technicals?.atr_14 ?? 4.25;
  const rsi14 = technicals?.rsi_14 ?? 52.0;

  return (
    <div className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-sans transition-colors ${
      isDayTrader ? "border-amber-900/40" : "border-[#243044]"
    }`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1b2434] pb-3">
        <div>
          <h3 className="text-sm sm:text-base font-bold text-slate-100 flex items-center gap-2">
            <svg className={`w-4 h-4 shrink-0 ${isDayTrader ? "text-amber-400" : "text-cyan-400"}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <span>
              {isDayTrader
                ? (isPlain ? "⚡ Day Trader Speedometer & Safety Stops" : "⚡ Intraday Volatility & Execution Risk Guardrails")
                : (isPlain ? "🛡️ Worst-Case Crash Test & Downside Protection" : "🏛️ Advanced Tail Risk & Benchmark Ratios")}
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5 font-normal">
            {isDayTrader
              ? (isPlain ? "Live day range speed, daily max loss limits, and momentum gauges." : "Real-time ATR volatility, single-day drawdown boundaries, and intraday risk-adjusted ratios.")
              : (isPlain ? "If bad news hits tomorrow, how much could this asset realistically drop?" : "Downside Crash Protection & Black-Swan Tail Risk Evaluation")}
          </p>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 inline-flex items-center gap-1 font-mono">
              <span>🧮</span> {isDayTrader ? "Live 14-Day Range Gauge" : (isPlain ? "Black Swan Downside Crash Model" : "Cornish-Fisher Non-Normal VaR Model")}
            </span>
          </div>
        </div>
        <span className={`text-xs px-2.5 py-0.5 rounded-md font-semibold border ${
          isDayTrader
            ? "text-amber-400 bg-amber-950/60 border-amber-800/80"
            : "text-cyan-400 bg-cyan-950/60 border-cyan-800/80"
        }`}>
          {isDayTrader ? (isPlain ? "Intraday Guardrails" : "Intraday Risk Engine") : (isPlain ? "Crash Test Engine" : "Crash-Adjusted VaR")}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Metric 1 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group ">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isDayTrader ? (isPlain ? "Typical Daily Swing" : "14-Day True Range (ATR)") : (isPlain ? "Standard Bad Day (95% VaR)" : "Modified VaR (95%)")}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Daily Move" : "95% Level"}
            </span>
          </div>
          <span className={`text-xl sm:text-2xl font-bold block tabular-nums ${isDayTrader ? "text-amber-400" : "text-rose-400"}`}>
            {isDayTrader ? `$${atr14.toFixed(2)}` : `${var95.toFixed(2)}%`}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className={`${isDayTrader ? "bg-amber-500" : "bg-rose-500"} h-full rounded-full`} style={{ width: `${Math.min(100, Math.abs(var95) * 15)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {isDayTrader ? "Average swing per trading day" : "Expected max drop 19 out of 20 days"}
          </span>
        </div>

        {/* Metric 2 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group ">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isDayTrader ? (isPlain ? "Max Session Budget" : "Single-Session Max Loss") : (isPlain ? "Worst-Case Shock (99% VaR)" : "Modified VaR (99%)")}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Hard Stop" : "99% Shock"}
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-rose-500 block tabular-nums">
            {var99.toFixed(2)}%
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-600 h-full rounded-full" style={{ width: `${Math.min(100, Math.abs(var99) * 12)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {isDayTrader ? "Do not hold through this loss level" : "Estimated worst 1-day crash in 100 sessions"}
          </span>
        </div>

        {/* Metric 3 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group ">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isPlain ? "Downside Armor (Sortino)" : "Sortino Ratio"}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Downside" : "Armor"}
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-emerald-400 block tabular-nums">
            {sortino.toFixed(2)}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.min(100, sortino * 35)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {sortino >= 1.5 ? "Excellent downside protection" : "Moderate downside turbulence"}
          </span>
        </div>

        {/* Metric 4 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group ">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isDayTrader ? (isPlain ? "Speedometer (RSI)" : "RSI (14-Period Momentum)") : (isPlain ? "Recovery Speed (Calmar)" : "Calmar Ratio")}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Momentum" : "Max DD"}
            </span>
          </div>
          <span className={`text-xl sm:text-2xl font-bold block tabular-nums ${
            isDayTrader
              ? rsi14 > 70 ? "text-rose-400" : rsi14 < 30 ? "text-emerald-400" : "text-amber-300"
              : "text-purple-400"
          }`}>
            {isDayTrader ? rsi14.toFixed(1) : calmar.toFixed(2)}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className={`${isDayTrader ? "bg-amber-500" : "bg-purple-500"} h-full rounded-full`} style={{ width: `${Math.min(100, isDayTrader ? rsi14 : calmar * 30)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {isDayTrader
              ? rsi14 > 70 ? "Overheating (Don't chase high)" : rsi14 < 30 ? "Oversold (Watch for bounce)" : "Cruising in middle gear"
              : "Historical return vs deepest drawdown"}
          </span>
        </div>
      </div>
    </div>
  );
}
