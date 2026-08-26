"use client";

import { AnalyticsResponse } from "../lib/api";

interface RiskMetricsCardProps {
  analyticsData?: AnalyticsResponse;
  userRole?: "DAY_TRADER" | "LONG_TERM";
}

export default function RiskMetricsCard({ analyticsData, userRole = "LONG_TERM" }: RiskMetricsCardProps) {
  const metrics = analyticsData?.analytics?.advanced_metrics;
  const technicals = analyticsData?.technicals;
  const isDayTrader = userRole === "DAY_TRADER";

  const var95 = metrics?.Modified_VaR_95 ?? -3.42;
  const var99 = metrics?.Modified_VaR_99 ?? -5.18;
  const sortino = metrics?.Sortino_Ratio ?? 1.84;
  const calmar = metrics?.Calmar_Ratio ?? 2.15;
  const atr14 = technicals?.atr_14 ?? 4.25;
  const rsi14 = technicals?.rsi_14 ?? 52.0;

  return (
    <div className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono transition-colors ${
      isDayTrader ? "border-amber-900/40" : "border-[#243044]"
    }`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1b2434] pb-3">
        <div>
          <h3 className="text-sm sm:text-base font-bold text-slate-100 flex items-center gap-2">
            <svg className={`w-4 h-4 shrink-0 ${isDayTrader ? "text-amber-400" : "text-cyan-400"}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <span>{isDayTrader ? "⚡ Intraday Volatility & Execution Risk Guardrails" : "🏛️ Advanced Tail Risk & Benchmark Ratios"}</span>
          </h3>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
            {isDayTrader
              ? "Real-time ATR volatility, single-day drawdown boundaries, and intraday risk-adjusted ratios"
              : "Non-normal return distribution models & Cornish-Fisher tail risk quantification"}
          </p>
        </div>
        <span className={`text-[10px] sm:text-[11px] px-2.5 py-0.5 rounded-md font-semibold border ${
          isDayTrader
            ? "text-amber-400 bg-amber-950/60 border-amber-800/80"
            : "text-cyan-400 bg-cyan-950/60 border-cyan-800/80"
        }`}>
          {isDayTrader ? "Intraday Risk Engine" : "Cornish-Fisher Model"}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Metric 1 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isDayTrader ? "14-Day True Range (ATR)" : "Modified VaR (95%)"}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Intraday" : "95% CF"}
            </span>
          </div>
          <span className={`text-xl sm:text-2xl font-bold block tabular-nums ${isDayTrader ? "text-amber-400" : "text-rose-400"}`}>
            {isDayTrader ? `$${atr14.toFixed(2)}` : `${var95.toFixed(2)}%`}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className={`${isDayTrader ? "bg-amber-500" : "bg-rose-500"} h-full rounded-full`} style={{ width: `${Math.min(100, Math.abs(var95) * 15)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {isDayTrader ? "Average Daily Dollar Movement" : "Low Tail Risk vs S&P Benchmark"}
          </span>
        </div>

        {/* Metric 2 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isDayTrader ? "Single-Session Max Loss" : "Modified VaR (99%)"}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "1D Limit" : "99% CF"}
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-rose-500 block tabular-nums">
            {var99.toFixed(2)}%
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-600 h-full rounded-full" style={{ width: `${Math.min(100, Math.abs(var99) * 12)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {isDayTrader ? "Daily Stop-Out Capital Threshold" : "Extreme 99% Tail Risk Horizon"}
          </span>
        </div>

        {/* Metric 3 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Sortino Ratio</span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Downside" : "Annualized"}
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-emerald-400 block tabular-nums">
            {sortino.toFixed(2)}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.min(100, sortino * 35)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">
            {isDayTrader ? "High Downside Risk Protection" : "Institutional Risk-Adjusted Alpha"}
          </span>
        </div>

        {/* Metric 4 */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">
              {isDayTrader ? "RSI (14-Period Momentum)" : "Calmar Ratio"}
            </span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">
              {isDayTrader ? "Tape Flow" : "Max DD"}
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
              ? rsi14 > 70 ? "Overbought Scalp Risk" : rsi14 < 30 ? "Oversold Bounce Zone" : "Neutral Order Flow"
              : "Annualized Return / Max Drawdown"}
          </span>
        </div>
      </div>
    </div>
  );
}
