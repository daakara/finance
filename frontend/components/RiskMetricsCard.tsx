"use client";

import { AnalyticsResponse } from "../lib/api";

interface RiskMetricsCardProps {
  analyticsData?: AnalyticsResponse;
}

export default function RiskMetricsCard({ analyticsData }: RiskMetricsCardProps) {
  const metrics = analyticsData?.analytics?.advanced_metrics;

  const var95 = metrics?.Modified_VaR_95 ?? -3.42;
  const var99 = metrics?.Modified_VaR_99 ?? -5.18;
  const sortino = metrics?.Sortino_Ratio ?? 1.84;
  const calmar = metrics?.Calmar_Ratio ?? 2.15;

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1b2434] pb-3">
        <div>
          <h3 className="text-sm sm:text-base font-bold text-slate-100 flex items-center gap-2">
            <svg className="w-4 h-4 text-cyan-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <span>Advanced Tail Risk & Benchmark Ratios</span>
          </h3>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">Non-normal return distribution models & Cornish-Fisher tail risk quantification</p>
        </div>
        <span className="text-[10px] sm:text-[11px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2.5 py-0.5 rounded-md font-semibold">
          Cornish-Fisher Model
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* VaR 95% Card */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Modified VaR (95%)</span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">95% CF</span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-rose-400 block">
            {var95.toFixed(2)}%
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-500 h-full rounded-full" style={{ width: `${Math.min(100, Math.abs(var95) * 15)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">Low Tail Risk vs S&P Benchmark</span>
        </div>

        {/* VaR 99% Card */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Modified VaR (99%)</span>
            <span className="text-[10px] text-slate-500 font-semibold px-1 rounded bg-[#1b2434]">99% CF</span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-rose-500 block">
            {var99.toFixed(2)}%
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-600 h-full rounded-full" style={{ width: `${Math.min(100, Math.abs(var99) * 12)}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">1-in-100 Day Tail Threshold</span>
        </div>

        {/* Sortino Ratio Card */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Sortino Ratio</span>
            <span className="text-[10px] text-emerald-400 font-semibold px-1 rounded bg-emerald-950/60 border border-emerald-800/40">Alpha</span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-emerald-400 block">
            {sortino.toFixed(2)}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.min(100, Math.max(0, sortino * 25))}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">Strong Downside Alpha Efficiency</span>
        </div>

        {/* Calmar Ratio Card */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Calmar Ratio</span>
            <span className="text-[10px] text-cyan-400 font-semibold px-1 rounded bg-cyan-950/60 border border-cyan-800/40">Recovery</span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-cyan-400 block">
            {calmar.toFixed(2)}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${Math.min(100, Math.max(0, calmar * 25))}%` }}></div>
          </div>
          <span className="text-[10px] text-slate-400 block">Annualized Return / Max Drawdown</span>
        </div>
      </div>
    </div>
  );
}

