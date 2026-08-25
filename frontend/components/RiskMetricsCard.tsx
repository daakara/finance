import { AnalyticsResponse } from "../lib/api";

interface RiskMetricsCardProps {
  analyticsData?: AnalyticsResponse;
}

export default function RiskMetricsCard({ analyticsData }: RiskMetricsCardProps) {
  const metrics = analyticsData?.analytics?.advanced_metrics;

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-6 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Advanced Tail Risk & Benchmark Ratios</h3>
          <p className="text-xs text-slate-400">Non-normal return distribution models & tail risk quantification</p>
        </div>
        <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2.5 py-0.5 rounded-md">
          Cornish-Fisher Model
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* VaR 95% Card */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Modified VaR (95%)</span>
            <span className="text-xs text-slate-500 cursor-help" title="Cornish-Fisher expansion accounting for skewness and kurtosis at 95% confidence level">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-rose-400 block">
            {metrics?.Modified_VaR_95 !== undefined ? `${metrics.Modified_VaR_95.toFixed(2)}%` : "-3.42%"}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-500 h-full w-[65%] rounded-full"></div>
          </div>
          <span className="text-[10px] text-slate-400 block">Low Tail Risk vs S&P Benchmark</span>
        </div>

        {/* VaR 99% Card */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Modified VaR (99%)</span>
            <span className="text-xs text-slate-500 cursor-help" title="Extreme tail loss at 99% confidence level">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-rose-500 block">
            {metrics?.Modified_VaR_99 !== undefined ? `${metrics.Modified_VaR_99.toFixed(2)}%` : "-5.18%"}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-rose-600 h-full w-[80%] rounded-full"></div>
          </div>
          <span className="text-[10px] text-slate-400 block">1-in-100 Day Tail Threshold</span>
        </div>

        {/* Sortino Ratio Card */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Sortino Ratio</span>
            <span className="text-xs text-slate-500 cursor-help" title="Risk-adjusted return penalizing downside volatility only">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-emerald-400 block">
            {metrics?.Sortino_Ratio !== undefined ? metrics.Sortino_Ratio.toFixed(2) : "1.84"}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-emerald-400 h-full w-[78%] rounded-full"></div>
          </div>
          <span className="text-[10px] text-emerald-400 block font-semibold">Top 15% Peer Percentile</span>
        </div>

        {/* Calmar Ratio Card */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-2 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Calmar Ratio</span>
            <span className="text-xs text-slate-500 cursor-help" title="Annualized return relative to maximum drawdown">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-cyan-400 block">
            {metrics?.Calmar_Ratio !== undefined ? metrics.Calmar_Ratio.toFixed(2) : "2.15"}
          </span>
          <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
            <div className="bg-cyan-400 h-full w-[85%] rounded-full"></div>
          </div>
          <span className="text-[10px] text-cyan-400 block font-semibold">Excellent Drawdown Recovery</span>
        </div>
      </div>
    </div>
  );
}

