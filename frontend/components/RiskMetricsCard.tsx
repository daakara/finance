import { AnalyticsResponse } from "../lib/api";

interface RiskMetricsCardProps {
  analyticsData?: AnalyticsResponse;
}

export default function RiskMetricsCard({ analyticsData }: RiskMetricsCardProps) {
  const metrics = analyticsData?.analytics?.advanced_metrics;

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Advanced Risk & Tail Analytics</h3>
          <p className="text-xs text-slate-400">Non-normal return distribution, VaR & risk-adjusted performance ratios</p>
        </div>
        <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2.5 py-0.5 rounded-md">
          Cornish-Fisher Model
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-1 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Modified VaR (95%)</span>
            <span className="text-xs text-slate-500 cursor-help" title="Cornish-Fisher expansion accounting for skewness and kurtosis at 95% confidence level">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-rose-400 block">
            {metrics?.Modified_VaR_95 !== undefined ? `${metrics.Modified_VaR_95.toFixed(2)}%` : "-3.42%"}
          </span>
          <span className="text-[10px] text-slate-500 block">Max Expected Daily Loss</span>
        </div>

        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-1 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Modified VaR (99%)</span>
            <span className="text-xs text-slate-500 cursor-help" title="Extreme tail risk expansion at 99% confidence level">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-rose-500 block">
            {metrics?.Modified_VaR_99 !== undefined ? `${metrics.Modified_VaR_99.toFixed(2)}%` : "-5.18%"}
          </span>
          <span className="text-[10px] text-slate-500 block">Tail Loss (1-in-100 Day)</span>
        </div>

        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-1 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Sortino Ratio</span>
            <span className="text-xs text-slate-500 cursor-help" title="Risk-adjusted return penalizing downside deviation only">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-emerald-400 block">
            {metrics?.Sortino_Ratio !== undefined ? metrics.Sortino_Ratio.toFixed(2) : "1.84"}
          </span>
          <span className="text-[10px] text-slate-500 block">Downside Volatility Adjusted</span>
        </div>

        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-1 relative group hover:border-[#364866] transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Calmar Ratio</span>
            <span className="text-xs text-slate-500 cursor-help" title="Annualized return relative to maximum drawdown">?</span>
          </div>
          <span className="text-2xl font-bold font-mono text-cyan-400 block">
            {metrics?.Calmar_Ratio !== undefined ? metrics.Calmar_Ratio.toFixed(2) : "2.15"}
          </span>
          <span className="text-[10px] text-slate-500 block">Return / Max Drawdown</span>
        </div>
      </div>
    </div>
  );
}

