import { AnalyticsResponse } from "../lib/api";

interface RiskMetricsCardProps {
  analyticsData?: AnalyticsResponse;
}

export default function RiskMetricsCard({ analyticsData }: RiskMetricsCardProps) {
  const metrics = analyticsData?.analytics?.advanced_metrics;

  return (
    <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-6 shadow-lg">
      <h3 className="text-lg font-semibold text-white mb-4">Advanced Risk Analytics</h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#0d1117] p-4 rounded-lg border border-[#21262d]">
          <span className="text-xs text-gray-400 block">Cornish-Fisher VaR (95%)</span>
          <span className="text-xl font-bold text-red-400">
            {metrics?.Modified_VaR_95 !== undefined ? `${metrics.Modified_VaR_95.toFixed(2)}%` : "N/A"}
          </span>
        </div>

        <div className="bg-[#0d1117] p-4 rounded-lg border border-[#21262d]">
          <span className="text-xs text-gray-400 block">Cornish-Fisher VaR (99%)</span>
          <span className="text-xl font-bold text-red-500">
            {metrics?.Modified_VaR_99 !== undefined ? `${metrics.Modified_VaR_99.toFixed(2)}%` : "N/A"}
          </span>
        </div>

        <div className="bg-[#0d1117] p-4 rounded-lg border border-[#21262d]">
          <span className="text-xs text-gray-400 block">Sortino Ratio</span>
          <span className="text-xl font-bold text-emerald-400">
            {metrics?.Sortino_Ratio !== undefined ? metrics.Sortino_Ratio.toFixed(2) : "N/A"}
          </span>
        </div>

        <div className="bg-[#0d1117] p-4 rounded-lg border border-[#21262d]">
          <span className="text-xs text-gray-400 block">Calmar Ratio</span>
          <span className="text-xl font-bold text-sky-400">
            {metrics?.Calmar_Ratio !== undefined ? metrics.Calmar_Ratio.toFixed(2) : "N/A"}
          </span>
        </div>
      </div>
    </div>
  );
}

