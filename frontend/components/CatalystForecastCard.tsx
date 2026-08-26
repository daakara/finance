"use client";

import { CatalystForecastData } from "../lib/api";

interface CatalystForecastCardProps {
  data?: CatalystForecastData;
}

export default function CatalystForecastCard({ data }: CatalystForecastCardProps) {
  if (!data) return null;

  return (
    <section aria-labelledby="catalyst-header" className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl font-mono space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1b2434] pb-3">
        <div className="flex items-center space-x-2">
          <span className="text-xl">🔬</span>
          <div>
            <h2 id="catalyst-header" className="text-sm sm:text-base font-bold text-white tracking-tight">
              Clinical Trials, Pipeline Catalysts & 5-Year Earnings Model
            </h2>
            <span className="text-[11px] text-cyan-400 font-semibold">{data.company_name} ({data.symbol}) • {data.sector}</span>
          </div>
        </div>
        <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-indigo-950/80 text-indigo-300 border border-indigo-800/80">
          {data.trial_phase}
        </span>
      </div>

      {/* Primary Pipeline Feature & Efficacy Banner */}
      <div className="bg-[#0b1019] border border-cyan-900/40 rounded-xl p-3.5 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs font-bold text-cyan-300 flex items-center space-x-1.5">
            <span>💊 Key Asset:</span>
            <span className="text-white font-extrabold">{data.primary_drug_trial}</span>
          </span>
          <span className="text-[11px] text-slate-400">Readout Horizon: <strong className="text-slate-200">{data.trial_readout_timeline}</strong></span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          <strong className="text-slate-100">Efficacy & Impact:</strong> {data.efficacy_summary}
        </p>
        <p className="text-[11px] text-slate-400 leading-snug">
          <strong className="text-cyan-400">Competitive Moat:</strong> {data.competitive_edge}
        </p>
      </div>

      {/* Grid: Upcoming Milestone Timeline + 5-Year Forecast Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Catalyst Timeline */}
        <div className="bg-[#0b1019] border border-[#1b2434] rounded-xl p-3.5 space-y-2.5">
          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
            <span>📅</span>
            <span>Upcoming Critical Milestone Schedule</span>
          </h3>
          <div className="space-y-2">
            {data.upcoming_milestones.map((m, idx) => (
              <div key={idx} className="flex items-start justify-between gap-2 p-2 rounded-lg bg-[#111722] border border-[#1e293b]">
                <div>
                  <span className="text-[10px] font-black text-cyan-400 block">{m.date}</span>
                  <span className="text-xs text-slate-200 leading-tight">{m.event}</span>
                </div>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded whitespace-nowrap ${
                  m.impact.includes("Transformational")
                    ? "bg-purple-950/80 text-purple-300 border border-purple-800"
                    : "bg-emerald-950/80 text-emerald-300 border border-emerald-800"
                }`}>
                  {m.impact}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: 5-Year DCF / EPS Valuation Trajectory */}
        <div className="bg-[#0b1019] border border-[#1b2434] rounded-xl p-3.5 space-y-2.5">
          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
            <span>📈</span>
            <span>5-Year Revenue & Earnings Simulation</span>
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#1b2434] text-[10px] text-slate-400">
                  <th className="pb-1.5">Year</th>
                  <th className="pb-1.5">Revenue</th>
                  <th className="pb-1.5">Net Margin</th>
                  <th className="pb-1.5">Proj. EPS</th>
                  <th className="pb-1.5 text-right">Target (P/E)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#162030] text-[11px] tabular-nums font-semibold">
                {data.multi_year_forecast.map((f) => (
                  <tr key={f.year} className="hover:bg-[#131c2c]">
                    <td className="py-2 text-cyan-400 font-bold">{f.year}</td>
                    <td className="py-2 text-slate-200">${f.revenue_billions}B</td>
                    <td className="py-2 text-slate-400">{f.net_margin_pct}%</td>
                    <td className="py-2 text-emerald-400 font-bold">${f.projected_eps.toFixed(2)}</td>
                    <td className="py-2 text-right text-purple-300 font-bold">${f.implied_target.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-500 leading-tight">
            *Simulation estimates future free cash flows, commercial penetration curves, and multiple compression upon patent expirations.
          </p>
        </div>
      </div>
    </section>
  );
}
