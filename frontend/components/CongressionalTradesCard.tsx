"use client";

import { CongressTradeItem, OptionsFlowItem } from "../lib/api";

interface CongressionalTradesCardProps {
  symbol?: string;
  congressTrades?: CongressTradeItem[];
  optionsFlow?: OptionsFlowItem[];
  userRole?: "DAY_TRADER" | "LONG_TERM";
}

export default function CongressionalTradesCard({
  symbol = "NVDA",
  congressTrades = [],
  optionsFlow = [],
  userRole = "LONG_TERM",
}: CongressionalTradesCardProps) {
  const isDayTrader = userRole === "DAY_TRADER";

  return (
    <div className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono transition-colors ${
      isDayTrader ? "border-amber-900/40" : "border-[#243044]"
    }`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className={`w-2.5 h-2.5 rounded-full ${isDayTrader ? "bg-amber-400" : "bg-purple-400"} animate-pulse`}></span>
            <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <svg className={`w-4 h-4 ${isDayTrader ? "text-amber-400" : "text-purple-400"} shrink-0`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              <span>{symbol} {isDayTrader ? "Institutional Options Flow Tape & Sweeps" : "Congressional STOCK Act Disclosures"}</span>
            </h3>
          </div>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
            {isDayTrader
              ? "Unusual call sweeps, dark pool blocks, and high-urgency order flow"
              : "Capitol Hill House & Senate insider transaction tracking and post-filing returns"}
          </p>
        </div>

        <span className={`text-[10px] sm:text-[11px] px-2.5 py-0.5 rounded-md font-semibold border ${
          isDayTrader
            ? "text-amber-400 bg-amber-950/60 border-amber-800/80"
            : "text-purple-400 bg-purple-950/60 border-purple-800/80"
        }`}>
          {isDayTrader ? "⚡ Real-Time Tape" : "🏛️ Capitol Hill Radar"}
        </span>
      </div>

      {/* Content Rendering based on Horizon */}
      {isDayTrader ? (
        /* Day Trader: Options Flow Sweeps Table */
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#1b2434] text-slate-400 text-[10px] uppercase">
                <th className="pb-2 font-semibold">Time</th>
                <th className="pb-2 font-semibold">Type</th>
                <th className="pb-2 font-semibold">Strike / Exp</th>
                <th className="pb-2 font-semibold text-right">Premium</th>
                <th className="pb-2 font-semibold text-right">Vol/OI</th>
                <th className="pb-2 font-semibold text-right">Sentiment</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#162032]">
              {optionsFlow.length > 0 ? (
                optionsFlow.map((flow, i) => (
                  <tr key={i} className="hover:bg-[#162030] transition-colors">
                    <td className="py-2.5 font-mono text-slate-400 text-[11px]">{flow.time}</td>
                    <td className="py-2.5">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        flow.type.includes("CALL")
                          ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/60"
                          : flow.type.includes("PUT")
                          ? "bg-rose-950/80 text-rose-400 border border-rose-800/60"
                          : "bg-cyan-950/80 text-cyan-400 border border-cyan-800/60"
                      }`}>
                        {flow.type}
                      </span>
                    </td>
                    <td className="py-2.5 text-slate-200">
                      <div className="font-semibold">{flow.strike}</div>
                      <div className="text-[10px] text-slate-400">{flow.expiration}</div>
                    </td>
                    <td className="py-2.5 font-bold text-amber-400 text-right tabular-nums">{flow.premium}</td>
                    <td className="py-2.5 font-semibold text-slate-300 text-right tabular-nums">{flow.volume_oi_ratio}x</td>
                    <td className="py-2.5 text-right">
                      <span className="text-[11px] font-bold text-emerald-400">{flow.sentiment}</span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="py-4 text-center text-slate-500 text-xs">
                    No unusual intraday sweeps detected in current session window.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* Long Term: Congressional STOCK Act Disclosures */
        <div className="space-y-3">
          {congressTrades.length > 0 ? (
            congressTrades.map((trade, idx) => (
              <div key={idx} className="bg-[#090d14] p-3 sm:p-4 rounded-lg border border-[#243044] flex flex-wrap items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-bold text-slate-100">{trade.politician}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1b2434] text-slate-400 font-semibold">{trade.chamber}</span>
                  </div>
                  <div className="text-[11px] text-slate-400 flex items-center space-x-2">
                    <span className="font-semibold text-cyan-400">{trade.transaction_type}</span>
                    <span>•</span>
                    <span>{trade.amount_range}</span>
                    <span>•</span>
                    <span>Filed: {trade.filing_date} ({trade.days_to_filing}d lag)</span>
                  </div>
                </div>

                <div className="flex items-center space-x-3 text-right">
                  <div>
                    <span className="text-[10px] text-slate-500 block uppercase">Return Since Filing</span>
                    <span className={`text-sm sm:text-base font-bold tabular-nums ${
                      trade.performance_since_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}>
                      {trade.performance_since_pct >= 0 ? `+${trade.performance_since_pct}%` : `${trade.performance_since_pct}%`}
                    </span>
                  </div>
                  <span className={`px-2.5 py-1 rounded text-[11px] font-bold border ${
                    trade.sentiment.includes("Bullish")
                      ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/80"
                      : "bg-slate-800 text-slate-300 border-slate-700"
                  }`}>
                    {trade.sentiment}
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] text-center text-slate-500 text-xs">
              No recent 30-day Congressional transactions disclosed for {symbol}.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
