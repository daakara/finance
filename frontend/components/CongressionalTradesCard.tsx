"use client";

import { useState } from "react";
import { CongressTradeItem, OptionsFlowItem } from "../lib/api";
import SmartMoneyDetailModal from "./SmartMoneyDetailModal";

interface CongressionalTradesCardProps {
  symbol?: string;
  congressTrades?: CongressTradeItem[];
  optionsFlow?: OptionsFlowItem[];
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onSelectSymbol?: (symbol: string) => void;
}

export default function CongressionalTradesCard({
  symbol = "NVDA",
  congressTrades = [],
  optionsFlow = [],
  userRole = "LONG_TERM",
  onSelectSymbol,
}: CongressionalTradesCardProps) {
  const isDayTrader = userRole === "DAY_TRADER";
  const [selectedCongress, setSelectedCongress] = useState<CongressTradeItem | null>(null);
  const [selectedOptions, setSelectedOptions] = useState<OptionsFlowItem | null>(null);

  return (
    <>
      <div className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono transition-colors ${
        isDayTrader ? "border-amber-900/40" : "border-[#243044]"
      }`}>
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className={`w-2.5 h-2.5 rounded-full ${
                isDayTrader ? "bg-amber-400" : "bg-purple-400"
              } animate-pulse`}></span>
              <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
                <span>{isDayTrader ? "⚡ Intraday Options Sweeps & Dark Pool Block Prints" : "🏛️ Capitol Hill & Institutional Order Flow Radar"}</span>
              </h3>
            </div>
            <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
              {isDayTrader
                ? `High-velocity OPRA options order flow & volume-to-open-interest anomalies for ${symbol}`
                : `STOCK Act Title I Article 105 disclosures & congressional committee alignment for ${symbol}`}
            </p>
          </div>

          <span className={`text-[11px] px-2.5 py-1 rounded-md font-semibold border ${
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
                  <th className="pb-2 font-semibold text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#162032]">
                {optionsFlow.length > 0 ? (
                  optionsFlow.map((flow, i) => (
                    <tr
                      key={i}
                      onClick={() => setSelectedOptions(flow)}
                      className="hover:bg-[#162030] cursor-pointer transition-colors group"
                    >
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
                      <td className="py-2.5 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedOptions(flow);
                          }}
                          className="bg-cyan-950 hover:bg-cyan-900 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded text-[10px] font-bold"
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-slate-500 text-xs">
                      No unusual intraday sweeps detected for {symbol} in current session window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        ) : (
          /* Long Term Investor: Congressional Trades Disclosure List */
          <div className="space-y-3">
            {congressTrades.length > 0 ? (
              congressTrades.map((trade, i) => (
                <div
                  key={i}
                  onClick={() => setSelectedCongress(trade)}
                  className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg bg-[#090d14] hover:bg-[#162030] border border-[#1e293b] hover:border-purple-500/40 transition-all cursor-pointer gap-3 group"
                >
                  <div className="space-y-1.5 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-white text-sm group-hover:text-purple-300">{trade.politician}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1b2434] text-slate-400 font-semibold">{trade.chamber}</span>
                      {trade.legislative_alignment_score !== undefined && (
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold border ${
                          trade.legislative_alignment_score >= 80
                            ? "bg-purple-950/80 text-purple-300 border-purple-700/80"
                            : trade.legislative_alignment_score >= 60
                            ? "bg-cyan-950/80 text-cyan-300 border-cyan-800/80"
                            : "bg-[#162030] text-slate-400 border-[#243044]"
                        }`}>
                          ⚖️ Alignment: {trade.legislative_alignment_score}/100
                        </span>
                      )}
                      {trade.staleness_badge && (
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold border ${
                          trade.staleness_status === "LATE_FILER"
                            ? "bg-rose-950/80 text-rose-300 border-rose-800/80"
                            : trade.staleness_status === "AGING"
                            ? "bg-amber-950/80 text-amber-300 border-amber-800/80"
                            : "bg-emerald-950/80 text-emerald-300 border-emerald-800/80"
                        }`}>
                          {trade.staleness_badge}
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="font-semibold text-cyan-400">{trade.transaction_type}</span>
                      <span>•</span>
                      <span>{trade.amount_range}</span>
                      <span>•</span>
                      <span>Filed: {trade.filing_date} ({trade.days_to_filing}d lag)</span>
                    </div>
                    {trade.staleness_warning && (
                      <div className="text-[10px] text-rose-400/90 font-sans font-medium flex items-center gap-1 bg-rose-950/30 border border-rose-900/40 px-2 py-0.5 rounded">
                        <span>⚠️</span>
                        <span>{trade.staleness_warning}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center space-x-3 text-right shrink-0">
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
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedCongress(trade);
                      }}
                      className="bg-purple-950 hover:bg-purple-900 text-purple-300 border border-purple-800 px-2 py-1 rounded text-[10px] font-bold"
                    >
                      Inspect
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <div className="p-4 bg-[#090d14] rounded-lg border border-[#243044] text-center text-xs text-slate-400">
                No recent Capitol Hill transactions reported for {symbol} within statutory filing windows.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Forensic Interactive Detail Drawer Modal */}
      <SmartMoneyDetailModal
        congressItem={selectedCongress}
        optionsItem={selectedOptions}
        onClose={() => {
          setSelectedCongress(null);
          setSelectedOptions(null);
        }}
        onSelectSymbol={onSelectSymbol}
      />
    </>
  );
}