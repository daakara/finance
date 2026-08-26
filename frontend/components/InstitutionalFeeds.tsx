"use client";

import { useState, useEffect } from "react";
import {
  FredMacroData,
  SecForm4Trade,
  fetchFredMacroRegime,
  fetchSecForm4Insiders,
} from "../lib/institutionalFeeds";

interface InstitutionalFeedsProps {
  activeSymbol: string;
}

export default function InstitutionalFeeds({ activeSymbol }: InstitutionalFeedsProps) {
  const [macro, setMacro] = useState<FredMacroData | null>(null);
  const [insiders, setInsiders] = useState<SecForm4Trade[]>([]);
  const [activeTab, setActiveTab] = useState<"FRED_MACRO" | "SEC_FORM4">("FRED_MACRO");

  useEffect(() => {
    fetchFredMacroRegime().then(setMacro);
    fetchSecForm4Insiders(activeSymbol).then(setInsiders);
  }, [activeSymbol]);

  const cleanSym = activeSymbol.toUpperCase().replace("-USD", "");
  const matchedInsider = insiders.find((i) => i.ticker === cleanSym);

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono">
      {/* Header with Sub-Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1b2434] pb-3">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <h3 className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center gap-1.5">
            <span>🏛️</span>
            <span>Federal Reserve (FRED) &amp; SEC EDGAR Corporate Insiders</span>
          </h3>
        </div>

        <div className="flex items-center space-x-1.5 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
          <button
            type="button"
            onClick={() => setActiveTab("FRED_MACRO")}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors ${
              activeTab === "FRED_MACRO"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-700"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            📊 FRED Macro Regime
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("SEC_FORM4")}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors ${
              activeTab === "SEC_FORM4"
                ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            🏢 SEC Form 4 (CEO/CFO)
          </button>
        </div>
      </div>

      {/* VIEW 1: FRED MACRO REGIME */}
      {activeTab === "FRED_MACRO" && macro && (
        <div className="space-y-4">
          <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
            <div className="flex items-center justify-between">
              <strong className="text-xs font-bold text-cyan-400">{macro.regimeTitle}</strong>
              <span className="text-[10px] text-emerald-400 font-bold bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
                1.1x Risk Budget Active
              </span>
            </div>
            <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
              {macro.regimeSubtitle}
            </p>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
            <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
              <span className="text-[10px] text-slate-400 block uppercase">10Y-2Y Yield Curve</span>
              <span className="text-sm sm:text-base font-extrabold text-emerald-400 tabular-nums">
                +{macro.yieldCurve10Y2Y.toFixed(2)}%
              </span>
              <span className="text-[9px] text-slate-500 block">Normal Upward Slope</span>
            </div>

            <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
              <span className="text-[10px] text-slate-400 block uppercase">High Yield OAS Spread</span>
              <span className="text-sm sm:text-base font-extrabold text-cyan-300 tabular-nums">
                {macro.highYieldCreditSpread.toFixed(2)}%
              </span>
              <span className="text-[9px] text-slate-500 block">Tight Default Risk</span>
            </div>

            <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
              <span className="text-[10px] text-slate-400 block uppercase">10Y Real Rate (TIPS)</span>
              <span className="text-sm sm:text-base font-extrabold text-slate-200 tabular-nums">
                {macro.realInterestRate10Y.toFixed(2)}%
              </span>
              <span className="text-[9px] text-slate-500 block">Positive Real Return</span>
            </div>

            <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
              <span className="text-[10px] text-slate-400 block uppercase">Effective Fed Funds</span>
              <span className="text-sm sm:text-base font-extrabold text-amber-400 tabular-nums">
                {macro.fedFundsRate.toFixed(2)}%
              </span>
              <span className="text-[9px] text-slate-500 block">Terminal Pivot Regime</span>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: SEC FORM 4 CORPORATE INSIDERS */}
      {activeTab === "SEC_FORM4" && (
        <div className="space-y-3">
          {matchedInsider ? (
            <div className="bg-emerald-950/30 border border-emerald-800/80 rounded-xl p-3.5 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-base">🟢</span>
                  <strong className="text-emerald-300 font-bold text-sm">
                    {matchedInsider.insiderName} ({matchedInsider.insiderRole})
                  </strong>
                </div>
                <a
                  href={matchedInsider.secEdgarUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[10px] font-bold text-cyan-400 hover:underline flex items-center gap-1"
                >
                  <span>SEC EDGAR Form 4</span>
                  <span>↗</span>
                </a>
              </div>
              <p className="text-[11px] text-slate-200 font-sans">
                Purchased <strong>{matchedInsider.sharesTraded.toLocaleString()} shares</strong> on the open market at <strong>${matchedInsider.pricePerShare.toFixed(2)}</strong> (Total: <strong>${(matchedInsider.totalValueUsd / 1000000).toFixed(2)}M USD</strong>).
              </p>
            </div>
          ) : (
            <div className="bg-[#090d14] p-3 rounded-lg text-xs text-slate-400 text-center">
              No recent C-Suite open-market purchases filed on SEC EDGAR for {cleanSym} in the last 30 days.
            </div>
          )}

          <div className="overflow-x-auto pt-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#090d14] text-slate-400 text-[10px] uppercase border-b border-[#1b2434]">
                <tr>
                  <th className="py-2.5 px-3">Ticker</th>
                  <th className="py-2.5 px-3">Executive / Role</th>
                  <th className="py-2.5 px-3">Shares</th>
                  <th className="py-2.5 px-3">Total ($)</th>
                  <th className="py-2.5 px-3">Filing Date</th>
                  <th className="py-2.5 px-3 text-right">EDGAR Source</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1b2434] tabular-nums">
                {insiders.map((trade, idx) => (
                  <tr key={idx} className="hover:bg-[#151e2d] transition-colors">
                    <td className="py-2 px-3 font-bold text-cyan-400">{trade.ticker}</td>
                    <td className="py-2 px-3 text-slate-200">
                      <span className="font-semibold">{trade.insiderName}</span>
                      <span className="text-[10px] text-slate-400 block font-normal">{trade.insiderRole}</span>
                    </td>
                    <td className="py-2 px-3 text-slate-300">{trade.sharesTraded.toLocaleString()}</td>
                    <td className="py-2 px-3 text-emerald-400 font-bold">
                      ${(trade.totalValueUsd / 1000000).toFixed(2)}M
                    </td>
                    <td className="py-2 px-3 text-slate-400 text-[11px]">{trade.filingDate}</td>
                    <td className="py-2 px-3 text-right">
                      <a
                        href={trade.secEdgarUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] font-bold text-cyan-400 hover:text-cyan-300 underline"
                      >
                        Form 4 ↗
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}