"use client";

import { useState } from "react";
import { OptimalExecutionPlan } from "../lib/api";

interface InsightProvenanceModalProps {
  symbol: string;
  executionPlan?: OptimalExecutionPlan;
  userRole?: "DAY_TRADER" | "LONG_TERM";
}

export default function InsightProvenanceModal({
  symbol,
  executionPlan,
  userRole = "LONG_TERM",
}: InsightProvenanceModalProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!executionPlan) return null;

  const isDayTrader = userRole === "DAY_TRADER";

  return (
    <>
      {/* Trigger Button: Progressive Disclosure */}
      <button
        onClick={() => setIsOpen(true)}
        type="button"
        aria-label={`Inspect ${symbol} quantitative thesis and source provenance`}
        className="w-full flex items-center justify-between px-3.5 py-2 rounded-xl bg-[#090d14] hover:bg-[#162030] border border-cyan-500/40 hover:border-cyan-400 text-cyan-300 hover:text-white transition-all text-xs font-mono font-bold shadow cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
      >
        <div className="flex items-center space-x-2">
          <span>📜</span>
          <span>Inspect Deep Quantitative Thesis & Verified Source URLs</span>
        </div>
        <span className="text-cyan-400 group-hover:translate-x-0.5 transition-transform text-xs">
          Read More ↗
        </span>
      </button>

      {/* Fullscreen Provenance Modal Backdrop */}
      {isOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`${symbol} Insight Provenance and Statutory Sources`}
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-[110] bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 font-mono animate-fadeIn"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-[#0c1017] border border-[#2b3a52] rounded-2xl max-w-2xl w-full p-4 sm:p-6 shadow-2xl space-y-4 relative max-h-[90vh] flex flex-col text-slate-200"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-[#1e2a3c] pb-3">
              <div className="flex items-center space-x-2">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/50 flex items-center justify-center text-cyan-300 font-bold text-xs">
                  {symbol}
                </div>
                <div>
                  <h3 className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center gap-2">
                    <span>{symbol} Quantitative Thesis & Verified Sources</span>
                  </h3>
                  <span className="text-[10px] text-cyan-400 uppercase tracking-wider font-semibold">
                    Statutory Grounding & Algorithmic Provenance
                  </span>
                </div>
              </div>

              <button
                onClick={() => setIsOpen(false)}
                aria-label="Close modal"
                className="text-slate-400 hover:text-white text-xs px-2.5 py-1 bg-[#162030] hover:bg-[#223147] rounded-lg border border-[#2b394f] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none transition-colors cursor-pointer"
              >
                ESC ✕
              </button>
            </div>

            {/* Scrollable Body */}
            <div className="overflow-y-auto space-y-3.5 flex-1 pr-1 text-xs">
              {/* 1. Algorithmic Technical Engine */}
              <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1e2a3c] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-cyan-300 flex items-center gap-1.5">
                    <span>🧮</span>
                    <span>1. Algorithmic Execution Formula</span>
                  </span>
                  <span className="text-[10px] bg-[#162030] text-cyan-300 px-2 py-0.5 rounded border border-[#2b394f]">
                    Weight: 40%
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  {isDayTrader
                    ? "Linda Raschke 20-period Exponential Moving Average (20 EMA) pullback model with intraday 14-period Turtle ATR volatility sizing."
                    : "Mark Minervini 3-Stage Volatility Contraction Pattern (VCP) detecting institutional accumulation and supply absorption."}
                </p>
                <div className="bg-[#111722] p-2.5 rounded-lg border border-[#1e2433] text-[11px] font-mono text-slate-400">
                  <span className="text-cyan-400 font-bold block mb-1">Mathematical Constraints:</span>
                  <div>• Stop-Loss: Entry Floor - 1.8x ATR14 (${executionPlan.stop_loss.toFixed(2)})</div>
                  <div>• Target 1: Spot + 2.5x ATR14 (${executionPlan.take_profit_1.toFixed(2)})</div>
                  <div>• Reward-to-Risk: {executionPlan.risk_reward_ratio}:1.0 Minimum Asymmetry Gate</div>
                </div>
              </div>

              {/* 2. Statutory Smart Money & Regulatory Filings */}
              <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1e2a3c] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-amber-300 flex items-center gap-1.5">
                    <span>🏛️</span>
                    <span>2. Smart Money & Regulatory Disclosures</span>
                  </span>
                  <span className="text-[10px] bg-[#162030] text-amber-300 px-2 py-0.5 rounded border border-[#2b394f]">
                    Weight: 35%
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Aggregated from mandatory public filings under the 2012 Stop Trading on Congressional Knowledge (STOCK) Act (Public Law 112-105) and SEC EDGAR Form 4 / 13F institutional ownership filings.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                  <a
                    href="https://disclosures-clerk.house.gov/PublicDisclosure/FinancialDisclosure"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded-lg bg-[#111722] hover:bg-[#162030] border border-[#243044] hover:border-amber-400/60 text-amber-300 hover:text-white transition-colors"
                  >
                    <span>US House Financial Disclosures</span>
                    <span>↗</span>
                  </a>
                  <a
                    href="https://efdsearch.senate.gov/search/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded-lg bg-[#111722] hover:bg-[#162030] border border-[#243044] hover:border-amber-400/60 text-amber-300 hover:text-white transition-colors"
                  >
                    <span>US Senate EFD Portal</span>
                    <span>↗</span>
                  </a>
                </div>
              </div>

              {/* 3. Macroeconomic & Fundamentals Grounding */}
              <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1e2a3c] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-emerald-300 flex items-center gap-1.5">
                    <span>🌐</span>
                    <span>3. Macro Regime & SEC Financial Tape</span>
                  </span>
                  <span className="text-[10px] bg-[#162030] text-emerald-300 px-2 py-0.5 rounded border border-[#2b394f]">
                    Weight: 25%
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Macro regime calculations utilize live Federal Reserve Economic Data (FRED) for yield curve dynamics (10Y-2Y spread) and CPI inflation momentum. Financial factor health scores derive from 10-K/10-Q SEC EDGAR filings.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                  <a
                    href={`https://www.sec.gov/edgar/searchedgar/companysearch?companyName=${symbol}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded-lg bg-[#111722] hover:bg-[#162030] border border-[#243044] hover:border-emerald-400/60 text-emerald-300 hover:text-white transition-colors"
                  >
                    <span>SEC EDGAR {symbol} Filings</span>
                    <span>↗</span>
                  </a>
                  <a
                    href="https://fred.stlouisfed.org/series/T10Y2Y"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2 rounded-lg bg-[#111722] hover:bg-[#162030] border border-[#243044] hover:border-emerald-400/60 text-emerald-300 hover:text-white transition-colors"
                  >
                    <span>FRED 10Y-2Y Yield Curve</span>
                    <span>↗</span>
                  </a>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="pt-2 border-t border-[#1e2a3c] flex items-center justify-between text-[11px] text-slate-400">
              <span>All external links open in secure new tabs</span>
              <span>100% Grounded Provenance</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}