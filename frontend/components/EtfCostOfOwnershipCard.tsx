"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Info, Calculator, Layers, AlertCircle } from "lucide-react";

export interface EtfCostOfOwnershipCardProps {
  symbol: string;
  expenseRatio?: number | null; // Given as percentage, e.g. 0.09 for 0.09% TER
  className?: string;
}

export type FeeTierCategory = "Very Low" | "Low" | "Moderate" | "High";

export function getFeeTierCategory(ratioPct: number): FeeTierCategory {
  if (ratioPct < 0.15) return "Very Low";
  if (ratioPct < 0.35) return "Low";
  if (ratioPct < 0.70) return "Moderate";
  return "High";
}

function getCategoryTheme(category: FeeTierCategory) {
  switch (category) {
    case "Very Low":
      return {
        badge: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
        dot: "bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]",
        accent: "text-emerald-400",
        trackHighlight: "bg-emerald-500",
      };
    case "Low":
      return {
        badge: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
        dot: "bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.6)]",
        accent: "text-cyan-400",
        trackHighlight: "bg-cyan-500",
      };
    case "Moderate":
      return {
        badge: "bg-amber-500/10 text-amber-400 border-amber-500/30",
        dot: "bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.6)]",
        accent: "text-amber-400",
        trackHighlight: "bg-amber-500",
      };
    case "High":
      return {
        badge: "bg-rose-500/10 text-rose-400 border-rose-500/30",
        dot: "bg-rose-400 shadow-[0_0_8px_rgba(244,63,94,0.6)]",
        accent: "text-rose-400",
        trackHighlight: "bg-rose-500",
      };
  }
}

export default function EtfCostOfOwnershipCard({
  symbol,
  expenseRatio,
  className = "",
}: EtfCostOfOwnershipCardProps) {
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    }
    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  const cleanSymbol = useMemo(() => {
    return (symbol || "").toUpperCase().replace(/.*:/, "").trim();
  }, [symbol]);

  const hasValidRatio = typeof expenseRatio === "number" && !isNaN(expenseRatio) && expenseRatio >= 0;

  // Safe missing-data fallback
  if (!hasValidRatio) {
    return (
      <div className={`bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-3 font-sans text-slate-300 ${className}`}>
        <div className="flex items-center justify-between border-b border-[#243044]/60 pb-3">
          <div className="flex items-center space-x-2 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <h3 className="text-sm font-bold text-slate-200">
              💰 {isPlain ? "Cost of Ownership" : "Total Expense Analysis"} • {cleanSymbol || "ETF"}
            </h3>
          </div>
          <span className="text-caption-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
            TER: UNVERIFIED
          </span>
        </div>
        <p className="text-body-ui text-slate-400 leading-relaxed">
          Certified Total Expense Ratio (TER) is not currently available in the runtime feed for {cleanSymbol}. Check the fund prospectus or official sponsor disclosures for confirmed fee schedules.
        </p>
        <div className="text-caption-mono text-slate-500 bg-[#090d14] p-2.5 rounded border border-[#243044]/60 flex items-center space-x-2">
          <AlertCircle className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
          <span>Status: NO_RUNTIME_TER_METADATA • Under ARX quantitative integrity rules, zero synthetic fees are imputed.</span>
        </div>
      </div>
    );
  }

  const ratio = expenseRatio;
  const category = getFeeTierCategory(ratio);
  const theme = getCategoryTheme(category);

  // Direct nominal fund cost calculations (undiscounted direct fee deductions)
  const principal = 10000;
  const annualCost = (ratio / 100) * principal; // Direct $ per $10,000 invested per year
  const tenYearCost = annualCost * 10; // Nominal 10-year direct fee sum without assumed return compounding
  const basisPoints = (ratio * 100).toFixed(1);

  // Spectrum positioning (0% to 1.00%)
  const clampedRatio = Math.min(Math.max(ratio, 0), 1.0);
  const spectrumPct = (clampedRatio / 1.0) * 100;

  return (
    <div className={`bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-5 font-sans ${className}`}>
      {/* 1. Header */}
      <div className="flex items-center justify-between border-b border-[#243044]/60 pb-3">
        <div className="flex items-center space-x-2">
          <h3 className="text-sm sm:text-base font-bold text-white tracking-wide flex items-center gap-1.5">
            <span>💰</span>
            <span>{isPlain ? "Cost of Ownership" : "Total Expense Analysis"}</span>
          </h3>
          <span className="text-caption-mono text-slate-400 font-normal">({cleanSymbol})</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`px-2.5 py-0.5 rounded-full text-caption-mono font-medium border ${theme.badge}`}>
            {category}
          </span>
        </div>
      </div>

      {/* 2. Expense Ratio Display */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 bg-[#0b101b] p-4 rounded-lg border border-[#243044]/80">
        <div>
          <div className="text-caption-mono text-slate-400 uppercase tracking-wider">
            {isPlain ? "Annual Expense Ratio" : "Gross Total Expense Ratio (TER)"}
          </div>
          <div className="flex items-baseline space-x-3 mt-1">
            <span className="text-3xl sm:text-4xl font-mono font-bold text-white tracking-tight">
              {ratio.toFixed(2)}%
            </span>
            <span className="text-data-mono-sm text-cyan-400 font-mono font-semibold">
              {basisPoints} bps
            </span>
          </div>
        </div>
        <div className="text-caption-mono text-slate-400">
          <span className="text-slate-500">Benchmark Principal: </span>
          <span className="text-slate-200 font-mono font-semibold">$10,000 USD</span>
        </div>
      </div>

      {/* 3. Direct Cost Projections Grid (2 columns: Annual and 10-Year Direct Nominal Fees) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Column 1: Annual Direct Cost */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044]/60 flex flex-col justify-between space-y-1">
          <div className="text-caption-mono text-slate-400">
            {isPlain ? "Annual Direct Cost / $10k" : "1-Yr Direct Fee (P₀ × TER)"}
          </div>
          <div className="text-2xl font-mono font-bold text-white tracking-tight">
            ${annualCost.toFixed(0)}
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            {isPlain ? "per $10,000 invested" : `$${annualCost.toFixed(2)} / year direct deduction`}
          </div>
        </div>

        {/* Column 2: 10-Year Nominal Cost */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044]/60 flex flex-col justify-between space-y-1">
          <div className="text-caption-mono text-slate-400">
            {isPlain ? "10-Year Direct Nominal Fee" : "10-Yr Undiscounted Sum (10 × Fee)"}
          </div>
          <div className="text-2xl font-mono font-bold text-white tracking-tight">
            ${tenYearCost.toFixed(0)}
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            {isPlain ? "cumulative direct fee" : `10 × $${annualCost.toFixed(2)} = $${tenYearCost.toFixed(2)}`}
          </div>
        </div>
      </div>

      {/* 4. Plain English insight / PRO_QUANT Details */}
      {isPlain ? (
        <div className="bg-[#0b101b] border border-[#243044]/80 rounded-lg p-3.5 text-body-ui text-slate-300 leading-relaxed flex items-start space-x-3">
          <Info className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
          <div>
            At {ratio.toFixed(2)}%, {cleanSymbol} incurs approximately ${annualCost.toFixed(0)} per year in direct fund deductions for every $10,000 invested (${tenYearCost.toFixed(0)} cumulative over 10 years without compounding).
          </div>
        </div>
      ) : (
        <div className="bg-[#0b101b] border border-[#243044]/80 rounded-lg p-3.5 space-y-2.5">
          <div className="flex items-center justify-between text-caption-mono">
            <span className="text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Calculator className="w-3.5 h-3.5 text-cyan-400" />
              <span>Total Expense Ratio Accounting:</span>
            </span>
            <span className="text-cyan-400 font-mono font-bold">
              {basisPoints} bps ({ratio.toFixed(2)}%)
            </span>
          </div>

          <div className="text-caption-mono font-mono text-slate-300 bg-[#090d14] p-3 rounded border border-[#243044]/60 space-y-1.5">
            <div className="flex items-center justify-between text-slate-400 text-[11px] font-sans border-b border-[#243044]/40 pb-1">
              <span>DIRECT FUND FEE SUMMARY</span>
              <span className="text-[10px] text-slate-500 font-mono">NOMINAL ACCOUNTING</span>
            </div>
            <div className="text-slate-100 font-semibold py-0.5">
              Annual Fee = P₀ × TER • 10-Yr Fee = 10 × (P₀ × TER)
            </div>
            <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
              Parameters: <code className="text-slate-200 font-mono">P₀ = $10,000</code>,{" "}
              <code className="text-slate-200 font-mono">TER = {(ratio / 100).toFixed(4)}</code> ({basisPoints} bps).
              Direct fee deductions assume constant portfolio balance; no hypothetical return distributions or compounding drag are imputed.
            </div>
          </div>
        </div>
      )}

      {/* 5. Fee Tier Comparison Bar */}
      <div className="space-y-2.5 bg-[#090d14] p-3.5 rounded-lg border border-[#243044]/60">
        <div className="flex items-center justify-between text-caption-mono">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            <span>{isPlain ? "Fee Tier Comparison (0.00% – 1.00%)" : "TER Expense Spectrum Positioning"}</span>
          </span>
          <span className="font-mono text-slate-300">
            {cleanSymbol}: <span className={`font-semibold ${theme.accent}`}>{ratio.toFixed(2)}%</span>
          </span>
        </div>

        {/* Visual Bar Track */}
        <div className="relative pt-6 pb-2">
          {/* Floating Marker / Pin above the track */}
          <div
            className="absolute top-0 -translate-x-1/2 z-20 flex flex-col items-center pointer-events-none transition-all duration-300"
            style={{ left: `${spectrumPct}%` }}
          >
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-[#162032] border border-[#243044] text-white shadow-md">
              {cleanSymbol} {ratio.toFixed(2)}%
            </span>
            <span className="w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[4px] border-t-[#243044]" />
          </div>

          {/* Spectrum Bar Segments */}
          <div className="relative h-2.5 w-full rounded-full bg-[#0b101b] border border-[#243044] overflow-hidden flex">
            {/* 0% - 15%: Very Low (Emerald) */}
            <div className="w-[15%] h-full bg-emerald-500/25 border-r border-[#243044]/50" title="Very Low (0% - 0.15%)" />
            {/* 15% - 35%: Low (Cyan) */}
            <div className="w-[20%] h-full bg-cyan-500/20 border-r border-[#243044]/50" title="Low (0.15% - 0.35%)" />
            {/* 35% - 70%: Moderate (Amber) */}
            <div className="w-[35%] h-full bg-amber-500/20 border-r border-[#243044]/50" title="Moderate (0.35% - 0.70%)" />
            {/* 70% - 100%: High (Rose) */}
            <div className="w-[30%] h-full bg-rose-500/25" title="High (0.70% - 1.00%+)" />
          </div>

          {/* Indicator Dot on the bar */}
          <div
            className="absolute top-[29px] -translate-x-1/2 -translate-y-1/2 z-10 pointer-events-none transition-all duration-300"
            style={{ left: `${spectrumPct}%` }}
          >
            <div className={`w-3.5 h-3.5 rounded-full border-2 border-[#111722] ${theme.dot}`} />
          </div>
        </div>

        {/* Spectrum Axis Labels */}
        <div className="flex justify-between text-[10px] font-mono text-slate-500 pt-0.5">
          <span>0.00% (Ultra Low)</span>
          <span>0.15%</span>
          <span>0.35%</span>
          <span>0.70%</span>
          <span>1.00%+ (High)</span>
        </div>
      </div>
    </div>
  );
}
