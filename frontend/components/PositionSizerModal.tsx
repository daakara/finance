"use client";

import { useState } from "react";

interface PositionSizerProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  entryPrice: number;
  stopLoss: number;
  takeProfit1: number;
  riskRewardRatio?: number;
  isStage4?: boolean;
}

export default function PositionSizerModal({
  isOpen,
  onClose,
  symbol,
  entryPrice = 100,
  stopLoss,
  takeProfit1,
  riskRewardRatio = 2.5,
  isStage4 = false,
}: PositionSizerProps) {
  const [accountSize, setAccountSize] = useState<number>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("FINANCE_USER_ACCOUNT_SIZE");
      if (saved) {
        const parsed = Number(saved);
        if (!isNaN(parsed) && parsed > 0) return parsed;
      }
    }
    return 25000;
  });
  const [riskPct, setRiskPct] = useState<number>(isStage4 ? 0.25 : 1.0);
  const [savedToast, setSavedToast] = useState<boolean>(false);

  if (!isOpen) return null;

  const safeEntry = typeof entryPrice === "number" && !isNaN(entryPrice) && entryPrice > 0 ? entryPrice : 100;
  const safeStop = typeof stopLoss === "number" && !isNaN(stopLoss) && stopLoss > 0 ? stopLoss : safeEntry * 0.95;
  const safeTarget = typeof takeProfit1 === "number" && !isNaN(takeProfit1) && takeProfit1 > 0 ? takeProfit1 : safeEntry * 1.10;

  const riskPerShare = Math.max(0.01, safeEntry - safeStop);
  const maxDollarRisk = accountSize * (riskPct / 100);
  const shares = Math.max(1, Math.floor(maxDollarRisk / riskPerShare));
  const totalAllocation = Number((shares * safeEntry).toFixed(2));
  const portfolioAllocPct = Number(((totalAllocation / (accountSize || 1)) * 100).toFixed(1));
  const actualDollarRisk = Number((shares * riskPerShare).toFixed(2));
  const projectedProfit = Number((shares * (safeTarget - safeEntry)).toFixed(2));

  // Half-Kelly calculation
  const b = Math.max(0.5, (safeTarget - safeEntry) / riskPerShare);
  const p = 0.55;
  const q = 0.45;
  const fullKelly = Math.max(0, (b * p - q) / b);
  const halfKellyPct = Math.min(25, Number(((fullKelly / 2) * 100).toFixed(1)));

  const handleSaveToPortfolio = () => {
    try {
      const raw = localStorage.getItem("FINANCE_USER_PORTFOLIO");
      let currentPositions = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(currentPositions)) currentPositions = [];

      const newPos = {
        symbol: symbol || "ASSET",
        name: `${symbol || "ASSET"} Corporation`,
        shares,
        entryPrice: safeEntry,
        currentPrice: safeEntry,
        targetPrice: safeTarget,
        stopLossPrice: safeStop,
        addedAt: new Date().toISOString().split("T")[0],
        assetType: "Stock",
      };

      const existingIndex = currentPositions.findIndex((p: any) => p.symbol === symbol);
      let updated;
      if (existingIndex >= 0) {
        updated = [...currentPositions];
        updated[existingIndex] = newPos;
      } else {
        updated = [newPos, ...currentPositions];
      }

      localStorage.setItem("FINANCE_USER_PORTFOLIO", JSON.stringify(updated));
      setSavedToast(true);
      setTimeout(() => setSavedToast(false), 3500);
      window.dispatchEvent(new Event("storage"));
    } catch (err) {
      console.warn("Failed to save to portfolio:", err);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono">
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-slate-100 font-sans">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422]">
          <div className="flex items-center space-x-2">
            <span className="text-xl">⚖️</span>
            <div>
              <h2 className="text-base font-black text-white tracking-tight">
                Institutional Position Sizer & Kelly Risk
              </h2>
              <p className="text-[11px] text-slate-400">
                Calibrated for <span className="text-cyan-400 font-bold font-mono">{symbol}</span> @ ${safeEntry.toFixed(2)}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-all text-sm"
          >
            ✕
          </button>
        </div>

        {/* Stage 4 Capital Protection Advisory */}
        {isStage4 && (
          <div className="p-3 bg-amber-950/40 border-b border-amber-800/40 text-amber-300 text-xs font-sans flex items-start gap-2">
            <span className="text-base leading-none">⚠️</span>
            <div>
              <strong className="font-bold block">Stage 4 Markdown Caution:</strong>
              <span>Capital deployment not recommended until a constructive base forms above the 50-day SMA. Suggested risk defaulted to 0.25% pilot sizing or paper trading.</span>
            </div>
          </div>
        )}

        {/* Inputs Body */}
        <div className="p-5 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[11px] text-slate-400 font-bold block mb-1">
                Account Equity ($)
              </label>
              <div className="relative">
                <span className="absolute left-3 top-2 text-xs text-slate-500 font-bold">$</span>
                <input
                  type="number"
                  value={accountSize}
                  onChange={(e) => {
                    const val = Math.max(100, Number(e.target.value));
                    setAccountSize(val);
                    if (typeof window !== "undefined") {
                      localStorage.setItem("FINANCE_USER_ACCOUNT_SIZE", String(val));
                    }
                  }}
                  className="w-full bg-[#070b13] border border-[#24334b] rounded-lg pl-7 pr-3 py-1.5 text-xs text-white font-bold focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="text-[11px] text-slate-400 font-bold block mb-1">
                Max Risk Per Trade (%)
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="0.25"
                  min="0.25"
                  max="10"
                  value={riskPct}
                  onChange={(e) => setRiskPct(Math.max(0.1, Math.min(10, Number(e.target.value))))}
                  className="w-full bg-[#070b13] border border-[#24334b] rounded-lg px-3 py-1.5 text-xs text-white font-bold focus:border-cyan-500 focus:outline-none"
                />
                <span className="absolute right-3 top-2 text-xs text-slate-500 font-bold">%</span>
              </div>
            </div>
          </div>

          {/* Quick Risk Buttons */}
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-slate-500 font-bold mr-1">Risk Presets:</span>
            {[0.5, 1.0, 1.5, 2.0].map((preset) => (
              <button
                key={preset}
                onClick={() => setRiskPct(preset)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-all ${
                  riskPct === preset
                    ? "bg-cyan-600 border-cyan-400 text-white"
                    : "bg-[#0c121e] border-[#1f2c42] text-slate-400 hover:text-slate-200"
                }`}
              >
                {preset}% (${((accountSize * preset) / 100).toFixed(0)})
              </button>
            ))}
          </div>

          {/* Sizing Results Card */}
          <div className="bg-[#070c16] border border-[#1b273b] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-[#141e30] pb-2">
              <span className="text-xs text-slate-400 font-bold">Recommended Position</span>
              <span className="text-xl font-black text-cyan-400 tabular-nums">
                {shares} <span className="text-xs text-slate-400 font-semibold">Shares</span>
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#172235]">
                <span className="text-[10px] text-slate-500 block">CAPITAL ALLOCATED</span>
                <span className="font-bold text-white tabular-nums">${totalAllocation.toLocaleString()}</span>
                <span className="text-[10px] text-slate-400 block mt-0.5">({portfolioAllocPct}% of portfolio)</span>
              </div>

              <div className="bg-[#140e11] p-2.5 rounded-lg border border-rose-950/60">
                <span className="text-[10px] text-rose-400 block font-semibold">HARD MAX LOSS</span>
                <span className="font-bold text-rose-300 tabular-nums">-${actualDollarRisk.toFixed(2)}</span>
                <span className="text-[10px] text-rose-500/80 block mt-0.5">at ${safeStop.toFixed(2)} stop</span>
              </div>

              <div className="bg-[#0a1414] p-2.5 rounded-lg border border-emerald-950/60">
                <span className="text-[10px] text-emerald-400 block font-semibold">TARGET PROFIT (TP1)</span>
                <span className="font-bold text-emerald-300 tabular-nums">+${projectedProfit.toFixed(2)}</span>
                <span className="text-[10px] text-emerald-500/80 block mt-0.5">at ${safeTarget.toFixed(2)} (+{(((safeTarget - safeEntry)/safeEntry)*100).toFixed(1)}%)</span>
              </div>

              <div className="bg-[#12110c] p-2.5 rounded-lg border border-amber-950/60">
                <span className="text-[10px] text-amber-400 block font-semibold">FRACTIONAL KELLY</span>
                <span className="font-bold text-amber-300 tabular-nums">{halfKellyPct}% Optimal</span>
                <span className="text-[10px] text-amber-500/80 block mt-0.5">Half-Kelly Cap: 25% max</span>
              </div>
            </div>
          </div>

          {/* Action Guidance */}
          <div className="text-[11px] text-slate-400 bg-[#0c121d] p-3 rounded-lg border border-[#1b2639] leading-relaxed">
            <span className="text-cyan-400 font-bold">Execution Plan: </span>
            Buy <span className="text-white font-bold">{shares} shares</span> of {symbol || "asset"} at ${safeEntry.toFixed(2)}. Place GTC Stop-Loss at ${safeStop.toFixed(2)}. Risk is locked at exactly ${actualDollarRisk.toFixed(2)} ({riskPct}% equity constraint).
          </div>

          {/* Success Toast */}
          {savedToast && (
            <div className="bg-emerald-950/90 border border-emerald-600 text-emerald-300 p-2.5 rounded-xl text-xs flex items-center justify-between font-bold animate-fade-in">
              <span>✅ Added {shares} shares of {symbol} to your private portfolio!</span>
              <a href="/portfolio" className="underline hover:text-white">View Portfolio →</a>
            </div>
          )}
        </div>

        {/* Footer with 1-Click Save to Portfolio */}
        <div className="p-4 border-t border-[#1b2537] bg-[#0e1422] flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={handleSaveToPortfolio}
            className="px-4 py-2 bg-[#172338] hover:bg-[#20314f] border border-[#2b3f63] text-cyan-300 hover:text-white rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 active:scale-95 shadow"
          >
            <span>💼</span>
            <span>Save to My Portfolio</span>
          </button>

          <button
            onClick={onClose}
            className="px-5 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold transition-all shadow active:scale-95"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
