"use client";

import { useState, useRef, useEffect } from "react";

import { SHARED_WATCHLIST_ITEMS } from "../lib/constants";
import { getCanonicalAssetName } from "../lib/assetRegistry";
import { trackPositionSizer } from "../lib/matomo";

interface PositionSizerProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  entryPrice: number;
  stopLoss: number;
  takeProfit1?: number;
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
  const [allowFractional, setAllowFractional] = useState<boolean>(false);
  const [savedToast, setSavedToast] = useState<boolean>(false);

  const modalRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedElementRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    if (typeof document !== "undefined") {
      previouslyFocusedElementRef.current = document.activeElement as HTMLElement | null;
    }

    const timer = setTimeout(() => {
      if (modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length > 0) {
          focusable[0].focus();
        } else {
          modalRef.current.focus();
        }
      }
    }, 50);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }

      if (e.key === "Tab" && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;

        const firstElement = focusable[0];
        const lastElement = focusable[focusable.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("keydown", handleKeyDown);
      if (previouslyFocusedElementRef.current && typeof previouslyFocusedElementRef.current.focus === "function") {
        previouslyFocusedElementRef.current.focus();
      }
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const safeEntry = typeof entryPrice === "number" && !isNaN(entryPrice) && entryPrice > 0 ? entryPrice : 100;
  const safeStop = typeof stopLoss === "number" && !isNaN(stopLoss) && stopLoss > 0 ? stopLoss : safeEntry * 0.95;
  const safeTarget = typeof takeProfit1 === "number" && !isNaN(takeProfit1) && takeProfit1 > 0 ? takeProfit1 : safeEntry * 1.10;

  const isSetupInvalid = safeEntry <= safeStop;
  const isMicroAccount = accountSize < safeEntry;
  const isFractionalActive = allowFractional || isMicroAccount;

  const riskPerShare = isSetupInvalid ? 0 : Math.max(0.01, safeEntry - safeStop);
  const maxDollarRisk = accountSize * (riskPct / 100);
  const rawShares = isSetupInvalid
    ? 0
    : isFractionalActive
    ? Number((maxDollarRisk / (riskPerShare || 1)).toFixed(4))
    : Math.floor(maxDollarRisk / (riskPerShare || 1));
  const shares = isSetupInvalid ? 0 : rawShares;
  const totalAllocation = Number((shares * safeEntry).toFixed(2));
  const portfolioAllocPct = Number(((totalAllocation / (accountSize || 1)) * 100).toFixed(1));
  const actualDollarRisk = Number((shares * riskPerShare).toFixed(2));
  const projectedProfit = isSetupInvalid ? 0 : Number((shares * (safeTarget - safeEntry)).toFixed(2));

  // Half-Kelly calculation
  const b = isSetupInvalid ? 0 : Math.max(0.5, (safeTarget - safeEntry) / (riskPerShare || 1));
  const p = 0.55;
  const q = 0.45;
  const fullKelly = isSetupInvalid ? 0 : Math.max(0, (b * p - q) / (b || 1));
  const halfKellyPct = isSetupInvalid ? 0 : Math.min(25, Number(((fullKelly / 2) * 100).toFixed(1)));

  const matchedItem = SHARED_WATCHLIST_ITEMS.find((i) => i.symbol.toUpperCase() === symbol.toUpperCase());
  const authenticName = getCanonicalAssetName(symbol, matchedItem?.name);

  const handleSaveToPortfolio = () => {
    try {
      const raw = localStorage.getItem("FINANCE_USER_PORTFOLIO");
      let currentPositions = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(currentPositions)) currentPositions = [];

      const newPos = {
        symbol: symbol || "ASSET",
        name: authenticName,
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
      trackPositionSizer(symbol, riskPct, shares);
      setTimeout(() => setSavedToast(false), 3500);
      window.dispatchEvent(new Event("storage"));
    } catch (err) {
      console.warn("Failed to save to portfolio:", err);
    }
  };

  return (
    <div className="fixed inset-0 z-[1200] flex items-center justify-center p-2 sm:p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono overflow-y-auto">
      <div
        ref={modalRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="position-sizer-modal-title"
        className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-slate-100 font-sans max-h-[92vh] flex flex-col my-auto focus:outline-none"
      >
        {/* Fixed Header */}
        <div className="flex items-center justify-between p-3.5 sm:p-4 border-b border-[#1b2537] bg-[#0e1422] shrink-0">
          <div className="flex items-center space-x-2">
            <span className="text-xl">⚖️</span>
            <div>
              <h2 id="position-sizer-modal-title" className="text-sm sm:text-base font-black text-white tracking-tight">
                Institutional Position Sizer & Kelly Risk
              </h2>
              <p className="text-[10px] sm:text-[11px] text-slate-400">
                Calibrated for <span className="text-cyan-400 font-bold font-mono">{symbol}</span> @ ${safeEntry.toFixed(2)}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close Position Sizer"
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition-all text-sm cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-4 sm:p-5 space-y-3.5 sm:space-y-4 overflow-y-auto flex-1">
          {/* Setup Invalidation Warning Banner */}
          {isSetupInvalid && (
            <div className="p-3.5 bg-rose-950/80 border border-rose-600 rounded-xl text-rose-200 text-xs font-sans flex items-start gap-2.5 shadow-lg">
              <span className="text-xl leading-none">🚨</span>
              <div className="space-y-1">
                <strong className="font-bold text-rose-300 block text-sm">Setup Invalidated — Sizing Disabled:</strong>
                <p className="leading-relaxed text-slate-200">
                  Current entry price (${safeEntry.toFixed(2)}) is at or below the stop loss floor (${safeStop.toFixed(2)}). Risk geometry is invalid and capital deployment is blocked. Zero shares recommended.
                </p>
              </div>
            </div>
          )}

          {/* Stage 4 Capital Protection Advisory */}
          {isStage4 && !isSetupInvalid && (
            <div className="p-3 bg-amber-950/40 border border-amber-800/40 rounded-xl text-amber-300 text-xs font-sans flex items-start gap-2">
              <span className="text-base leading-none">⚠️</span>
              <div>
                <strong className="font-bold block">Stage 4 Markdown Caution:</strong>
                <span>Capital deployment not recommended until a constructive base forms above the 50-day SMA. Suggested risk defaulted to 0.25% pilot sizing or paper trading.</span>
              </div>
            </div>
          )}

          {/* Micro-Wallet Auto-Adaptation Banner */}
          {isMicroAccount && (
            <div className="p-2.5 bg-cyan-950/40 border border-cyan-700/50 rounded-xl text-cyan-200 text-xs font-sans flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-base">🎯</span>
                <div>
                  <strong className="font-bold text-cyan-300">Fractional Precision Active:</strong>
                  <span className="text-[11px] text-slate-300 block">Sized for ${accountSize.toLocaleString()} wallet. You are never priced out of high-conviction assets.</span>
                </div>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-900/60 border border-cyan-600/60 text-cyan-200 font-mono font-bold shrink-0">
                0.0001 Precision
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[11px] text-slate-400 font-bold block mb-1">
                Account Equity ($)
              </label>
              <div className="relative">
                <span className="absolute left-3 top-2 text-xs text-slate-500 font-bold">$</span>
                <input
                  type="number"
                  min="1"
                  step="10"
                  value={accountSize}
                  onChange={(e) => {
                    const val = Math.max(1, Number(e.target.value));
                    setAccountSize(val);
                    if (typeof window !== "undefined") {
                      localStorage.setItem("FINANCE_USER_ACCOUNT_SIZE", String(val));
                    }
                  }}
                  className="w-full bg-[#070b13] border border-[#24334b] rounded-lg pl-7 pr-3 py-1.5 text-xs text-white font-bold focus:border-cyan-500 focus:outline-none"
                />
              </div>

              {/* Quick Wallet Presets */}
              <div className="flex flex-wrap items-center gap-1 mt-1.5">
                {[50, 100, 500, 2500, 10000, 25000].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => {
                      setAccountSize(preset);
                      if (typeof window !== "undefined") {
                        localStorage.setItem("FINANCE_USER_ACCOUNT_SIZE", String(preset));
                      }
                    }}
                    className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border transition-all cursor-pointer ${
                      accountSize === preset
                        ? "bg-cyan-600 border-cyan-400 text-white"
                        : "bg-[#0c121e] border-[#1f2c42] text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    ${preset >= 1000 ? `${preset / 1000}k` : preset}
                  </button>
                ))}
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
                  min="0.1"
                  max="10"
                  value={riskPct}
                  onChange={(e) => setRiskPct(Math.max(0.1, Math.min(10, Number(e.target.value))))}
                  className="w-full bg-[#070b13] border border-[#24334b] rounded-lg px-3 py-1.5 text-xs text-white font-bold focus:border-cyan-500 focus:outline-none"
                />
                <span className="absolute right-3 top-2 text-xs text-slate-500 font-bold">%</span>
              </div>

              {/* Risk Presets */}
              <div className="flex flex-wrap items-center gap-1 mt-1.5">
                {[0.5, 1.0, 1.5, 2.0].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setRiskPct(preset)}
                    className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border transition-all cursor-pointer ${
                      riskPct === preset
                        ? "bg-cyan-600 border-cyan-400 text-white"
                        : "bg-[#0c121e] border-[#1f2c42] text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {preset}%
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Mode Selector */}
          <div className="flex items-center justify-between pt-1 border-t border-[#172235]">
            <span className="text-[10px] text-slate-400 font-mono">
              Risk Budget: <strong className="text-cyan-300 font-bold">${maxDollarRisk.toFixed(2)}</strong>
            </span>

            <button
              type="button"
              onClick={() => setAllowFractional(!allowFractional)}
              className={`px-2 py-1 rounded text-[10px] font-bold border transition-all cursor-pointer flex items-center gap-1 ${
                isFractionalActive
                  ? "bg-cyan-950 border-cyan-600 text-cyan-300"
                  : "bg-[#0c121e] border-[#1f2c42] text-slate-400 hover:text-slate-200"
              }`}
            >
              <span>{isFractionalActive ? "🔢 Fractional Units" : "📦 Whole Shares"}</span>
            </button>
          </div>

          {/* Sizing Results Card */}
          <div className="bg-[#070c16] border border-[#1b273b] rounded-xl p-3.5 sm:p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-[#141e30] pb-2">
              <span className="text-xs text-slate-400 font-bold">Recommended Position</span>
              <span className="text-lg sm:text-xl font-black text-cyan-400 tabular-nums">
                {shares} <span className="text-xs text-slate-400 font-semibold">{shares === 1 ? "Share" : "Shares"}</span>
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2.5 sm:gap-3 text-xs">
              <div className="bg-[#0b101b] p-2.5 rounded-lg border border-[#172235]">
                <span className="text-[10px] text-slate-500 block">CAPITAL ALLOCATED</span>
                <span className="font-bold text-white tabular-nums">${totalAllocation.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
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
            {isSetupInvalid ? (
              <span className="text-rose-400 font-bold">
                Setup Invalidated. Entry price (${safeEntry.toFixed(2)}) is at or below stop loss floor (${safeStop.toFixed(2)}). Position sizing is disabled and zero orders are permitted.
              </span>
            ) : shares > 0 ? (
              <>
                Buy <span className="text-white font-bold">{shares} {shares === 1 ? "share" : "shares"}</span> of {symbol || "asset"} at ${safeEntry.toFixed(2)}. Place GTC Stop-Loss at ${safeStop.toFixed(2)}. Risk is locked at exactly ${actualDollarRisk.toFixed(2)} ({riskPct}% equity constraint).
              </>
            ) : (
              <>
                Budget of ${maxDollarRisk.toFixed(2)} is less than 1 whole share risk. Switch to <span className="text-cyan-300 font-bold">🔢 Fractional Units</span> to invest with fractional precision.
              </>
            )}
          </div>

          {/* Success Toast */}
          {savedToast && (
            <div className="bg-emerald-950/90 border border-emerald-600 text-emerald-300 p-2.5 rounded-xl text-xs flex items-center justify-between font-bold animate-fade-in">
              <span>✅ Added {shares} shares of {symbol} to your private portfolio!</span>
              <a href="/portfolio" className="underline hover:text-white">View Portfolio →</a>
            </div>
          )}
        </div>

        {/* Fixed Footer with 1-Click Save to Portfolio */}
        <div className="p-3.5 sm:p-4 border-t border-[#1b2537] bg-[#0e1422] flex items-center justify-between gap-3 shrink-0">
          <button
            type="button"
            disabled={isSetupInvalid || shares <= 0}
            onClick={handleSaveToPortfolio}
            className={`px-3.5 sm:px-4 py-2 border rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 shadow ${
              isSetupInvalid || shares <= 0
                ? "bg-[#0e1420] border-[#182334] text-slate-600 cursor-not-allowed"
                : "bg-[#172338] hover:bg-[#20314f] border border-[#2b3f63] text-cyan-300 hover:text-white active:scale-95 cursor-pointer"
            }`}
          >
            <span>💼</span>
            <span>{isSetupInvalid ? "Sizing Disabled (Invalid Setup)" : "Save to My Portfolio"}</span>
          </button>

          <button
            type="button"
            onClick={onClose}
            className="px-4 sm:px-5 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold transition-all shadow active:scale-95 cursor-pointer"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
