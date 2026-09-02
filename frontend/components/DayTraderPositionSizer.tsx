"use client";

import { useState } from "react";
import { AnalyticsResponse } from "../lib/api";
import { loadPortfolioPositions, savePortfolioPositions, PortfolioPosition } from "../lib/portfolio";
import { getCanonicalAssetName } from "../lib/assetRegistry";

interface DayTraderPositionSizerProps {
  symbol: string;
  data: AnalyticsResponse;
}

export default function DayTraderPositionSizer({ symbol, data }: DayTraderPositionSizerProps) {
  const [accountSize, setAccountSize] = useState<number>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("FINANCE_USER_ACCOUNT_SIZE");
      if (saved) {
        const parsed = Number(saved);
        if (!isNaN(parsed) && parsed >= 1000) return parsed;
      }
    }
    return 25000;
  });
  const [riskPct, setRiskPct] = useState<number>(1.0);
  const [tradeDirection, setTradeDirection] = useState<"LONG" | "SHORT">("LONG");
  const [accountType, setAccountType] = useState<"CASH" | "MARGIN">("CASH");
  const [allowFractional, setAllowFractional] = useState<boolean>(false);
  const [addedFeedback, setAddedFeedback] = useState<boolean>(false);

  const handleAccountSizeChange = (val: number) => {
    setAccountSize(val);
    if (typeof window !== "undefined") {
      localStorage.setItem("FINANCE_USER_ACCOUNT_SIZE", val.toString());
    }
  };

  const currentPrice = data.currentPrice || 100.0;
  const metrics = data.analytics?.advanced_metrics || {};
  const technicals = data.technicals || { vwap: currentPrice, rsi_14: 55.0, ema_20: currentPrice, atr_14: currentPrice * 0.015 };
  const rsi = technicals.rsi_14 ?? 50.0;

  // Calculate Dollar Risk Budget
  const dollarRisk = accountSize * (riskPct / 100);

  // Statistical stop distance derived from Cornish-Fisher Modified VaR 95% and ATR 14
  const mVaR95Pct = Math.abs(metrics.Modified_VaR_95 || 2.5);
  const stopDistancePct = Math.max(0.8, Math.min(6.0, mVaR95Pct * 0.65));
  const stopDistanceDollar = currentPrice * (stopDistancePct / 100);

  // Unconstrained statistical units based on volatility stop
  const rawVolatilityUnits = stopDistanceDollar > 0
    ? allowFractional
      ? Number((dollarRisk / stopDistanceDollar).toFixed(3))
      : Math.max(1, Math.floor(dollarRisk / stopDistanceDollar))
    : 1;

  // Maximum units constrained by 100% cash buying power
  const maxCashUnits = allowFractional
    ? Number((accountSize / currentPrice).toFixed(3))
    : Math.max(1, Math.floor(accountSize / currentPrice));

  // In CASH mode, clamp position units to available cash equity
  const positionUnits = accountType === "CASH" ? Math.min(maxCashUnits, rawVolatilityUnits) : rawVolatilityUnits;
  const isCappedByCash = accountType === "CASH" && rawVolatilityUnits > maxCashUnits;

  const totalPositionValue = Number((positionUnits * currentPrice).toFixed(2));
  const leverageRatio = accountSize > 0 ? (totalPositionValue / accountSize).toFixed(1) : "1.0";

  // Price targets based on Risk-to-Reward multiples
  const stopPrice = tradeDirection === "LONG"
    ? Math.max(0.01, currentPrice - stopDistanceDollar)
    : currentPrice + stopDistanceDollar;

  const target15 = tradeDirection === "LONG"
    ? currentPrice + stopDistanceDollar * 1.5
    : Math.max(0.01, currentPrice - stopDistanceDollar * 1.5);

  const target20 = tradeDirection === "LONG"
    ? currentPrice + stopDistanceDollar * 2.0
    : Math.max(0.01, currentPrice - stopDistanceDollar * 2.0);

  const target30 = tradeDirection === "LONG"
    ? currentPrice + stopDistanceDollar * 3.0
    : Math.max(0.01, currentPrice - stopDistanceDollar * 3.0);

    const handleSaveToPortfolio = () => {
    const symKey = symbol.toUpperCase().replace("-USD", "");
    const existing = loadPortfolioPositions();
    const newPos: PortfolioPosition = {
      symbol: symKey,
      name: `${getCanonicalAssetName(symKey)} Intraday Trade`,
      shares: positionUnits,
      entryPrice: currentPrice,
      currentPrice: currentPrice,
      stopLossPrice: stopPrice,
      targetPrice: target20,
      addedAt: new Date().toISOString().split("T")[0],
      assetType: "Stock",
    };
    // Replace if exists, or prepend
    const updated = [newPos, ...existing.filter((p) => p.symbol !== symKey)];
    savePortfolioPositions(updated);
    setAddedFeedback(true);
    setTimeout(() => setAddedFeedback(false), 3000);
  };

  return (
    <section aria-labelledby="position-sizer-title" className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span aria-hidden="true" className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse motion-reduce:animate-none"></span>
            <h3 id="position-sizer-title" className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <svg aria-hidden="true" className="w-4 h-4 text-amber-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="22" y1="12" x2="18" y2="12" />
                <line x1="6" y1="12" x2="2" y2="12" />
                <line x1="12" y1="6" x2="12" y2="2" />
                <line x1="12" y1="22" x2="12" y2="18" />
              </svg>
              <span>Rule #1: Protect The Castle ({symbol} Position Sizer)</span>
            </h3>
          </div>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
            Tells you exactly how many shares to buy so a single bad trade never blows up your account.
          </p>
        </div>

        {/* Direction & Account Type Toggles */}
        <div className="flex items-center gap-2">
          {/* Account Mode Toggle */}
          <div role="radiogroup" aria-label="Account leverage mode" className="flex items-center bg-[#090d14] p-1 rounded-lg border border-[#243044]">
            <button
              type="button"
              role="radio"
              aria-checked={accountType === "CASH"}
              onClick={() => setAccountType("CASH")}
              className={`px-2.5 py-1 text-[11px] font-bold rounded transition-colors active:scale-[0.96] ${
                accountType === "CASH"
                  ? "bg-cyan-500 text-slate-950 font-extrabold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              💵 Cash (1.0x)
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={accountType === "MARGIN"}
              onClick={() => setAccountType("MARGIN")}
              className={`px-2.5 py-1 text-[11px] font-bold rounded transition-colors active:scale-[0.96] ${
                accountType === "MARGIN"
                  ? "bg-purple-500 text-white font-extrabold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              ⚡ Margin (PDT)
            </button>
          </div>

          {/* Fractional Units Toggle */}
          <div className="flex items-center bg-[#090d14] p-1 rounded-lg border border-[#243044]">
            <button
              type="button"
              onClick={() => setAllowFractional(!allowFractional)}
              className={`px-2 py-1 text-[11px] font-bold rounded transition-colors active:scale-[0.96] flex items-center gap-1 ${
                allowFractional
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-700"
                  : "text-slate-500 hover:text-slate-300"
              }`}
              title="Toggle fractional share units for high-priced stocks"
            >
              <span>{allowFractional ? "🔢 Fractional Units" : "📦 Whole Shares"}</span>
            </button>
          </div>

          {/* Long / Short Toggle */}
          <div role="radiogroup" aria-label="Trade direction" className="flex items-center bg-[#090d14] p-1 rounded-lg border border-[#243044]">
            <button
              role="radio"
              aria-checked={tradeDirection === "LONG"}
              onClick={() => setTradeDirection("LONG")}
              className={`px-3 py-1.5 min-h-[32px] text-xs font-bold rounded-md transition-colors active:scale-[0.96] transition-transform duration-100 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:outline-none ${
                tradeDirection === "LONG"
                  ? "bg-emerald-500 text-black shadow-md shadow-emerald-950/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              BUY / LONG
            </button>
            <button
              role="radio"
              aria-checked={tradeDirection === "SHORT"}
              onClick={() => setTradeDirection("SHORT")}
              className={`px-3 py-1.5 min-h-[32px] text-xs font-bold rounded-md transition-colors active:scale-[0.96] transition-transform duration-100 focus-visible:ring-2 focus-visible:ring-rose-400 focus-visible:outline-none ${
                tradeDirection === "SHORT"
                  ? "bg-rose-500 text-white shadow-md shadow-rose-950/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              SELL / SHORT
            </button>
          </div>
        </div>
      </div>

      {/* Buying Power / Margin Advisory Banner */}
      {isCappedByCash && (
        <div className="bg-cyan-950/40 border border-cyan-700/60 p-2.5 rounded-lg text-xs flex items-center justify-between gap-2 text-cyan-300">
          <span>🛡️ <strong>Cash Buying Power Cap Active:</strong> Position sized to {positionUnits} units (${totalPositionValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}) to stay strictly within 100% cash capital. Unconstrained volatility sizing would require {rawVolatilityUnits} units (${(rawVolatilityUnits * currentPrice).toLocaleString(undefined, { maximumFractionDigits: 0 })}).</span>
          <button
            type="button"
            onClick={() => setAccountType("MARGIN")}
            className="px-2 py-1 bg-cyan-600 hover:bg-cyan-500 text-white text-[10px] font-bold rounded shrink-0 cursor-pointer"
          >
            Switch to Margin Mode →
          </button>
        </div>
      )}

      {accountType === "MARGIN" && totalPositionValue > accountSize && (
        <div className="bg-amber-950/50 border border-amber-600/70 p-2.5 rounded-lg text-xs text-amber-300">
          ⚠️ <strong>Margin Leverage Active ({leverageRatio}x Capital Ratio):</strong> Total position exposure is ${totalPositionValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}, exceeding ${accountSize.toLocaleString()} cash equity. Requires ${(totalPositionValue - accountSize).toLocaleString(undefined, { maximumFractionDigits: 0 })} broker margin borrowing. Volatility risk remains strictly clamped to ${dollarRisk.toFixed(0)} ({riskPct}% portfolio risk).
        </div>
      )}

      {accountType === "MARGIN" && accountSize < 25000 && (
        <div className="bg-rose-950/50 border border-rose-600/70 p-2.5 rounded-lg text-xs text-rose-300">
          ⚠️ <strong>FINRA Rule 4210 (PDT) Warning:</strong> Margin day trading requires a minimum equity balance of $25,000. Accounts below $25,000 (${accountSize.toLocaleString()} currently) are legally restricted to 3 day trades per rolling 5 business days unless upgraded to a cash account.
        </div>
      )}

      {/* Interactive Sliders with Full ARIA Accessibility */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Capital Slider */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2">
          <label htmlFor="capital-slider" className="flex justify-between items-center text-xs cursor-pointer">
            <span className="text-slate-300">Portfolio Capital</span>
            <span className="text-amber-400 font-bold text-sm tabular-nums">${accountSize.toLocaleString()}</span>
          </label>
          <input
            id="capital-slider"
            type="range"
            min={1000}
            max={250000}
            step={1000}
            value={accountSize}
            aria-label="Portfolio Capital Amount"
            aria-valuemin={1000}
            aria-valuemax={250000}
            aria-valuenow={accountSize}
            aria-valuetext={`$${accountSize.toLocaleString()}`}
            onChange={(e) => handleAccountSizeChange(Number(e.target.value))}
            className="w-full accent-amber-500 cursor-pointer h-2 bg-[#1b2434] rounded-lg focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none"
          />
          <div className="flex justify-between text-[10px] text-slate-400 tabular-nums">
            <span>$1k</span>
            <span>$50k</span>
            <span>$100k</span>
            <span>$250k</span>
          </div>
        </div>

        {/* Risk Budget Slider */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2">
          <label htmlFor="risk-slider" className="flex justify-between items-center text-xs cursor-pointer">
            <span className="text-slate-300">Risk Budget Per Trade</span>
            <span className="text-rose-400 font-bold text-sm tabular-nums">{riskPct.toFixed(2)}% (${dollarRisk.toFixed(0)})</span>
          </label>
          <input
            id="risk-slider"
            type="range"
            min={0.25}
            max={3.0}
            step={0.25}
            value={riskPct}
            aria-label="Risk Budget Percentage"
            aria-valuemin={0.25}
            aria-valuemax={3.0}
            aria-valuenow={riskPct}
            aria-valuetext={`${riskPct.toFixed(2)} percent, equals $${dollarRisk.toFixed(0)} maximum risk`}
            onChange={(e) => setRiskPct(Number(e.target.value))}
            className="w-full accent-rose-500 cursor-pointer h-2 bg-[#1b2434] rounded-lg focus-visible:ring-2 focus-visible:ring-rose-400 focus-visible:outline-none"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>0.25% (Conservative)</span>
            <span>1.0% (Standard)</span>
            <span>3.0% (Aggressive)</span>
          </div>
        </div>
      </div>

      {/* Sizing Outputs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block">Recommended Size</span>
          <span className="text-base sm:text-lg font-bold text-cyan-400 tabular-nums">{positionUnits} Units</span>
          <span className="text-[9px] text-slate-400 block mt-0.5">Shares / Tokens</span>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block">Total Exposure</span>
          <span className="text-base sm:text-lg font-bold text-slate-200 tabular-nums">${totalPositionValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          <span className="text-[9px] text-slate-400 block mt-0.5 tabular-nums">{leverageRatio}x Capital Ratio</span>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block">Statistical Stop Dist</span>
          <span className="text-base sm:text-lg font-bold text-purple-400 tabular-nums">${stopDistanceDollar.toFixed(2)}</span>
          <span className="text-[9px] text-slate-400 block mt-0.5 tabular-nums">{stopDistancePct.toFixed(2)}% mVaR Vol</span>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block">Intraday RSI (14)</span>
          <span className={`text-base sm:text-lg font-bold tabular-nums ${rsi > 70 ? "text-rose-400" : rsi < 30 ? "text-emerald-400" : "text-amber-400"}`}>
            {rsi.toFixed(1)}
          </span>
          <span className="text-[9px] text-slate-400 block mt-0.5">{rsi > 70 ? "Overbought" : rsi < 30 ? "Oversold" : "Neutral"}</span>
        </div>
      </div>

      {/* 💡 Plain-English Sizing Takeaway */}
      <div className="bg-[#090d14] border border-amber-500/30 p-3 rounded-xl text-xs font-sans text-slate-200 leading-relaxed shadow-sm">
        🛡️ <strong>The Math Made Simple:</strong> If you buy <strong>{positionUnits} shares</strong> at <strong>${currentPrice.toFixed(2)}</strong> and the trade goes wrong, hitting your stop at <strong>${stopPrice.toFixed(2)}</strong>, you lose exactly <strong>${dollarRisk.toFixed(0)} ({riskPct}% of account)</strong>. Your profit targets are <strong>+${(dollarRisk * 1.5).toFixed(0)}</strong> / <strong>+${(dollarRisk * 2.0).toFixed(0)}</strong>. You survive to fight another day.
      </div>

      {/* Execution Targets */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center font-mono">
        {/* 1.0R Stop-Loss */}
        <div className="bg-rose-950/40 border border-rose-800/80 p-3 rounded-lg">
          <span className="text-[10px] text-rose-300 uppercase font-bold block">1.0R Stop-Loss</span>
          <span className="text-sm sm:text-base font-bold text-rose-400 block tabular-nums">${stopPrice.toFixed(2)}</span>
          <span className="text-[9px] text-rose-300/90 block mt-0.5 tabular-nums">-${dollarRisk.toFixed(0)} Max Risk</span>
        </div>

        {/* 1.5R Scalp Target */}
        <div className="bg-cyan-950/40 border border-cyan-800/80 p-3 rounded-lg">
          <span className="text-[10px] text-cyan-300 uppercase font-bold block">1.5R Scalp</span>
          <span className="text-sm sm:text-base font-bold text-cyan-400 block tabular-nums">${target15.toFixed(2)}</span>
          <span className="text-[9px] text-cyan-300/90 block mt-0.5 tabular-nums">+${(dollarRisk * 1.5).toFixed(0)} Profit</span>
        </div>

        {/* 2.0R Optimal Target */}
        <div className="bg-emerald-950/40 border border-emerald-800/80 p-3 rounded-lg">
          <span className="text-[10px] text-emerald-300 uppercase font-bold block">2.0R Optimal</span>
          <span className="text-sm sm:text-base font-bold text-emerald-400 block tabular-nums">${target20.toFixed(2)}</span>
          <span className="text-[9px] text-emerald-300/90 block mt-0.5 tabular-nums">+${(dollarRisk * 2.0).toFixed(0)} Profit</span>
        </div>

        {/* 3.0R Momentum Runner */}
        <div className="bg-purple-950/40 border border-purple-800/80 p-3 rounded-lg">
          <span className="text-[10px] text-purple-300 uppercase font-bold block">3.0R Runner</span>
          <span className="text-sm sm:text-base font-bold text-purple-400 block tabular-nums">${target30.toFixed(2)}</span>
          <span className="text-[9px] text-purple-300/90 block mt-0.5 tabular-nums">+${(dollarRisk * 3.0).toFixed(0)} Profit</span>
        </div>
      </div>
      {/* 🎯 STAGE 3 EXECUTION: 1-Click Send Trade to Portfolio */}
      <div className="pt-2">
        <button
          type="button"
          onClick={handleSaveToPortfolio}
          className={`w-full py-3 rounded-xl font-bold text-xs flex items-center justify-center space-x-2 transition-all cursor-pointer shadow-lg active:scale-95 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:outline-none ${
            addedFeedback
              ? "bg-emerald-600 text-white font-extrabold shadow-emerald-950/60"
              : "bg-cyan-600 hover:bg-cyan-500 text-white"
          }`}
        >
          <span>{addedFeedback ? "✓ SIZED TRADE SAVED TO MY PORTFOLIO!" : "💼 Send Sized Trade to My Portfolio (1-Click)"}</span>
        </button>
      </div>
    </section>
  );
}

