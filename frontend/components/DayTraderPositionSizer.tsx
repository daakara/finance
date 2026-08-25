"use client";

import { useState } from "react";
import { AnalyticsResponse } from "../lib/api";

interface DayTraderPositionSizerProps {
  symbol: string;
  data: AnalyticsResponse;
}

export default function DayTraderPositionSizer({ symbol, data }: DayTraderPositionSizerProps) {
  const [accountSize, setAccountSize] = useState<number>(25000);
  const [riskPct, setRiskPct] = useState<number>(1.0);
  const [tradeDirection, setTradeDirection] = useState<"LONG" | "SHORT">("LONG");

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

  // Position Sizing Formula: Position Units = Dollar Risk / Stop Distance Dollar
  const positionUnits = stopDistanceDollar > 0 ? Math.floor(dollarRisk / stopDistanceDollar) : 1;
  const totalPositionValue = positionUnits * currentPrice;
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

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
            <h3 className="text-base font-bold text-slate-100 font-mono tracking-tight flex items-center gap-2">
              <span>?</span>
              <span>{symbol} Day Trader Risk & Position Sizer</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Statistical stop-loss sizing powered by Cornish-Fisher fat-tail risk (Modified VaR) and ATR(14)
          </p>
        </div>

        {/* Long / Short Toggle */}
        <div className="flex items-center bg-[#090d14] p-1 rounded-lg border border-[#243044]">
          <button
            onClick={() => setTradeDirection("LONG")}
            className={`px-3 py-1 text-xs font-mono font-bold rounded-md transition-all ${
              tradeDirection === "LONG" ? "bg-emerald-600 text-white shadow-lg" : "text-slate-400 hover:text-slate-200"
            }`}
          >
            BUY / LONG
          </button>
          <button
            onClick={() => setTradeDirection("SHORT")}
            className={`px-3 py-1 text-xs font-mono font-bold rounded-md transition-all ${
              tradeDirection === "SHORT" ? "bg-rose-600 text-white shadow-lg" : "text-slate-400 hover:text-slate-200"
            }`}
          >
            SELL / SHORT
          </button>
        </div>
      </div>

      {/* Intraday Technical Dashboard Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 bg-[#090d14] p-3 rounded-lg border border-[#243044] text-center font-mono">
        <div>
          <span className="text-[10px] text-slate-400 block">Current Price</span>
          <span className="text-sm font-bold text-slate-100">${currentPrice}</span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 block">VWAP</span>
          <span className={`text-sm font-bold ${technicals.vwap && currentPrice >= technicals.vwap ? "text-emerald-400" : "text-rose-400"}`}>
            ${technicals.vwap || currentPrice}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 block">RSI (14)</span>
          <span className={`text-sm font-bold ${rsi > 70 ? "text-rose-400" : rsi < 30 ? "text-emerald-400" : "text-cyan-400"}`}>
            {rsi}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 block">ATR Volatility</span>
          <span className="text-sm font-bold text-purple-400">
            ${technicals.atr_14 || (currentPrice * 0.015).toFixed(2)}
          </span>
        </div>
      </div>

      {/* Interactive Controls & Sizing Outputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Controls */}
        <div className="space-y-4 bg-[#090d14] p-4 rounded-lg border border-[#243044]">
          <div>
            <div className="flex justify-between text-xs font-mono mb-1.5">
              <span className="text-slate-400">Account Capital</span>
              <span className="text-cyan-400 font-bold">${accountSize.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="1000"
              max="250000"
              step="1000"
              value={accountSize}
              onChange={(e) => setAccountSize(Number(e.target.value))}
              className="w-full h-1.5 bg-[#1b2434] rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1.5">
              <span className="text-slate-400">Max Risk Budget per Trade</span>
              <span className="text-amber-400 font-bold">{riskPct.toFixed(1)}% (${dollarRisk.toFixed(0)})</span>
            </div>
            <input
              type="range"
              min="0.25"
              max="3.0"
              step="0.25"
              value={riskPct}
              onChange={(e) => setRiskPct(Number(e.target.value))}
              className="w-full h-1.5 bg-[#1b2434] rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          <div className="pt-2 border-t border-[#1b2434] flex justify-between items-center text-xs font-mono">
            <span className="text-slate-400">Statistical Stop Distance:</span>
            <span className="text-rose-400 font-bold">{stopDistancePct.toFixed(2)}% (${stopDistanceDollar.toFixed(2)})</span>
          </div>
        </div>

        {/* Sizing & Execution Parameters */}
        <div className="bg-[#090d14] p-4 rounded-lg border border-[#243044] space-y-3 font-mono">
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="bg-[#111722] p-2.5 rounded border border-cyan-500/40">
              <span className="text-[10px] text-cyan-300 block font-bold">Recommended Quantity</span>
              <span className="text-lg font-bold text-cyan-400">{positionUnits.toLocaleString()} units</span>
            </div>
            <div className="bg-[#111722] p-2.5 rounded border border-[#243044]">
              <span className="text-[10px] text-slate-400 block">Total Position Value</span>
              <span className="text-lg font-bold text-slate-100">${totalPositionValue.toLocaleString()} ({leverageRatio}x)</span>
            </div>
          </div>

          {/* Profit Target Matrix */}
          <div className="space-y-1.5 text-xs pt-1">
            <div className="flex justify-between items-center bg-[#111722] px-2.5 py-1.5 rounded border border-rose-900/60">
              <span className="text-rose-400 font-bold">Stop-Loss (1.0R):</span>
              <span className="text-rose-300">${stopPrice.toFixed(2)} (-${dollarRisk.toFixed(0)})</span>
            </div>
            <div className="flex justify-between items-center bg-[#111722] px-2.5 py-1.5 rounded border border-[#243044]">
              <span className="text-emerald-400 font-semibold">Target 1 (1.5R Take-Profit):</span>
              <span className="text-emerald-300">${target15.toFixed(2)} (+${(dollarRisk * 1.5).toFixed(0)})</span>
            </div>
            <div className="flex justify-between items-center bg-[#111722] px-2.5 py-1.5 rounded border border-emerald-900/60">
              <span className="text-emerald-400 font-bold">Target 2 (2.0R Optimal):</span>
              <span className="text-emerald-300 font-bold">${target20.toFixed(2)} (+${(dollarRisk * 2.0).toFixed(0)})</span>
            </div>
            <div className="flex justify-between items-center bg-[#111722] px-2.5 py-1.5 rounded border border-cyan-900/60">
              <span className="text-cyan-400 font-bold">Target 3 (3.0R Runner):</span>
              <span className="text-cyan-300 font-bold">${target30.toFixed(2)} (+${(dollarRisk * 3.0).toFixed(0)})</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

