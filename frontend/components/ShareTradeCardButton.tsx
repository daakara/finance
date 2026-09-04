"use client";

import { useState } from "react";

interface ShareTradeCardButtonProps {
  ticker: string;
  name: string;
  spotPrice: number;
  entryMin: number;
  entryMax: number;
  target1?: number;
  stopLoss: number;
  compositeScore: number;
  piotroskiScore: number;
  posture?: string;
}

export default function ShareTradeCardButton({
  ticker,
  name,
  spotPrice,
  entryMin,
  entryMax,
  target1,
  stopLoss,
  compositeScore,
  piotroskiScore,
  posture = "IN_BUY_ZONE",
}: ShareTradeCardButtonProps) {
  const [copied, setCopied] = useState(false);
  const hasValidLevels = Boolean(
    spotPrice && spotPrice > 0 &&
    entryMin && entryMin > 0 &&
    entryMax && entryMax > 0 &&
    stopLoss && stopLoss > 0 &&
    stopLoss < entryMin &&
    entryMin <= entryMax
  );
  const isAvailable = Boolean(
    hasValidLevels &&
    posture !== "UNAVAILABLE" &&
    posture !== "UNVERIFIED_ASSET" &&
    posture !== "INSUFFICIENT_HISTORY"
  );

  const handleShare = () => {
    if (!isAvailable) return;
    const postureLabel =
      posture === "IN_BUY_ZONE" || posture === "ACQUIRE"
        ? `🟢 IN_BUY_ZONE ($${entryMin.toFixed(2)} - $${entryMax.toFixed(2)})`
        : posture === "WAIT_FOR_TRIGGER"
        ? `⏳ WAIT_FOR_TRIGGER (Stage 4 Correction)`
        : `🔍 RESEARCH (Evidence Incomplete)`;

    const targetLabel =
      typeof target1 === "number" && !isNaN(target1) && target1 > 0
        ? `🎯 Target 1: $${target1.toFixed(2)} (+2.5x ATR)`
        : `🎯 Target 1: N/A (< 50 sessions)`;

    const text = `📊 ${ticker} (${name}) Quantitative Trade Setup
Spot: $${spotPrice.toFixed(2)} | ${postureLabel}
${targetLabel} | 🛑 Stop Loss: $${stopLoss.toFixed(2)}
Health: ${compositeScore}/100 | Piotroski: ${piotroskiScore}/9
Analyze on ARX Terminal: https://www.arxterminal.com/stock/${ticker.toLowerCase()}/`;

    try {
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.warn("Could not copy trade card:", err);
    }
  };

  return (
    <button
      type="button"
      onClick={handleShare}
      disabled={!isAvailable}
      className={`px-3 py-1.5 rounded-lg border text-xs font-bold flex items-center gap-1.5 shadow transition-all ${
        isAvailable
          ? "bg-[#141b29] hover:bg-[#1c2638] border-[#24334a] text-cyan-300 hover:text-white active:scale-95"
          : "bg-slate-900/60 border-slate-800 text-slate-500 cursor-not-allowed"
      }`}
      title={isAvailable ? "Copy formatted trade card to clipboard for Discord, X, or Reddit" : "Trade setup unavailable for unverified asset"}
    >
      <span>🔗</span>
      <span>{!isAvailable ? "Setup Unavailable" : copied ? "✅ Trade Card Copied!" : "Share Setup"}</span>
    </button>
  );
}
