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

  const handleShare = () => {
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
      className="px-3 py-1.5 rounded-lg bg-[#141b29] hover:bg-[#1c2638] border border-[#24334a] text-xs font-bold text-cyan-300 hover:text-white transition-all active:scale-95 flex items-center gap-1.5 shadow"
      title="Copy formatted trade card to clipboard for Discord, X, or Reddit"
    >
      <span>🔗</span>
      <span>{copied ? "✅ Trade Card Copied!" : "Share Setup"}</span>
    </button>
  );
}
