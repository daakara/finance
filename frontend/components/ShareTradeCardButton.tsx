"use client";

import { useState } from "react";

interface ShareTradeCardButtonProps {
  ticker: string;
  name: string;
  spotPrice: number;
  entryMin: number;
  entryMax: number;
  target1: number;
  stopLoss: number;
  compositeScore: number;
  piotroskiScore: number;
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
}: ShareTradeCardButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleShare = () => {
    const text = `📊 ${ticker} (${name}) Quantitative Trade Setup
Spot: $${spotPrice.toFixed(2)} | 🟢 IN_BUY_ZONE ($${entryMin.toFixed(2)} - $${entryMax.toFixed(2)})
🎯 Target 1: $${target1.toFixed(2)} (+2.5x ATR) | 🛑 Stop Loss: $${stopLoss.toFixed(2)}
Health: ${compositeScore}/100 | Piotroski: ${piotroskiScore}/9
Analyze on Finance Terminal: https://finance-xp8.pages.dev/stock/${ticker.toLowerCase()}/`;

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
