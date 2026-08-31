"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";
import { SpotPriceRegistry } from "../lib/api";
import { getPersistedMarketSnapshot } from "../lib/marketDatabase";

export interface PriceAlert {
  id: string;
  symbol: string;
  targetPrice?: number;
  condition: "ABOVE" | "BELOW" | "BUY_ZONE" | "STOP_LOSS";
  optimalMin?: number;
  optimalMax?: number;
  stopPrice?: number;
  createdPrice: number;
  createdAt: string;
  triggered?: boolean;
  triggeredAt?: string;
  triggerMessage?: string;
}

export default function RealTimeAlertEngine() {
  const [activeNotification, setActiveNotification] = useState<PriceAlert | null>(null);
  const router = useRouter();

  // Evaluate alerts against live market price
  const evaluateAlerts = useCallback(() => {
    if (typeof window === "undefined") return;

    try {
      const stored = localStorage.getItem("FINANCE_PRICE_ALERTS");
      if (!stored) return;

      const alerts: PriceAlert[] = JSON.parse(stored);
      let updated = false;

      for (const alert of alerts) {
        if (alert.triggered) continue;

        const upper = alert.symbol.toUpperCase();
        const reg = SpotPriceRegistry.get(upper);
        const snap = getPersistedMarketSnapshot(upper);
        const asset = MASTER_ASSET_CATALOG[upper];
        const livePrice = (reg?.price && reg.price > 0)
          ? reg.price
          : (snap?.currentPrice && snap.currentPrice > 0)
          ? snap.currentPrice
          : (asset ? asset.price : alert.createdPrice);

        let isTriggered = false;
        let msg = "";

        if (alert.condition === "BUY_ZONE" && alert.optimalMin && alert.optimalMax) {
          if (livePrice >= alert.optimalMin && livePrice <= alert.optimalMax) {
            isTriggered = true;
            msg = `🎯 ${alert.symbol} is inside the Optimal Buy Zone ($${alert.optimalMin.toFixed(2)} – $${alert.optimalMax.toFixed(2)}) at $${livePrice.toFixed(2)}!`;
          }
        } else if (alert.condition === "STOP_LOSS" && alert.stopPrice) {
          if (livePrice <= alert.stopPrice) {
            isTriggered = true;
            msg = `🛑 ${alert.symbol} breached Stop Loss safety floor ($${alert.stopPrice.toFixed(2)}) at $${livePrice.toFixed(2)}!`;
          }
        } else if (alert.condition === "ABOVE" && alert.targetPrice) {
          if (livePrice >= alert.targetPrice) {
            isTriggered = true;
            msg = `🚀 ${alert.symbol} reached Profit Target ($${alert.targetPrice.toFixed(2)}) at $${livePrice.toFixed(2)}!`;
          }
        } else if (alert.condition === "BELOW" && alert.targetPrice) {
          if (livePrice <= alert.targetPrice) {
            isTriggered = true;
            msg = `⚠️ ${alert.symbol} dropped below trigger level ($${alert.targetPrice.toFixed(2)}) at $${livePrice.toFixed(2)}!`;
          }
        }

        if (isTriggered) {
          alert.triggered = true;
          alert.triggeredAt = new Date().toISOString();
          alert.triggerMessage = msg;
          updated = true;

          // Dispatch event to navbar badge
          window.dispatchEvent(new CustomEvent("finance:alert-triggered", { detail: alert }));

          // Show floating toast
          setActiveNotification({ ...alert, triggerMessage: msg });
          break; // Show one toast at a time
        }
      }

      if (updated) {
        localStorage.setItem("FINANCE_PRICE_ALERTS", JSON.stringify(alerts));
      }
    } catch (err) {
      console.warn("Error evaluating alerts:", err);
    }
  }, []);

  // Run on mount and periodically every 10 seconds
  useEffect(() => {
    evaluateAlerts();
    const intervalId = setInterval(evaluateAlerts, 10000);
    window.addEventListener("finance:alert-created", evaluateAlerts);
    return () => {
      clearInterval(intervalId);
      window.removeEventListener("finance:alert-created", evaluateAlerts);
    };
  }, [evaluateAlerts]);

  if (!activeNotification) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="fixed bottom-20 right-4 sm:bottom-6 sm:right-6 z-50 max-w-md w-full bg-[#0d1422] border-2 border-cyan-500/80 rounded-2xl p-4 shadow-[0_0_40px_rgba(6,182,212,0.25)] text-white font-mono animate-slideInRight"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <span className="text-2xl animate-bounce shrink-0">🔔</span>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-black text-sm text-cyan-300">PRICE TRIGGER FIRED</span>
              <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-700">
                {activeNotification.symbol}
              </span>
            </div>
            <p className="text-xs text-slate-200 mt-1 leading-relaxed">
              {activeNotification.triggerMessage}
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setActiveNotification(null)}
          className="text-slate-400 hover:text-white p-1 rounded-lg text-xs"
          aria-label="Dismiss alert notification"
        >
          ✕
        </button>
      </div>

      <div className="mt-3 pt-3 border-t border-[#1e293b] flex items-center justify-between gap-2">
        <span className="text-[10px] text-slate-400">
          Triggered {new Date().toLocaleTimeString()}
        </span>

        <button
          type="button"
          onClick={() => {
            router.push(`/?symbol=${activeNotification.symbol}`);
            setActiveNotification(null);
          }}
          className="px-3 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 text-xs font-black transition-all active:scale-95 flex items-center gap-1 shadow"
        >
          <span>Execute Plan</span>
          <span>→</span>
        </button>
      </div>
    </div>
  );
}
