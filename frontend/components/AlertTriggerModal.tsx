"use client";

import { useEffect, useState } from "react";
import { AlertManager, ExecutionAlertRule } from "../lib/alertManager";
import { trackAlertSet } from "../lib/matomo";

interface AlertTriggerModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  currentPrice: number;
  optimalEntryMin: number;
  optimalEntryMax: number;
  stopLoss: number;
  takeProfit1: number;
  isStage4?: boolean;
  breakoutPivot?: number;
}

export default function AlertTriggerModal({
  isOpen,
  onClose,
  symbol,
  currentPrice,
  optimalEntryMin,
  optimalEntryMax,
  stopLoss,
  takeProfit1,
  isStage4 = false,
  breakoutPivot,
}: AlertTriggerModalProps) {
  const [permission, setPermission] = useState<NotificationPermission>("default");
  const [notifyBuyZone, setNotifyBuyZone] = useState<boolean>(true);
  const [notifyStopLoss, setNotifyStopLoss] = useState<boolean>(true);
  const [notifyTakeProfit, setNotifyTakeProfit] = useState<boolean>(true);
  const [isSaved, setIsSaved] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window !== "undefined" && "Notification" in window) {
      setPermission(Notification.permission);
    }
    const rules = AlertManager.getAlertRules();
    const existing = rules[symbol.toUpperCase()];
    if (existing) {
      setNotifyBuyZone(existing.notifyOnBuyZone);
      setNotifyStopLoss(existing.notifyOnStopLossWarning);
      setNotifyTakeProfit(existing.notifyOnTakeProfit1);
      setIsSaved(true);
    } else {
      setIsSaved(false);
    }
  }, [isOpen, symbol]);

  if (!isOpen) return null;

  const safeCurrent = typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0 ? currentPrice : 100;
  const safeEntryMin = typeof optimalEntryMin === "number" && !isNaN(optimalEntryMin) && optimalEntryMin > 0 ? optimalEntryMin : safeCurrent * 0.98;
  const safeEntryMax = typeof optimalEntryMax === "number" && !isNaN(optimalEntryMax) && optimalEntryMax > 0 ? optimalEntryMax : safeCurrent * 1.02;
  const safeStop = typeof stopLoss === "number" && !isNaN(stopLoss) && stopLoss > 0 ? stopLoss : safeCurrent * 0.95;
  const safeTarget = typeof takeProfit1 === "number" && !isNaN(takeProfit1) && takeProfit1 > 0 ? takeProfit1 : safeCurrent * 1.10;

  const handleRequestPermission = async () => {
    const res = await AlertManager.requestPermission();
    setPermission(res);
  };

  const handleSaveAlerts = () => {
    if (permission !== "granted") {
      handleRequestPermission();
    }
    const rule: ExecutionAlertRule = {
      symbol: (symbol || "ASSET").toUpperCase(),
      notifyOnBuyZone: notifyBuyZone,
      notifyOnStopLossWarning: notifyStopLoss,
      notifyOnTakeProfit1: notifyTakeProfit,
      optimalEntryMin: safeEntryMin,
      optimalEntryMax: safeEntryMax,
      stopLoss: safeStop,
      takeProfit1: safeTarget,
      isStage4: isStage4,
      breakoutPivotPrice: breakoutPivot || safeCurrent * 1.072,
      createdAt: Date.now(),
    };
    AlertManager.saveAlertRule(rule);
    setIsSaved(true);
    trackAlertSet(symbol, isStage4 ? (breakoutPivot || safeCurrent * 1.072) : safeEntryMin, isStage4);
    AlertManager.playAlertSound("BUY");
    onClose();
  };

  const handleRemoveAlerts = () => {
    AlertManager.removeAlertRule(symbol || "ASSET");
    setIsSaved(false);
    onClose();
  };

  const handleTestSound = (type: "BUY" | "WARNING" | "SUCCESS") => {
    AlertManager.playAlertSound(type);
  };

  return (
    <div className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono">
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-md shadow-2xl overflow-hidden text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422]">
          <div className="flex items-center space-x-2">
            <span className="text-xl">🔔</span>
            <div>
              <h2 className="text-base font-black text-white tracking-tight">
                Execution Price & Invalidation Alerts
              </h2>
              <p className="text-[11px] text-slate-400">
                Live monitoring for <span className="text-cyan-400 font-bold">{symbol || "Asset"}</span> @ ${safeCurrent.toFixed(2)}
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

        {/* Permission Status Banner */}
        <div className="p-5 space-y-4">
          {permission !== "granted" && (
            <div className="bg-[#191508] border border-amber-900/60 p-3 rounded-xl flex items-center justify-between gap-3 text-xs">
              <div>
                <span className="text-amber-400 font-bold block">Desktop Notifications Required</span>
                <p className="text-[11px] text-slate-400">Enable browser notifications to receive alerts when off-tab.</p>
              </div>
              <button
                onClick={handleRequestPermission}
                className="px-3 py-1 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-lg text-xs transition-all shrink-0"
              >
                Enable
              </button>
            </div>
          )}

          {/* Trigger Toggles */}
          <div className="space-y-2.5">
            {/* Buy Zone or 50-SMA Breakout Pivot */}
            <label className={`flex items-center justify-between p-3 rounded-xl bg-[#070b13] border transition-all cursor-pointer ${
              isStage4 ? "border-amber-700/60 hover:border-amber-500" : "border-[#1b273b] hover:border-cyan-500/50"
            }`}>
              <div>
                <span className={`text-xs font-bold flex items-center gap-1.5 ${isStage4 ? "text-amber-300" : "text-emerald-400"}`}>
                  <span>{isStage4 ? "⏳ 50-Day SMA Breakout Pivot Alert" : "🎯 Optimal Buy Zone Entry"}</span>
                </span>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {isStage4
                    ? `Alert the instant price reclaims the 50-day moving average breakout pivot at $${(breakoutPivot || safeCurrent * 1.072).toFixed(2)}`
                    : `Alert when price touches $${safeEntryMin.toFixed(2)} – $${safeEntryMax.toFixed(2)}`}
                </p>
              </div>
              <input
                type="checkbox"
                checked={notifyBuyZone}
                onChange={(e) => setNotifyBuyZone(e.target.checked)}
                className={`w-4 h-4 rounded border-slate-700 bg-slate-900 focus:ring-0 cursor-pointer ${
                  isStage4 ? "text-amber-500" : "text-cyan-500"
                }`}
              />
            </label>

            {/* Stop Loss Warning */}
            <label className="flex items-center justify-between p-3 rounded-xl bg-[#070b13] border border-[#1b273b] cursor-pointer hover:border-rose-500/50 transition-all">
              <div>
                <span className="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                  <span>🛑 Stop-Loss Warning Line</span>
                </span>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  Alert within 1.0% of statutory stop at ${safeStop.toFixed(2)}
                </p>
              </div>
              <input
                type="checkbox"
                checked={notifyStopLoss}
                onChange={(e) => setNotifyStopLoss(e.target.checked)}
                className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-rose-500 focus:ring-0 cursor-pointer"
              />
            </label>

            {/* Take Profit */}
            <label className="flex items-center justify-between p-3 rounded-xl bg-[#070b13] border border-[#1b273b] cursor-pointer hover:border-amber-500/50 transition-all">
              <div>
                <span className="text-xs font-bold text-amber-400 flex items-center gap-1.5">
                  <span>🚀 Take-Profit 1 Expansion</span>
                </span>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  Alert when price approaches ${safeTarget.toFixed(2)} target
                </p>
              </div>
              <input
                type="checkbox"
                checked={notifyTakeProfit}
                onChange={(e) => setNotifyTakeProfit(e.target.checked)}
                className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-0 cursor-pointer"
              />
            </label>
          </div>

          {/* Test Sound Chime */}
          <div className="flex items-center justify-between pt-2 border-t border-[#182335] text-[11px] text-slate-400">
            <span>Test Sound Synthesis:</span>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={() => handleTestSound("BUY")}
                className="px-2 py-0.5 rounded bg-[#101827] border border-[#23334d] text-cyan-300 hover:bg-slate-800 text-[10px] font-bold"
              >
                🎯 Buy Chime
              </button>
              <button
                type="button"
                onClick={() => handleTestSound("WARNING")}
                className="px-2 py-0.5 rounded bg-[#101827] border border-[#23334d] text-rose-300 hover:bg-slate-800 text-[10px] font-bold"
              >
                🛑 Warning
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#1b2537] bg-[#0e1422] flex items-center justify-between">
          {isSaved ? (
            <button
              onClick={handleRemoveAlerts}
              className="text-xs text-rose-400 hover:text-rose-300 font-bold transition-colors"
            >
              Remove Alerts
            </button>
          ) : (
            <div />
          )}

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 font-bold transition-all"
            >
              Cancel
            </button>
            <button
              onClick={handleSaveAlerts}
              className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-bold transition-all shadow"
            >
              Save Active Alerts
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
