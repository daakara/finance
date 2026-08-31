"use client";

import React, { useState, useEffect } from "react";

interface TradeExecutionStepperProps {
  symbol: string;
  isPreFlightCleared?: boolean;
  onOpenPreFlight: () => void;
  onOpenSizer: () => void;
  onOpenAlert: () => void;
  onLogPortfolio: () => void;
  logStatus?: string | null;
}

export default function TradeExecutionStepper({
  symbol,
  isPreFlightCleared = false,
  onOpenPreFlight,
  onOpenSizer,
  onOpenAlert,
  onLogPortfolio,
  logStatus,
}: TradeExecutionStepperProps) {
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");
  const [stepStates, setStepStates] = useState({
    discovery: true,
    preflight: isPreFlightCleared,
    sizer: false,
    portfolio: false,
    alert: false,
  });

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

  useEffect(() => {
    if (isPreFlightCleared) {
      setStepStates((prev) => ({ ...prev, preflight: true }));
    }
  }, [isPreFlightCleared]);

  useEffect(() => {
    if (logStatus && logStatus.includes("Logged")) {
      setStepStates((prev) => ({ ...prev, portfolio: true }));
    }
  }, [logStatus]);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  const steps = [
    {
      id: "step-1",
      number: 1,
      title: isPlain ? "Setup Discovery" : "Pattern Confluence",
      subtitle: isPlain ? `${symbol} Identified` : "VCP / Level Calculated",
      isComplete: true,
      onClick: () => {},
      icon: "🎯",
    },
    {
      id: "step-2",
      number: 2,
      title: isPlain ? "Safety Clearance" : "Pre-Flight Gate",
      subtitle: stepStates.preflight ? "🟢 Cleared" : "5 Risk Checks",
      isComplete: stepStates.preflight,
      onClick: () => {
        setStepStates((prev) => ({ ...prev, preflight: true }));
        onOpenPreFlight();
      },
      icon: "✈️",
    },
    {
      id: "step-3",
      number: 3,
      title: isPlain ? "Position Sizing" : "Risk-Budget Sizer",
      subtitle: stepStates.sizer ? "🟢 Sized" : "1% Account Risk",
      isComplete: stepStates.sizer,
      onClick: () => {
        setStepStates((prev) => ({ ...prev, sizer: true }));
        onOpenSizer();
      },
      icon: "⚖️",
    },
    {
      id: "step-4",
      number: 4,
      title: isPlain ? "Paper Portfolio" : "Paper Execution",
      subtitle: stepStates.portfolio ? "🟢 In Portfolio" : (logStatus || "1-Click Log"),
      isComplete: stepStates.portfolio,
      onClick: () => {
        setStepStates((prev) => ({ ...prev, portfolio: true }));
        onLogPortfolio();
      },
      icon: "💼",
    },
    {
      id: "step-5",
      number: 5,
      title: isPlain ? "Trigger Alert" : "Price Invalidation",
      subtitle: stepStates.alert ? "🟢 Active" : "Set Notifications",
      isComplete: stepStates.alert,
      onClick: () => {
        setStepStates((prev) => ({ ...prev, alert: true }));
        onOpenAlert();
      },
      icon: "🔔",
    },
  ];

  const completedCount = steps.filter((s) => s.isComplete).length;
  const progressPct = Math.round((completedCount / steps.length) * 100);

  return (
    <div className="bg-[#0b1019] border border-[#1e293b] rounded-xl p-3 sm:p-4 space-y-3 shadow-inner">
      {/* Header & Progress Bar */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-bold font-mono text-[11px] uppercase tracking-wider">
            {isPlain ? "Guided Trade Execution Journey:" : "Institutional Execution Stepper:"}
          </span>
          <span className="font-mono font-black text-white text-xs">
            {completedCount} / 5 Done ({progressPct}%)
          </span>
        </div>

        <span className="text-[10px] font-mono text-slate-400 hidden sm:inline">
          {isPlain ? "Step-by-step risk management" : "Monotonic execution protocol"}
        </span>
      </div>

      {/* Visual Progress Track */}
      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 via-emerald-400 to-indigo-400 transition-all duration-500 rounded-full"
          style={{ width: `${progressPct}%` }}
        />
      </div>

      {/* Interactive 5-Step Node Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-1">
        {steps.map((step) => {
          return (
            <button
              key={step.id}
              type="button"
              onClick={step.onClick}
              className={`p-2.5 rounded-lg border text-left transition-all active:scale-[0.97] flex flex-col justify-between ${
                step.isComplete
                  ? "bg-emerald-950/40 border-emerald-700/80 hover:bg-emerald-900/40"
                  : "bg-[#111724] border-[#1e2a3c] hover:border-cyan-500/60 hover:bg-[#141e2e]"
              }`}
            >
              <div className="flex items-center justify-between w-full">
                <span className="text-sm">{step.icon}</span>
                <span
                  className={`text-[9px] font-mono font-black px-1.5 py-0.5 rounded ${
                    step.isComplete
                      ? "bg-emerald-500 text-slate-950 font-bold"
                      : "bg-slate-800 text-slate-400"
                  }`}
                >
                  {step.isComplete ? "✓" : `Step ${step.number}`}
                </span>
              </div>

              <div className="mt-2">
                <strong className={`text-[11px] block font-mono font-bold truncate ${
                  step.isComplete ? "text-emerald-300" : "text-white"
                }`}>
                  {step.title}
                </strong>
                <span className="text-[10px] text-slate-400 block truncate mt-0.5 font-sans">
                  {step.subtitle}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
