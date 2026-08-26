"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { trackMatomoEvent } from "../lib/matomo";

export default function OnboardingModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [step, setStep] = useState(1);
  const [selectedStyle, setSelectedStyle] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");

  useEffect(() => {
    try {
      const hasCompleted = localStorage.getItem("FINANCE_ONBOARDING_COMPLETED");
      if (!hasCompleted) {
        // Automatically show to first-time visitors after short delay
        const timer = setTimeout(() => setIsOpen(true), 1200);
        return () => clearTimeout(timer);
      }
    } catch {
      // Ignore storage errors
    }
  }, []);

  const handleComplete = () => {
    try {
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      localStorage.setItem("FINANCE_USER_ROLE", selectedStyle);
    } catch {}
    setIsOpen(false);
    trackMatomoEvent("User Journey", "Complete Onboarding", selectedStyle);
    window.location.reload();
  };

  const handleSkip = () => {
    try {
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
    } catch {}
    setIsOpen(false);
    trackMatomoEvent("User Journey", "Skip Onboarding");
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/85 backdrop-blur-md z-50 flex items-center justify-center p-4 font-mono">
      <div className="bg-[#111722] border border-[#243044] rounded-2xl max-w-lg w-full p-5 sm:p-7 shadow-2xl space-y-5 text-slate-100 relative">
        {/* Step Indicator */}
        <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
              Terminal Setup • Step {step} of 3
            </span>
          </div>
          <button onClick={handleSkip} className="text-xs text-slate-400 hover:text-white transition-colors">
            Skip Intro ✕
          </button>
        </div>

                {/* STEP 1: WELCOME & PLATFORM INTRODUCTION */}
        {step === 1 && (
          <div className="space-y-4">
            <div className="text-center space-y-2 py-1">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-cyan-600 to-indigo-600 mx-auto flex items-center justify-center text-xl font-bold shadow-lg shadow-cyan-950/50">
                FT
              </div>
              <h2 className="text-lg sm:text-xl font-extrabold text-white tracking-tight">
                Welcome to Finance Terminal
              </h2>
              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
                Professional market intelligence, political insider tracking, and automated trade execution levels — with zero sign-up required.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
              <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
                <strong className="text-cyan-400 block font-bold text-xs">🏛️ Political Insiders</strong>
                <span className="text-slate-300 text-[11px] font-sans leading-snug block">
                  Track real-time stock purchases by US Congress members and top politicians.
                </span>
              </div>
              <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
                <strong className="text-purple-400 block font-bold text-xs">🎯 Clear Buy &amp; Sell Zones</strong>
                <span className="text-slate-300 text-[11px] font-sans leading-snug block">
                  Automated entry prices, stop-loss protection, and multi-tier profit targets.
                </span>
              </div>
            </div>

            <button
              onClick={() => setStep(2)}
              className="w-full py-3 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-bold rounded-xl text-xs shadow-lg transition-transform active:scale-95 cursor-pointer"
            >
              Choose Trading Style →
            </button>
          </div>
        )}

        {/* STEP 2: CHOOSE TRADING HORIZON STYLE */}
        {step === 2 && (
          <div className="space-y-4">
            <div className="space-y-1 text-center">
              <h3 className="text-base font-bold text-white">Select Your Primary Market Horizon</h3>
              <p className="text-xs text-slate-400">The interface adapts charts, indicators, and risk metrics automatically.</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setSelectedStyle("DAY_TRADER")}
                className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
                  selectedStyle === "DAY_TRADER"
                    ? "bg-amber-950/40 border-amber-500 text-white shadow-lg shadow-amber-950/40"
                    : "bg-[#090d14] border-[#1b2434] text-slate-400 hover:border-slate-600"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-base">⚡</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">
                    INTRADAY
                  </span>
                </div>
                <strong className="text-xs font-bold text-slate-200 block">Day Trader / Scalper</strong>
                <span className="text-[11px] text-slate-400 block mt-1 leading-snug">
                  5-minute VWAP bands, fast position sizer, and real-time options tape.
                </span>
              </button>

              <button
                type="button"
                onClick={() => setSelectedStyle("LONG_TERM")}
                className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
                  selectedStyle === "LONG_TERM"
                    ? "bg-cyan-950/40 border-cyan-500 text-white shadow-lg shadow-cyan-950/40"
                    : "bg-[#090d14] border-[#1b2434] text-slate-400 hover:border-slate-600"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-base">🏛️</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                    LONG-TERM
                  </span>
                </div>
                <strong className="text-xs font-bold text-slate-200 block">Wealth Compounder</strong>
                <span className="text-[11px] text-slate-400 block mt-1 leading-snug">
                  5-year multi-year horizons, 20 EMA pullbacks, and Piotroski F-Scores.
                </span>
              </button>
            </div>

            <div className="flex items-center space-x-2 pt-2">
              <button
                onClick={() => setStep(1)}
                className="w-1/3 py-2 bg-[#162030] text-slate-300 rounded-xl text-xs hover:bg-[#202d40]"
              >
                ← Back
              </button>
              <button
                onClick={() => setStep(3)}
                className="w-2/3 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs"
              >
                Next: Hotkeys & Shortcuts →
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: KEYBOARD SHORTCUTS & LAUNCH */}
        {step === 3 && (
          <div className="space-y-4">
            <div className="space-y-1 text-center">
              <h3 className="text-base font-bold text-white">Keyboard Hotkeys & Navigation</h3>
              <p className="text-xs text-slate-400">Master rapid institutional terminal navigation.</p>
            </div>

            <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Global Omni-Search</span>
                <span className="bg-[#162030] text-cyan-300 font-bold px-2 py-0.5 rounded border border-[#2b394f]">
                  ⌘K or /
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Toggle Dark / Light Theme</span>
                <span className="text-slate-400 font-bold">Top Navbar Switcher 🌙</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">My Portfolio & Allocations</span>
                <span className="text-slate-400 font-bold">Navbar &apos;💼 Portfolio&apos;</span>
              </div>
            </div>

            <button
              onClick={handleComplete}
              className="w-full py-3 bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white font-extrabold rounded-xl text-xs shadow-lg transition-transform active:scale-95 cursor-pointer"
            >
              🚀 Launch Quantitative Terminal
            </button>
          </div>
        )}
      </div>
    </div>
  );
}