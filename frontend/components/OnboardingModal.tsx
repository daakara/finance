"use client";

import { useState, useEffect, useCallback } from "react";
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
        const timer = setTimeout(() => setIsOpen(true), 1000);
        return () => clearTimeout(timer);
      }
    } catch {
      // Ignore storage errors
    }
  }, []);

  // Listen for custom trigger to open onboarding tour from anywhere in the app
  useEffect(() => {
    const handleOpen = () => {
      setStep(1);
      setIsOpen(true);
    };
    window.addEventListener("open-onboarding", handleOpen);
    return () => window.removeEventListener("open-onboarding", handleOpen);
  }, []);

  const handleComplete = useCallback(() => {
    try {
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      localStorage.setItem("FINANCE_USER_ROLE", selectedStyle);
    } catch {}
    setIsOpen(false);
    trackMatomoEvent("User Journey", "Complete Onboarding", selectedStyle);
    window.dispatchEvent(new CustomEvent("finance:role-change", { detail: selectedStyle }));
  }, [selectedStyle]);

  const handleSkip = useCallback(() => {
    try {
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
    } catch {}
    setIsOpen(false);
    trackMatomoEvent("User Journey", "Skip Onboarding");
  }, []);

  // Escape key to dismiss
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        handleSkip();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, handleSkip]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-modal-title"
      className="fixed inset-0 bg-black/85 backdrop-blur-md z-[100] flex items-center justify-center p-3 sm:p-4 font-mono animate-fadeIn"
      onClick={handleSkip}
    >
      <div
        className="bg-[#111722] border border-[#243044] rounded-2xl max-w-lg w-full p-5 sm:p-7 shadow-2xl space-y-5 text-slate-100 relative pointer-events-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Step Indicator */}
        <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
              Terminal Setup • Step {step} of 2
            </span>
          </div>
          <button
            type="button"
            onClick={handleSkip}
            className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-[#162030] hover:bg-[#202d40] transition-colors focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer"
            aria-label="Skip onboarding intro"
          >
            Skip Intro ✕
          </button>
        </div>

        {/* STEP 1: INTERACTIVE TRADING STYLE SELECTION (Directly Clickable Cards) */}
        {step === 1 && (
          <div className="space-y-4">
            <div className="text-center space-y-1 py-1">
              <div className="w-12 h-12 rounded-2xl bg-cyan-600 mx-auto flex items-center justify-center text-xl font-bold shadow-sm text-white">
                ARX
              </div>
              <h2 id="onboarding-modal-title" className="text-lg sm:text-xl font-extrabold text-white tracking-tight">
                Welcome to ARX Terminal
              </h2>
              <p className="text-xs text-slate-300 font-sans">
                Select your primary trading horizon. The terminal customizes indicators and risk ladders automatically:
              </p>
            </div>

            {/* Clickable Option 1: Day Trader */}
            <button
              type="button"
              onClick={() => setSelectedStyle("DAY_TRADER")}
              className={`w-full p-4 rounded-xl border text-left transition-all cursor-pointer focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none ${
                selectedStyle === "DAY_TRADER"
                  ? "bg-amber-950/40 border-amber-500 text-white shadow-lg shadow-amber-950/50 ring-1 ring-amber-500/50"
                  : "bg-[#090d14] border-[#1b2434] text-slate-400 hover:border-slate-600 hover:bg-[#121924]"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">⚡</span>
                  <strong className="text-sm font-bold text-white">Day Trader / Scalper</strong>
                </div>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  selectedStyle === "DAY_TRADER"
                    ? "bg-amber-500 text-black font-extrabold"
                    : "bg-amber-950/80 text-amber-300 border border-amber-800"
                }`}>
                  {selectedStyle === "DAY_TRADER" ? "✓ SELECTED" : "SELECT"}
                </span>
              </div>
              <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                5-minute VWAP bands, fast position sizing calculator, and real-time options tape.
              </p>
            </button>

            {/* Clickable Option 2: Long Term */}
            <button
              type="button"
              onClick={() => setSelectedStyle("LONG_TERM")}
              className={`w-full p-4 rounded-xl border text-left transition-all cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                selectedStyle === "LONG_TERM"
                  ? "bg-cyan-950/40 border-cyan-500 text-white shadow-lg shadow-cyan-950/50 ring-1 ring-cyan-500/50"
                  : "bg-[#090d14] border-[#1b2434] text-slate-400 hover:border-slate-600 hover:bg-[#121924]"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">🏛️</span>
                  <strong className="text-sm font-bold text-white">Wealth Compounder</strong>
                </div>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  selectedStyle === "LONG_TERM"
                    ? "bg-cyan-500 text-black font-extrabold"
                    : "bg-cyan-950/80 text-cyan-300 border border-cyan-800"
                }`}>
                  {selectedStyle === "LONG_TERM" ? "✓ SELECTED" : "SELECT"}
                </span>
              </div>
              <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                Multi-year chart horizons, 20 EMA pullbacks, and 9-point Piotroski F-Scores.
              </p>
            </button>

            <button
              type="button"
              onClick={() => setStep(2)}
              className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs shadow-sm transition-transform active:scale-95 cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
            >
              Continue to Hotkeys &amp; Shortcuts →
            </button>
          </div>
        )}

        {/* STEP 2: KEYBOARD SHORTCUTS & LAUNCH */}
        {step === 2 && (
          <div className="space-y-4">
            <div className="space-y-1 text-center">
              <h3 id="onboarding-modal-title" className="text-base font-bold text-white">Keyboard Hotkeys &amp; Shortcuts</h3>
              <p className="text-xs text-slate-400 font-sans">Quick navigation tips for rapid quantitative research.</p>
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
                <span className="text-slate-300">My Portfolio &amp; Allocations</span>
                <span className="text-slate-400 font-bold">Navbar &apos;💼 Portfolio&apos;</span>
              </div>
            </div>

            <div className="flex items-center space-x-2 pt-2">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="w-1/3 py-3 bg-[#162030] text-slate-300 rounded-xl text-xs hover:bg-[#202d40] font-bold cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none transition-colors"
              >
                ← Back
              </button>
              <button
                type="button"
                onClick={handleComplete}
                className="w-2/3 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-extrabold rounded-xl text-xs shadow-sm transition-transform active:scale-95 cursor-pointer focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:outline-none"
              >
                🚀 Launch Terminal
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}