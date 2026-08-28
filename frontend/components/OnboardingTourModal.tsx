"use client";

import { useState, useEffect } from "react";

interface OnboardingTourModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const TOUR_SLIDES = [
  {
    step: 1,
    badge: "DUAL-HORIZON ADAPTIVE WORKSPACE",
    title: "⚡ Day Trader vs. 🏛️ Long-Term Compounder",
    icon: "🎛️",
    content: "Switch execution lenses instantly. Day Trader mode activates intraday ATR stops, 5m VWAP anchors, and opening range breakout scalps. Long-Term mode prioritizes Minervini Stage 2 setups, ROIC capital efficiency, and Piotroski F-Scores.",
    highlight: "Toggle anytime via the lens switcher in the navbar or execution cards."
  },
  {
    step: 2,
    badge: "EXECUTION & RISK INLINE MATH",
    title: "🎯 4 Mathematical ATR Execution States",
    icon: "📐",
    content: "Every tracked asset is categorized into an explicit mathematical execution state: 🟢 IN_BUY_ZONE, 🔵 APPROACHING_TARGET, 🟡 WAITING_PULLBACK, or 🛑 STOPPED_OUT, backed by Mark Minervini VCP invalidation ladders.",
    highlight: "Stop losses are strictly pinned to 1.25x ATR below structural accumulation pivots."
  },
  {
    step: 3,
    badge: "CONGRESSIONAL ALPHA ENGINE",
    title: "🏛️ STOCK Act Legislative Alignment Index",
    icon: "⚖️",
    content: "We track US House & Senate disclosures under Public Law 112-105. Trades are scored (0–100) on committee jurisdiction oversight conflicts and penalized up to -32 points for late filings exceeding the statutory 45-day window.",
    highlight: "Filter by Fresh (<15d lag) vs. Aging/Late Filers in the Smart Money feed."
  },
  {
    step: 4,
    badge: "ZERO-LOGIN PRIVATE STORAGE",
    title: "🔒 Client-Side Encrypted Risk & Cornish-Fisher VaR",
    icon: "💼",
    content: "Your portfolio and watchlists are saved entirely in your local browser storage. We compute Cornish-Fisher Modified Value-at-Risk (M-VaR) to protect your capital from fat-tailed black swan market crashes.",
    highlight: "1-Click 'Save to Portfolio' directly from the Position Sizer modal."
  }
];

export default function OnboardingTourModal({ isOpen, onClose }: OnboardingTourModalProps) {
  const [currentSlide, setCurrentSlide] = useState(0);

  if (!isOpen) return null;

  const slide = TOUR_SLIDES[currentSlide];
  const isLast = currentSlide === TOUR_SLIDES.length - 1;

  const handleNext = () => {
    if (isLast) {
      try {
        localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      } catch {}
      onClose();
    } else {
      setCurrentSlide(prev => prev + 1);
    }
  };

  const handlePrev = () => {
    setCurrentSlide(prev => Math.max(0, prev - 1));
  };

  return (
    <div className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono">
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-slate-100 flex flex-col justify-between">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422]">
          <div className="flex items-center space-x-2">
            <span className="text-xl">{slide.icon}</span>
            <div>
              <span className="text-[10px] text-cyan-400 font-bold tracking-wider uppercase block">
                {slide.badge} ({slide.step}/4)
              </span>
              <h2 className="text-sm sm:text-base font-bold text-white tracking-tight">
                Finance Terminal Quick Tour
              </h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-all text-sm"
          >
            ✕
          </button>
        </div>

        {/* Slide Body */}
        <div className="p-6 space-y-4">
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded bg-[#1b2639] text-slate-300 font-bold text-xs">
              Step {slide.step}
            </span>
            <h3 className="text-base font-extrabold text-white">
              {slide.title}
            </h3>
          </div>

          <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
            {slide.content}
          </p>

          <div className="bg-[#070c16] p-3 rounded-xl border border-cyan-900/40 text-xs text-cyan-300 font-sans flex items-start gap-2">
            <span className="text-base shrink-0">💡</span>
            <span><strong>Pro Tip:</strong> {slide.highlight}</span>
          </div>

          {/* Dots Indicator */}
          <div className="flex items-center justify-center space-x-2 pt-2">
            {TOUR_SLIDES.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentSlide(idx)}
                className={`h-2 rounded-full transition-all ${
                  currentSlide === idx ? "w-6 bg-cyan-400" : "w-2 bg-[#223147] hover:bg-slate-500"
                }`}
              />
            ))}
          </div>
        </div>

        {/* Footer Navigation */}
        <div className="p-4 border-t border-[#1b2537] bg-[#0e1422] flex items-center justify-between">
          <button
            type="button"
            onClick={handlePrev}
            disabled={currentSlide === 0}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              currentSlide === 0
                ? "opacity-30 cursor-not-allowed text-slate-500"
                : "text-slate-300 hover:text-white bg-[#151f2e] border border-[#223147]"
            }`}
          >
            ← Previous
          </button>

          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              Skip
            </button>
            <button
              type="button"
              onClick={handleNext}
              className="px-5 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-extrabold rounded-xl text-xs transition-transform active:scale-95 shadow"
            >
              {isLast ? "Get Started 🚀" : "Next →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
