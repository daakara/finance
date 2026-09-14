"use client";

import { useState, useEffect, useRef } from "react";
import { trackOnboardingCompleted } from "../lib/matomo";

interface OnboardingTourModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const TOUR_SLIDES = [
  {
    step: 1,
    badge: "1. FIND · RADAR",
    title: "📡 Radar: Find Opportunities",
    icon: "📡",
    content: "Scan and filter the market universe for momentum leaders, volume contraction, and Minervini Stage 2 breakouts that warrant further investigation.",
    highlight: "Use confluence scoring and multi-factor filters to isolate high-probability candidates.",
  },
  {
    step: 2,
    badge: "2. UNDERSTAND · ANALYSIS",
    title: "🔬 Analysis: Understand the Asset",
    icon: "🔬",
    content: "Evaluate fundamental strength, market regime bias, multi-factor scores, and institutional accumulation footprints to determine if an asset is worthy of capital.",
    highlight: "Inspect the multi-factor confluence trace and macroeconomic regime context before committing capital.",
  },
  {
    step: 3,
    badge: "3. PLAN · TRADE PLAN",
    title: "⚡ Trade Plan: Define the Plan",
    icon: "⚡",
    content: "Prepare and size your execution ticket according to governed risk limits. The Behavioral Governor clamps position size during drawdowns and establishes asymmetric R:R brackets.",
    highlight: "Stop losses and profit targets are strictly pinned before trade execution is recorded.",
  },
  {
    step: 4,
    badge: "4. MANAGE · PORTFOLIO",
    title: "💼 Portfolio: Manage the Position",
    icon: "💼",
    content: "Track live capital at risk, monitor Cornish-Fisher Value-at-Risk (M-VaR), and manage the full trade lifecycle with disciplined execution exits.",
    highlight: "Reconcile manual holdings with recorded fills and monitor active risk heat.",
  },
];

export default function OnboardingTourModal({ isOpen, onClose }: OnboardingTourModalProps) {
  const [currentSlide, setCurrentSlide] = useState(0);
  const triggerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    triggerRef.current = (document.activeElement as HTMLElement) || null;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "Tab") {
        const dialog = document.querySelector('[role="dialog"][aria-labelledby="tour-modal-title"]');
        if (!dialog) return;
        const focusables = Array.from(
          dialog.querySelectorAll<HTMLElement>('button, input, select, textarea, a[href], [tabindex="0"]')
        ).filter((el) => !el.hasAttribute('disabled') && el.tabIndex !== -1);
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    // Initial focus on the next button or close button
    setTimeout(() => {
      const dialog = document.querySelector('[role="dialog"][aria-labelledby="tour-modal-title"]');
      const focusTarget = dialog?.querySelector<HTMLElement>('#tour-next-btn') || dialog?.querySelector<HTMLElement>('button');
      focusTarget?.focus();
    }, 50);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      const trigger = triggerRef.current;
      setTimeout(() => trigger?.focus(), 20);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const slide = TOUR_SLIDES[currentSlide];
  const isLast = currentSlide === TOUR_SLIDES.length - 1;

  const handleNext = () => {
    if (isLast) {
      try {
        localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      } catch {}
      trackOnboardingCompleted(slide.title);
      onClose();
    } else {
      setCurrentSlide(prev => prev + 1);
    }
  };

  const handlePrev = () => {
    setCurrentSlide(prev => Math.max(0, prev - 1));
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="tour-modal-title"
      className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono"
    >
      <div className="bg-[#0b101b] border border-[#223147] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-slate-100 flex flex-col justify-between">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1b2537] bg-[#0e1422]">
          <div className="flex items-center space-x-2">
            <span className="text-xl">{slide.icon}</span>
            <div>
              <span className="text-[10px] text-cyan-400 font-bold tracking-wider uppercase block">
                {slide.badge} ({slide.step}/4)
              </span>
              <h2 id="tour-modal-title" className="text-sm sm:text-base font-bold text-white tracking-tight">
                ARX Terminal Quick Tour
              </h2>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close tour modal"
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-all text-sm focus-ring"
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
              id="tour-next-btn"
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
