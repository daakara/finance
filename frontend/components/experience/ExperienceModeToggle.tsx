"use client";

import React, { useRef } from "react";
import { ExperienceMode } from "../../state/experience-store";
import { useExperienceMode } from "../../hooks/useExperienceMode";

interface ModeItem {
  id: ExperienceMode;
  label: string;
  shortLabel: string;
  dotClass: string;
  activeClass: string;
  title: string;
}

const MODES: ModeItem[] = [
  {
    id: "GUIDED",
    label: "Guided",
    shortLabel: "Guided",
    dotClass: "bg-emerald-300",
    activeClass: "bg-emerald-600 text-white shadow-md font-extrabold",
    title: "Guided Mode: Plain English, step-by-step guidance",
  },
  {
    id: "STANDARD",
    label: "Standard",
    shortLabel: "Std",
    dotClass: "bg-cyan-300",
    activeClass: "bg-cyan-600 text-white shadow-md font-extrabold",
    title: "Standard Mode: Balanced confluence metrics and decision triggers",
  },
  {
    id: "QUANT",
    label: "Quant",
    shortLabel: "Quant",
    dotClass: "bg-purple-300",
    activeClass: "bg-purple-600 text-white shadow-md font-extrabold",
    title: "Quant Mode: Maximum quantitative density and deep models",
  },
];

export function ExperienceModeToggle() {
  const { mode, changeMode, isHydrated } = useExperienceMode();
  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  if (!isHydrated) {
    return (
      <div
        role="region"
        aria-label="Experience Mode Loading"
        data-testid="experience-mode-skeleton"
        className="h-8 w-44 sm:w-64 bg-slate-800/80 rounded-xl animate-pulse"
      />
    );
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, currentIndex: number) => {
    let nextIndex = currentIndex;

    if (e.key === "ArrowRight") {
      e.preventDefault();
      nextIndex = (currentIndex + 1) % MODES.length;
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      nextIndex = (currentIndex - 1 + MODES.length) % MODES.length;
    } else if (e.key === "Home") {
      e.preventDefault();
      nextIndex = 0;
    } else if (e.key === "End") {
      e.preventDefault();
      nextIndex = MODES.length - 1;
    } else {
      return;
    }

    const nextMode = MODES[nextIndex].id;
    changeMode(nextMode);
    buttonRefs.current[nextIndex]?.focus();
  };

  return (
    <div
      role="tablist"
      aria-label="Experience Mode"
      data-testid="experience-mode-toggle"
      className="flex bg-[#070b13] p-0.5 rounded-xl border border-[#243044] items-center shadow-inner shrink-0"
    >
      {MODES.map((m, index) => {
        const isSelected = mode === m.id;
        return (
          <button
            key={m.id}
            ref={(el) => {
              buttonRefs.current[index] = el;
            }}
            type="button"
            role="tab"
            aria-selected={isSelected}
            tabIndex={isSelected ? 0 : -1}
            data-testid={`mode-${m.id}`}
            id={`tab-mode-${m.id.toLowerCase()}`}
            title={m.title}
            aria-label={`Switch to ${m.label} Mode`}
            onClick={() => changeMode(m.id)}
            onKeyDown={(e) => handleKeyDown(e, index)}
            className={`flex items-center space-x-1 px-2 py-1 min-h-[28px] sm:min-h-[30px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer ${
              isSelected
                ? m.activeClass
                : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${m.dotClass}`} aria-hidden="true" />
            <span className="font-mono tracking-tight text-[10px] sm:text-xs">
              <span className="hidden xl:inline">{m.label}</span>
              <span className="xl:hidden">{m.shortLabel}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default ExperienceModeToggle;
