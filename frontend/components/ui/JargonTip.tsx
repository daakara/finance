"use client";

import React, { useState, useRef, useEffect, useId } from "react";
import { getJargonDefinition, JargonDefinition } from "@/lib/jargonGlossary";
import { useVernacularMode, VernacularMode } from "@/lib/useVernacularMode";

export interface JargonTipProps {
  /**
   * The term key matching JARGON_GLOSSARY (e.g. "VCP", "SMA50", "VWAP", "PIOTROSKI")
   * or a custom label.
   */
  term: string;
  /**
   * Optional custom plain-English definition override.
   */
  definition?: string;
  /**
   * Optional custom label to display instead of the default term/shortName.
   */
  children?: React.ReactNode;
  /**
   * Optional explicit vernacular mode override. If omitted, uses global useVernacularMode().
   */
  mode?: VernacularMode;
  /**
   * Optional extra CSS classes for the trigger.
   */
  className?: string;
}

/**
 * JargonTip: Accessible, progressive terminology disclosure component.
 * Complies with WCAG 2.1 AA accessibility guidelines:
 * - Keyboard focusable (Tab)
 * - Enter/Space to toggle tooltip, Escape to dismiss
 * - Explicit ARIA roles (`role="tooltip"`, `aria-describedby`, `aria-expanded`)
 * - High contrast visual cues (dotted underline + hover highlight)
 */
export function JargonTip({
  term,
  definition: customDefinition,
  children,
  mode: explicitMode,
  className = "",
}: JargonTipProps) {
  const globalVernacular = useVernacularMode();
  const activeMode = explicitMode || globalVernacular.mode;
  const isPlain = activeMode === "PLAIN_ENGLISH";

  const [isOpen, setIsOpen] = useState<boolean>(false);
  const containerRef = useRef<HTMLSpanElement | null>(null);
  const tooltipId = useId();

  const jargon: JargonDefinition | undefined = getJargonDefinition(term);
  const displayDefinition = customDefinition || jargon?.plainEnglishSummary || jargon?.detailedDefinition || "Specialized quantitative metric.";
  const detailedContext = jargon?.detailedDefinition;
  const institutionalNote = jargon?.institutionalContext;

  // Close on Click Outside
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  const handleToggle = () => setIsOpen((prev) => !prev);

  const displayText = children || (isPlain && jargon?.term ? jargon.term : jargon?.shortName || term);

  return (
    <span
      ref={containerRef}
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button
        type="button"
        aria-describedby={isOpen ? tooltipId : undefined}
        aria-expanded={isOpen}
        onClick={handleToggle}
        className="inline-flex items-center text-left border-b border-dotted border-slate-500 hover:border-cyan-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:ring-offset-1 focus-visible:ring-offset-[#090d14] rounded-sm transition-colors cursor-help group"
      >
        <span className="group-hover:text-cyan-300 transition-colors">{displayText}</span>
        <span className="sr-only"> (Click for plain English explanation)</span>
      </button>

      {isOpen && (
        <span
          id={tooltipId}
          role="tooltip"
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 sm:w-80 p-3 rounded-lg bg-[#0c121d] border border-cyan-800/80 shadow-2xl shadow-cyan-950/50 text-slate-200 text-xs z-50 pointer-events-auto transition-opacity"
        >
          {/* Header */}
          <span className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1.5 mb-1.5">
            <span className="font-bold text-cyan-400 font-mono text-[11px] uppercase tracking-wider">
              {jargon?.shortName || term} · {jargon?.category || "METRIC"}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">
              {isPlain ? "Plain English" : "Pro Quant"}
            </span>
          </span>

          {/* Plain English Summary */}
          <span className="block text-slate-200 leading-relaxed font-sans mb-1.5 font-medium">
            {displayDefinition}
          </span>

          {/* Detailed Context if in Pro Quant or when available */}
          {detailedContext && detailedContext !== displayDefinition && (
            <span className="block text-[11px] text-slate-400 leading-normal border-t border-slate-800/60 pt-1.5 mt-1.5">
              <strong className="text-slate-300 font-semibold">Technical Detail: </strong>
              {detailedContext}
            </span>
          )}

          {/* Institutional Note */}
          {institutionalNote && (
            <span className="block text-[10px] text-cyan-400/90 leading-tight pt-1 mt-1 font-mono">
              ARX Context: {institutionalNote}
            </span>
          )}

          {/* Tooltip caret arrow */}
          <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent border-t-[#0c121d]" />
        </span>
      )}
    </span>
  );
}

export default JargonTip;
