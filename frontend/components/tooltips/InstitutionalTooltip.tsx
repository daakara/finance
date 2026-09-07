"use client";

import React, { useState, useRef, useEffect, useId } from "react";
import { trackTelemetryEvent } from "../../telemetry/tracker";

export interface InstitutionalTooltipProps {
  title: string;
  formula?: string;
  explanation: string;
  benchmark?: string;
  provenance?: string;
  children?: React.ReactNode;
  align?: "left" | "center" | "right";
  side?: "top" | "bottom";
  className?: string;
}

export default function InstitutionalTooltip({
  title,
  formula,
  explanation,
  benchmark,
  provenance,
  children,
  align = "center",
  side = "top",
  className = "",
}: InstitutionalTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipId = useId();

  const handleOpen = () => {
    setIsVisible(true);
    trackTelemetryEvent(
      "ORIENTATION",
      "institutional_tooltip_viewed",
      { metric: title }
    );
  };

  const handleClose = () => {
    setIsVisible(false);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isVisible) {
        handleClose();
      }
    };
    if (isVisible) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isVisible]);

  const alignClass =
    align === "left"
      ? "left-0"
      : align === "right"
      ? "right-0"
      : "left-1/2 -translate-x-1/2";

  const sideClass =
    side === "top"
      ? "bottom-full mb-2"
      : "top-full mt-2";

  return (
    <div
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={handleOpen}
      onMouseLeave={handleClose}
    >
      <button
        type="button"
        ref={triggerRef}
        aria-describedby={isVisible ? tooltipId : undefined}
        onFocus={handleOpen}
        onBlur={handleClose}
        className="inline-flex items-center text-text-muted hover:text-text-primary focus-visible:text-accent-info focus-visible:outline-none transition-colors cursor-help p-0.5 rounded"
      >
        {children || (
          <span className="text-[11px] font-mono leading-none px-1 py-0.5 rounded border border-border-subtle bg-bg-surface-raised text-text-secondary hover:text-accent-info">
            ⓘ
          </span>
        )}
      </button>

      {isVisible && (
        <div
          id={tooltipId}
          role="tooltip"
          className={`absolute z-50 w-72 p-3 rounded-xl bg-bg-surface-elevated border border-border-subtle text-text-primary shadow-2xl text-left pointer-events-none animate-fadeIn ${alignClass} ${sideClass}`}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-border-subtle pb-1.5 mb-2">
            <span className="text-caption-mono font-bold text-text-primary">
              {title}
            </span>
            <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-bg-surface text-accent-info border border-accent-info/20">
              Methodology
            </span>
          </div>

          {/* Formula */}
          {formula && (
            <div className="mb-2 p-1.5 rounded bg-bg-app border border-border-subtle font-mono text-[11px] text-accent-info overflow-x-auto">
              <code>{formula}</code>
            </div>
          )}

          {/* Explanation */}
          <p className="text-caption-mono text-text-secondary leading-relaxed mb-2 font-sans text-xs">
            {explanation}
          </p>

          {/* Benchmark & Provenance */}
          {(benchmark || provenance) && (
            <div className="space-y-1 pt-1.5 border-t border-border-subtle/60 text-[10px] font-mono text-text-muted">
              {benchmark && (
                <div className="flex items-start gap-1">
                  <span className="text-text-secondary font-semibold">Benchmark:</span>
                  <span>{benchmark}</span>
                </div>
              )}
              {provenance && (
                <div className="flex items-start gap-1">
                  <span className="text-text-secondary font-semibold">Source:</span>
                  <span>{provenance}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
