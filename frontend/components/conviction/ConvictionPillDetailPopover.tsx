"use client";

import React, { useState, useRef, useEffect, useId } from "react";
import { ConvictionItem, ConvictionStatus } from "../../types/workstation";
import { trackTelemetryEvent } from "../../telemetry/tracker";
import InstitutionalTooltip from "../tooltips/InstitutionalTooltip";

export interface ConvictionPillDetailPopoverProps {
  item: ConvictionItem;
  ticker?: string;
  className?: string;
}

export default function ConvictionPillDetailPopover({
  item,
  ticker,
  className = "",
}: ConvictionPillDetailPopoverProps) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const popoverId = useId();
  const openTimeRef = useRef<number>(0);

  const {
    dimension,
    status,
    label,
    value,
    summary,
    reasons = [],
    provenanceSource,
    methodologyNote,
    metrics = [],
  } = item;

  const handleToggle = () => {
    if (!isOpen) {
      setIsOpen(true);
      openTimeRef.current = performance.now();
      trackTelemetryEvent(
        "DECISION",
        "conviction_popover_opened",
        { dimension, status, ticker },
        ticker
      );
    } else {
      handleClose();
    }
  };

  const handleClose = () => {
    if (isOpen) {
      setIsOpen(false);
      const durationMs = Math.round(performance.now() - openTimeRef.current);
      trackTelemetryEvent(
        "DECISION",
        "conviction_popover_closed",
        { dimension, durationMs, ticker },
        ticker
      );
    }
  };

  // Close on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (
        isOpen &&
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node)
      ) {
        handleClose();
      }
    };
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        handleClose();
        triggerRef.current?.focus();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleOutsideClick);
      window.addEventListener("keydown", handleEscape);
    }
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen]);

  // Anti-Cyan Semantic Colors
  const getStatusColors = (st: ConvictionStatus) => {
    switch (st) {
      case "FAVORABLE":
        return {
          pill: "bg-emerald-950/40 text-emerald-400 border-emerald-800/60 hover:bg-emerald-900/40 hover:border-emerald-700",
          dot: "bg-emerald-400",
          badge: "bg-emerald-950/60 text-emerald-300 border-emerald-700/80",
        };
      case "CAUTION":
        return {
          pill: "bg-amber-950/40 text-amber-400 border-amber-800/60 hover:bg-amber-900/40 hover:border-amber-700",
          dot: "bg-amber-400",
          badge: "bg-amber-950/60 text-amber-300 border-amber-700/80",
        };
      case "UNFAVORABLE":
        return {
          pill: "bg-rose-950/40 text-rose-400 border-rose-800/60 hover:bg-rose-900/40 hover:border-rose-700",
          dot: "bg-rose-400",
          badge: "bg-rose-950/60 text-rose-300 border-rose-700/80",
        };
      case "NEUTRAL":
      default:
        return {
          pill: "bg-slate-900/60 text-slate-300 border-slate-700/60 hover:bg-slate-800/60 hover:border-slate-600",
          dot: "bg-slate-400",
          badge: "bg-slate-900 text-slate-300 border-slate-700",
        };
    }
  };

  const colors = getStatusColors(status);

  return (
    <div className={`relative ${className}`}>
      {/* Trigger Button: Self-Explanatory Label & Status Value */}
      <button
        ref={triggerRef}
        type="button"
        onClick={handleToggle}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        aria-controls={isOpen ? popoverId : undefined}
        className={`w-full p-3 rounded-xl border text-left transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-info flex flex-col justify-between ${colors.pill} ${
          isOpen ? "ring-2 ring-accent-info/50" : ""
        }`}
      >
        <div className="flex items-center justify-between gap-1.5 w-full">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full shrink-0 ${colors.dot}`} />
            <span className="text-body-ui font-semibold text-text-primary">
              {label}
            </span>
          </div>
          <span className="text-caption-mono text-text-muted text-[11px]">
            {isOpen ? "▲" : "▼"}
          </span>
        </div>

        <div className="mt-2 flex items-baseline justify-between w-full">
          <span className="text-caption-mono font-bold text-text-primary text-sm truncate">
            {value}
          </span>
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border uppercase font-bold shrink-0 ${colors.badge}`}>
            {status}
          </span>
        </div>
      </button>

      {/* Popover Card */}
      {isOpen && (
        <div
          ref={popoverRef}
          id={popoverId}
          role="dialog"
          aria-label={`${label} Details`}
          className="absolute z-40 left-0 sm:left-auto right-0 sm:right-auto sm:w-84 mt-2 p-4 rounded-xl bg-bg-surface-elevated border border-border-subtle text-text-primary shadow-2xl animate-fadeIn font-sans"
          style={{ minWidth: "280px", maxWidth: "360px" }}
        >
          {/* Header */}
          <div className="flex items-start justify-between pb-2 border-b border-border-subtle mb-3">
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-caption-mono font-bold text-text-primary">
                  {label}
                </span>
                {methodologyNote && (
                  <InstitutionalTooltip
                    title={`${label} Methodology`}
                    explanation={methodologyNote}
                    provenance={provenanceSource}
                  />
                )}
              </div>
              <p className="text-[11px] text-text-muted font-mono mt-0.5">
                Pillar: {dimension}
              </p>
            </div>
            <button
              type="button"
              onClick={handleClose}
              className="text-text-muted hover:text-text-primary text-xs p-1 rounded hover:bg-bg-surface"
              aria-label="Close popover"
            >
              ✕
            </button>
          </div>

          {/* High-Salience Summary */}
          <p className="text-body-ui text-text-secondary mb-3 leading-relaxed">
            {summary}
          </p>

          {/* Contextual Plain-English Reasons */}
          {reasons.length > 0 && (
            <div className="mb-3 space-y-1.5">
              <span className="text-[10px] font-mono uppercase text-text-muted font-semibold tracking-wider block">
                Key Causal Drivers:
              </span>
              <ul className="space-y-1">
                {reasons.map((r, idx) => (
                  <li key={idx} className="flex items-start gap-1.5 text-xs text-text-secondary leading-snug">
                    <span className="text-accent-positive shrink-0 mt-0.5">•</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Quantitative Metrics Grid */}
          {metrics.length > 0 && (
            <div className="mb-3 p-2 rounded-lg bg-bg-surface border border-border-subtle/80 grid grid-cols-2 gap-2">
              {metrics.map((m, idx) => (
                <div key={idx} className="text-[11px]">
                  <span className="text-text-muted font-mono block text-[10px]">
                    {m.label}
                  </span>
                  <span className="font-mono font-bold text-text-primary">
                    {m.value}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Data Provenance Footer */}
          <div className="pt-2 border-t border-border-subtle flex items-center justify-between text-[10px] font-mono text-text-muted">
            <span>Source: {provenanceSource}</span>
            <span className="text-accent-positive font-semibold">Verified Model Signal</span>
          </div>
        </div>
      )}
    </div>
  );
}
