"use client";

import React, { useState, useEffect } from "react";
import { Compass, Target, Layers, ShieldCheck, Zap, ChevronUp, ChevronDown } from "lucide-react";

export default function SynthesisGuideBanner() {
  // Default collapsed — prevents layout shift on SSR (window undefined → false).
  // On first client mount: restore stored preference; if none, expand on desktop only.
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  useEffect(() => {
    if (typeof localStorage !== "undefined") {
      const stored = localStorage.getItem("ARX_GUIDE_EXPANDED");
      if (stored !== null) {
        setIsExpanded(stored === "true");
      } else {
        // First visit: expanded on desktop (≥768px), collapsed on mobile
        setIsExpanded(window.innerWidth >= 768);
      }
    }
  }, []);

  const handleToggle = () => {
    const next = !isExpanded;
    setIsExpanded(next);
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("ARX_GUIDE_EXPANDED", String(next));
    }
  };

  const steps = [
    { num: 1, label: "Discover", desc: "Filter factors & smart money in Radar", icon: Compass, href: "/radar" },
    { num: 2, label: "Thesis", desc: "Verify setup conviction & stage breakout", icon: Target, href: "#verdict-card" },
    { num: 3, label: "Evidence", desc: "Inspect 4 pillars at 25% equal weight", icon: Layers, href: "#panel-pillars" },
    { num: 4, label: "Risk", desc: "Pass 5 mandatory preflight checks", icon: ShieldCheck, href: "#preflight-readiness-bar" },
    { num: 5, label: "Execution", desc: "Plan conditional stage ladder & sizing", icon: Zap, href: "#optimal-entry-exit-card" },
  ];

  return (
    <section
      id="guide-banner"
      data-testid="synthesis-guide-banner"
      role="region"
      aria-label="Workflow Guide Navigation"
      className="w-full bg-[var(--surface-1)] border-b border-[var(--border-subtle)] text-xs font-mono transition-all duration-200"
    >
      <div className="max-w-[1750px] mx-auto px-3 sm:px-6 py-2 flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
              SYNTHESIS E V3 WORKFLOW
            </span>
            <span className="text-[var(--text-muted)] text-[11px] hidden sm:inline">
              Discovery → Thesis → Evidence → Risk → Execution
            </span>
          </div>

          <button
            id="toggle-guide-btn"
            type="button"
            onClick={handleToggle}
            aria-expanded={isExpanded}
            aria-controls="guide-steps-body"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[var(--text-2)] hover:text-white hover:bg-[var(--surface-hover)] transition-colors min-h-[32px] cursor-pointer"
          >
            <span id="guide-toggle-label" className="text-[11px]">
              {isExpanded ? "Collapse Guide" : "View 5-Stage Guide"}
            </span>
            {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>

        {isExpanded && (
          <div
            id="guide-steps-body"
            className="grid grid-cols-1 sm:grid-cols-5 gap-2 pt-1 pb-1 animate-fade-in"
          >
            {steps.map((s) => {
              const Icon = s.icon;
              return (
                <a
                  key={s.num}
                  href={s.href}
                  className="flex items-center gap-2.5 p-2 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] hover:border-cyan-500/50 hover:bg-[var(--surface-hover)] transition-all group min-h-[44px]"
                >
                  <div className="w-6 h-6 rounded-md bg-[var(--surface-elevated)] border border-[var(--border-strong)] flex items-center justify-center shrink-0 font-bold text-cyan-400 group-hover:scale-105 transition-transform">
                    {s.num}
                  </div>
                  <div className="min-w-0">
                    <div className="font-bold text-[var(--text-1)] text-xs flex items-center gap-1">
                      <Icon className="w-3 h-3 text-cyan-400 shrink-0" />
                      <span>{s.label}</span>
                    </div>
                    <div className="text-[10px] text-[var(--text-muted)] truncate font-sans">
                      {s.desc}
                    </div>
                  </div>
                </a>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
