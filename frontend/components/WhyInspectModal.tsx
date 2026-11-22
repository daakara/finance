"use client";

import React from "react";
import { TerminalViewState, FactorAttributionItem } from "../types/insight";
import { ConfluenceData } from "../lib/api";
import { Search, BarChart3, Sliders, RefreshCw } from "lucide-react";

export interface WhyInspectModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  setupScore: number;
  terminalState?: TerminalViewState;
  items?: FactorAttributionItem[];
  catalystToIncreaseScore?: string;
  whatWouldChangeAssessment?: string;
  confluence?: ConfluenceData;
}

function formatPillarScore(status?: string, score?: number | null): string {
  const norm = (status || "").toLowerCase();
  if (norm === "unverified" || norm === "synthetic") return "UNVERIFIED";
  if (norm === "unavailable" || norm === "pipeline_held" || score == null) return "UNAVAILABLE";
  if (norm === "not_applicable" || norm === "exempt") return "EXEMPT";
  return `${Math.round(score)} / 100`;
}

function getPillarScoreColor(status?: string, score?: number | null): string {
  const norm = (status || "").toLowerCase();
  if (
    norm === "unverified" ||
    norm === "synthetic" ||
    norm === "unavailable" ||
    norm === "pipeline_held" ||
    norm === "not_applicable" ||
    norm === "exempt" ||
    score == null
  ) {
    return "text-[var(--text-2)] font-semibold";
  }
  if (score >= 75) return "text-emerald-400";
  if (score >= 50) return "text-cyan-400";
  if (score >= 35) return "text-amber-400";
  return "text-rose-400";
}

function getPillarStatusBadge(status?: string): { label: string; className: string } {
  const norm = (status || "").toLowerCase();
  if (norm === "positive" || norm === "favorable") {
    return { label: "FAVORABLE", className: "bg-emerald-950 text-emerald-400 border border-emerald-800" };
  }
  if (norm === "warning" || norm === "unfavorable") {
    return { label: "UNFAVORABLE", className: "bg-rose-950 text-rose-400 border border-rose-800" };
  }
  if (norm === "neutral" || norm === "mixed") {
    return { label: "NEUTRAL", className: "bg-cyan-950 text-cyan-300 border border-cyan-800" };
  }
  if (norm === "unverified" || norm === "synthetic") {
    return { label: "UNVERIFIED", className: "bg-slate-900 text-slate-400 border border-slate-700" };
  }
  if (norm === "not_applicable" || norm === "exempt") {
    return { label: "EXEMPT", className: "bg-slate-900/60 text-slate-500 border border-slate-800" };
  }
  return { label: "UNAVAILABLE", className: "bg-slate-900 text-slate-400 border border-slate-700" };
}

function getPillarModelRule(pillarId: string): string {
  const norm = pillarId.toUpperCase();
  if (norm.includes("TECHNICAL")) {
    return "25% base weight: Evaluated via multi-timeframe moving averages (20 EMA, 50 SMA), momentum (RSI), and stage analysis.";
  }
  if (norm.includes("FUNDAMENTAL") || norm.includes("SOLVENCY")) {
    return "25% base weight: Evaluated via audited SEC 10-K/10-Q financial statements, solvency ratios, and capital efficiency.";
  }
  if (norm.includes("SMART") || norm.includes("FLOW")) {
    return "25% base weight: Evaluated via SEC Form 4 insider transactions, 13F institutional disclosures, and options order flow.";
  }
  if (norm.includes("MACRO") || norm.includes("REGIME")) {
    return "25% base weight: Evaluated via Federal Reserve yield curves (10Y-2Y), credit spreads, and systemic volatility floor.";
  }
  return "25% base weight in multi-factor confluence model; normalized across verified pillars.";
}

export default function WhyInspectModal({
  isOpen,
  onClose,
  symbol,
  setupScore,
  terminalState,
  items = [],
  catalystToIncreaseScore,
  whatWouldChangeAssessment,
  confluence,
}: WhyInspectModalProps) {
  const modalRef = React.useRef<HTMLDivElement>(null);
  const previouslyFocusedElementRef = React.useRef<HTMLElement | null>(null);

  React.useEffect(() => {
    if (!isOpen) return;

    if (typeof document !== "undefined") {
      previouslyFocusedElementRef.current = document.activeElement as HTMLElement | null;
    }

    const timer = setTimeout(() => {
      if (modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length > 0) {
          focusable[0].focus();
        } else {
          modalRef.current.focus();
        }
      }
    }, 50);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }

      if (e.key === "Tab" && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;

        const firstElement = focusable[0];
        const lastElement = focusable[focusable.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("keydown", handleKeyDown);
      if (previouslyFocusedElementRef.current && typeof previouslyFocusedElementRef.current.focus === "function") {
        previouslyFocusedElementRef.current.focus();
      }
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const changeCondition =
    terminalState?.whatWouldChangeAssessment ||
    whatWouldChangeAssessment ||
    catalystToIncreaseScore;

  const agreementLabel =
    terminalState?.factorAgreement?.displayLabel ||
    `${items.filter((i) => i.impact > 0).length} of ${items.length} evaluated factors are favorable`;

  const domains = terminalState?.domains;
  const modelProv = terminalState?.modelProvenance;

  const effectiveScore = confluence?.confluenceScore !== undefined
    ? Math.round(confluence.confluenceScore)
    : setupScore;

  const rawScore = confluence?.rawConfluenceScore ?? confluence?.rawComposite ?? effectiveScore;
  const coverageCap = confluence?.coverageCap ?? 96;

  return (
    <div
      className="fixed inset-0 z-[1300] flex items-center justify-center p-3 sm:p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-sans"
      role="dialog"
      aria-modal="true"
      aria-labelledby="why-modal-title"
    >
      <div
        ref={modalRef}
        tabIndex={-1}
        className="bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden text-[var(--text-1)] max-h-[90vh] flex flex-col focus:outline-none"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-subtle)] bg-[var(--surface-2)] shrink-0">
          <div className="flex items-center space-x-2.5">
            <Search className="w-5 h-5 text-cyan-400 shrink-0" aria-hidden="true" />
            <div>
              <h2 id="why-modal-title" className="text-sm sm:text-base font-black text-white tracking-tight">
                Traceable Evidence & Score Breakdown
              </h2>
              <p className="text-xs text-[var(--text-2)] font-mono">
                Multi-factor evaluation provenance for <span className="text-cyan-400 font-bold">{symbol}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close Breakdown"
            className="text-[var(--text-2)] hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition-all text-sm cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-4 sm:p-5 space-y-4 overflow-y-auto font-sans text-xs">
          {/* Factor Agreement & Coverage Header Box */}
          <div className="bg-[var(--surface-2)] p-3.5 rounded-xl border border-[var(--border-subtle)] space-y-2.5">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[var(--text-2)] font-bold block text-xs">
                  {confluence?.confluenceBadge ? `Confluence: ${confluence.confluenceBadge}` : "Factor Agreement State:"}
                </span>
                <span className="text-xs text-cyan-300 font-mono font-bold">
                  {confluence?.verifiedPillarCount !== undefined
                    ? `${confluence.verifiedPillarCount} of 4 verified evidence pillars`
                    : agreementLabel}
                </span>
              </div>
              <div className="flex items-center gap-1.5 font-mono">
                <span className={`text-xl font-black ${effectiveScore >= 75 ? "text-emerald-400" : effectiveScore >= 55 ? "text-amber-400" : "text-rose-400"}`}>
                  {effectiveScore}
                </span>
                <span className="text-xs text-[var(--text-muted)]">/ 100</span>
              </div>
            </div>

            {/* Score Before Coverage Cap & Coverage Cap Rows */}
            {confluence && (
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[var(--border-subtle)] text-xs font-mono">
                <div
                  data-testid="confluence-score-before-cap"
                  className="p-2 rounded-lg bg-[var(--surface-1)] border border-[var(--border-subtle)] space-y-0.5"
                >
                  <span className="text-[10px] text-[var(--text-2)] uppercase font-bold block">
                    Score Before Cap
                  </span>
                  <span className="text-xs font-bold text-white tabular-nums">
                    {typeof rawScore === "number" ? rawScore.toFixed(1) : rawScore} / 100
                  </span>
                </div>
                <div
                  data-testid="confluence-coverage-cap"
                  className="p-2 rounded-lg bg-[var(--surface-1)] border border-[var(--border-subtle)] space-y-0.5"
                >
                  <span className="text-[10px] text-[var(--text-2)] uppercase font-bold block">
                    Coverage Cap
                  </span>
                  <span className="text-xs font-bold text-cyan-300 tabular-nums">
                    {coverageCap} / 100
                    {confluence.verifiedPillarCount !== undefined && (
                      <span className="text-[10px] text-[var(--text-muted)] font-normal ml-1">
                        ({confluence.verifiedPillarCount}/4)
                      </span>
                    )}
                  </span>
                </div>
              </div>
            )}

            {terminalState?.overallEligibility && (
              <div className="flex items-center justify-between pt-2 border-t border-[var(--border-subtle)] text-xs font-mono text-[var(--text-2)]">
                <span>Data Eligibility:</span>
                <span className={`px-2 py-0.5 rounded font-bold ${
                  terminalState.overallEligibility === "ELIGIBLE"
                    ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                    : terminalState.overallEligibility === "LIMITED"
                    ? "bg-amber-950 text-amber-400 border border-amber-800"
                    : "bg-rose-950 text-rose-400 border border-rose-800"
                }`}>
                  {terminalState.overallEligibility}
                </span>
              </div>
            )}

            {terminalState?.decisionState && (
              <div className="flex items-center justify-between pt-1 text-xs font-mono text-[var(--text-2)]">
                <span>Decision State:</span>
                <span className={`px-2 py-0.5 rounded font-bold ${
                  terminalState.decisionState === "ACTIONABLE_SETUP"
                    ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                    : terminalState.decisionState === "VALID_SETUP"
                    ? "bg-cyan-950 text-cyan-300 border border-cyan-700"
                    : terminalState.decisionState === "EVIDENCE_INCOMPLETE"
                    ? "bg-amber-950 text-amber-300 border border-amber-700"
                    : terminalState.decisionState === "INSUFFICIENT_DATA"
                    ? "bg-amber-950 text-amber-200 border border-amber-600"
                    : "bg-slate-900 text-[var(--text-2)] border border-slate-700"
                }`}>
                  {terminalState.decisionState.replace(/_/g, " ")}
                </span>
              </div>
            )}
          </div>

          {/* Earnings / Binary Risk Warning */}
          {confluence?.warnings && confluence.warnings.length > 0 && (
            <div
              data-testid="confluence-drawer-earnings-warning"
              className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/80 text-amber-300 font-mono text-xs flex items-start gap-2.5 shadow-sm"
              role="alert"
            >
              <span className="text-base shrink-0" aria-hidden="true">⚠️</span>
              <div className="space-y-0.5">
                <span className="font-bold block uppercase text-[11px] text-amber-400 tracking-wider">
                  Earnings & Binary Event Warning:
                </span>
                {confluence.warnings.map((warning, wIdx) => (
                  <p key={wIdx} className="text-xs font-sans leading-snug text-amber-200">
                    {warning}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Multi-Pillar Breakdown (Backend Pillar Scores in place of conflicting point labels) */}
          <div className="space-y-3">
            <span className="text-xs uppercase font-mono tracking-wider text-[var(--text-2)] font-bold block">
              {confluence?.pillars && confluence.pillars.length > 0
                ? "Backend Confluence Pillars (25% Weight Each)"
                : "Evaluated Domains (Fact vs. Model Rule)"}
            </span>

            {confluence?.pillars && confluence.pillars.length > 0 ? (
              confluence.pillars.map((pillar) => {
                const scoreDisplay = formatPillarScore(pillar.status, pillar.score);
                const scoreColor = getPillarScoreColor(pillar.status, pillar.score);
                const statusBadge = getPillarStatusBadge(pillar.status);

                return (
                  <div
                    key={pillar.pillar}
                    data-testid={`pillar-card-${pillar.pillar.toLowerCase()}`}
                    className="bg-[var(--surface-2)] border border-[var(--border-subtle)] rounded-xl p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-sm shrink-0">{pillar.icon}</span>
                        <span className="font-bold text-white text-xs">{pillar.label}</span>
                        <span className="text-[10px] font-mono text-[var(--text-muted)] font-semibold hidden xs:inline">
                          (25% Weight)
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`font-mono font-black text-sm tabular-nums ${scoreColor}`}>
                          {scoreDisplay}
                        </span>
                        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${statusBadge.className}`}>
                          {statusBadge.label}
                        </span>
                      </div>
                    </div>

                    {/* Factual Observation */}
                    <div className="space-y-0.5">
                      <span className="text-xs text-[var(--text-2)] font-mono font-bold flex items-center gap-1">
                        <BarChart3 className="w-3.5 h-3.5 text-cyan-400 inline" aria-hidden="true" />
                        <span>Observation (Fact):</span>
                      </span>
                      <p className="text-xs text-[var(--text-1)] leading-relaxed font-sans pl-1">
                        {pillar.detail || pillar.plainDetail}
                      </p>
                    </div>

                    {/* Model Governance */}
                    <div className="space-y-0.5 pt-1 border-t border-[var(--border-subtle)]">
                      <span className="text-xs text-cyan-400 font-mono font-bold flex items-center gap-1">
                        <Sliders className="w-3.5 h-3.5 text-cyan-400 inline" aria-hidden="true" />
                        <span>Model Governance:</span>
                      </span>
                      <p className="text-xs text-[var(--text-2)] leading-relaxed font-sans pl-1">
                        {getPillarModelRule(pillar.pillar)}
                      </p>
                    </div>

                    {pillar.pillar === "SMART_MONEY_FLOW" && (
                      <div className="pt-1.5 flex justify-end">
                        <button
                          type="button"
                          data-testid="why-modal-sec-disclosure-link"
                          onClick={() => {
                            if (typeof window !== "undefined") {
                              window.dispatchEvent(new CustomEvent("arx:open-sec-disclosures", { detail: { symbol } }));
                              onClose();
                              setTimeout(() => {
                                const el = document.getElementById("sec-disclosures-feed") || document.getElementById("institutional-feeds");
                                if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
                              }, 100);
                            }
                          }}
                          className="text-[11px] text-cyan-400 hover:text-cyan-300 font-mono font-semibold flex items-center gap-1 cursor-pointer py-1 px-2 rounded bg-cyan-950/40 border border-cyan-800/60 transition-colors focus-visible:ring-2 focus-visible:ring-cyan-400"
                        >
                          <span>Open SEC Form 4 Disclosures</span>
                          <span>→</span>
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
            ) : domains && domains.length > 0 ? (
              domains.map((dom, idx) => (
                <div
                  key={idx}
                  className="bg-[var(--surface-2)] border border-[var(--border-subtle)] rounded-xl p-3 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">{dom.domainName}</span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-[var(--surface-2)] text-[var(--text-2)] border border-[var(--border-subtle)]">
                        {dom.availability}
                      </span>
                      <span className={`text-[11px] font-mono px-1.5 py-0.5 rounded font-bold ${
                        dom.status === "FAVORABLE"
                          ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                          : dom.status === "UNFAVORABLE"
                          ? "bg-rose-950 text-rose-400 border border-rose-800"
                          : "bg-amber-950 text-amber-400 border border-amber-800"
                      }`}>
                        {dom.status}
                      </span>
                    </div>
                  </div>

                  {/* Factual Observation */}
                  <div className="space-y-0.5">
                    <span className="text-xs text-[var(--text-2)] font-mono font-bold flex items-center gap-1">
                      <BarChart3 className="w-3.5 h-3.5 text-cyan-400 inline" aria-hidden="true" />
                      <span>Observation (Fact):</span>
                    </span>
                    <p className="text-xs text-[var(--text-1)] leading-relaxed font-sans pl-1">
                      {dom.observation}
                    </p>
                  </div>

                  {/* Model Rule */}
                  <div className="space-y-0.5 pt-1 border-t border-[var(--border-subtle)]">
                    <span className="text-xs text-cyan-400 font-mono font-bold flex items-center gap-1">
                      <Sliders className="w-3.5 h-3.5 text-cyan-400 inline" aria-hidden="true" />
                      <span>Model Governance:</span>
                    </span>
                    <p className="text-xs text-[var(--text-2)] leading-relaxed font-sans pl-1">
                      {dom.modelRule}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              items.map((item, idx) => {
                const name = item.factorName || item.category || "Factor";
                return (
                  <div
                    key={idx}
                    className="bg-[var(--surface-2)] border border-[var(--border-subtle)] rounded-xl p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white text-xs">{name}</span>
                      {item.importanceLevel && (
                        <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-[var(--surface-2)] text-[var(--text-2)] border border-[var(--border-subtle)]">
                          {item.importanceLevel} IMPACT
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-[var(--text-2)] leading-relaxed">
                      {item.plainEnglishReason || item.reason}
                    </p>
                  </div>
                );
              })
            )}
          </div>

          {/* Overall What Would Change ARX's Assessment */}
          {changeCondition && (
            <div className="bg-[var(--surface-2)] border border-cyan-800/50 p-3.5 rounded-xl space-y-1.5">
              <span className="text-xs font-mono uppercase text-cyan-400 font-bold flex items-center gap-1.5">
                <RefreshCw className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" />
                <span>What would change this assessment?</span>
              </span>
              <p className="text-xs text-[var(--text-1)] font-sans leading-relaxed">
                {changeCondition}
              </p>
            </div>
          )}
        </div>

        {/* Footer with Model Provenance */}
        <div className="p-3.5 bg-[var(--surface-2)] border-t border-[var(--border-subtle)] flex flex-wrap justify-between items-center gap-2 shrink-0">
          <div className="text-xs text-[var(--text-muted)] font-mono">
            {modelProv ? (
              <span>Engine: <span className="text-[var(--text-2)]">{modelProv.modelId}</span> (v{modelProv.modelVersion} · ruleset {modelProv.rulesetVersion})</span>
            ) : (
              <span>Engine: <span className="text-[var(--text-2)]">arx-confluence-engine</span> (v2.5.3 · frozen manifest)</span>
            )}
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 min-h-[36px] bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-black rounded-lg text-xs font-mono transition-all cursor-pointer"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
