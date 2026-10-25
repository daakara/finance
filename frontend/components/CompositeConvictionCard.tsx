"use client";

import { useMemo, useState, useEffect } from "react";
import { AnalyticsResponse } from "../lib/api";
import { FredMacroData, SecForm4Trade } from "../lib/institutionalFeeds";

interface CompositeConvictionCardProps {
  symbol: string;
  data: AnalyticsResponse | null;
  macro: FredMacroData | null;
  insiders: SecForm4Trade[];
  userRole: "DAY_TRADER" | "LONG_TERM";
  onOpenWhy?: () => void;
}

export interface ConfluencePillarDisplay {
  id: "TECHNICAL_STRUCTURE" | "FUNDAMENTAL_SOLVENCY" | "SMART_MONEY_FLOW" | "MACRO_SAFETY_FLOOR";
  label: string;
  weightPct: number;
  score: number | null;
  scoreDisplay: string;
  status: "positive" | "neutral" | "warning" | "unavailable";
  detail: string;
  icon: string;
}

function getPillarScoreColor(scoreDisplay: string, score: number | null): string {
  if (score === null || scoreDisplay === "UNAVAILABLE" || scoreDisplay === "PENDING") {
    return "text-slate-400 font-semibold text-sm";
  }
  if (score >= 75) return "text-emerald-400";
  if (score >= 50) return "text-cyan-400";
  if (score >= 35) return "text-amber-400";
  return "text-rose-400";
}

function getPillarStatusDot(status: string): string {
  switch (status) {
    case "positive":
      return "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]";
    case "warning":
      return "bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.6)]";
    case "neutral":
      return "bg-cyan-400";
    case "unavailable":
    default:
      return "bg-slate-600";
  }
}

export default function CompositeConvictionCard({
  symbol,
  data,
  macro,
  insiders,
  userRole,
  onOpenWhy,
}: CompositeConvictionCardProps) {
  const cleanSym = symbol.toUpperCase().replace("-USD", "");
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };

    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  // Pure View Lens Projection: Consume Canonical Backend Confluence (Single Source of Truth)
  const synthesis = useMemo(() => {
    // 1. SSOT: Direct projection of canonical backend confluence engine
    if (data?.confluence) {
      const conf = data.confluence;
      const score = Math.round(conf.confluenceScore);
      const confluenceBadge = isPlain ? conf.plainBadge || conf.confluenceBadge : conf.confluenceBadge;
      const bottomLineText = conf.bottomLine;
      const verdictColor =
        conf.badgeColor === "emerald"
          ? "text-emerald-400"
          : conf.badgeColor === "rose"
          ? "text-rose-400"
          : conf.badgeColor === "amber"
          ? "text-amber-400"
          : "text-cyan-400";

      const backendPillars = conf.pillars || [];
      const techP = backendPillars.find((p) => p.pillar === "TECHNICAL_STRUCTURE");
      const fundP = backendPillars.find((p) => p.pillar === "FUNDAMENTAL_SOLVENCY");
      const smartP = backendPillars.find((p) => p.pillar === "SMART_MONEY_FLOW");
      const macroP = backendPillars.find((p) => p.pillar === "MACRO_SAFETY_FLOOR");

      const pillars: ConfluencePillarDisplay[] = [
        {
          id: "TECHNICAL_STRUCTURE",
          label: "Technical Confluence",
          weightPct: 25,
          score: techP?.score != null ? Math.round(techP.score) : null,
          scoreDisplay: techP?.score != null ? `${Math.round(techP.score)} / 100` : "UNAVAILABLE",
          status: techP?.status || "unavailable",
          detail: (isPlain ? techP?.plainDetail : techP?.detail) || techP?.detail || "Chart pattern and volume accumulation evaluated.",
          icon: techP?.icon || "📈",
        },
        {
          id: "FUNDAMENTAL_SOLVENCY",
          label: "Fundamental Moat",
          weightPct: 25,
          score: fundP?.score != null ? Math.round(fundP.score) : null,
          scoreDisplay: fundP?.score != null ? `${Math.round(fundP.score)} / 100` : "UNAVAILABLE",
          status: fundP?.status || "unavailable",
          detail: (isPlain ? fundP?.plainDetail : fundP?.detail) || fundP?.detail || "Balance sheet solvency and financial quality evaluated.",
          icon: fundP?.icon || "🏢",
        },
        {
          id: "SMART_MONEY_FLOW",
          label: "Institutional Flow",
          weightPct: 25,
          score: smartP?.score != null ? Math.round(smartP.score) : null,
          scoreDisplay: smartP?.score != null ? `${Math.round(smartP.score)} / 100` : "UNAVAILABLE",
          status: smartP?.status || "unavailable",
          detail: (isPlain ? smartP?.plainDetail : smartP?.detail) || smartP?.detail || "SEC Form 4 insider transactions and options order flow evaluated.",
          icon: smartP?.icon || "🏛️",
        },
        {
          id: "MACRO_SAFETY_FLOOR",
          label: "Macro Environment",
          weightPct: 25,
          score: macroP?.score != null ? Math.round(macroP.score) : null,
          scoreDisplay: macroP?.score != null ? `${Math.round(macroP.score)} / 100` : "UNAVAILABLE",
          status: macroP?.status || "unavailable",
          detail: (isPlain ? macroP?.plainDetail : macroP?.detail) || macroP?.detail || "Treasury yield curve and credit spread tailwinds evaluated.",
          icon: macroP?.icon || "🛡️",
        },
      ];

      return {
        score,
        verdictColor,
        confluenceBadge,
        bottomLineText,
        pillars,
      };
    }

    // 2. Strict Fail-Closed Check: Degraded Mode or Unverified State
    if (
      data?.degradedMode ||
      data?.decisionUnavailable ||
      data?.decisionTrace?.decisionState === "UNVERIFIED" ||
      data?.optimalExecution?.execution_status === "UNVERIFIED_ASSET"
    ) {
      const pillars: ConfluencePillarDisplay[] = [
        {
          id: "TECHNICAL_STRUCTURE",
          label: "Technical Confluence",
          weightPct: 25,
          score: null,
          scoreDisplay: "UNAVAILABLE",
          status: "unavailable",
          detail: "Live exchange feed unavailable: Quantitative technical evaluation offline.",
          icon: "📈",
        },
        {
          id: "FUNDAMENTAL_SOLVENCY",
          label: "Fundamental Moat",
          weightPct: 25,
          score: null,
          scoreDisplay: "UNAVAILABLE",
          status: "unavailable",
          detail: "Audited SEC financial statements unavailable for this unverified asset.",
          icon: "🏢",
        },
        {
          id: "SMART_MONEY_FLOW",
          label: "Institutional Flow",
          weightPct: 25,
          score: null,
          scoreDisplay: "UNAVAILABLE",
          status: "unavailable",
          detail: "Verified insider filings and options order flow unavailable for this asset.",
          icon: "🏛️",
        },
        {
          id: "MACRO_SAFETY_FLOOR",
          label: "Macro Environment",
          weightPct: 25,
          score: null,
          scoreDisplay: "UNAVAILABLE",
          status: "unavailable",
          detail: "Federal Reserve macroeconomic indicators and credit yield spreads uncalibrated.",
          icon: "🛡️",
        },
      ];

      return {
        score: 0,
        verdictColor: "text-slate-400",
        confluenceBadge: isPlain ? "Tape Only (Degraded)" : "Display Only (Unverified)",
        bottomLineText: "Analytical backend decision authority is unreachable. Displaying market data tape only.",
        pillars,
      };
    }

    // 3. Fallback: Refuse to synthesize speculative local score or actionable recommendations
    const pendingPillars: ConfluencePillarDisplay[] = [
      {
        id: "TECHNICAL_STRUCTURE",
        label: "Technical Confluence",
        weightPct: 25,
        score: null,
        scoreDisplay: "PENDING",
        status: "neutral",
        detail: "Awaiting multi-timeframe chart structure and volatility contraction confirmation.",
        icon: "📈",
      },
      {
        id: "FUNDAMENTAL_SOLVENCY",
        label: "Fundamental Moat",
        weightPct: 25,
        score: null,
        scoreDisplay: "PENDING",
        status: "neutral",
        detail: "Awaiting audited balance sheet solvency and margin metrics from fundamental database.",
        icon: "🏢",
      },
      {
        id: "SMART_MONEY_FLOW",
        label: "Institutional Flow",
        weightPct: 25,
        score: null,
        scoreDisplay: "PENDING",
        status: "neutral",
        detail: "Awaiting corporate insider SEC Form 4 and options tape synchronization.",
        icon: "🏛️",
      },
      {
        id: "MACRO_SAFETY_FLOOR",
        label: "Macro Environment",
        weightPct: 25,
        score: null,
        scoreDisplay: "PENDING",
        status: "neutral",
        detail: "Awaiting Federal Reserve yield curve and credit spread calibration.",
        icon: "🛡️",
      },
    ];

    return {
      score: 0,
      verdictColor: "text-slate-400",
      confluenceBadge: isPlain ? "Awaiting Backend Confluence" : "Evaluation Pending",
      bottomLineText: "CONFLUENCE EVALUATION PENDING: Quantitative confluence model results pending from analytical backend.",
      pillars: pendingPillars,
    };
  }, [data, isPlain]);

  return (
    <div className="bg-[#0b101b] border border-[#1d293d] rounded-2xl p-4 sm:p-5 shadow-xl space-y-4 font-sans text-xs">
      {/* Top Banner: Composite Confluence Title & Score Gauge */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#182335] pb-3 sm:pb-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className={`w-2.5 h-2.5 rounded-full ${synthesis.score >= 75 ? "bg-emerald-400 animate-ping" : "bg-slate-500"}`}></span>
            <span className="text-xs font-extrabold uppercase tracking-wider text-slate-400 font-mono">
              {isPlain ? `Multi-Pillar Evidence • ${cleanSym}` : `Quantitative Confluence • ${cleanSym}`}
            </span>
          </div>
          <h3 className="text-base sm:text-lg font-bold tracking-tight text-white font-mono">
            Multi-Model Confluence Attribution
          </h3>
          <p className="text-xs text-slate-400 font-sans">
            Four distinct, equally weighted analytical dimensions evaluated for structural conviction.
          </p>
        </div>

        {/* Big Pill Score Gauge & Inspect Why button */}
        <div className="flex items-center gap-3">
          <div className="flex items-center space-x-3 bg-[#06090f] px-3.5 py-2 rounded-xl border border-[#24334b]">
            <div className="text-right">
              <span className="text-[11px] uppercase font-bold text-slate-400 block font-mono">
                {isPlain ? "Confluence Evidence" : "Confluence Score"}
              </span>
              <span className="text-xs text-slate-300 font-medium">
                {synthesis.confluenceBadge}
              </span>
            </div>
            <div className={`text-2xl sm:text-3xl font-black font-mono tabular-nums ${synthesis.verdictColor}`}>
              {synthesis.score > 0 ? (
                <>
                  {synthesis.score}<span className="text-xs text-slate-500 font-normal">/100</span>
                </>
              ) : (
                <span className="text-sm font-bold text-slate-400">PENDING</span>
              )}
            </div>
          </div>

          {onOpenWhy && (
            <button
              id="why-score-btn"
              type="button"
              onClick={onOpenWhy}
              aria-label={`Why Score ${synthesis.score}? Inspect Confluence Attribution`}
              className="text-xs text-cyan-400 hover:text-cyan-300 font-mono font-bold cursor-pointer py-2 px-3 rounded-xl border border-cyan-800/60 bg-[#090d14] hover:bg-[#121b2a] transition-colors inline-flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-cyan-400"
            >
              <span>Inspect Telemetry</span>
              <span>→</span>
            </button>
          )}
        </div>
      </div>

      {/* 4 Distinct, Equally Weighted Evidence Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {synthesis.pillars.map((pillar) => (
          <div
            key={pillar.id}
            className="bg-[#090d14] border border-[#1e293b] rounded-xl p-3.5 flex flex-col justify-between gap-2 shadow-sm"
          >
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-1.5 min-w-0">
                <span className="text-sm shrink-0">{pillar.icon}</span>
                <span className="font-bold text-slate-200 uppercase tracking-wider font-mono text-[11px] truncate">
                  {pillar.label}
                </span>
              </div>
              <span className="text-[10px] font-mono text-slate-500 font-semibold shrink-0">
                {pillar.weightPct}% Weight
              </span>
            </div>

            <div className="flex items-center justify-between my-0.5">
              <span className={`text-lg sm:text-xl font-black font-mono tabular-nums ${getPillarScoreColor(pillar.scoreDisplay, pillar.score)}`}>
                {pillar.scoreDisplay}
              </span>
              <span
                className={`w-2.5 h-2.5 rounded-full ${getPillarStatusDot(pillar.status)}`}
                title={`Status: ${pillar.status.toUpperCase()}`}
              />
            </div>

            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {pillar.detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
