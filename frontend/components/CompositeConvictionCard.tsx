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
}

export default function CompositeConvictionCard({
  symbol,
  data,
  macro,
  insiders,
  userRole,
}: CompositeConvictionCardProps) {
  const cleanSym = symbol.toUpperCase().replace("-USD", "");
  const matchedInsider = insiders.find((i) => i.ticker === cleanSym);
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
      const verdictTitle = isPlain ? conf.plainRating || conf.confluenceRating : conf.confluenceRating;
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

      const reasons = (conf.pillars || []).map((p) => ({
        label: p.label,
        plainLabel: p.plainLabel,
        detail: p.detail,
        plainDetail: p.plainDetail,
        status: p.status,
        icon: p.icon,
      }));

      return {
        score,
        verdictTitle,
        verdictColor,
        confluenceBadge,
        bottomLineText,
        reasons,
      };
    }

    // 2. Strict Fail-Closed Check: Degraded Mode or Unverified State
    if (
      data?.degradedMode ||
      data?.decisionUnavailable ||
      data?.decisionTrace?.decisionState === "UNVERIFIED" ||
      data?.optimalExecution?.execution_status === "UNVERIFIED_ASSET"
    ) {
      return {
        score: 0,
        verdictTitle: isPlain ? "UNVERIFIED ASSET" : "DECISION ENGINE UNREACHABLE",
        verdictColor: "text-slate-400",
        confluenceBadge: isPlain ? "Tape Only (Degraded)" : "Display Only (Unverified)",
        bottomLineText: "Analytical backend decision authority is unreachable. Displaying market data tape only.",
        reasons: [
          {
            label: "Analytical Engine",
            plainLabel: "Decision System",
            detail: "Backend analytical authority unreachable. Platform refuses to synthesize speculative confluence.",
            plainDetail: "Decision engine offline. Showing authentic market tape only.",
            status: "warning" as const,
            icon: "⚠️",
          },
        ],
      };
    }

    // 3. Fallback: Refuse to synthesize speculative local score or actionable recommendations
    return {
      score: 0,
      verdictTitle: isPlain ? "CONFLUENCE EVALUATION PENDING" : "CONFLUENCE EVIDENCE PENDING",
      verdictColor: "text-slate-400",
      confluenceBadge: isPlain ? "Awaiting Backend Confluence" : "Evaluation Pending",
      bottomLineText: "Quantitative confluence model results pending from analytical backend.",
      reasons: [
        {
          label: "Quantitative Confluence",
          plainLabel: "Evidence Confluence",
          detail: "Backend confluence engine evaluation not available for this asset.",
          plainDetail: "Awaiting multi-pillar confluence score from analytical server.",
          status: "neutral" as const,
          icon: "⏳",
        },
      ],
    };
  }, [data, isPlain]);

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-2xl p-4 sm:p-5 shadow-2xl space-y-4 font-sans">
      {/* Top Banner: Composite Score & Actionable Consensus */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className={`w-2.5 h-2.5 rounded-full ${synthesis.score >= 75 ? "bg-emerald-400 animate-ping" : "bg-slate-500"}`}></span>
            <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400 font-mono">
              {isPlain ? `Multi-Pillar Evidence • ${cleanSym}` : `Quantitative Confluence • ${cleanSym}`}
            </span>
          </div>
          <h2 className={`text-base sm:text-lg font-black tracking-tight ${synthesis.verdictColor}`}>
            {synthesis.verdictTitle}
          </h2>
        </div>

        {/* Big Circular/Pill Score Gauge */}
        <div className="flex items-center space-x-3 bg-[#090d14] px-3.5 py-2 rounded-xl border border-[#243044]">
          <div className="text-right">
            <span className="text-[9px] uppercase font-bold text-slate-400 block font-mono">
              {isPlain ? "Confluence Evidence" : "Confluence Score"}
            </span>
            <span className="text-xs text-slate-300 font-medium">
              {synthesis.confluenceBadge}
            </span>
          </div>
          <div className={`text-2xl sm:text-3xl font-black font-mono tabular-nums ${synthesis.verdictColor}`}>
            {synthesis.score}<span className="text-xs text-slate-500 font-normal">/100</span>
          </div>
        </div>
      </div>

      {/* 💡 The Bottom Line Callout Box */}
      <div className="bg-[#090d14] border border-cyan-500/30 p-3 sm:p-3.5 rounded-xl flex items-start gap-2.5 shadow-sm">
        <span className="text-base sm:text-lg shrink-0 select-none">🎯</span>
        <div className="space-y-0.5 min-w-0">
          <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider block font-mono">
            The Bottom Line (No Wall Street Fluff)
          </span>
          <p className="text-xs sm:text-sm text-slate-200 leading-relaxed">
            {synthesis.bottomLineText}
          </p>
        </div>
      </div>

      {/* 4 Multi-Source Evidence Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {synthesis.reasons.map((r, idx) => (
          <div
            key={idx}
            className="bg-[#090d14] p-3 rounded-xl border border-[#1e293b] space-y-1.5 flex flex-col justify-between"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-1.5">
                <span className="text-sm">{r.icon}</span>
                <strong className="text-xs font-bold text-slate-200">
                  {isPlain ? r.plainLabel : r.label}
                </strong>
              </div>
              <span
                className={`w-2 h-2 rounded-full ${
                  r.status === "positive"
                    ? "bg-emerald-400"
                    : r.status === "warning"
                    ? "bg-rose-400"
                    : "bg-cyan-400"
                }`}
              ></span>
            </div>
            <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
              {isPlain ? r.plainDetail : r.detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}