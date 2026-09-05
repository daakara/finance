import React from "react";
import Link from "next/link";

interface AuthorEeatBadgeProps {
  lastUpdated?: string;
  topic?: string;
  className?: string;
}

export default function AuthorEeatBadge({
  lastUpdated = "September 2026",
  topic = "Quantitative Finance & Algorithmic Systems",
  className = "",
}: AuthorEeatBadgeProps) {
  return (
    <div
      className={`bg-[#0b1322] border border-[#1d2d47] rounded-xl p-4 text-xs font-sans text-slate-300 ${className}`}
      aria-label="Editorial Credibility & Methodology Attribution"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#162238] pb-3 mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/30 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-mono font-bold text-xs shrink-0">
            ARX
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-white">ARX Quantitative Research Group</span>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-cyan-950/80 border border-cyan-700/50 text-cyan-300">
                Institutional Review Board
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Authored & Audited by Chartered Financial Analysts (CFA) & Econometric Systems Engineers
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400 sm:text-right shrink-0">
          <div>
            <span className="text-slate-400">Topic:</span>{" "}
            <span className="text-slate-200 font-medium">{topic}</span>
          </div>
          <div className="hidden sm:block text-slate-600">•</div>
          <div>
            <span className="text-slate-400">Audited:</span>{" "}
            <span className="text-cyan-400">{lastUpdated}</span>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400">
        <p className="leading-relaxed">
          <strong className="text-slate-200">Methodology Standard:</strong> All mathematical models, statutory STOCK Act disclosures, and execution geometries are continuously audited via automated Kupiec POF backtests and walk-forward RMSE tracking.
        </p>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <Link href="/guide/" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2">
            Field Manual
          </Link>
          <span className="text-slate-600">•</span>
          <Link href="/glossary/" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2">
            Quant Glossary
          </Link>
        </div>
      </div>
    </div>
  );
}
