"use client";

import React from "react";

export interface IntelligenceSuccessStateProps {
  title: string;
  message: string;
  auditHash?: string;
  certificationId?: string;
  onProceed?: () => void;
}

export default function IntelligenceSuccessState({
  title,
  message,
  auditHash,
  certificationId = "CERT-PHASE-31-M11",
  onProceed,
}: IntelligenceSuccessStateProps) {
  return (
    <div className="rounded-2xl border border-emerald-500/40 bg-emerald-950/30 p-6 md:p-8 max-w-2xl mx-auto my-6 text-center space-y-4">
      <div className="w-12 h-12 rounded-full bg-emerald-900/60 border border-emerald-500/50 flex items-center justify-center text-emerald-400 mx-auto">
        <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>

      <h3 className="text-lg font-bold text-white tracking-tight">{title}</h3>
      <p className="text-xs font-mono text-emerald-200/90 leading-relaxed max-w-lg mx-auto">
        {message}
      </p>

      <div className="flex flex-wrap items-center justify-center gap-2 font-mono text-[11px] text-slate-300 pt-2">
        <span className="px-2 py-0.5 rounded bg-emerald-900/40 border border-emerald-500/30 text-emerald-300">
          Gate: {certificationId}
        </span>
        {auditHash && (
          <span className="px-2 py-0.5 rounded bg-black/40 border border-[#24324A] text-slate-400 truncate max-w-xs">
            Hash: {auditHash}
          </span>
        )}
      </div>

      {onProceed && (
        <div className="pt-2">
          <button
            onClick={onProceed}
            className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-mono text-xs font-semibold transition-colors"
          >
            Acknowledge & Continue
          </button>
        </div>
      )}
    </div>
  );
}
