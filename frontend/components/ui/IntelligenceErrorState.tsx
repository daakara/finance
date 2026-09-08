"use client";

import React from "react";

export interface IntelligenceErrorStateProps {
  errorCode: string;
  failureClass: string;
  message: string;
  recoverySteps?: string[];
  onRetry?: () => void;
  onActivateSafeMode?: () => void;
}

export default function IntelligenceErrorState({
  errorCode,
  failureClass,
  message,
  recoverySteps = ["Verify telemetry invariant integrity", "Trigger L1 cache refresh", "Escalate to committee administrator"],
  onRetry,
  onActivateSafeMode,
}: IntelligenceErrorStateProps) {
  return (
    <div
      role="alert"
      className="rounded-2xl border border-red-500/50 bg-red-950/40 p-6 md:p-8 max-w-3xl mx-auto my-6 text-left space-y-5"
    >
      <div className="flex items-center justify-between border-b border-red-500/30 pb-3">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
          <span className="text-xs font-mono font-bold text-red-400 uppercase tracking-wider">
            FAIL-CLOSED DEFENSE ACTIVATED
          </span>
        </div>
        <span className="text-xs font-mono px-2 py-0.5 rounded bg-red-900/60 text-red-200 border border-red-500/40 font-bold">
          {errorCode}
        </span>
      </div>

      <div>
        <h3 className="text-base font-semibold text-white mb-1">
          {failureClass}
        </h3>
        <p className="text-xs font-mono text-red-200/90 leading-relaxed bg-black/40 p-3 rounded-lg border border-red-500/30">
          {message}
        </p>
      </div>

      {recoverySteps.length > 0 && (
        <div>
          <h4 className="text-xs font-mono text-slate-300 uppercase tracking-wider font-semibold mb-2">
            Recommended Recovery Procedures:
          </h4>
          <ul className="list-disc list-inside space-y-1 text-xs font-mono text-slate-300">
            {recoverySteps.map((step, idx) => (
              <li key={idx}>{step}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex items-center gap-3 pt-3 border-t border-red-500/30">
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-mono text-xs font-semibold border border-slate-600 transition-colors"
          >
            Retry Telemetry Ingestion
          </button>
        )}
        {onActivateSafeMode && (
          <button
            onClick={onActivateSafeMode}
            className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-mono text-xs font-semibold shadow-lg shadow-red-950/60 transition-colors"
          >
            Engage L4 Safe Mode
          </button>
        )}
      </div>
    </div>
  );
}
