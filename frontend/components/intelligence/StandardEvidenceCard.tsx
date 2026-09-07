'use client';

import React, { useState } from 'react';
import { StandardEvidenceData } from '@/types/ux-foundations';

interface StandardEvidenceCardProps {
  evidence: StandardEvidenceData;
}

export const StandardEvidenceCard: React.FC<StandardEvidenceCardProps> = ({
  evidence,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopyHash = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(evidence.ledgerHash);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div
      role="region"
      aria-label={`Evidence Verification for ${evidence.claim}`}
      className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm"
    >
      {/* Top Banner */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400" />
          <span className="text-xs font-semibold text-slate-300">
            Cryptographic Evidence & Statistical Audit
          </span>
        </div>
        <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
          SEC / FINRA Immutable Audit Trail
        </span>
      </div>

      {/* Claim statement */}
      <div className="text-sm font-medium text-white mb-3">
        &ldquo;{evidence.claim}&rdquo;
      </div>

      {/* Statistical Evidence 4-Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        <div className="bg-slate-950 p-2 rounded border border-slate-800">
          <span className="text-[10px] uppercase font-mono text-slate-400 block">Sample Size</span>
          <span className="text-xs font-mono font-bold text-white">N = {evidence.sampleSize}</span>
        </div>
        <div className="bg-slate-950 p-2 rounded border border-slate-800">
          <span className="text-[10px] uppercase font-mono text-slate-400 block">P-Value</span>
          <span className="text-xs font-mono font-bold text-emerald-400">
            {evidence.pValue < 0.001 ? 'p < 0.001' : `p = ${evidence.pValue.toFixed(3)}`}
          </span>
        </div>
        <div className="bg-slate-950 p-2 rounded border border-slate-800">
          <span className="text-[10px] uppercase font-mono text-slate-400 block">Sharpe Impact</span>
          <span className="text-xs font-mono font-bold text-cyan-400">{evidence.sharpeImpact}</span>
        </div>
        <div className="bg-slate-950 p-2 rounded border border-slate-800">
          <span className="text-[10px] uppercase font-mono text-slate-400 block">Lookback</span>
          <span className="text-xs font-mono font-bold text-slate-300">{evidence.lookbackPeriod}</span>
        </div>
      </div>

      {/* Ledger Hash & Copy */}
      <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between gap-2">
        <div className="flex items-center space-x-2 truncate">
          <span className="text-[10px] font-mono text-slate-400 uppercase">SHA-256:</span>
          <code className="text-[11px] font-mono text-cyan-400 truncate bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
            {evidence.ledgerHash}
          </code>
        </div>
        <button
          type="button"
          onClick={handleCopyHash}
          className="shrink-0 text-[11px] font-mono font-medium px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500"
        >
          {copied ? '✓ Copied' : 'Copy Hash'}
        </button>
      </div>
    </div>
  );
};
