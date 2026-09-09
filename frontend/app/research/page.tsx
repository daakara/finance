"use client";

import React from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";

export default function ResearchPage() {
  const researchAssets = [
    { ticker: 'GOOGL', name: 'Alphabet Inc.', roic: '31.4%', debtToEbitda: '0.2x', insiderFootprint: 'Net Inflow +$1.2M', catalyst: 'Enterprise AI & Cloud Operating Margin Expansion' },
    { ticker: 'NVDA', name: 'NVIDIA Corp.', roic: '68.2%', debtToEbitda: '0.4x', insiderFootprint: 'Congress Committee Accumulation', catalyst: 'Data Center Architecture Dominance' },
    { ticker: 'ANET', name: 'Arista Networks', roic: '34.2%', debtToEbitda: '0.0x (Net Cash)', insiderFootprint: 'Form 4 Zero Sales', catalyst: '800G / 1.6T AI Networking Infrastructure Cycle' },
  ];

  return (
    <TerminalShell activeHub="research">
      <div className="space-y-6">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400">
              Fundamental Catalysts & SEC Form 4
            </div>
            <div className="text-sm font-semibold text-slate-200 mt-0.5">
              Deep institutional diagnostics: Return on Capital (ROIC), debt buffers, and insider accumulation.
            </div>
          </div>
          <span className="text-xs font-mono text-emerald-400 bg-emerald-950/80 px-3 py-1.5 rounded-lg border border-emerald-800">
            SEC Form 4 Feed: Realtime
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {researchAssets.map((asset) => (
            <div key={asset.ticker} className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold font-mono text-white">{asset.ticker}</h3>
                  <p className="text-xs text-slate-400">{asset.name}</p>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  ROIC {asset.roic}
                </span>
              </div>

              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Debt/EBITDA:</span>
                  <span className="text-white">{asset.debtToEbitda}</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Insider Activity:</span>
                  <span className="text-emerald-400 font-bold">{asset.insiderFootprint}</span>
                </div>
              </div>

              <div className="pt-2 text-xs text-slate-300">
                <span className="font-semibold text-slate-400 block mb-1">Primary Catalyst:</span>
                {asset.catalyst}
              </div>

              <div className="pt-2 border-t border-slate-800">
                <Link
                  href={`/stock/${asset.ticker.toLowerCase()}`}
                  className="text-xs font-mono text-cyan-400 hover:text-cyan-300 font-bold flex items-center justify-between"
                >
                  <span>Detailed SEC Disclosures</span>
                  <span>→</span>
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </TerminalShell>
  );
}
