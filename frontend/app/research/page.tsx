"use client";

import React, { useState } from 'react';
import Link from 'next/link';

export default function ResearchPage() {
  const [ticker, setTicker] = useState<string>('GOOGL');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="text-xs font-mono font-semibold text-cyan-400 mb-1">
              ARX Terminal / Research Diagnostics
            </div>
            <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2">
              <span>Fundamental &amp; Smart Money Diagnostics</span>
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Deep-dive into corporate SEC filings, Congressional trade clusters, and return on invested capital.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/radar"
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs font-mono text-slate-300 border border-slate-800 transition-colors"
            >
              ← Back to Radar
            </Link>
          </div>
        </div>

        {/* Ticker Search & Diagnosis Card */}
        <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/60 space-y-5 font-mono">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400">Analyze Symbol:</span>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="px-3 py-1.5 rounded bg-slate-950 border border-slate-700 text-white font-bold text-sm w-32 focus:outline-none focus:border-cyan-400"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs pt-2">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-slate-400">Return on Invested Capital (ROIC)</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">31.4%</div>
              <div className="text-[10px] text-slate-500">Top Decile Quality (Magic Formula)</div>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-slate-400">Net Debt to EBITDA</div>
              <div className="text-xl font-bold text-cyan-400 mt-1">-0.42x</div>
              <div className="text-[10px] text-slate-500">Fortress Balance Sheet (Net Cash)</div>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-slate-400">Congressional Accumulation</div>
              <div className="text-xl font-bold text-amber-400 mt-1">3 Purchases</div>
              <div className="text-[10px] text-slate-500">Senate Commerce Committee Members</div>
            </div>
          </div>

          <div className="p-4 rounded bg-slate-950 border border-slate-800/80 text-xs space-y-2">
            <div className="text-cyan-400 font-bold uppercase tracking-wider text-[11px]">
              SEC Form 4 Insider Clustered Transactions:
            </div>
            <div className="text-slate-300 text-[11px] leading-relaxed">
              • 2026-08-24: Director purchased 5,000 shares at $178.10 ($890,500 open market purchase).<br />
              • 2026-08-22: Chief Accounting Officer acquired 2,200 shares at $177.90 ($391,380 open market).<br />
              • Confluence status: <strong>High Insider Conviction</strong> matching stage 2 base breakout.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
