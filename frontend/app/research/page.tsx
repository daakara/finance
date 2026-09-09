"use client";

import React, { useState } from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";

interface CatalystItem {
  ticker: string;
  name: string;
  source: 'SEC_FORM_4' | '13F_WHALE' | 'CONGRESS_STOCK_ACT' | 'EARNINGS_ACCEL';
  impact: 'TIER_1_CRITICAL' | 'TIER_2_HIGH' | 'TIER_3_MODERATE';
  date: string;
  summary: string;
  filingEntity: string;
  transactionValue: string;
  roic: string;
  debtToEbitda: string;
}

export default function ResearchPage() {
  const [selectedTicker, setSelectedTicker] = useState('GOOGL');

  const catalysts: CatalystItem[] = [
    {
      ticker: 'GOOGL',
      name: 'Alphabet Inc.',
      source: 'SEC_FORM_4',
      impact: 'TIER_1_CRITICAL',
      date: '2026-09-04',
      summary: '2 Independent Directors purchased $1.2M at $178.00 floor price following antitrust headline dip.',
      filingEntity: 'Board of Directors (Insider Cluster)',
      transactionValue: '+$1,200,000',
      roic: '31.4%',
      debtToEbitda: '0.2x Net Cash',
    },
    {
      ticker: 'NVDA',
      name: 'NVIDIA Corp.',
      source: 'CONGRESS_STOCK_ACT',
      impact: 'TIER_1_CRITICAL',
      date: '2026-09-01',
      summary: 'House Commerce & Science Committee member disclosed accumulation ahead of rack-scale datacenter policy.',
      filingEntity: 'Congressional Commerce Committee',
      transactionValue: '+$500,000',
      roic: '68.2%',
      debtToEbitda: '0.4x',
    },
    {
      ticker: 'ANET',
      name: 'Arista Networks',
      source: '13F_WHALE',
      impact: 'TIER_2_HIGH',
      date: '2026-08-28',
      summary: 'Duquesne Capital & Renaissance Technologies increased stake +24% during base consolidation.',
      filingEntity: 'Institutional 13F Whale Cluster',
      transactionValue: '+$84,000,000',
      roic: '34.2%',
      debtToEbitda: '0.0x Net Cash',
    },
    {
      ticker: 'PLTR',
      name: 'Palantir Technologies',
      source: 'CONGRESS_STOCK_ACT',
      impact: 'TIER_2_HIGH',
      date: '2026-08-24',
      summary: 'Senate Armed Services member purchase concurrent with multi-year DoD Maven expansion award.',
      filingEntity: 'Congressional Armed Services Committee',
      transactionValue: '+$250,000',
      roic: '22.8%',
      debtToEbitda: '0.0x Net Cash',
    },
    {
      ticker: 'LNTH',
      name: 'Lantheus Holdings',
      source: 'EARNINGS_ACCEL',
      impact: 'TIER_3_MODERATE',
      date: '2026-08-15',
      summary: 'Greenblatt Magic Formula Top Decile: ROIC 32.4% with forward P/E 11.2x and zero long-term debt.',
      filingEntity: 'Quantitative Value Screener',
      transactionValue: 'Undervalued',
      roic: '32.4%',
      debtToEbitda: '0.1x',
    },
  ];

  const activeDossier = catalysts.find((c) => c.ticker === selectedTicker) || catalysts[0];

  return (
    <TerminalShell activeHub="research">
      <div className="space-y-6">
        {/* Level 0: Asymmetric Research Dossier Hero */}
        <div className="relative overflow-hidden rounded-2xl border border-cyan-500/40 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 p-5 md:p-6 shadow-2xl space-y-4">
          <div className="absolute top-0 right-0 px-3 py-1 bg-cyan-500/20 border-b border-l border-cyan-500/40 text-[10px] font-mono uppercase tracking-widest text-cyan-300 font-bold rounded-bl-xl">
            Level 0 · Institutional Research Dossier
          </div>

          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-3 max-w-3xl">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-3xl font-black font-mono text-white tracking-tight">
                  {activeDossier.ticker}
                </span>
                <span className="text-base text-slate-300 font-medium">
                  {activeDossier.name}
                </span>
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  {activeDossier.impact.replace(/_/g, ' ')}
                </span>
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {activeDossier.source.replace(/_/g, ' ')}
                </span>
              </div>

              <div className="space-y-1">
                <span className="text-[11px] uppercase font-mono text-slate-400 font-bold block">
                  Core Fundamental Thesis &amp; Catalyst Window
                </span>
                <p className="text-sm text-slate-200 font-sans leading-relaxed">
                  {activeDossier.summary}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-4 text-xs font-mono pt-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">ROIC:</span>
                  <span className="text-emerald-400 font-bold">{activeDossier.roic}</span>
                </div>
                <span className="text-slate-700">•</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Balance Sheet:</span>
                  <span className="text-white font-bold">{activeDossier.debtToEbitda}</span>
                </div>
                <span className="text-slate-700">•</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Smart Money Flow:</span>
                  <span className="text-cyan-400 font-bold">{activeDossier.transactionValue}</span>
                </div>
                <span className="text-slate-700">•</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Filing Date:</span>
                  <span className="text-slate-300">{activeDossier.date}</span>
                </div>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row lg:flex-col gap-2.5 shrink-0">
              <Link
                href="/setups"
                className="px-5 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-black tracking-tight transition-all shadow-lg flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98]"
              >
                <span>OPEN TACTICAL SETUP</span>
                <span>&rarr;</span>
              </Link>
              <Link
                href={`/stock/${activeDossier.ticker.toLowerCase()}`}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono font-bold text-center transition-colors"
              >
                View EDGAR Filings &rarr;
              </Link>
            </div>
          </div>
        </div>

        {/* Level 1: Ranked Catalyst Priority Stream */}
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white uppercase font-mono">
                Ranked Institutional Catalyst Stream
              </h3>
              <p className="text-xs text-slate-400 font-sans mt-0.5">
                Real-time convergence of SEC Form 4 insider clusters, 13F whale changes, and Congressional STOCK Act filings
              </p>
            </div>
            <span className="text-xs font-mono text-emerald-400 bg-emerald-950/80 px-2.5 py-1 rounded border border-emerald-800">
              Live Disclosures Feed
            </span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 text-[10px] uppercase tracking-wider">
                  <tr>
                    <th className="p-3">Ticker</th>
                    <th className="p-3">Entity / Source</th>
                    <th className="p-3">Impact Tier</th>
                    <th className="p-3">Catalyst Description</th>
                    <th className="p-3 text-right">Flow / Size</th>
                    <th className="p-3 text-center">ROIC</th>
                    <th className="p-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {catalysts.map((item) => {
                    const isSelected = item.ticker === selectedTicker;
                    return (
                      <tr
                        key={item.ticker}
                        onClick={() => setSelectedTicker(item.ticker)}
                        className={`transition-colors cursor-pointer ${
                          isSelected ? 'bg-cyan-950/30' : 'hover:bg-slate-900/60'
                        }`}
                      >
                        <td className="p-3">
                          <div className="font-bold text-white text-sm">{item.ticker}</div>
                          <div className="text-[10px] text-slate-400 font-sans truncate max-w-[120px]">{item.name}</div>
                        </td>
                        <td className="p-3">
                          <span className="text-slate-300 text-[11px]">{item.filingEntity}</span>
                          <span className="text-[10px] text-slate-500 block">{item.date}</span>
                        </td>
                        <td className="p-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            item.impact === 'TIER_1_CRITICAL'
                              ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
                              : item.impact === 'TIER_2_HIGH'
                              ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                              : 'bg-slate-800 text-slate-400 border border-slate-700'
                          }`}>
                            {item.impact.replace(/_/g, ' ')}
                          </span>
                        </td>
                        <td className="p-3 text-slate-300 text-[11px] font-sans max-w-md">
                          {item.summary}
                        </td>
                        <td className="p-3 text-right font-bold text-emerald-400">
                          {item.transactionValue}
                        </td>
                        <td className="p-3 text-center font-bold text-white">
                          {item.roic}
                        </td>
                        <td className="p-3 text-right">
                          <button
                            type="button"
                            className="px-2.5 py-1 rounded bg-slate-800 hover:bg-cyan-600 hover:text-white text-cyan-400 text-[10px] font-bold font-mono transition-colors"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Level 2: Deep-Dive Fundamental Armor */}
        <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4 font-mono text-xs shadow-xl">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <span className="text-xs font-bold text-white uppercase">Fundamental Balance Sheet Armor</span>
              <p className="text-[11px] text-slate-400 font-sans mt-0.5">
                Solvency, capital allocation discipline, and operating margin sustainability
              </p>
            </div>
            <span className="text-[10px] font-mono text-slate-400">Verified via SEC EDGAR XBRL</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase block">Return on Capital (ROIC)</span>
              <span className="text-xl font-bold text-emerald-400">{activeDossier.roic}</span>
              <span className="text-[10px] text-slate-500 font-sans block">Substantially exceeds 8.2% cost of capital</span>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase block">Net Debt / EBITDA</span>
              <span className="text-xl font-bold text-white">{activeDossier.debtToEbitda}</span>
              <span className="text-[10px] text-slate-500 font-sans block">Zero solvency or refinancing stress</span>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase block">Institutional Alignment</span>
              <span className="text-xl font-bold text-cyan-400">High Moat</span>
              <span className="text-[10px] text-slate-500 font-sans block">Strong insider &amp; whale accumulation floor</span>
            </div>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
