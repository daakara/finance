"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import TerminalShell from '../../components/terminal/TerminalShell';

interface RadarAsset {
  ticker: string;
  name: string;
  price: number;
  rsRating: number;
  vcpStage: string;
  volumeDryUpPct: number;
  confluenceScore: number;
  catalyst: string;
  category: 'VCP' | 'SMART_MONEY' | 'VALUE';
}

const CANONICAL_RADAR_ASSETS: RadarAsset[] = [
  {
    ticker: 'GOOGL',
    name: 'Alphabet Inc.',
    price: 181.90,
    rsRating: 92,
    vcpStage: '4T (-1.8% on Pivot)',
    volumeDryUpPct: -64,
    confluenceScore: 94,
    catalyst: '2 Corporate Directors purchased $1.2M at $178 floor',
    category: 'VCP',
  },
  {
    ticker: 'NVDA',
    name: 'NVIDIA Corporation',
    price: 128.40,
    rsRating: 96,
    vcpStage: '3T (-2.4% on Pivot)',
    volumeDryUpPct: -48,
    confluenceScore: 91,
    catalyst: 'Congressional Commerce Committee Accumulation',
    category: 'VCP',
  },
  {
    ticker: 'ANET',
    name: 'Arista Networks',
    price: 312.10,
    rsRating: 89,
    vcpStage: '2T (-3.1% on Pivot)',
    volumeDryUpPct: -55,
    confluenceScore: 87,
    catalyst: 'Magic Formula Top Decile ROIC (34.2%)',
    category: 'SMART_MONEY',
  },
  {
    ticker: 'LLY',
    name: 'Eli Lilly & Co.',
    price: 945.20,
    rsRating: 94,
    vcpStage: '3T (-2.1% on Pivot)',
    volumeDryUpPct: -52,
    confluenceScore: 90,
    catalyst: 'Institutional Accumulation + PEG 1.1',
    category: 'VALUE',
  },
];

export default function RadarPage() {
  const [activeFilter, setActiveFilter] = useState<'ALL' | 'VCP' | 'SMART_MONEY' | 'VALUE'>('ALL');

  const filteredAssets = CANONICAL_RADAR_ASSETS.filter((a) =>
    activeFilter === 'ALL' ? true : a.category === activeFilter
  );

  return (
    <TerminalShell activeHub="radar">
      <div className="space-y-6">
        {/* Top Market Regime Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-xl border border-slate-800 bg-slate-900/80 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className="h-3 w-3 rounded-full bg-emerald-400 animate-pulse" />
            <div>
              <div className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
                Market Regime: Confirmed Uptrend
              </div>
              <div className="text-sm font-bold text-white">
                S&P 500 &gt; 21-EMA (+3.2% Spread) · Institutional Distribution: Low · Market Poise: 91/100
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/setups"
              className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold font-mono tracking-tight transition-colors shadow-sm"
            >
              Open Tactical Setups →
            </Link>
          </div>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            {(['ALL', 'VCP', 'SMART_MONEY', 'VALUE'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setActiveFilter(filter)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                  activeFilter === filter
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                {filter === 'ALL' ? 'All Confluences' : filter.replace('_', ' ')}
              </button>
            ))}
          </div>

          <div className="text-xs font-mono text-slate-400">
            Scanning 60+ Assets · <span className="text-emerald-400 font-bold">{filteredAssets.length} Qualified</span>
          </div>
        </div>

        {/* Radar Assets Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredAssets.map((asset) => (
            <div
              key={asset.ticker}
              className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/70 transition-all space-y-4"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-black font-mono tracking-tight text-white">
                      {asset.ticker}
                    </span>
                    <span className="text-xs text-slate-400 font-medium">
                      {asset.name}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-slate-500 mt-0.5">
                    Stage 2 Uptrend · Contraction: {asset.vcpStage}
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-sm font-bold font-mono text-white">
                    ${asset.price.toFixed(2)}
                  </div>
                  <div className="text-xs font-mono text-emerald-400 font-semibold">
                    Score: {asset.confluenceScore}/100
                  </div>
                </div>
              </div>

              {/* Confluence Metrics */}
              <div className="grid grid-cols-3 gap-2 p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60 text-xs font-mono">
                <div>
                  <div className="text-[10px] uppercase text-slate-400">Relative Strength</div>
                  <div className="font-bold text-cyan-400">{asset.rsRating}/99</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-slate-400">Vol Contraction</div>
                  <div className="font-bold text-emerald-400">{asset.volumeDryUpPct}%</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-slate-400">Strategy Model</div>
                  <div className="font-bold text-purple-400">{asset.category}</div>
                </div>
              </div>

              {/* Primary Catalyst */}
              <div className="text-xs text-slate-300 flex items-center gap-1.5">
                <span className="text-amber-400">⚡</span>
                <span>{asset.catalyst}</span>
              </div>

              {/* Action Trigger */}
              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between">
                <span className="text-[11px] font-mono text-slate-400">
                  Tight Risk Envelope Verified
                </span>
                <Link
                  href="/setups"
                  className="text-xs font-mono font-bold text-cyan-400 hover:text-cyan-300 transition-colors"
                >
                  View Setup Ticket →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </TerminalShell>
  );
}
