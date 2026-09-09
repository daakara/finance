"use client";

import React, { useState } from 'react';
import Link from 'next/link';

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
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-6">
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

        {/* Radar Section Title & Filters */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2">
              <span>Market Confluence Radar</span>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/80">
                Top 1% Setups
              </span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Screening 6,400+ equities across Mark Minervini VCP criteria, Congressional STOCK Act disclosures, and Magic Formula quality.
            </p>
          </div>

          <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono">
            {(['ALL', 'VCP', 'SMART_MONEY', 'VALUE'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setActiveFilter(filter)}
                className={`px-3 py-1 rounded-md transition-colors ${
                  activeFilter === filter
                    ? 'bg-cyan-600 text-white font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {filter === 'ALL' ? 'All Signals' : filter.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        {/* High-Density Data Grid */}
        <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-900/40 shadow-sm">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="p-3.5">Asset</th>
                <th className="p-3.5">Price</th>
                <th className="p-3.5">RS Rating</th>
                <th className="p-3.5">Contraction (VCP)</th>
                <th className="p-3.5">Volume Dry-Up</th>
                <th className="p-3.5">Confluence</th>
                <th className="p-3.5">Catalyst &amp; Smart Money</th>
                <th className="p-3.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredAssets.map((asset) => (
                <tr key={asset.ticker} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-3.5 font-bold text-white text-sm">
                    {asset.ticker}
                    <div className="text-[10px] text-slate-400 font-normal">{asset.name}</div>
                  </td>
                  <td className="p-3.5 text-slate-200 font-semibold">${asset.price.toFixed(2)}</td>
                  <td className="p-3.5">
                    <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 font-bold border border-emerald-800/60">
                      RS {asset.rsRating}
                    </span>
                  </td>
                  <td className="p-3.5 text-cyan-400 font-semibold">{asset.vcpStage}</td>
                  <td className="p-3.5 text-amber-400 font-semibold">{asset.volumeDryUpPct}%</td>
                  <td className="p-3.5">
                    <div className="flex items-center gap-1.5">
                      <div className="w-12 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-cyan-400 h-full rounded-full"
                          style={{ width: `${asset.confluenceScore}%` }}
                        />
                      </div>
                      <span className="font-bold text-white">{asset.confluenceScore}</span>
                    </div>
                  </td>
                  <td className="p-3.5 text-slate-300 text-[11px] max-w-xs truncate">{asset.catalyst}</td>
                  <td className="p-3.5 text-right">
                    <Link
                      href={`/setups?symbol=${asset.ticker}`}
                      className="px-3 py-1.5 rounded bg-slate-800 hover:bg-cyan-600 hover:text-white text-slate-300 transition-colors font-bold text-[11px]"
                    >
                      View Setup
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
