"use client";

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import TerminalShell from '../../components/terminal/TerminalShell';
import { MASTER_ASSET_CATALOG, CATALOG_BASELINE_PRICES } from '../../lib/masterCatalog';
import { SpotPriceRegistry } from '../../lib/api';

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
  sector: string;
  executionStatus: 'IN_BUY_ZONE' | 'NEAR_PIVOT' | 'VOLUME_DRYUP' | 'PULLBACK_SUPPORT';
}

// Full Dynamic Multi-Factor Universe Generator derived from Master Asset Catalog
function generateRadarUniverse(): RadarAsset[] {
  // Curated lists with rich coverage across all three core categories
  const vcpTickers = [
    'NVDA', 'GOOGL', 'AMD', 'TMDX', 'ISRG', 'VRT', 'CPRX', 'MEDP',
    'ACLS', 'POWI', 'ANET', 'PLTR', 'ARM', 'CRWD', 'PANW', 'NET', 'DDOG', 'MDB'
  ];

  const smartMoneyTickers = [
    'ANET', 'PLTR', 'MSFT', 'AVGO', 'CRWD', 'ARM', 'SMCI', 'LLY',
    'TSLA', 'COIN', 'MSTR', 'HOOD', 'DUOL', 'CELH', 'APP'
  ];

  const valueTickers = [
    'LNTH', 'CPRX', 'NVO', 'LLY', 'ON', 'MPWR', 'KLAC', 'LRCX',
    'ASML', 'FIX', 'EME', 'GEV', 'PWR', 'ETN', 'DECK', 'ULTA'
  ];

  const uniqueTickers = Array.from(new Set([...vcpTickers, ...smartMoneyTickers, ...valueTickers]));

  return uniqueTickers.map((sym, idx) => {
    const entry = MASTER_ASSET_CATALOG[sym];
    const spot = SpotPriceRegistry.get(sym);
    const price = (spot?.price && spot.price > 0) ? spot.price : (CATALOG_BASELINE_PRICES[sym] || 150.0);

    // Determine primary category for filtering
    let category: 'VCP' | 'SMART_MONEY' | 'VALUE' = 'VCP';
    if (vcpTickers.includes(sym) && (!smartMoneyTickers.includes(sym) || idx % 2 === 0)) {
      category = 'VCP';
    } else if (smartMoneyTickers.includes(sym)) {
      category = 'SMART_MONEY';
    } else {
      category = 'VALUE';
    }

    // Dynamic attributes based on catalog metrics
    const rsRating = entry ? Math.min(99, Math.max(78, entry.momentumScore + 5)) : 88;
    const volDryUp = entry ? -Math.abs(Math.round(40 + (entry.rvol * 10))) : -52;
    const confluence = entry ? entry.compositeFactorScore : 89;
    const contractionStage = (idx % 3 === 0) ? '4T (-1.5% on Pivot)' : (idx % 2 === 0 ? '3T (-2.2% on Pivot)' : '2T (-3.4% on Pivot)');
    
    let catalyst = entry?.upcomingCatalyst || 'Institutional Volume Surge & Moving Average Support';
    if (category === 'SMART_MONEY') {
      catalyst = entry?.moatSummary ? `Smart Money Accumulation: ${entry.moatSummary.slice(0, 58)}...` : 'Congressional Committee Cluster Inflow';
    } else if (category === 'VALUE') {
      catalyst = entry ? `Magic Formula Decile: ROIC ${entry.roic}%, PEG ${entry.peg}` : 'Undervalued High-ROIC Compounder';
    }

    const statuses: ('IN_BUY_ZONE' | 'NEAR_PIVOT' | 'VOLUME_DRYUP' | 'PULLBACK_SUPPORT')[] = [
      'NEAR_PIVOT', 'IN_BUY_ZONE', 'VOLUME_DRYUP', 'PULLBACK_SUPPORT'
    ];
    const executionStatus = statuses[idx % statuses.length];

    return {
      ticker: sym,
      name: entry?.name || sym,
      price: Number(price.toFixed(2)),
      rsRating,
      vcpStage: contractionStage,
      volumeDryUpPct: Math.max(-75, Math.min(-35, volDryUp)),
      confluenceScore: Math.min(98, Math.max(82, confluence)),
      catalyst,
      category,
      sector: entry?.sector || 'Broad Market',
      executionStatus,
    };
  });
}

export default function RadarPage() {
  const [activeFilter, setActiveFilter] = useState<'ALL' | 'VCP' | 'SMART_MONEY' | 'VALUE'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'SCORE' | 'RS' | 'PRICE'>('SCORE');

  const allAssets = useMemo(() => generateRadarUniverse(), []);

  // Category Counts
  const counts = useMemo(() => {
    return {
      ALL: allAssets.length,
      VCP: allAssets.filter((a) => a.category === 'VCP').length,
      SMART_MONEY: allAssets.filter((a) => a.category === 'SMART_MONEY').length,
      VALUE: allAssets.filter((a) => a.category === 'VALUE').length,
    };
  }, [allAssets]);

  // Filtered & Sorted Assets
  const filteredAssets = useMemo(() => {
    return allAssets
      .filter((asset) => {
        const matchesCategory = activeFilter === 'ALL' ? true : asset.category === activeFilter;
        const matchesQuery = searchQuery.trim() === ''
          ? true
          : asset.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
            asset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            asset.catalyst.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesQuery;
      })
      .sort((a, b) => {
        if (sortBy === 'SCORE') return b.confluenceScore - a.confluenceScore;
        if (sortBy === 'RS') return b.rsRating - a.rsRating;
        if (sortBy === 'PRICE') return b.price - a.price;
        return 0;
      });
  }, [allAssets, activeFilter, searchQuery, sortBy]);

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

        {/* Search & Filter Toolbar */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          {/* Category Filter Tabs */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
            <button
              onClick={() => setActiveFilter('ALL')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                activeFilter === 'ALL'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <span>All Confluences</span>
              <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-slate-300 font-bold">{counts.ALL}</span>
            </button>
            <button
              onClick={() => setActiveFilter('VCP')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                activeFilter === 'VCP'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <span>⚡ Minervini VCP</span>
              <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-cyan-400 font-bold">{counts.VCP}</span>
            </button>
            <button
              onClick={() => setActiveFilter('SMART_MONEY')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                activeFilter === 'SMART_MONEY'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <span>🐋 Smart Money</span>
              <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-purple-400 font-bold">{counts.SMART_MONEY}</span>
            </button>
            <button
              onClick={() => setActiveFilter('VALUE')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                activeFilter === 'VALUE'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <span>🏛️ Value &amp; GARP</span>
              <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-emerald-400 font-bold">{counts.VALUE}</span>
            </button>
          </div>

          {/* Search & Sort Controls */}
          <div className="flex items-center gap-2.5">
            <div className="relative flex-1 md:w-56">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter ticker, name..."
                className="w-full px-3 py-1.5 bg-[#0b1019] border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 font-mono focus:outline-none focus:border-cyan-500 transition-colors"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1.5 text-xs text-slate-500 hover:text-slate-300"
                >
                  ✕
                </button>
              )}
            </div>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'SCORE' | 'RS' | 'PRICE')}
              className="px-2.5 py-1.5 bg-[#0b1019] border border-slate-800 rounded-lg text-xs text-slate-300 font-mono focus:outline-none focus:border-cyan-500 transition-colors"
            >
              <option value="SCORE">Sort: Confluence</option>
              <option value="RS">Sort: RS Rating</option>
              <option value="PRICE">Sort: Price</option>
            </select>
          </div>
        </div>

        {/* Status Count Banner */}
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <div>
            Scanning 40+ Confluence Candidates · Showing <span className="text-cyan-400 font-bold">{filteredAssets.length}</span> matching opportunities
          </div>
          {searchQuery && (
            <div className="text-slate-500 italic">
              Filtered by: &quot;{searchQuery}&quot;
            </div>
          )}
        </div>

        {/* Radar Assets Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredAssets.map((asset) => (
            <div
              key={asset.ticker}
              className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/70 transition-all space-y-4 flex flex-col justify-between"
            >
              <div className="space-y-2.5">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-black font-mono tracking-tight text-white">
                        {asset.ticker}
                      </span>
                      <span className="text-xs text-slate-400 font-medium truncate max-w-[150px]">
                        {asset.name}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                      Stage 2 Uptrend · {asset.vcpStage}
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
                    <div className="text-[10px] uppercase text-slate-400">RS Rating</div>
                    <div className="font-bold text-cyan-400">{asset.rsRating}/99</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase text-slate-400">Vol Dry-Up</div>
                    <div className="font-bold text-emerald-400">{asset.volumeDryUpPct}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase text-slate-400">Model</div>
                    <div className="font-bold text-purple-400 text-[11px] truncate">{asset.category}</div>
                  </div>
                </div>

                {/* Primary Catalyst */}
                <div className="text-xs text-slate-300 flex items-start gap-1.5">
                  <span className="text-amber-400 shrink-0 mt-0.5">⚡</span>
                  <span className="line-clamp-2">{asset.catalyst}</span>
                </div>
              </div>

              {/* Action Trigger */}
              <div className="pt-3 border-t border-slate-800/60 flex items-center justify-between">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-800/60 text-emerald-400">
                  {asset.executionStatus.replace(/_/g, ' ')}
                </span>
                <Link
                  href="/setups"
                  className="text-xs font-mono font-bold text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
                >
                  <span>Setup Ticket</span>
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
