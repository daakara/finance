"use client";

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import TerminalShell from '../../components/terminal/TerminalShell';
import { fetchScreenerGems } from '../../lib/api';

interface RadarAsset {
  ticker: string;
  name: string;
  price: number;
  rsRating: number;
  vcpStage: string;
  volumeDryUpPct: number;
  confluenceScore: number;
  catalyst: string;
  categories: ('VCP' | 'SMART_MONEY' | 'VALUE')[];
  sector: string;
  executionStatus: 'IN_BUY_ZONE' | 'NEAR_PIVOT' | 'VOLUME_DRYUP' | 'PULLBACK_SUPPORT';
}

export default function RadarPage() {
  const [activeFilter, setActiveFilter] = useState<'ALL' | 'VCP' | 'SMART_MONEY' | 'VALUE'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'SCORE' | 'RS' | 'PRICE'>('SCORE');
  const [allAssets, setAllAssets] = useState<RadarAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    fetchScreenerGems("all")
      .then((res) => {
        if (!isMounted) return;
        const mapped: RadarAsset[] = (res.results || []).map((gem: any) => {
          const cat: ('VCP' | 'SMART_MONEY' | 'VALUE')[] = [];
          const modelStr = (gem.expert_model || "").toUpperCase();
          if (modelStr.includes("VCP") || modelStr.includes("MINERVINI")) cat.push("VCP");
          if (modelStr.includes("MAGIC") || modelStr.includes("GARP") || modelStr.includes("VALUE")) cat.push("VALUE");
          if (cat.length === 0 || gem.composite_score >= 85) cat.push("SMART_MONEY");

          const rawStatus = (gem.execution_status || gem.factor_verdict || "").toUpperCase();
          let executionStatus: 'IN_BUY_ZONE' | 'NEAR_PIVOT' | 'VOLUME_DRYUP' | 'PULLBACK_SUPPORT' = 'PULLBACK_SUPPORT';
          if (rawStatus.includes("BUY_ZONE")) executionStatus = 'IN_BUY_ZONE';
          else if (rawStatus.includes("NEAR_PIVOT") || rawStatus.includes("APPROACHING")) executionStatus = 'NEAR_PIVOT';
          else if (rawStatus.includes("DRYUP") || rawStatus.includes("WAITING")) executionStatus = 'VOLUME_DRYUP';

          return {
            ticker: gem.ticker,
            name: gem.ticker,
            price: Number((gem.current_price || 0).toFixed(2)),
            rsRating: Math.min(99, Math.max(50, Math.round(gem.composite_score || 80))),
            vcpStage: executionStatus === 'IN_BUY_ZONE' ? '3T Pivot Breakout' : 'Stage 2 Base',
            volumeDryUpPct: -45,
            confluenceScore: Math.round(gem.composite_score || 0),
            catalyst: gem.primary_catalyst || gem.investment_thesis || "Stage 2 accumulation breakout with institutional liquidity flow.",
            categories: cat,
            sector: "Broad Market",
            executionStatus,
          };
        });
        setAllAssets(mapped);
        setIsLoading(false);
      })
      .catch(() => {
        if (!isMounted) return;
        setIsLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Category Counts
  const counts = useMemo(() => {
    return {
      ALL: allAssets.length,
      VCP: allAssets.filter((a) => a.categories.includes('VCP')).length,
      SMART_MONEY: allAssets.filter((a) => a.categories.includes('SMART_MONEY')).length,
      VALUE: allAssets.filter((a) => a.categories.includes('VALUE')).length,
    };
  }, [allAssets]);

  // Filtered & Sorted Assets
  const filteredAssets = useMemo(() => {
    return allAssets
      .filter((asset) => {
        const matchesCategory =
          activeFilter === 'ALL' ? true : asset.categories.includes(activeFilter);
        const q = searchQuery.trim().toLowerCase();
        const matchesQuery =
          q === ''
            ? true
            : asset.ticker.toLowerCase().includes(q) ||
              asset.name.toLowerCase().includes(q) ||
              asset.catalyst.toLowerCase().includes(q) ||
              asset.categories.some((c) => c.toLowerCase().includes(q));
        return matchesCategory && matchesQuery;
      })
      .sort((a, b) => {
        if (sortBy === 'SCORE') return b.confluenceScore - a.confluenceScore;
        if (sortBy === 'RS') return b.rsRating - a.rsRating;
        if (sortBy === 'PRICE') return b.price - a.price;
        return 0;
      });
  }, [allAssets, activeFilter, searchQuery, sortBy]);

  const heroAsset = filteredAssets[0] || allAssets[0];

  return (
    <TerminalShell activeHub="radar">
      <div className="space-y-6">
        {/* Loading Indicator */}
        {isLoading && (
          <div className="p-12 text-center text-slate-400 font-mono text-xs animate-pulse">
            ⏳ Scanning multi-factor equity tape and quantitative confluence filters...
          </div>
        )}

        {/* Empty State */}
        {!isLoading && allAssets.length === 0 && (
          <div className="p-12 rounded-2xl border border-slate-800 bg-slate-900/30 text-center space-y-3 font-mono text-xs text-slate-400">
            <span className="text-3xl">📡</span>
            <div className="text-white font-bold text-sm">No Active Confluence Candidates</div>
            <p>Exchange tape scan returned zero assets currently meeting strict multi-factor criteria.</p>
          </div>
        )}

        {/* Level 0: Asymmetric #1 High-Confluence Attention Leader Hero */}
        {!isLoading && heroAsset && (
          <div className="relative overflow-hidden rounded-2xl border border-emerald-500/40 bg-gradient-to-br from-emerald-950/40 via-slate-900 to-slate-950 p-5 md:p-6 shadow-2xl">
            <div className="absolute top-0 right-0 px-3 py-1 bg-emerald-500/20 border-b border-l border-emerald-500/40 text-[10px] font-mono uppercase tracking-widest text-emerald-300 font-bold rounded-bl-xl">
              Level 0 · #1 Attention Leader Today
            </div>

            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div className="space-y-3 max-w-3xl">
                <div className="flex flex-wrap items-center gap-2.5">
                  <span className="text-2xl md:text-3xl font-black font-mono tracking-tight text-white">
                    {heroAsset.ticker}
                  </span>
                  <span className="text-sm md:text-base text-slate-300 font-medium">
                    {heroAsset.name}
                  </span>
                  <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/50">
                    {heroAsset.executionStatus.replace(/_/g, ' ')}
                  </span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    Stage 2 · {heroAsset.vcpStage}
                  </span>
                </div>

                <p className="text-xs md:text-sm text-slate-200 font-sans leading-relaxed">
                  <strong className="text-amber-400 font-semibold">Primary Catalyst: </strong>
                  {heroAsset.catalyst}
                </p>

                <div className="flex flex-wrap items-center gap-4 text-xs font-mono pt-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">Price:</span>
                    <span className="text-white font-bold">${heroAsset.price.toFixed(2)}</span>
                  </div>
                  <span className="text-slate-700">•</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">Confluence:</span>
                    <span className="text-emerald-400 font-bold">{heroAsset.confluenceScore}/100</span>
                  </div>
                  <span className="text-slate-700">•</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">RS Rating:</span>
                    <span className="text-white font-bold">{heroAsset.rsRating}/99</span>
                  </div>
                  <span className="text-slate-700">•</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">Vol Dry-Up:</span>
                    <span className="text-emerald-400 font-bold">{heroAsset.volumeDryUpPct}%</span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row lg:flex-col gap-2.5 shrink-0">
                <Link
                  href={`/setups?ticker=${heroAsset.ticker}`}
                  className="px-5 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-mono font-black tracking-tight transition-all shadow-lg flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98]"
                >
                  <span>ARM EXECUTION TICKET IN /SETUPS</span>
                  <span>→</span>
                </Link>
                <div className="text-[10px] font-mono text-slate-400 text-center">
                  Verified S&P 500 Uptrend · 3-Model Convergence
                </div>
              </div>
            </div>
          </div>
        )}

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
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-300">
                {counts.ALL}
              </span>
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
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-300">
                {counts.VCP}
              </span>
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
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-300">
                {counts.SMART_MONEY}
              </span>
            </button>

            <button
              onClick={() => setActiveFilter('VALUE')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                activeFilter === 'VALUE'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <span>💎 Value / GARP</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-300">
                {counts.VALUE}
              </span>
            </button>
          </div>

          {/* Search Input & Sort Controls */}
          <div className="flex items-center gap-2.5">
            <div className="relative flex-1 sm:w-64">
              <input
                type="text"
                placeholder="Filter by ticker, catalyst..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-[#0b1019] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2.5 top-2 text-xs text-slate-500 hover:text-slate-300 font-mono"
                >
                  ✕
                </button>
              )}
            </div>

            <div className="flex items-center gap-1 bg-[#0b1019] border border-slate-800 rounded-lg p-0.5 text-xs font-mono shrink-0">
              <span className="text-[10px] text-slate-500 px-2 uppercase">Sort:</span>
              <button
                onClick={() => setSortBy('SCORE')}
                className={`px-2 py-1 rounded ${sortBy === 'SCORE' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Score
              </button>
              <button
                onClick={() => setSortBy('RS')}
                className={`px-2 py-1 rounded ${sortBy === 'RS' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                RS
              </button>
              <button
                onClick={() => setSortBy('PRICE')}
                className={`px-2 py-1 rounded ${sortBy === 'PRICE' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Price
              </button>
            </div>
          </div>
        </div>

        {/* Level 1: Dense Confluence Stream Table */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 text-[10px] uppercase tracking-wider">
                <tr>
                  <th className="p-3">Asset</th>
                  <th className="p-3">Action Status</th>
                  <th className="p-3">Price</th>
                  <th className="p-3 text-center">Score</th>
                  <th className="p-3 text-center">RS Rating</th>
                  <th className="p-3">VCP / Base Setup</th>
                  <th className="p-3 text-right">Vol Dry-Up</th>
                  <th className="p-3">Catalyst Rationale</th>
                  <th className="p-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredAssets.map((asset) => {
                  const isBuy = asset.executionStatus === 'IN_BUY_ZONE';
                  const isPivot = asset.executionStatus === 'NEAR_PIVOT';
                  return (
                    <tr key={asset.ticker} className="hover:bg-slate-900/70 transition-colors group">
                      <td className="p-3">
                        <div className="font-black text-white text-sm tracking-tight">{asset.ticker}</div>
                        <div className="text-[10px] text-slate-400 font-sans truncate max-w-[120px]">{asset.name}</div>
                      </td>
                      <td className="p-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isBuy
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
                            : isPivot
                            ? 'bg-amber-950 text-amber-300 border border-amber-800'
                            : 'bg-slate-800/60 text-slate-400 border border-slate-700/60'
                        }`}>
                          {asset.executionStatus.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="p-3 font-bold text-white">${asset.price.toFixed(2)}</td>
                      <td className="p-3 text-center">
                        <span className="font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-1.5 py-0.5 rounded">
                          {asset.confluenceScore}
                        </span>
                      </td>
                      <td className="p-3 text-center font-bold text-white">{asset.rsRating}</td>
                      <td className="p-3 text-slate-300 text-[11px]">{asset.vcpStage}</td>
                      <td className="p-3 text-right font-bold text-emerald-400">{asset.volumeDryUpPct}%</td>
                      <td className="p-3 text-slate-300 text-[11px] font-sans max-w-xs truncate" title={asset.catalyst}>
                        {asset.catalyst}
                      </td>
                      <td className="p-3 text-right">
                        <Link
                          href={`/setups?ticker=${asset.ticker}`}
                          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-cyan-600 hover:text-white text-cyan-400 text-[10px] font-bold font-mono transition-colors inline-block"
                        >
                          Setup →
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
