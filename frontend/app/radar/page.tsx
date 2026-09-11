"use client";

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import TerminalShell from '../../components/terminal/TerminalShell';
import PageIntro from '../../components/PageIntro';
import { fetchScreenerGems, fetchAssetAnalytics } from '../../lib/api';

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

type CategoryFilter = 'ALL' | 'VCP' | 'SMART_MONEY' | 'VALUE';

export default function RadarPage() {
  const [activeFilter, setActiveFilter] = useState<CategoryFilter>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'SCORE' | 'RS' | 'PRICE'>('SCORE');
  const [allAssets, setAllAssets] = useState<RadarAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isOnDemandLoading, setIsOnDemandLoading] = useState(false);
  const [onDemandError, setOnDemandError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    fetchScreenerGems("all")
      .then((res) => {
        if (!isMounted) return;
        const mapped: RadarAsset[] = (res.results || []).map((gem: any) => {
          const cat: ('VCP' | 'SMART_MONEY' | 'VALUE')[] = [];
          const modelStr = (gem.expert_model || "").toUpperCase();
          if (modelStr.includes("VCP") || modelStr.includes("MINERVINI")) cat.push("VCP");
          if (modelStr.includes("MAGIC") || modelStr.includes("GARP") || modelStr.includes("VALUE") || modelStr.includes("GREENBLATT") || modelStr.includes("LYNCH") || modelStr.includes("GARDNER")) cat.push("VALUE");
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

  const cleanQ = searchQuery.trim().toUpperCase();
  const isSearching = searchQuery.trim().length > 0;
  const isTickerQuery = /^[A-Z0-9\.\-]{1,6}$/i.test(searchQuery.trim());

  const handleOnDemandScan = async (tickerOverride?: string) => {
    const raw = tickerOverride || searchQuery;
    const clean = raw.trim().toUpperCase();
    if (!clean) return;

    // Fast-path: Check if already in allAssets
    const existing = allAssets.find((a) => a.ticker === clean);
    if (existing) {
      setSearchQuery(clean);
      setOnDemandError(null);
      return;
    }

    setIsOnDemandLoading(true);
    setOnDemandError(null);
    try {
      const data = await fetchAssetAnalytics(clean, "1y", "1d");
      if (!data || !data.currentPrice || data.currentPrice <= 0) {
        setOnDemandError(`Asset "${clean}" is not recognized on the exchange tape or has zero trading history.`);
        return;
      }

      const opt = data.optimalExecution;
      const rawStatus = (opt?.execution_status || "").toUpperCase();
      let executionStatus: 'IN_BUY_ZONE' | 'NEAR_PIVOT' | 'VOLUME_DRYUP' | 'PULLBACK_SUPPORT' = 'PULLBACK_SUPPORT';
      if (rawStatus.includes("BUY_ZONE")) executionStatus = 'IN_BUY_ZONE';
      else if (rawStatus.includes("NEAR_PIVOT") || rawStatus.includes("APPROACHING")) executionStatus = 'NEAR_PIVOT';
      else if (rawStatus.includes("DRYUP") || rawStatus.includes("WAITING")) executionStatus = 'VOLUME_DRYUP';

      const cat: ('VCP' | 'SMART_MONEY' | 'VALUE')[] = ['VCP'];
      if ((data.confluence?.confluenceScore || 0) >= 70) cat.push('SMART_MONEY');

      const newAsset: RadarAsset = {
        ticker: clean,
        name: clean,
        price: Number(data.currentPrice.toFixed(2)),
        rsRating: Math.min(99, Math.max(50, Math.round(data.confluence?.confluenceScore || 75))),
        vcpStage: opt?.setup_pattern || 'Stage 2 Continuation',
        volumeDryUpPct: -35,
        confluenceScore: Math.round(data.confluence?.confluenceScore || 50),
        catalyst: opt?.entry_thesis || "On-demand quantitative exchange tape discovery.",
        categories: cat,
        sector: "On-Demand Discovery",
        executionStatus,
      };

      setAllAssets((prev) => [newAsset, ...prev.filter((a) => a.ticker !== clean)]);
      setSearchQuery(clean);
      setOnDemandError(null);
    } catch (err: any) {
      setOnDemandError(`Asset "${clean}" is not recognized on the exchange tape: ${err.message || 'Data unavailable'}.`);
    } finally {
      setIsOnDemandLoading(false);
    }
  };

  const handleFilterKeyDown = (e: React.KeyboardEvent, current: CategoryFilter) => {
    const filters: CategoryFilter[] = ['ALL', 'VCP', 'SMART_MONEY', 'VALUE'];
    const idx = filters.indexOf(current);
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      const next = filters[(idx + 1) % filters.length];
      setActiveFilter(next);
      document.getElementById(`tab-radar-filter-${next.toLowerCase()}`)?.focus();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = filters[(idx - 1 + filters.length) % filters.length];
      setActiveFilter(prev);
      document.getElementById(`tab-radar-filter-${prev.toLowerCase()}`)?.focus();
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActiveFilter(filters[0]);
      document.getElementById(`tab-radar-filter-${filters[0].toLowerCase()}`)?.focus();
    } else if (e.key === 'End') {
      e.preventDefault();
      setActiveFilter(filters[filters.length - 1]);
      document.getElementById(`tab-radar-filter-${filters[filters.length - 1].toLowerCase()}`)?.focus();
    }
  };

  const heroAsset = isSearching ? (filteredAssets[0] || null) : (filteredAssets[0] || allAssets[0]);

  return (
    <TerminalShell
      activeHub="radar"
      activeSymbol={searchQuery.trim() ? searchQuery.trim().toUpperCase() : null}
    >
      <div className="space-y-6">
        {/* Hub Guidance & Orientation (A3-AC1, A3-AC2, A3-AC3) */}
        <PageIntro
          hubId="radar"
          title="Radar"
          purpose="Scan and filter the market universe for momentum and breakout candidates that warrant further analysis."
          badge="Market Discovery"
          symbol={cleanQ || null}
          primaryAction={{
            label: heroAsset ? `Analyze ${heroAsset.ticker} →` : "Analyze Candidate →",
            href: heroAsset ? `/?symbol=${heroAsset.ticker}` : "/",
          }}
          secondaryAction={{
            label: "Review Setups →",
            href: heroAsset ? `/setups?symbol=${heroAsset.ticker}` : "/setups",
          }}
        />

        {/* Loading Indicator */}
        {isLoading && (
          <div className="p-12 text-center text-slate-400 font-mono text-xs animate-pulse">
            ⏳ Scanning multi-factor equity tape and quantitative confluence filters...
          </div>
        )}

        {/* Actionable Empty State */}
        {!isLoading && allAssets.length === 0 && (
          <div className="p-12 rounded-2xl border border-slate-800 bg-slate-900/30 text-center space-y-4 font-mono text-xs text-slate-400">
            <span className="text-3xl">📡</span>
            <div className="space-y-1">
              <div className="text-white font-bold text-sm">No Active Confluence Candidates</div>
              <p className="text-slate-400 max-w-lg mx-auto font-sans">
                No market assets currently meet the combined Minervini Stage 2, volume dry-up, and relative strength thresholds. This is normal during market pullbacks or broad consolidation regimes.
              </p>
            </div>
            <div className="flex items-center justify-center gap-3 pt-2">
              <Link
                href="/"
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs font-sans transition-transform active:scale-95"
              >
                Analyze Any Ticker →
              </Link>
              <Link
                href="/setups"
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-xl text-xs font-sans transition-colors"
              >
                Review Active Setups →
              </Link>
            </div>
          </div>
        )}

        {/* Search Discovery / Empty Status Banner when filteredAssets is 0 */}
        {!isLoading && isSearching && !heroAsset && (
          <div className="p-5 md:p-6 rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md space-y-3 font-mono text-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="text-2xl">🔍</span>
                <div>
                  <div className="text-white font-bold text-sm">
                    Searching for &quot;{cleanQ}&quot; · 0 matching candidates in current pre-scanned batch
                  </div>
                  <div className="text-slate-400 text-[11px] mt-0.5 font-sans">
                    {isTickerQuery
                      ? `Asset "${cleanQ}" is not in today's top 24 pre-screened confluence batch. You can evaluate it live across the exchange tape.`
                      : `No radar candidates matched your keyword or catalyst filter.`}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {isTickerQuery && (
                  <button
                    onClick={() => handleOnDemandScan(cleanQ)}
                    disabled={isOnDemandLoading}
                    className="px-3.5 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs flex items-center gap-1.5 transition-all shadow-md shadow-cyan-950/50 disabled:opacity-50 cursor-pointer"
                  >
                    <span>{isOnDemandLoading ? "⏳ Scanning..." : "⚡ Run On-Demand Scan"}</span>
                  </button>
                )}
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setOnDemandError(null);
                  }}
                  className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold text-xs transition-all border border-slate-700 cursor-pointer"
                >
                  Clear Filter
                </button>
              </div>
            </div>

            {onDemandError && (
              <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs flex items-center gap-2">
                <span>⚠️</span>
                <span>{onDemandError}</span>
              </div>
            )}
          </div>
        )}

        {/* Attention Candidate Hero Card */}
        {!isLoading && heroAsset && (
          <div className="relative overflow-hidden rounded-2xl border border-emerald-500/40 bg-gradient-to-br from-emerald-950/40 via-slate-900 to-slate-950 p-5 md:p-6 shadow-2xl">
            <div className="absolute top-0 right-0 px-3 py-1 bg-emerald-500/20 border-b border-l border-emerald-500/40 text-[10px] font-mono uppercase tracking-widest text-emerald-300 font-bold rounded-bl-xl">
              Attention Candidate · Top Pre-Screened Confluence
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
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                    ATTENTION CANDIDATE
                  </span>
                  <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/50">
                    {heroAsset.executionStatus.replace(/_/g, ' ')}
                  </span>
                  {heroAsset.vcpStage && (
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      {heroAsset.vcpStage}
                    </span>
                  )}
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
                    <span className="text-white font-bold">{heroAsset.rsRating !== null ? `${heroAsset.rsRating}/99` : "--"}</span>
                  </div>
                  <span className="text-slate-700">•</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">Vol Dry-Up:</span>
                    <span className="text-emerald-400 font-bold">{heroAsset.volumeDryUpPct !== null ? `${heroAsset.volumeDryUpPct}%` : "--"}</span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row lg:flex-col gap-2.5 shrink-0">
                <Link
                  href={`/?symbol=${heroAsset.ticker}`}
                  className="px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-black tracking-tight transition-all shadow-lg flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98]"
                >
                  <span>Analyze {heroAsset.ticker}</span>
                  <span>→</span>
                </Link>
                <Link
                  href={`/setups?symbol=${heroAsset.ticker}`}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-bold transition-all border border-slate-700 flex items-center justify-center gap-1.5 hover:border-slate-600"
                >
                  <span>Inspect Setup in /setups</span>
                  <span>→</span>
                </Link>
                <div className="text-[10px] font-mono text-slate-400 text-center">
                  Discovery Candidate · Not an Execution Recommendation
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Search & Filter Toolbar */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          {/* Category Filter Tabs (WAI-ARIA Tablist) */}
          <div
            role="tablist"
            aria-label="Radar Confluence Categories"
            className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0"
          >
            <button
              type="button"
              role="tab"
              id="tab-radar-filter-all"
              aria-selected={activeFilter === 'ALL'}
              aria-controls="panel-radar-candidates"
              tabIndex={activeFilter === 'ALL' ? 0 : -1}
              onKeyDown={(e) => handleFilterKeyDown(e, 'ALL')}
              onClick={() => setActiveFilter('ALL')}
              className={`focus-ring px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 cursor-pointer ${
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
              type="button"
              role="tab"
              id="tab-radar-filter-vcp"
              aria-selected={activeFilter === 'VCP'}
              aria-controls="panel-radar-candidates"
              tabIndex={activeFilter === 'VCP' ? 0 : -1}
              onKeyDown={(e) => handleFilterKeyDown(e, 'VCP')}
              onClick={() => setActiveFilter('VCP')}
              className={`focus-ring px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 cursor-pointer ${
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
              type="button"
              role="tab"
              id="tab-radar-filter-smart_money"
              aria-selected={activeFilter === 'SMART_MONEY'}
              aria-controls="panel-radar-candidates"
              tabIndex={activeFilter === 'SMART_MONEY' ? 0 : -1}
              onKeyDown={(e) => handleFilterKeyDown(e, 'SMART_MONEY')}
              onClick={() => setActiveFilter('SMART_MONEY')}
              className={`focus-ring px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 cursor-pointer ${
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
              type="button"
              role="tab"
              id="tab-radar-filter-value"
              aria-selected={activeFilter === 'VALUE'}
              aria-controls="panel-radar-candidates"
              tabIndex={activeFilter === 'VALUE' ? 0 : -1}
              onKeyDown={(e) => handleFilterKeyDown(e, 'VALUE')}
              onClick={() => setActiveFilter('VALUE')}
              className={`focus-ring px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 shrink-0 cursor-pointer ${
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
          <div className="flex flex-wrap items-center gap-2.5">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (filteredAssets.length === 0 && isTickerQuery) {
                  handleOnDemandScan(cleanQ);
                }
              }}
              className="relative flex-1 sm:w-72 min-w-[220px]"
            >
              <input
                type="text"
                aria-label="Search candidate, ticker, or catalyst"
                placeholder="Search candidate, ticker (e.g. NVDA, TSLA)..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setOnDemandError(null);
                }}
                className="focus-ring w-full bg-[#0b1019] border border-slate-800 rounded-lg pl-3 pr-20 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
              />
              <div className="absolute right-2 top-1.5 flex items-center gap-1">
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => {
                      setSearchQuery('');
                      setOnDemandError(null);
                    }}
                    className="focus-ring text-xs text-slate-500 hover:text-slate-300 font-mono px-1 rounded"
                    title="Clear filter"
                    aria-label="Clear search"
                  >
                    ✕
                  </button>
                )}
                {isTickerQuery && filteredAssets.length === 0 && (
                  <button
                    type="submit"
                    disabled={isOnDemandLoading}
                    className="focus-ring text-[10px] bg-cyan-900/80 hover:bg-cyan-800 text-cyan-300 px-1.5 py-0.5 rounded font-mono font-bold border border-cyan-700 cursor-pointer disabled:opacity-50"
                    title="Scan on-demand"
                  >
                    {isOnDemandLoading ? "..." : "Scan ↵"}
                  </button>
                )}
              </div>
            </form>

            {searchQuery && (
              <span className="text-[11px] font-mono text-slate-400 shrink-0">
                {filteredAssets.length} of {allAssets.length}
              </span>
            )}

            <div className="flex items-center gap-1 bg-[#0b1019] border border-slate-800 rounded-lg p-0.5 text-xs font-mono shrink-0">
              <span className="text-[10px] text-slate-500 px-2 uppercase">Sort:</span>
              <button
                type="button"
                onClick={() => setSortBy('SCORE')}
                className={`focus-ring px-2 py-1 rounded ${sortBy === 'SCORE' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Score
              </button>
              <button
                type="button"
                onClick={() => setSortBy('RS')}
                className={`focus-ring px-2 py-1 rounded ${sortBy === 'RS' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                RS
              </button>
              <button
                type="button"
                onClick={() => setSortBy('PRICE')}
                className={`focus-ring px-2 py-1 rounded ${sortBy === 'PRICE' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Price
              </button>
            </div>
          </div>
        </div>

        {/* Level 1: Dense Confluence Stream Table (Tabpanel) */}
        <div
          role="tabpanel"
          id="panel-radar-candidates"
          aria-label="Radar Confluence Candidates"
          tabIndex={0}
          className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden shadow-xl focus-ring"
        >
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
                {filteredAssets.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="p-8 text-center text-slate-400 space-y-4 font-mono">
                      <div className="max-w-md mx-auto space-y-3">
                        <span className="text-3xl block">🔍</span>
                        <div className="space-y-1">
                          <h4 className="text-sm font-bold text-white">
                            {searchQuery
                              ? `No assets match "${searchQuery}" in current confluence scan`
                              : `No candidates currently in "${activeFilter}"`}
                          </h4>
                          <p className="text-xs text-slate-400 font-sans leading-relaxed">
                            {searchQuery
                              ? (isTickerQuery
                                  ? `Asset "${cleanQ}" is not in today's top 24 pre-scanned confluence batch. You can scan it live across the exchange tape or review its tactical setup.`
                                  : `No confluence candidates match this search. Try searching by symbol (e.g. NVDA, CPRX, ASML, TSLA) or reset the filter.`)
                              : `Try switching to "All Confluences" to view all active setups.`}
                          </p>
                        </div>

                        {onDemandError && (
                          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
                            {onDemandError}
                          </div>
                        )}

                        <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                          {isTickerQuery && (
                            <button
                              type="button"
                              onClick={() => handleOnDemandScan(cleanQ)}
                              disabled={isOnDemandLoading}
                              className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold font-mono transition-transform active:scale-95 shadow-lg shadow-cyan-950/50 flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                            >
                              <span>{isOnDemandLoading ? "⏳ Scanning Tape..." : "⚡ Scan " + cleanQ + " On-Demand"}</span>
                            </button>
                          )}

                          {isTickerQuery && (
                            <Link
                              href={`/setups?ticker=${cleanQ}`}
                              className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-cyan-300 text-xs font-bold font-mono border border-slate-700"
                            >
                              Inspect in /setups →
                            </Link>
                          )}

                          {isTickerQuery && (
                            <Link
                              href={`/?symbol=${cleanQ}`}
                              className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold font-mono border border-slate-700"
                            >
                              Open in Terminal →
                            </Link>
                          )}

                          <button
                            type="button"
                            onClick={() => {
                              setSearchQuery('');
                              setOnDemandError(null);
                            }}
                            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white text-xs font-bold font-mono border border-slate-700 cursor-pointer"
                          >
                            ✕ Clear Filter (Show All {allAssets.length})
                          </button>
                        </div>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredAssets.map((asset) => {
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
                            className={`px-2.5 py-1 rounded text-[10px] font-bold font-mono transition-colors inline-block ${
                              isBuy
                                ? 'bg-emerald-500/20 hover:bg-emerald-500 hover:text-black text-emerald-300 border border-emerald-500/40'
                                : 'bg-slate-800 hover:bg-cyan-600 hover:text-white text-cyan-400'
                            }`}
                          >
                            {isBuy ? 'Ticket →' : 'Setup →'}
                          </Link>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
