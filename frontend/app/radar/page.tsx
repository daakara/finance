"use client";

import React, { useState, useEffect, useMemo, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import TerminalShell from '../../components/terminal/TerminalShell';
import PageIntro from '../../components/PageIntro';
import {
  fetchScreenerGems,
  fetchAssetAnalytics,
  CanonicalRadarCategory,
  RadarCapabilities,
  RadarCapabilityStatus,
  CandidatePopulationModel,
  DEFAULT_RADAR_CAPABILITIES,
  normalizeRadarCapabilities,
  getPublicationIntegrityLabel,
  formatPriceObservationDate,
} from '../../lib/api';
import { trackRadarAssetClick } from '../../lib/matomo';
import { MASTER_ASSET_CATALOG } from '../../lib/masterCatalog';
import { usePortfolioContext, OwnershipFilter } from '../../hooks/usePortfolioContext';
import { RadarPortfolioBadge } from '../../components/radar/RadarPortfolioBadge';
import { ExecutionStatus } from '../../types/decisionContract';
import SynthesisGuideBanner from '../../components/synthesis/SynthesisGuideBanner';
import SynthesisMobileDock from '../../components/synthesis/SynthesisMobileDock';

interface RadarAsset {
  ticker: string;
  name: string;
  price: number;
  priceBasis?: "DAILY_CLOSE" | null;
  priceDate?: string | null;
  rvol?: string | null;
  riskRewardRatio?: number | null;
  vcpStage: string;
  volumeDryUpPct?: number | null;
  confluenceScore: number;
  catalyst: string;
  categories: CanonicalRadarCategory[];
  sector: string;
  executionStatus: ExecutionStatus;
  screeningStatus?: string;
  screeningGeometry?: string;
  decisionState?: string;
  decisionStateLabel?: string;
  isActionable?: boolean;
  canSizeTrade?: boolean;
  disqualificationReason?: string | null;
  decisionContextId?: string;
  allowedActions: string[];
  screeningEligible?: boolean;
}

type CategoryFilter = 'ALL' | CanonicalRadarCategory;


function parseCanonicalExecutionStatus(raw: string): ExecutionStatus {
  const s = raw.trim().toUpperCase();
  if (s === 'IN_BUY_ZONE' || s === 'READY_TO_BUY') return s;
  if (s === 'IN_BUY_ZONE_AWAITING_TRIGGER') return s;
  if (s === 'WAITING_PULLBACK') return s;
  if (s === 'APPROACHING_TARGET') return s;
  if (s === 'STOPPED_OUT') return s;
  if (s === 'INSUFFICIENT_HISTORY') return s;
  if (s === 'DELISTED_SECURITY' || s.includes('DELISTED')) return 'DELISTED_SECURITY';
  if (s === 'UNVERIFIED_ASSET') return s;
  if (s === 'STALE_MARKET_DATA') return s;
  if (s.includes('BUY_ZONE') && !s.includes('AWAITING')) return 'IN_BUY_ZONE';
  if (s.includes('AWAITING') || s.includes('TRIGGER')) return 'IN_BUY_ZONE_AWAITING_TRIGGER';
  if (s.includes('APPROACHING') || s.includes('TARGET')) return 'APPROACHING_TARGET';
  if (s.includes('PULLBACK') || s.includes('WAITING')) return 'WAITING_PULLBACK';
  return 'UNKNOWN';
}

function formatRadarExecutionStatus(status: ExecutionStatus): string {
  switch (status) {
    case 'DELISTED_SECURITY':
      return 'DELISTED / INACTIVE';
    case 'IN_BUY_ZONE':
      return 'NEAR SCREENING ZONE';
    case 'READY_TO_BUY':
      return 'READY TO BUY';
    case 'IN_BUY_ZONE_AWAITING_TRIGGER':
      return 'AWAITING TRIGGER';
    case 'APPROACHING_TARGET':
      return 'APPROACHING TARGET';
    case 'WAITING_PULLBACK':
      return 'PULLBACK PENDING';
    case 'STOPPED_OUT':
      return 'STOPPED OUT';
    case 'INSUFFICIENT_HISTORY':
      return 'INSUFFICIENT HISTORY';
    case 'UNVERIFIED_ASSET':
      return 'UNVERIFIED ASSET';
    case 'STALE_MARKET_DATA':
      return 'STALE MARKET DATA';
    default:
      return 'UNKNOWN';
  }
}

function RadarContent() {
  const searchParams = useSearchParams();
  const initialQ = searchParams?.get('q') || searchParams?.get('symbol') || '';
  const [activeFilter, setActiveFilter] = useState<CategoryFilter>('ALL');
  const [ownershipFilter, setOwnershipFilter] = useState<OwnershipFilter>('ALL');
  const [searchQuery, setSearchQuery] = useState(initialQ);
  const [sortBy, setSortBy] = useState<'SCORE' | 'RVOL' | 'PRICE'>('SCORE');
  const [allAssets, setAllAssets] = useState<RadarAsset[]>([]);
  const [capabilities, setCapabilities] = useState<RadarCapabilities>(DEFAULT_RADAR_CAPABILITIES);
  const [population, setPopulation] = useState<CandidatePopulationModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [isOnDemandLoading, setIsOnDemandLoading] = useState(false);
  const [onDemandError, setOnDemandError] = useState<string | null>(null);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const latestRequestIdRef = React.useRef(0);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).__arxTriggerRadarScan = () => setRefreshNonce((n) => n + 1);
    }
    return () => {
      if (typeof window !== 'undefined') {
        delete (window as any).__arxTriggerRadarScan;
      }
    };
  }, []);

  const {
    isVerified: isPortfolioVerified,
    isDegraded: isPortfolioDegraded,
    getOwnershipState,
    getHolding,
  } = usePortfolioContext();

  // Synchronize searchQuery when URL search params change while component remains mounted (clears when query is removed)
  const handleOwnershipFilterSelect = (filter: OwnershipFilter, buttonId: string) => {
    setOwnershipFilter(filter);
    if (typeof document !== 'undefined') {
      const btn = document.getElementById(buttonId);
      if (btn && typeof btn.scrollIntoView === 'function') {
        btn.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
      }
    }
  };

  const handleCategoryFilterSelect = (filter: CategoryFilter, buttonId: string) => {
    setActiveFilter(filter);
    if (typeof document !== 'undefined') {
      const btn = document.getElementById(buttonId);
      if (btn && typeof btn.scrollIntoView === 'function') {
        btn.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
      }
    }
  };

  useEffect(() => {
    const q = (searchParams?.get('q') || searchParams?.get('symbol') || '').trim().toUpperCase();
    setSearchQuery((prev) => (prev.trim().toUpperCase() !== q ? q : prev));
  }, [searchParams]);

  useEffect(() => {
    let isMounted = true;
    const reqId = ++latestRequestIdRef.current;
    setIsLoading(true);

    fetchScreenerGems("all")
      .then((res) => {
        if (!isMounted || reqId !== latestRequestIdRef.current) return;
        if (res.status === "ERROR" || res.status === "TIMEOUT" || res.error) {
          setFetchError(res.error || "Unable to reach the Radar screening service");
          setCapabilities(DEFAULT_RADAR_CAPABILITIES);
          setPopulation(null);
          setAllAssets([]);
          setIsLoading(false);
          return;
        }
        setFetchError(null);
        setCapabilities(normalizeRadarCapabilities(res.capabilities));
        setPopulation(res.population || null);
        const mapped: RadarAsset[] = (res.results || []).map((gem: any) => {
          // Canonical typed categories: strictly consume explicit backend categories.
          // Zero substring-based inference from arbitrary strings (e.g. "Smart Growth" or "Flow Traders").
          const cat: CanonicalRadarCategory[] = Array.isArray(gem.categories)
            ? (gem.categories.filter((c: any) => ["VALUE_GARP", "VCP", "SMART_MONEY"].includes(c)) as CanonicalRadarCategory[])
            : [];

          const rawStatus = (gem.executionStatus || gem.execution_status || gem.factor_verdict || "").toUpperCase();
          const executionStatus = parseCanonicalExecutionStatus(rawStatus);

          const dryUp = typeof gem.volume_dry_up === 'number' ? gem.volume_dry_up : (typeof gem.volumeDryUpPct === 'number' ? gem.volumeDryUpPct : null);
          const rvolVal = typeof gem.rvol === 'string' && gem.rvol !== 'N/A' ? gem.rvol : null;
          const rrVal = typeof gem.riskRewardRatio === 'number' ? gem.riskRewardRatio : null;
          const stageStr = gem.setup_pattern || (gem.stage_phase ? `Stage ${gem.stage_phase} Base` : (executionStatus === 'IN_BUY_ZONE' ? 'Pivot Breakout' : 'Base Under Evaluation'));

          const cleanTicker = (gem.ticker || '').toUpperCase();
          const catalogEntry = MASTER_ASSET_CATALOG[cleanTicker];
          const resolvedName = gem.companyName || catalogEntry?.name || '';

          return {
            ticker: cleanTicker,
            name: resolvedName,
            price: Number(((gem.currentPrice ?? gem.current_price) || 0).toFixed(2)),
            priceBasis: gem.priceBasis ?? (gem.price_basis as any) ?? null,
            priceDate: gem.priceDate ?? gem.price_date ?? null,
            rvol: rvolVal,
            riskRewardRatio: rrVal,
            vcpStage: stageStr,
            volumeDryUpPct: dryUp,
            confluenceScore: typeof gem.confluenceScore === 'number'
              ? Math.round(gem.confluenceScore)
              : (typeof gem.composite_score === 'number' ? Math.round(gem.composite_score) : 0),
            catalyst: gem.primary_catalyst || gem.investment_thesis || "Stage 2 accumulation breakout with institutional liquidity flow.",
            categories: cat,
            sector: catalogEntry?.sector || "Broad Market",
            executionStatus,
            screeningStatus: gem.screeningStatus || (executionStatus === 'IN_BUY_ZONE' ? 'SCREENING_ZONE' : (executionStatus !== 'UNKNOWN' ? executionStatus : 'UNKNOWN')),
            screeningGeometry: gem.screeningGeometry || (executionStatus === 'IN_BUY_ZONE' ? 'WITHIN_TOLERANCE' : 'OUTSIDE_TOLERANCE'),
            decisionState: gem.decisionState || 'VALID_SETUP',
            decisionStateLabel: gem.decisionStateLabel || 'Discovery Candidate — Analyze for Trigger',
            isActionable: false, // Invariant: Radar is a discovery scanner; canonical actionability reserved for DecisionTrace
            canSizeTrade: false,
            disqualificationReason: gem.disqualificationReason || null,
            decisionContextId: gem.decisionContextId,
            allowedActions: Array.isArray(gem.allowedActions) ? gem.allowedActions : [],
            screeningEligible: typeof gem.screeningEligible === 'boolean' ? gem.screeningEligible : Boolean(executionStatus === 'IN_BUY_ZONE'),
          };
        });
        setAllAssets(mapped);
        setIsLoading(false);
      })
      .catch((err) => {
        if (!isMounted || reqId !== latestRequestIdRef.current) return;
        setFetchError(err instanceof Error ? err.message : "Failed to load screener candidates");
        setCapabilities(DEFAULT_RADAR_CAPABILITIES);
        setPopulation(null);
        setAllAssets([]);
        setIsLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [refreshNonce]);

  // Category Counts & Invariant Enforcement
  // Zero Semantics Invariant: Numerical count (including 0) is permitted ONLY when
  // capability is AVAILABLE. For PIPELINE_PENDING or NOT_IMPLEMENTED, return null.
  // Missing API connectivity is not represented as zero qualifying candidates.
  const categoryMeta = useMemo(() => {
    const isAvailable = (cat: CanonicalRadarCategory) =>
      !fetchError && capabilities[cat]?.universeScreening === 'AVAILABLE';

    return {
      ALL: {
        count: fetchError ? null : allAssets.length,
        badge: fetchError ? '—' : `${allAssets.length}`,
        status: (fetchError ? 'PIPELINE_PENDING' : 'AVAILABLE') as RadarCapabilityStatus,
      },
      VALUE_GARP: {
        count: isAvailable('VALUE_GARP')
          ? (typeof capabilities.VALUE_GARP?.matchedCount === 'number'
              ? capabilities.VALUE_GARP.matchedCount
              : allAssets.filter((a) => a.categories.includes('VALUE_GARP')).length)
          : null,
        badge: fetchError
          ? '—'
          : isAvailable('VALUE_GARP')
          ? `${typeof capabilities.VALUE_GARP?.matchedCount === 'number'
              ? capabilities.VALUE_GARP.matchedCount
              : allAssets.filter((a) => a.categories.includes('VALUE_GARP')).length}`
          : 'Pending',
        status: capabilities.VALUE_GARP?.status || 'AVAILABLE',
      },
      VCP: {
        count: isAvailable('VCP')
          ? (typeof capabilities.VCP?.matchedCount === 'number' ? capabilities.VCP.matchedCount : allAssets.filter((a) => a.categories.includes('VCP')).length)
          : null,
        badge: isAvailable('VCP') ? `${allAssets.filter((a) => a.categories.includes('VCP')).length}` : 'Universe Scanner Pending',
        status: capabilities.VCP?.status || 'PIPELINE_PENDING',
      },
      SMART_MONEY: {
        count: isAvailable('SMART_MONEY') ? allAssets.filter((a) => a.categories.includes('SMART_MONEY')).length : null,
        badge: isAvailable('SMART_MONEY') ? `${allAssets.filter((a) => a.categories.includes('SMART_MONEY')).length}` : 'Universe Scanner Pending',
        status: capabilities.SMART_MONEY?.status || 'PIPELINE_PENDING',
      },
    };
  }, [allAssets, capabilities, fetchError]);

  // Candidate Population Model: Helper to determine if an asset is eligible for active opportunity metrics
  const isCandidateEligibleOpportunity = (asset: RadarAsset): boolean => {
    if (asset.executionStatus === 'DELISTED_SECURITY') return false;
    if (asset.executionStatus === 'UNVERIFIED_ASSET') return false;
    if (asset.executionStatus === 'UNKNOWN') return false;
    return true;
  };

  // Ownership & Opportunity Counts (Derived dynamically from candidate eligibility contract)
  const ownershipCounts = useMemo(() => {
    let held = 0;
    let notHeld = 0;
    let delistedCount = 0;
    let unverifiedCount = 0;

    for (const asset of allAssets) {
      const isEligible = isCandidateEligibleOpportunity(asset);
      if (asset.executionStatus === 'DELISTED_SECURITY') {
        delistedCount++;
      } else if (asset.executionStatus === 'UNVERIFIED_ASSET') {
        unverifiedCount++;
      }

      const st = getOwnershipState(asset.ticker);
      if (st === 'HELD') {
        held++;
      } else if (st === 'NOT_HELD' && isEligible) {
        // Delisted and unverified securities CANNOT inflate New Opportunities!
        notHeld++;
      }
    }
    const totalActive = allAssets.filter(isCandidateEligibleOpportunity).length;
    return {
      held,
      notHeld,
      total: allAssets.length,
      totalActive,
      delistedCount,
      unverifiedCount,
    };
  }, [allAssets, getOwnershipState]);

  // Filtered & Sorted Assets (Canonical sort order strictly preserved per INV-RADAR-PORTFOLIO-04)
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

        const ownState = getOwnershipState(asset.ticker);
        const matchesOwnership =
          ownershipFilter === 'ALL'
            ? true
            : (ownershipFilter === 'NEW_OPPORTUNITIES' || (ownershipFilter as any) === 'NOT_HELD')
            ? ownState === 'NOT_HELD' && isCandidateEligibleOpportunity(asset)
            : ownershipFilter === 'MY_HOLDINGS' || (ownershipFilter as any) === 'HELD'
            ? ownState === 'HELD'
            : true;

        return matchesCategory && matchesQuery && matchesOwnership;
      })
      .sort((a, b) => {
        // Scenario 10 in Gate H: Sorting never promotes delisted assets into eligible opportunity rankings
        const aDelisted = a.executionStatus === 'DELISTED_SECURITY' ? 1 : 0;
        const bDelisted = b.executionStatus === 'DELISTED_SECURITY' ? 1 : 0;
        if (aDelisted !== bDelisted) {
          return aDelisted - bDelisted; // Delisted securities sink to bottom
        }
        if (sortBy === 'SCORE') return b.confluenceScore - a.confluenceScore;
        if (sortBy === 'RVOL') {
          const aRvol = a.rvol ? parseFloat(a.rvol.replace('x', '')) : 0;
          const bRvol = b.rvol ? parseFloat(b.rvol.replace('x', '')) : 0;
          return bRvol - aRvol;
        }
        if (sortBy === 'PRICE') return b.price - a.price;
        return 0;
      });
  }, [allAssets, activeFilter, searchQuery, sortBy, ownershipFilter, getOwnershipState]);

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
      const data = await fetchAssetAnalytics(clean, "1y", "1d", "SWING_TRADER");
      if (!data || !data.currentPrice || data.currentPrice <= 0) {
        setOnDemandError(`Asset "${clean}" is not recognized on the exchange tape or has zero trading history.`);
        return;
      }

      const opt = data.optimalExecution;
      const rawStatus = (opt?.execution_status || "").toUpperCase();
      const executionStatus = parseCanonicalExecutionStatus(rawStatus);

      const cat: CanonicalRadarCategory[] = [];
      // On-demand asset is evaluated across single-asset execution and confluence engines.
      // If single-asset setup is verified VCP pattern:
      if (opt?.vcp_contraction_status === "VCP 3-Stage Compression Confirmed" || opt?.setup_pattern === "Minervini Volatility Contraction Pattern (VCP)") {
        cat.push("VCP");
      }
      const smartPillar = data.confluence?.pillars?.find((p) => p.pillar.toLowerCase().includes("smart") || p.pillar.toLowerCase().includes("flow"));
      if (smartPillar && smartPillar.status === "positive") {
        cat.push("SMART_MONEY");
      }
      const fundPillar = data.confluence?.pillars?.find((p) => p.pillar.toLowerCase().includes("fundamental") || p.pillar.toLowerCase().includes("solvency"));
      if (fundPillar && fundPillar.status === "positive") {
        cat.push("VALUE_GARP");
      }

      const rrVal = typeof opt?.risk_reward_ratio === 'number' ? opt.risk_reward_ratio : null;
      const stageStr = opt?.setup_pattern || (opt?.stage_phase ? `Stage ${opt.stage_phase} Base` : 'Base Under Evaluation');

      const dec = data.decisionTrace;

      const newAsset: RadarAsset = {
        ticker: clean,
        name: clean,
        price: Number(data.currentPrice.toFixed(2)),
        rvol: null,
        riskRewardRatio: rrVal,
        vcpStage: stageStr,
        volumeDryUpPct: null,
        confluenceScore: Math.round(data.confluence?.confluenceScore || 50),
        catalyst: opt?.entry_thesis || "On-demand quantitative exchange tape discovery.",
        categories: cat,
        sector: "On-Demand Discovery",
        executionStatus,
        screeningStatus: executionStatus === 'IN_BUY_ZONE' ? 'SCREENING_ZONE' : (executionStatus !== 'UNKNOWN' ? executionStatus : 'UNKNOWN'),
        screeningGeometry: executionStatus === 'IN_BUY_ZONE' ? 'WITHIN_TOLERANCE' : 'OUTSIDE_TOLERANCE',
        decisionState: dec?.decisionState || (data.degradedMode ? "UNVERIFIED" : "VALID_SETUP"),
        decisionStateLabel: dec?.stateLabel || "Discovery Candidate — Analyze for Trigger",
        isActionable: false, // Invariant: Radar is a discovery scanner; canonical actionability reserved for DecisionTrace
        canSizeTrade: false,
        disqualificationReason: dec?.disqualificationReason || null,
        decisionContextId: data.decisionId,
        allowedActions: executionStatus === 'DELISTED_SECURITY' ? [] : ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
        screeningEligible: Boolean(executionStatus === 'IN_BUY_ZONE'),
        priceBasis: "DAILY_CLOSE",
        priceDate: (data as any)?.tradeDate ?? null,
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
    const filters: CategoryFilter[] = ['ALL', 'VALUE_GARP', 'VCP', 'SMART_MONEY'];
    const idx = filters.indexOf(current);
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      const next = filters[(idx + 1) % filters.length];
      const buttonId = `tab-radar-filter-${next.toLowerCase()}`;
      handleCategoryFilterSelect(next, buttonId);
      document.getElementById(buttonId)?.focus();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = filters[(idx - 1 + filters.length) % filters.length];
      const buttonId = `tab-radar-filter-${prev.toLowerCase()}`;
      handleCategoryFilterSelect(prev, buttonId);
      document.getElementById(buttonId)?.focus();
    } else if (e.key === 'Home') {
      e.preventDefault();
      const first = filters[0];
      const buttonId = `tab-radar-filter-${first.toLowerCase()}`;
      handleCategoryFilterSelect(first, buttonId);
      document.getElementById(buttonId)?.focus();
    } else if (e.key === 'End') {
      e.preventDefault();
      const last = filters[filters.length - 1];
      const buttonId = `tab-radar-filter-${last.toLowerCase()}`;
      handleCategoryFilterSelect(last, buttonId);
      document.getElementById(buttonId)?.focus();
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

        <SynthesisGuideBanner />

        {/* Loading Indicator */}
        {isLoading && (
          <div className="p-12 text-center text-[var(--text-2)] font-mono text-xs animate-pulse">
            ⏳ Scanning multi-factor equity tape and quantitative confluence filters...
          </div>
        )}

        {/* Explicit Connection Error State */}
        {!isLoading && fetchError && (
          <div className="p-12 rounded-2xl border border-rose-900/60 bg-rose-950/20 text-center space-y-4 font-mono text-xs text-[var(--text-2)]">
            <span className="text-3xl">⚠️</span>
            <div className="space-y-1">
              <div className="text-rose-400 font-bold text-sm">Radar Tape Connection Error</div>
              <p className="text-[var(--text-2)] max-w-lg mx-auto font-sans">
                Unable to load live screening data: {fetchError}. Missing API connectivity is not represented as zero qualifying candidates.
              </p>
            </div>
            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                type="button"
                onClick={() => setRefreshNonce((n) => n + 1)}
                className="px-4 py-2.5 min-h-[44px] min-w-[44px] inline-flex items-center justify-center rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-medium text-xs transition-colors shadow-lg shadow-rose-950/40 focus:outline-none focus:ring-2 focus:ring-rose-400"
              >
                ↻ Retry Connection
              </button>
            </div>
          </div>
        )}

        {/* Actionable Empty State */}
        {!isLoading && !fetchError && allAssets.length === 0 && (
          <div className="p-12 rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-1)] text-center space-y-4 font-mono text-xs text-[var(--text-2)]">
            <span className="text-3xl">📡</span>
            <div className="space-y-1">
              <div className="text-[var(--text-1)] font-bold text-sm">No Active Confluence Candidates</div>
              <p className="text-[var(--text-2)] max-w-lg mx-auto font-sans">
                No market assets currently meet the active multi-factor confluence, valuation, or execution trigger thresholds. This is normal during market pullbacks or broad consolidation regimes.
              </p>
            </div>
            <div className="flex items-center justify-center gap-3 pt-2">
              <Link
                href="/"
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-black rounded-xl text-xs font-sans transition-transform active:scale-95"
              >
                Analyze Any Ticker →
              </Link>
              <Link
                href="/setups"
                className="px-4 py-2 bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-2)] font-bold rounded-xl text-xs font-sans transition-colors"
              >
                Review Active Setups →
              </Link>
            </div>
          </div>
        )}

        {/* Search Discovery / Empty Status Banner when filteredAssets is 0 */}
        {!isLoading && isSearching && !heroAsset && (
          <div className="p-5 md:p-6 rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-2)] backdrop-blur-md space-y-3 font-mono text-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="text-2xl">🔍</span>
                <div>
                  <div className="text-[var(--text-1)] font-bold text-sm">
                    Searching for &quot;{cleanQ}&quot; · 0 matching candidates in current pre-scanned batch
                  </div>
                  <div className="text-[var(--text-2)] text-[11px] mt-0.5 font-sans">
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
                  className="px-3 py-2 rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-2)] font-bold text-xs transition-all border border-[var(--border-subtle)] cursor-pointer"
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
          <div className="rounded-2xl border border-emerald-500/40 bg-gradient-to-br from-emerald-950/40 via-slate-900 to-slate-950 p-4 sm:p-5 md:p-6 shadow-2xl space-y-4">
            {/* Top Flow Header: Ribbon & Context Badge in Normal Document Flow (Zero Overlap on Mobile) */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-emerald-500/20">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 bg-emerald-500/15 border border-emerald-500/30 text-[10px] sm:text-xs font-mono uppercase tracking-widest text-emerald-300 font-bold rounded-lg">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Attention Candidate · Top Pre-Screened Confluence
              </div>
              <div
                className="inline-flex items-center gap-1.5 text-[10px] sm:text-[11px] font-mono text-[var(--text-2)] bg-[var(--surface-1)] px-2 py-0.5 rounded border border-[var(--border-subtle)]"
                title="Radar displays pre-calculated batch screening snapshot. Individual asset live indicators are calculated in Analysis hub."
              >
                <span>📡</span>
                <span>Screening Snapshot</span>
                <span className="text-[var(--text-muted)]">|</span>
                <span className="text-cyan-400">Batch Universe</span>
              </div>
            </div>

            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div className="space-y-3 max-w-3xl">
                <div className="flex flex-wrap items-center gap-2.5">
                  <span className="text-2xl md:text-3xl font-black font-mono tracking-tight text-[var(--text-1)]">
                    {heroAsset.ticker}
                  </span>
                  {heroAsset.name && heroAsset.name !== heroAsset.ticker && (
                    <span className="text-sm md:text-base text-[var(--text-2)] font-medium">
                      {heroAsset.name}
                    </span>
                  )}
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                    ATTENTION CANDIDATE
                  </span>
                  <RadarPortfolioBadge
                    ownershipState={heroAsset ? getOwnershipState(heroAsset.ticker) : 'UNKNOWN'}
                    shares={heroAsset ? getHolding(heroAsset.ticker)?.shares : undefined}
                  />
                  <span className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full border ${
                    heroAsset.executionStatus === 'IN_BUY_ZONE' || heroAsset.executionStatus === 'READY_TO_BUY'
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
                      : heroAsset.executionStatus === 'IN_BUY_ZONE_AWAITING_TRIGGER'
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                      : 'bg-[var(--surface-2)] text-[var(--text-2)] border-[var(--border-strong)]'
                  }`}>
                    {formatRadarExecutionStatus(heroAsset.executionStatus)}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--surface-1)] border border-[var(--border-strong)] text-[var(--text-2)]">
                    Posture: Analyze for Trigger
                  </span>
                  {heroAsset.vcpStage && (
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-[var(--surface-2)] text-[var(--text-2)]">
                      {heroAsset.vcpStage}
                    </span>
                  )}
                </div>

                <p className="text-xs md:text-sm text-[var(--text-1)] font-sans leading-relaxed">
                  <strong className="text-amber-400 font-semibold">Primary Catalyst: </strong>
                  {heroAsset.catalyst}
                </p>

                {heroAsset && getOwnershipState(heroAsset.ticker) === 'HELD' && (
                  <div className="p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-700/60 text-indigo-200 text-xs font-mono flex items-center gap-2">
                    <span className="text-sm">💼</span>
                    <span>
                      <strong>Existing Portfolio Holding:</strong> {getHolding(heroAsset.ticker)?.shares ?? 0} shares held. Aligns with top scanner confluence.
                    </span>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-4 text-xs font-mono pt-1">
                  <div
                    className="flex items-center gap-1.5"
                    title={
                      heroAsset.executionStatus === 'DELISTED_SECURITY'
                        ? "Historical final price. Ceased public trading."
                        : heroAsset.priceBasis === 'DAILY_CLOSE' && heroAsset.priceDate
                        ? `Daily Close · ${heroAsset.priceDate}. Live spot quotes active in Analysis hub.`
                        : "Observation date and/or price basis unverified. Live spot quotes active in Analysis hub."
                    }
                  >
                    <span className="text-[var(--text-2)]">Screen Price:</span>
                    <span className="text-[var(--text-1)] font-bold">
                      {heroAsset.price && heroAsset.price > 0 ? `$${heroAsset.price.toFixed(2)}` : <span className="text-amber-400">Price Unavailable</span>}
                    </span>
                    <span className={`text-[10px] font-sans ${!heroAsset.priceDate || heroAsset.priceBasis !== 'DAILY_CLOSE' ? 'text-amber-400 font-medium' : 'text-[var(--text-2)]'}`}>
                      · {heroAsset.executionStatus === 'DELISTED_SECURITY'
                        ? "Historical Final Price (Ceased Trading)"
                        : formatPriceObservationDate(heroAsset.priceDate, heroAsset.priceBasis)}
                    </span>
                  </div>
                  <span className="text-slate-700">•</span>
                  <div className="flex items-center gap-1.5" title="Pre-screened multi-factor confluence conviction score (0-100) from batch screener pipeline">
                    <span className="text-[var(--text-2)]">Screen Confluence:</span>
                    <span className="text-emerald-400 font-bold">{heroAsset.confluenceScore}/100</span>
                  </div>
                  {heroAsset.rvol && (
                    <>
                      <span className="text-slate-700">•</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[var(--text-2)]">RVOL:</span>
                        <span className="text-cyan-300 font-bold">{heroAsset.rvol}</span>
                      </div>
                    </>
                  )}
                  {heroAsset.riskRewardRatio !== null && heroAsset.riskRewardRatio !== undefined && (
                    <>
                      <span className="text-slate-700">•</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[var(--text-2)]">Risk:Reward:</span>
                        <span className="text-emerald-400 font-bold">{heroAsset.riskRewardRatio.toFixed(1)}:1</span>
                      </div>
                    </>
                  )}
                  {typeof heroAsset.volumeDryUpPct === 'number' && (
                    <>
                      <span className="text-slate-700">•</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[var(--text-2)]">Vol Dry-Up:</span>
                        <span className="text-emerald-400 font-bold">{heroAsset.volumeDryUpPct}%</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="flex flex-col sm:flex-row lg:flex-col gap-2.5 shrink-0">
                {(() => {
                  const allowed = Array.isArray(heroAsset.allowedActions) ? heroAsset.allowedActions : [];
                  const isDelisted = heroAsset.executionStatus === 'DELISTED_SECURITY';
                  const isUnverified = heroAsset.executionStatus === 'UNVERIFIED_ASSET';
                  const isHeroHeld = getOwnershipState(heroAsset.ticker) === 'HELD';

                  // Gate C: Prevent false Analyze affordance on delisted or prohibited securities
                  if (isDelisted || allowed.length === 0) {
                    return (
                      <span
                        className="px-5 py-2.5 min-h-[44px] min-w-[44px] rounded-xl text-xs font-mono font-bold border border-zinc-800 bg-zinc-900/80 text-zinc-500 inline-flex items-center justify-center cursor-not-allowed select-none"
                        title={
                          isDelisted
                            ? "Security ceased public trading pursuant to confirmed corporate action. Trade execution and technical triggers disabled."
                            : "Actionability unavailable: No authorized actions permitted for this security."
                        }
                      >
                        {isDelisted ? "Delisted / Inactive" : "Action Unavailable"}
                      </span>
                    );
                  }

                  if (isUnverified) {
                    if (allowed.includes("RESEARCH_PROFILE")) {
                      return (
                        <Link
                          href={`/?symbol=${heroAsset.ticker}`}
                          onClick={() => trackRadarAssetClick(heroAsset.ticker, 1)}
                          className="px-5 py-2.5 min-h-[44px] min-w-[44px] rounded-xl text-xs font-mono font-bold transition-all border bg-amber-950/60 hover:bg-amber-900/80 text-amber-200 border-amber-700 inline-flex items-center justify-center gap-2"
                          title="Open research profile in Analysis hub. Market data and execution clearance unverified."
                        >
                          <span>Research {heroAsset.ticker} Profile</span>
                          <span>→</span>
                        </Link>
                      );
                    }
                    return (
                      <span
                        className="px-5 py-2.5 min-h-[44px] min-w-[44px] rounded-xl text-xs font-mono font-bold border border-zinc-800 bg-zinc-900/80 text-zinc-500 inline-flex items-center justify-center cursor-not-allowed select-none"
                        title="Market data unverified. Analysis unavailable."
                      >
                        Unverified Asset
                      </span>
                    );
                  }

                  return (
                    <>
                      <Link
                        href={`/?symbol=${heroAsset.ticker}`}
                        onClick={() => trackRadarAssetClick(heroAsset.ticker, 1)}
                        className={`px-5 py-2.5 min-h-[44px] min-w-[44px] rounded-xl text-xs font-mono font-black tracking-tight transition-all shadow-lg flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] ${
                          isHeroHeld
                            ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-950/50'
                            : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-950/50'
                        }`}
                        title={isHeroHeld ? "Review existing position in Analysis hub" : `Inspect ${heroAsset.ticker} setup and structure in Analysis hub`}
                      >
                        <span>{isHeroHeld ? `Review ${heroAsset.ticker} Position` : `Analyze ${heroAsset.ticker}`}</span>
                        <span>→</span>
                      </Link>
                      <Link
                        href={`/setups?symbol=${heroAsset.ticker}`}
                        onClick={() => trackRadarAssetClick(heroAsset.ticker, 1)}
                        className="px-4 py-2.5 min-h-[44px] min-w-[44px] rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-1)] text-xs font-mono font-bold transition-all border border-[var(--border-strong)] flex items-center justify-center gap-1.5 hover:border-[var(--border-strong)] active:scale-95"
                      >
                        <span>Inspect Setup in /setups</span>
                        <span>→</span>
                      </Link>
                    </>
                  );
                })()}
                <div className="text-[10px] font-mono text-[var(--text-2)] text-center">
                  Discovery Candidate · Not an Execution Recommendation
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Search & Filter Toolbar */}
        <div className="space-y-3 border-b border-[var(--border-subtle)] pb-4">
          <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
            {/* Category Filter Tabs (WAI-ARIA Tablist) */}
            <div
              className="w-full md:w-auto overflow-x-auto overflow-y-hidden overscroll-x-contain touch-pan-x pb-1 md:pb-0"
              style={{ WebkitOverflowScrolling: 'touch' }}
            >
              <div
                role="tablist"
                aria-label="Radar Confluence Categories"
                className="flex items-center gap-1.5 min-w-max"
              >
                <button
                  type="button"
                  role="tab"
                  id="tab-radar-filter-all"
                  aria-selected={activeFilter === 'ALL'}
                  aria-controls="panel-radar-candidates"
                  tabIndex={activeFilter === 'ALL' ? 0 : -1}
                  onKeyDown={(e) => handleFilterKeyDown(e, 'ALL')}
                  onClick={() => handleCategoryFilterSelect('ALL', 'tab-radar-filter-all')}
                  className={`focus-ring px-3.5 py-2 min-h-[44px] shrink-0 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeFilter === 'ALL'
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>All Confluences</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--surface-2)] text-[var(--text-2)] font-mono">
                    {categoryMeta.ALL.badge}
                  </span>
                </button>

                <button
                  type="button"
                  role="tab"
                  id="tab-radar-filter-value_garp"
                  aria-selected={activeFilter === 'VALUE_GARP'}
                  aria-controls="panel-radar-candidates"
                  tabIndex={activeFilter === 'VALUE_GARP' ? 0 : -1}
                  onKeyDown={(e) => handleFilterKeyDown(e, 'VALUE_GARP')}
                  onClick={() => handleCategoryFilterSelect('VALUE_GARP', 'tab-radar-filter-value_garp')}
                  title="16 Value / GARP matches identified from 35 curated reference catalog candidates evaluated. Whole-market automated screening is pending deployment."
                  className={`focus-ring px-3.5 py-2 min-h-[44px] shrink-0 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeFilter === 'VALUE_GARP'
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>💎 Value / GARP</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--surface-2)] text-[var(--text-2)] font-mono">
                    {categoryMeta.VALUE_GARP.badge}
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
                  onClick={() => handleCategoryFilterSelect('VCP', 'tab-radar-filter-vcp')}
                  title={categoryMeta.VCP.status === 'PIPELINE_PENDING' ? "Automated market-wide volatility contraction screening is not active yet. Single-asset VCP geometry is still available in the Analysis (/) and Setups (/setups) hubs." : undefined}
                  className={`focus-ring px-3.5 py-2 min-h-[44px] shrink-0 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeFilter === 'VCP'
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>⚡ Minervini VCP</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono ${
                    categoryMeta.VCP.status === 'PIPELINE_PENDING'
                      ? 'bg-amber-950/60 text-amber-300 border border-amber-800/60'
                      : 'bg-[var(--surface-2)] text-[var(--text-2)]'
                  }`}>
                    {categoryMeta.VCP.badge}
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
                  onClick={() => handleCategoryFilterSelect('SMART_MONEY', 'tab-radar-filter-smart_money')}
                  title={categoryMeta.SMART_MONEY.status === 'PIPELINE_PENDING' ? "Automated market-wide institutional-flow screening is not active yet. Single-asset insider and congressional filing analysis is still available for individual symbols on /smart-money." : undefined}
                  className={`focus-ring px-3.5 py-2 min-h-[44px] shrink-0 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeFilter === 'SMART_MONEY'
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>🐋 Smart Money</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono ${
                    categoryMeta.SMART_MONEY.status === 'PIPELINE_PENDING'
                      ? 'bg-amber-950/60 text-amber-300 border border-amber-800/60'
                      : 'bg-[var(--surface-2)] text-[var(--text-2)]'
                  }`}>
                    {categoryMeta.SMART_MONEY.badge}
                  </span>
                </button>
              </div>
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
                  className="focus-ring w-full bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-lg pl-3 pr-20 py-2.5 min-h-[44px] text-xs text-[var(--text-1)] placeholder-[var(--text-muted)] focus:outline-none focus:border-cyan-500 font-mono"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                  {searchQuery && (
                    <button
                      type="button"
                      onClick={() => {
                        setSearchQuery('');
                        setOnDemandError(null);
                      }}
                      className="focus-ring text-xs text-[var(--text-muted)] hover:text-[var(--text-1)] font-mono px-2 min-h-[36px] min-w-[36px] flex items-center justify-center rounded cursor-pointer"
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
                      className="focus-ring text-[10px] bg-cyan-900/80 hover:bg-cyan-800 text-cyan-300 px-2 py-1 min-h-[36px] min-w-[44px] flex items-center justify-center rounded font-mono font-bold border border-cyan-700 cursor-pointer disabled:opacity-50"
                      title="Scan on-demand"
                    >
                      {isOnDemandLoading ? "..." : "Scan ↵"}
                    </button>
                  )}
                </div>
              </form>

              {searchQuery && (
                <span className="text-[11px] font-mono text-[var(--text-2)] shrink-0">
                  {filteredAssets.length} of {allAssets.length}
                </span>
              )}

              <div className="flex flex-wrap items-center gap-1 bg-[var(--surface-1)] border border-[var(--border-subtle)] rounded-lg p-1 text-xs font-mono shrink-0">
                <span className="text-[10px] text-[var(--text-muted)] px-2 uppercase">Sort:</span>
                <button
                  type="button"
                  onClick={() => setSortBy('SCORE')}
                  title="Sort by Confluence Score (Technical execution setup, RSI momentum, and smart money flow — not intrinsic valuation discount)"
                  className={`focus-ring px-3 py-2 min-h-[44px] inline-flex items-center justify-center rounded ${sortBy === 'SCORE' ? 'bg-[var(--surface-2)] text-[var(--text-1)] font-bold' : 'text-[var(--text-2)] hover:text-[var(--text-1)]'}`}
                >
                  Score (Confluence)
                </button>
                <button
                  type="button"
                  onClick={() => setSortBy('RVOL')}
                  className={`focus-ring px-3 py-2 min-h-[44px] min-w-[44px] inline-flex items-center justify-center rounded ${sortBy === 'RVOL' ? 'bg-[var(--surface-2)] text-[var(--text-1)] font-bold' : 'text-[var(--text-2)] hover:text-[var(--text-1)]'}`}
                >
                  RVOL
                </button>
                <button
                  type="button"
                  onClick={() => setSortBy('PRICE')}
                  className={`focus-ring px-3 py-2 min-h-[44px] min-w-[44px] inline-flex items-center justify-center rounded ${sortBy === 'PRICE' ? 'bg-[var(--surface-2)] text-[var(--text-1)] font-bold' : 'text-[var(--text-2)] hover:text-[var(--text-1)]'}`}
                >
                  Price
                </button>
              </div>
            </div>
          </div>

          {/* Secondary Row: Portfolio Ownership Filter Chips & Degraded Status */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[var(--border-subtle)]">
            <div
              className="w-full sm:w-auto overflow-x-auto overflow-y-hidden overscroll-x-contain touch-pan-x pb-1 sm:pb-0"
              style={{ WebkitOverflowScrolling: 'touch' }}
            >
              <div className="flex items-center gap-1.5 min-w-max" role="group" aria-label="Portfolio Ownership Filter">
                <span className="text-[10px] uppercase font-mono text-[var(--text-muted)] mr-1 shrink-0">Portfolio:</span>
                <button
                  type="button"
                  id="btn-ownership-filter-all"
                  aria-pressed={ownershipFilter === 'ALL'}
                  onClick={() => handleOwnershipFilterSelect('ALL', 'btn-ownership-filter-all')}
                  className={`focus-ring px-2.5 py-1 min-h-[44px] shrink-0 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
                    ownershipFilter === 'ALL'
                      ? 'bg-[var(--surface-2)] text-[var(--text-1)] font-bold border border-[var(--border-strong)]'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>All Candidates</span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-[var(--surface-2)] text-[var(--text-2)] font-mono">
                    {ownershipCounts.total}
                  </span>
                </button>

                <button
                  type="button"
                  id="btn-ownership-filter-new"
                  aria-pressed={ownershipFilter === 'NEW_OPPORTUNITIES'}
                  onClick={() => handleOwnershipFilterSelect('NEW_OPPORTUNITIES', 'btn-ownership-filter-new')}
                  className={`focus-ring px-2.5 py-1 min-h-[44px] shrink-0 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
                    ownershipFilter === 'NEW_OPPORTUNITIES'
                      ? 'bg-emerald-950/80 text-emerald-300 font-bold border border-emerald-700/70 shadow-sm'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>Unheld Candidates</span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-[var(--surface-2)] text-[var(--text-2)] font-mono">
                    {isPortfolioVerified ? ownershipCounts.notHeld : '—'}
                  </span>
                </button>

                <button
                  type="button"
                  id="btn-ownership-filter-holdings"
                  aria-pressed={ownershipFilter === 'MY_HOLDINGS'}
                  onClick={() => handleOwnershipFilterSelect('MY_HOLDINGS', 'btn-ownership-filter-holdings')}
                  className={`focus-ring px-2.5 py-1 min-h-[44px] shrink-0 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
                    ownershipFilter === 'MY_HOLDINGS'
                      ? 'bg-indigo-950/80 text-indigo-300 font-bold border border-indigo-700/70 shadow-sm'
                      : 'text-[var(--text-2)] hover:text-[var(--text-1)] hover:bg-[var(--surface-hover)] border border-transparent'
                  }`}
                >
                  <span>My Holdings</span>
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    ownershipCounts.held > 0 ? 'bg-indigo-900/60 text-indigo-200 font-bold' : 'bg-[var(--surface-2)] text-[var(--text-muted)]'
                  }`}>
                    {isPortfolioVerified ? ownershipCounts.held : '—'}
                  </span>
                </button>

                {typeof population?.currentOpportunityCount === 'number' && (
                  <div
                    className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2.5 py-1 min-h-[44px] rounded-lg shrink-0 ml-1"
                    title="Authoritative count of candidates currently meeting active buy zone screening criteria (IN_BUY_ZONE)"
                  >
                    <span className="text-[var(--text-2)]">Current Opportunities:</span>
                    <span className="font-bold text-emerald-300">{population.currentOpportunityCount}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Degraded State Indicator */}
            {isPortfolioDegraded && (
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-amber-400/90 bg-amber-950/30 border border-amber-800/40 px-2 py-0.5 rounded">
                <span>⚠️</span>
                <span>Portfolio sync unavailable — ownership state UNKNOWN</span>
              </div>
            )}
          </div>
        </div>

        {/* Scope and Coverage Disclosure Banner */}
        {activeFilter === 'VALUE_GARP' && (
          <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] text-xs font-mono">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-bold text-cyan-400">💎 Value / GARP</span>
              <span className="text-slate-600">•</span>
              <span className="text-[var(--text-2)]">
                Baseline: <strong className="text-white">Curated Fundamental Reference Catalog</strong>
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-[var(--text-2)]">
                Sorted by: {sortBy === 'SCORE' ? (
                  <>
                    <strong className="text-cyan-300">Technical Confluence</strong> (Confluence Score descending), <span className="text-amber-300/90 font-medium">not an intrinsic valuation discount ranking</span>
                  </>
                ) : sortBy === 'RVOL' ? (
                  <>
                    <strong className="text-cyan-300">Relative Volume</strong> (RVOL descending), <span className="text-amber-300/90 font-medium">not an intrinsic valuation discount ranking</span>
                  </>
                ) : (
                  <>
                    <strong className="text-cyan-300">Share Price</strong> (Nominal price descending), <span className="text-amber-300/90 font-medium">not an intrinsic valuation discount ranking</span>
                  </>
                )}
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-[var(--text-2)]">
                {filteredAssets.length} of 16 Value / GARP Matches · 100% of Defined Reference Catalog Evaluated ({allAssets.length} Assets)
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] bg-[var(--surface-2)] text-[var(--text-2)] border border-[var(--border-strong)]" title="100% of 35-asset curated reference catalog evaluated. Whole-market automated screening is pending VCP and Smart Money pipeline activation.">
                REFERENCE CATALOG (16 MATCHES / 35 EVALUATED)
              </span>
            </div>
          </div>
        )}

        {activeFilter === 'VCP' && (
          <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-xl bg-[var(--surface-2)] border border-[var(--border-subtle)] text-xs font-mono">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-bold text-cyan-400">⚡ Minervini VCP</span>
              <span className="text-slate-600">•</span>
              <span className="text-[var(--text-2)]">
                Scope: <strong className="text-white">{capabilities.VCP?.universeId || 'ARX-eligible US equities'}</strong>
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-[var(--text-2)]">
                {(() => {
                  const hasValidProvenance = Boolean(
                    typeof capabilities.VCP?.populationDenominator === 'number' &&
                    capabilities.VCP.populationDenominator > 0 &&
                    capabilities.VCP?.universeBuildId &&
                    capabilities.VCP?.publicationDigest
                  );
                  const coverageText = hasValidProvenance
                    ? `${capabilities.VCP?.populationDenominator} scannable population`
                    : 'Coverage Pending';

                  const authMatches = typeof capabilities.VCP?.matchedCount === 'number' && !capabilities.VCP?.isRollback
                    ? capabilities.VCP.matchedCount
                    : null;

                  if (capabilities.VCP?.universeScreening === 'AVAILABLE' && authMatches !== null) {
                    if (filteredAssets.length !== authMatches) {
                      return (
                        <>
                          Showing <strong className="text-white">{filteredAssets.length}</strong> of <strong className="text-white">{authMatches}</strong> scanner matches · {coverageText}
                        </>
                      );
                    }
                    return (
                      <>
                        <strong className="text-white">{authMatches}</strong> scanner matches · {coverageText}
                      </>
                    );
                  }

                  return (
                    <>
                      Scanner Pending · {coverageText}
                    </>
                  );
                })()}
              </span>
              {capabilities.VCP?.universeBuildId && (
                <>
                  <span className="text-slate-600">•</span>
                  <span className="text-[var(--text-2)]" title={capabilities.VCP?.publicationDigest ? `Digest: ${capabilities.VCP.publicationDigest}` : undefined}>
                    Build: <strong className="text-[var(--text-1)]">{capabilities.VCP.universeBuildId}</strong>
                    {capabilities.VCP?.isRollback && <span className="ml-1 text-amber-400 font-bold">(Rollback)</span>}
                  </span>
                </>
              )}
              {capabilities.VCP?.dataAsOf && (
                <>
                  <span className="text-slate-600">•</span>
                  <span className="text-[var(--text-2)]">
                    Data As Of: <strong className="text-[var(--text-1)]">{capabilities.VCP.dataAsOf}</strong>
                  </span>
                </>
              )}
            </div>
            <div className="flex items-center gap-2">
              {(() => {
                const asOfStr = capabilities.VCP?.dataAsOf;
                if (!asOfStr) return null;
                const asOfDate = new Date(asOfStr + 'T00:00:00Z');
                const now = new Date();
                const diffDays = Math.floor((now.getTime() - asOfDate.getTime()) / (1000 * 60 * 60 * 24));
                if (diffDays > 3) {
                  return (
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20"
                      title={`VCP batch evaluation based on market data as of ${asOfStr}; live ticker quotes are updated real-time.`}
                    >
                      ⚠️ Prior Evaluation: {asOfStr} · Quotes Live
                    </span>
                  );
                }
                return null;
              })()}
              {(() => {
                const cap = capabilities.VCP;
                const label = getPublicationIntegrityLabel(cap);
                if (cap?.isRollback || cap?.publicationIntegrity === 'ROLLBACK') {
                  return (
                    <div className="text-[11px] text-amber-400/90 font-mono flex items-center gap-1.5" data-testid="publication-integrity-badge">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                      {label}
                    </div>
                  );
                }
                if (cap?.publicationIntegrity === 'VERIFIED' || cap?.publicationIntegrityVerified === true) {
                  return (
                    <div className="text-[11px] text-emerald-400 font-mono flex items-center gap-1.5" data-testid="publication-integrity-badge">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                      {label}
                    </div>
                  );
                }
                return (
                  <div className="text-[11px] text-[var(--text-muted)] font-mono flex items-center gap-1.5" data-testid="publication-integrity-badge">
                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-muted)]"></span>
                    {label}
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {/* Level 1: Dense Confluence Stream Table (Tabpanel) */}
        {!fetchError && (
          <div
            role="tabpanel"
            id="panel-radar-candidates"
            aria-label="Radar Confluence Candidates"
            tabIndex={0}
            className="rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-1)] overflow-hidden shadow-xl focus-ring"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-[var(--bg-app)] border-b border-[var(--border-subtle)] text-[var(--text-2)] text-[10px] uppercase tracking-wider">
                <tr>
                  <th className="p-3 sticky left-0 bg-[var(--bg-app)] z-10 min-w-[130px]">Asset</th>
                  <th className="p-3 min-w-[140px]">Screening Status</th>
                  <th
                    className="p-3 min-w-[100px]"
                    title="Batch screening reference price from daily session close. Live spot quotes active in Analysis hub."
                  >
                    Screen Price <span className="text-[9px] font-normal text-[var(--text-muted)] normal-case">· Daily Close</span>
                  </th>
                  <th className="p-3 text-center min-w-[70px]" title="Screening Snapshot Confluence Conviction Score">Screen Score</th>
                  <th className="p-3 text-center min-w-[70px]">RVOL</th>
                  <th className="p-3 min-w-[120px]">VCP / Base Setup</th>
                  <th className="p-3 text-center min-w-[80px]">R:R Ratio</th>
                  <th className="p-3 min-w-[200px]">Catalyst Rationale</th>
                  <th className="p-3 text-right min-w-[90px]">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)]">
                {filteredAssets.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="p-8 text-center text-[var(--text-2)] space-y-4 font-mono">
                      <div className="max-w-md mx-auto space-y-3">
                        <span className="text-3xl block">🔍</span>
                        <div className="space-y-1">
                          <p className="text-sm font-bold text-[var(--text-1)]">
                            {searchQuery
                              ? `No assets match "${searchQuery}" in current confluence scan`
                              : ownershipFilter === 'MY_HOLDINGS'
                              ? (!isPortfolioVerified
                                  ? "Portfolio sync unavailable (Ownership UNKNOWN)"
                                  : "No active Radar candidates currently held in portfolio")
                              : ownershipFilter === 'NEW_OPPORTUNITIES' && !isPortfolioVerified
                              ? "Portfolio sync unavailable (Ownership UNKNOWN)"
                              : activeFilter === 'VCP' && categoryMeta.VCP.status === 'PIPELINE_PENDING'
                              ? "Minervini VCP — Universe Scanner Pending"
                              : activeFilter === 'SMART_MONEY' && categoryMeta.SMART_MONEY.status === 'PIPELINE_PENDING'
                              ? "Smart Money — Universe Scanner Pending"
                              : `No qualifying candidates currently in "${activeFilter === 'VALUE_GARP' ? 'Value / GARP' : activeFilter}"`}
                          </p>
                          <p className="text-xs text-[var(--text-2)] font-sans leading-relaxed">
                            {searchQuery
                              ? (isTickerQuery
                                  ? `Asset "${cleanQ}" is not in today's top 24 pre-scanned confluence batch. You can scan it live across the exchange tape or review its tactical setup.`
                                  : `No confluence candidates match this search. Try searching by symbol (e.g. NVDA, CPRX, ASML, TSLA) or reset the filter.`)
                              : ownershipFilter === 'MY_HOLDINGS'
                              ? (!isPortfolioVerified
                                  ? "The authoritative portfolio connection could not be verified. Radar renders canonically with ownership state UNKNOWN."
                                  : "None of your current portfolio positions meet the active Radar screening confluence criteria. Switch to 'All Candidates' or 'Unheld Candidates' to view market setups.")
                              : ownershipFilter === 'NEW_OPPORTUNITIES' && !isPortfolioVerified
                              ? "Filtering for unheld candidates requires verified portfolio state. Currently running in degraded UNKNOWN mode."
                              : activeFilter === 'VCP' && categoryMeta.VCP.status === 'PIPELINE_PENDING'
                              ? (capabilities.VCP?.populationDenominator && capabilities.VCP?.universeBuildId && capabilities.VCP?.publicationDigest
                                  ? `${capabilities.VCP.populationDenominator} securities registered in universe build ${capabilities.VCP.universeBuildId}. Automated market-wide volatility contraction screening is pending operational activation.`
                                  : "Automated market-wide volatility contraction screening is not active yet. Single-asset VCP geometry is still available in the Analysis (/) and Setups (/setups) hubs.")
                              : activeFilter === 'SMART_MONEY' && categoryMeta.SMART_MONEY.status === 'PIPELINE_PENDING'
                              ? (capabilities.SMART_MONEY?.populationDenominator && capabilities.SMART_MONEY?.universeBuildId && capabilities.SMART_MONEY?.publicationDigest
                                  ? `${capabilities.SMART_MONEY.populationDenominator} securities registered in universe build ${capabilities.SMART_MONEY.universeBuildId}. Automated market-wide institutional-flow screening is pending operational activation.`
                                  : "Automated market-wide institutional-flow screening is not active yet. Single-asset insider and congressional filing analysis is still available for individual symbols on /smart-money.")
                              : activeFilter === 'ALL'
                              ? "Try clearing your search query or refreshing the tape to discover active confluence setups."
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
                              className="px-4 py-2.5 min-h-[44px] rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold font-mono transition-transform active:scale-95 shadow-lg shadow-cyan-950/50 inline-flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-50"
                            >
                              <span>{isOnDemandLoading ? "⏳ Scanning Tape..." : "⚡ Scan " + cleanQ + " On-Demand"}</span>
                            </button>
                          )}

                          {isTickerQuery && (
                            <Link
                              href={`/setups?ticker=${cleanQ}`}
                              className="px-3.5 py-2.5 min-h-[44px] rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-cyan-300 text-xs font-bold font-mono border border-[var(--border-strong)] inline-flex items-center justify-center"
                            >
                              Inspect in /setups →
                            </Link>
                          )}

                          {isTickerQuery && (
                            <Link
                              href={`/?symbol=${cleanQ}`}
                              className="px-3.5 py-2.5 min-h-[44px] rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-2)] text-xs font-bold font-mono border border-[var(--border-strong)] inline-flex items-center justify-center"
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
                            className="px-3.5 py-2.5 min-h-[44px] rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-2)] hover:text-[var(--text-1)] text-xs font-bold font-mono border border-[var(--border-strong)] inline-flex items-center justify-center cursor-pointer"
                          >
                            ✕ Clear Filter (Show All {allAssets.length})
                          </button>
                        </div>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredAssets.map((asset) => {
                    const isNearZone = asset.executionStatus === 'IN_BUY_ZONE' || asset.executionStatus === 'READY_TO_BUY';
                    const isTarget = asset.executionStatus === 'APPROACHING_TARGET';
                    const screeningLabel = formatRadarExecutionStatus(asset.executionStatus);

                    return (
                      <tr key={asset.ticker} className="hover:bg-[var(--surface-hover)] transition-colors group">
                        <td className="p-3 sticky left-0 bg-[var(--surface-1)] z-10">
                          <div className="flex items-center gap-1.5">
                            <Link
                              href={`/?symbol=${asset.ticker}`}
                              onClick={() => trackRadarAssetClick(asset.ticker)}
                              className="font-black text-[var(--text-1)] text-sm tracking-tight hover:text-cyan-400 transition-colors inline-flex items-center min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded"
                            >
                              {asset.ticker}
                            </Link>
                            <RadarPortfolioBadge
                              ownershipState={getOwnershipState(asset.ticker)}
                              shares={getHolding(asset.ticker)?.shares}
                            />
                          </div>
                          {asset.name && asset.name !== asset.ticker && (
                            <div className="text-[10px] text-[var(--text-2)] font-sans truncate max-w-[130px]" title={asset.name}>{asset.name}</div>
                          )}
                        </td>
                        <td className="p-3 whitespace-nowrap">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            asset.executionStatus === 'DELISTED_SECURITY'
                              ? 'bg-rose-950/80 text-rose-300 border border-rose-700/80'
                              : isNearZone
                              ? 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/60'
                              : asset.executionStatus === 'IN_BUY_ZONE_AWAITING_TRIGGER'
                              ? 'bg-amber-950 text-amber-300 border border-amber-800'
                              : isTarget
                              ? 'bg-blue-950 text-blue-300 border border-blue-800'
                              : 'bg-[var(--surface-2)] text-[var(--text-2)] border border-[var(--border-subtle)]'
                          }`}>
                            {screeningLabel}
                          </span>
                        </td>
                        <td
                          className="p-3 font-mono text-xs font-bold text-[var(--text-1)] whitespace-nowrap"
                          title={
                            asset.executionStatus === 'DELISTED_SECURITY'
                              ? "Historical final price. Ceased public trading."
                              : asset.priceBasis === 'DAILY_CLOSE'
                              ? (asset.priceDate ? `Daily Close · ${asset.priceDate}. Open Analysis hub for live tape.` : "Daily Close · Date Unverified.")
                              : (asset.priceDate ? `Price · ${asset.priceDate} (Basis Unverified).` : "Observation date & price basis unverified.")
                          }
                        >
                          <div>
                            {asset.executionStatus === 'DELISTED_SECURITY' ? (
                              <span className="text-[var(--text-2)] font-mono text-xs">
                                {asset.price > 0 ? `$${asset.price.toFixed(2)}` : 'Price Unavailable'}
                              </span>
                            ) : asset.price > 0 ? (
                              `$${asset.price.toFixed(2)}`
                            ) : (
                              <span className="text-amber-400 font-normal text-xs">Price Unavailable</span>
                            )}
                          </div>
                          {asset.executionStatus === 'DELISTED_SECURITY' ? (
                            <div className="text-[9px] font-normal text-rose-400/90 font-mono" title="Ceased public trading">
                              Historical Close
                            </div>
                          ) : asset.priceBasis === 'DAILY_CLOSE' && asset.priceDate ? (
                            <div className="text-[9px] font-normal text-[var(--text-muted)] font-mono">
                              {asset.priceDate}
                            </div>
                          ) : asset.priceDate ? (
                            <div className="text-[9px] font-normal text-[var(--text-muted)] font-mono" title="Price basis unverified">
                              {asset.priceDate} (Unverified Basis)
                            </div>
                          ) : (
                            <div className="text-[9px] font-medium text-amber-400 font-mono" title="Price observation date and basis unverified">
                              Date & Basis Unverified
                            </div>
                          )}
                        </td>
                        <td className="p-3 text-center whitespace-nowrap">
                          <span className={`font-bold px-1.5 py-0.5 rounded ${
                            asset.executionStatus === 'DELISTED_SECURITY'
                              ? 'text-rose-400 bg-rose-950/60 border border-rose-800/60'
                              : 'text-emerald-400 bg-emerald-950/60 border border-emerald-800/60'
                          }`} title="Screening Snapshot Score">
                            {asset.confluenceScore}
                          </span>
                        </td>
                        <td className="p-3 text-center font-bold text-cyan-300 whitespace-nowrap">
                          {asset.rvol ? asset.rvol : <span className="text-[var(--text-muted)] font-normal">—</span>}
                        </td>
                        <td className="p-3 text-[var(--text-2)] text-[11px] whitespace-nowrap">{asset.vcpStage}</td>
                        <td className="p-3 text-center font-bold text-emerald-400 whitespace-nowrap">
                          {asset.riskRewardRatio !== null && asset.riskRewardRatio !== undefined ? `${asset.riskRewardRatio.toFixed(1)}:1` : <span className="text-[var(--text-muted)] font-normal">—</span>}
                        </td>
                        <td className="p-3 text-[var(--text-2)] text-[11px] font-sans max-w-xs truncate" title={asset.catalyst}>
                          {asset.catalyst}
                        </td>
                        <td className="p-3 text-right whitespace-nowrap">
                          {(() => {
                            const isAssetHeld = getOwnershipState(asset.ticker) === 'HELD';
                            const allowed = Array.isArray(asset.allowedActions) ? asset.allowedActions : [];
                            const isDelisted = asset.executionStatus === 'DELISTED_SECURITY';
                            const isUnverified = asset.executionStatus === 'UNVERIFIED_ASSET';

                            // Gate C: Prevent false Analyze affordance on delisted or prohibited securities
                            if (isDelisted || allowed.length === 0) {
                              return (
                                <span
                                  className="px-3 py-2 min-h-[44px] min-w-[44px] rounded text-[11px] font-mono text-zinc-500 bg-zinc-900/80 border border-zinc-800 inline-flex items-center justify-center cursor-not-allowed select-none"
                                  title={
                                    isDelisted
                                      ? "Security ceased public trading pursuant to confirmed corporate action. Live technical triggers and trade execution are disabled."
                                      : "Actionability unavailable: No authorized actions permitted for this security."
                                  }
                                >
                                  {isDelisted ? "Delisted" : "Unavailable"}
                                </span>
                              );
                            }

                            if (isUnverified) {
                              if (allowed.includes("RESEARCH_PROFILE")) {
                                return (
                                  <Link
                                    href={`/?symbol=${asset.ticker}`}
                                    onClick={() => trackRadarAssetClick(asset.ticker)}
                                    className="px-3 py-2 min-h-[44px] min-w-[44px] rounded text-[11px] font-bold font-mono transition-colors inline-flex items-center justify-center focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none border bg-amber-950/40 hover:bg-amber-900/50 text-amber-300 border-amber-800/60"
                                    title="Open research profile in Analysis hub. Market data and execution clearance unverified."
                                  >
                                    Research Profile →
                                  </Link>
                                );
                              }
                              return (
                                <span
                                  className="px-3 py-2 min-h-[44px] min-w-[44px] rounded text-[11px] font-mono text-zinc-500 bg-zinc-900/80 border border-zinc-800 inline-flex items-center justify-center cursor-not-allowed select-none"
                                  title="Market data unverified. Analysis unavailable."
                                >
                                  Unverified
                                </span>
                              );
                            }

                            return (
                              <Link
                                href={`/?symbol=${asset.ticker}`}
                                onClick={() => trackRadarAssetClick(asset.ticker)}
                                className={`px-3 py-2 min-h-[44px] min-w-[44px] rounded text-[11px] font-bold font-mono transition-colors inline-flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none border ${
                                  isAssetHeld
                                    ? 'bg-indigo-950/80 hover:bg-indigo-600 text-indigo-200 hover:text-white border-indigo-700/80 hover:border-indigo-500'
                                    : 'bg-[var(--surface-2)] hover:bg-cyan-600 hover:text-white text-cyan-300 border-[var(--border-strong)] hover:border-cyan-500'
                                }`}
                                title={
                                  isAssetHeld
                                    ? `Review existing ${asset.ticker} position in Analysis hub`
                                    : `Inspect ${asset.ticker} setup and technical structure in Analysis hub`
                                }
                              >
                                {isAssetHeld ? "Review Position →" : "Analyze →"}
                              </Link>
                            );
                          })()}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
        )}
      </div>
      <SynthesisMobileDock />
    </TerminalShell>
  );
}

export default function RadarPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-2)] font-mono text-xs p-6">Loading Radar Stream...</div>}>
      <RadarContent />
    </Suspense>
  );
}
