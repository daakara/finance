"use client";

import React, { useState, useMemo, useEffect } from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";
import PageIntro from "../../components/PageIntro";
import { fetchJournalTrades, JournalTradeRecord } from "../../lib/api";
import { loadPortfolioPositions, PortfolioPosition } from "../../lib/portfolio";
import {
  filterEligibleLiveTrades,
  computeRealizedMetrics,
  computeChronologicalTrajectory,
  groupSetupsByPattern,
  EligibleLiveTrade,
} from "../../lib/performanceMetrics";

export default function PerformancePage() {
  const [activeTab, setActiveTab] = useState<'OVERVIEW' | 'LEDGER' | 'EDGE'>('OVERVIEW');
  const [searchTicker, setSearchTicker] = useState('');
  const [livePositions, setLivePositions] = useState<PortfolioPosition[]>([]);
  const [liveTrades, setLiveTrades] = useState<JournalTradeRecord[]>([]);
  const [liveLoading, setLiveLoading] = useState<boolean>(true);
  const [liveError, setLiveError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const positions = loadPortfolioPositions();
      setLivePositions(positions);

      const params = new URLSearchParams(window.location.search);
      const sym = params.get("symbol") || params.get("ticker");
      if (sym) {
        setSearchTicker(sym.trim().toUpperCase());
      }
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    setLiveLoading(true);
    setLiveError(null);
    fetchJournalTrades(200)
      .then((trades) => {
        if (!isMounted) return;
        setLiveTrades(trades || []);
        setLiveLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        console.warn("Failed to fetch journal trades for performance:", err);
        setLiveError("Failed to load live execution records from API.");
        setLiveLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Strict validation of completed live trades (Option A Canonical)
  // An explicitly CLOSED status is mandatory. A positive exit price alone cannot establish completion.
  // Requires valid, finite entry price, exit price, shares, and explicit realized outcome.
  const eligibleLiveTrades = useMemo<EligibleLiveTrade[]>(() => {
    return filterEligibleLiveTrades(liveTrades);
  }, [liveTrades]);

  // Derived genuine empirical live metrics (zero fabricated attribution)
  const liveSummary = useMemo(() => {
    return computeRealizedMetrics(eligibleLiveTrades, liveTrades.length);
  }, [eligibleLiveTrades, liveTrades.length]);

  // Realized Cumulative Equity Trajectory (sorted strictly by verified exitDate)
  const trajectoryResult = useMemo(() => {
    return computeChronologicalTrajectory(eligibleLiveTrades);
  }, [eligibleLiveTrades]);

  // Genuine Setup Breakdown (grouped by raw recorded setup name, zero keyword guessing)
  const setupGroups = useMemo(() => {
    return groupSetupsByPattern(eligibleLiveTrades);
  }, [eligibleLiveTrades]);

  // Filtered Live Ledger
  const filteredLiveTrades = useMemo(() => {
    return eligibleLiveTrades.filter((t) => {
      if (searchTicker.trim() === '') return true;
      const q = searchTicker.toLowerCase();
      const matchTicker = t.ticker ? t.ticker.toLowerCase().includes(q) : false;
      const matchSetup = t.setupName ? t.setupName.toLowerCase().includes(q) : false;
      return matchTicker || matchSetup;
    });
  }, [eligibleLiveTrades, searchTicker]);

  const handleTabKeyDown = (e: React.KeyboardEvent, current: 'OVERVIEW' | 'EDGE' | 'LEDGER') => {
    const tabs: ('OVERVIEW' | 'EDGE' | 'LEDGER')[] = ['OVERVIEW', 'EDGE', 'LEDGER'];
    const idx = tabs.indexOf(current);
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      const next = tabs[(idx + 1) % tabs.length];
      setActiveTab(next);
      document.getElementById(`tab-${next.toLowerCase()}`)?.focus();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = tabs[(idx - 1 + tabs.length) % tabs.length];
      setActiveTab(prev);
      document.getElementById(`tab-${prev.toLowerCase()}`)?.focus();
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActiveTab(tabs[0]);
      document.getElementById(`tab-${tabs[0].toLowerCase()}`)?.focus();
    } else if (e.key === 'End') {
      e.preventDefault();
      setActiveTab(tabs[tabs.length - 1]);
      document.getElementById(`tab-${tabs[tabs.length - 1].toLowerCase()}`)?.focus();
    }
  };

  return (
    <TerminalShell
      activeHub="performance"
      activeSymbol={searchTicker.trim() ? searchTicker.trim().toUpperCase() : null}
    >
      <div className="space-y-6">
        {/* Hub Guidance & Orientation (A3-AC1, A3-AC2, A3-AC8) */}
        <PageIntro
          hubId="performance"
          title="Performance"
          purpose="Review realized trade outcomes, historical return metrics, and execution attribution across closed positions."
          badge="Realized Attribution"
          symbol={searchTicker.trim() ? searchTicker.trim().toUpperCase() : null}
          primaryAction={{
            label: "Review Journal Logs →",
            href: "/journal",
          }}
          secondaryAction={{
            label: "Explore Setups →",
            href: "/setups",
          }}
        />

        {/* Data Provenance Banner */}
        <div className="p-3.5 rounded-xl border border-slate-800 bg-slate-950/90 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400 font-bold">ℹ️ DATA PROVENANCE:</span>
            <span className="text-slate-300">
              Live Trader Account Mode ({eligibleLiveTrades.length} Verified Closed Executions)
            </span>
          </div>
          <div className="text-slate-400 text-[11px]">
            Server-Authoritative API · {liveSummary.coverageNotice}
          </div>
        </div>

        {liveLoading ? (
          <div className="p-12 rounded-2xl border border-slate-800 bg-slate-900/30 text-center space-y-4 max-w-xl mx-auto font-mono animate-pulse">
            <span className="text-3xl">⏳</span>
            <h2 className="text-lg font-bold text-slate-300">Loading Live Execution History…</h2>
            <p className="text-xs text-slate-500">Querying authoritative trade journal API records.</p>
          </div>
        ) : liveError ? (
          <div className="p-12 rounded-2xl border border-rose-900/50 bg-rose-950/20 text-center space-y-4 max-w-xl mx-auto font-mono">
            <span className="text-3xl">⚠️</span>
            <h2 className="text-lg font-bold text-rose-300">Live Execution Data Unavailable</h2>
            <p className="text-xs text-slate-400 leading-relaxed">
              {liveError} Unable to derive live performance attribution.
            </p>
          </div>
        ) : eligibleLiveTrades.length === 0 ? (
          <div className="p-12 rounded-2xl border border-slate-800 bg-slate-900/30 text-center space-y-4 max-w-xl mx-auto font-mono">
            <span className="text-3xl">📝</span>
            <h2 className="text-lg font-bold text-white">
              {liveTrades.length > 0 ? `${liveTrades.length} Trade Records Found — 0 Meet Completed Eligibility` : "0 Completed Executions Logged Yet"}
            </h2>
            <p className="text-xs text-slate-400 leading-relaxed">
              {liveTrades.length > 0
                ? `Your trade journal contains ${liveTrades.length} record(s), but none qualify as completed executions. Completed performance strictly requires explicitly CLOSED status with valid positive entry price, exit price, shares, and a verified realized net P&L. OPEN trade plans and records missing exit data are excluded.`
                : livePositions.length > 0
                ? `You have ${livePositions.length} active holding${livePositions.length > 1 ? "s" : ""} in your portfolio, but 0 completed executions have been closed. Performance activates once positions are closed and realized outcomes are recorded.`
                : `Your live account does not contain completed executions yet. Execute trade plans in /setups or record closed trades in /journal to build your empirical personal edge.`}
            </p>
            <div className="text-[11px] text-slate-500 font-mono">
              {liveSummary.coverageNotice}
            </div>
            <div className="flex items-center justify-center gap-3 pt-2">
              <Link
                href="/setups"
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg text-xs"
              >
                View Trade Setups →
              </Link>
              <Link
                href="/journal"
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-lg text-xs"
              >
                View Trade Journal →
              </Link>
            </div>
          </div>
        ) : (
          <>
            {/* Limited Sample Maturity Guard (A3-AC8, A3-CLOSE-AC3) */}
            {eligibleLiveTrades.length < 30 && (
              <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/80 font-mono text-xs text-amber-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-md">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl shrink-0">⚠️</span>
                  <div>
                    <span className="font-bold text-amber-300">
                      Limited closed-trade sample (N = {eligibleLiveTrades.length} &lt; 30)
                    </span>
                    <p className="text-[11px] text-slate-300 font-sans mt-0.5 leading-relaxed">
                      Realized results are factual historical observations, but the sample is too limited for reliable conclusions about persistent performance.
                    </p>
                  </div>
                </div>
                <span className="text-[10px] px-2.5 py-1 rounded bg-amber-900/60 border border-amber-700 text-amber-200 font-bold shrink-0 self-start sm:self-center">
                  Sample: {eligibleLiveTrades.length}/30 Closed Trades
                </span>
              </div>
            )}

            {/* Proof of Edge Headline Card */}
            <div className="p-6 rounded-2xl bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-800/50 flex flex-col lg:flex-row lg:items-center justify-between gap-6 shadow-xl">
              <div className="space-y-1.5 max-w-2xl">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase font-bold text-emerald-400 bg-emerald-950 px-2.5 py-0.5 rounded border border-emerald-800">
                    Verified Live Execution Metrics
                  </span>
                  <span className="text-xs font-mono text-slate-400">
                    Server-Authoritative API
                  </span>
                </div>
                <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                  Net Realized P&L: {liveSummary.totalRealizedPnL >= 0 ? `+$${liveSummary.totalRealizedPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : `-$${Math.abs(liveSummary.totalRealizedPnL).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                </h1>
                <p className="text-xs sm:text-sm text-slate-300">
                  Derived strictly from {eligibleLiveTrades.length} verified closed journal executions. Counterfactual Capital Preserved is unavailable (requires live Governor clamp logging).
                </p>
              </div>

              {/* Quick Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950/90 border border-slate-800 p-4 rounded-xl font-mono text-center shrink-0">
                <div className="px-2">
                  <span className="text-[10px] text-slate-400 uppercase block">Win Rate</span>
                  <span className="text-base sm:text-lg font-bold text-emerald-400">
                    {liveSummary.winRatePct !== null ? `${liveSummary.winRatePct}%` : '--'}
                  </span>
                  <span className="text-[9px] text-slate-500 block">
                    {liveSummary.wins}W / {liveSummary.losses}L / {liveSummary.scratches}S
                  </span>
                </div>
                <div className="px-2 border-l border-slate-800">
                  <span className="text-[10px] text-slate-400 uppercase block">Profit Factor</span>
                  <span className="text-base sm:text-lg font-bold text-cyan-400">
                    {liveSummary.profitFactor ?? '--'}
                  </span>
                  <span className="text-[9px] text-slate-500 block">Gross W/L Ratio</span>
                </div>
                <div className="px-2 border-l border-slate-800">
                  <span className="text-[10px] text-slate-400 uppercase block">Avg R-Multiple</span>
                  <span className="text-base sm:text-lg font-bold text-emerald-400">
                    {liveSummary.avgR !== null ? `${liveSummary.avgR}R` : '--'}
                  </span>
                  <span className="text-[9px] text-slate-500 block">
                    {liveSummary.eligibleRTradesCount} trades with R
                  </span>
                </div>
                <div className="px-2 border-l border-slate-800">
                  <span className="text-[10px] text-slate-400 uppercase block">Capital Preserved</span>
                  <span className="text-base sm:text-lg font-bold text-slate-400">--</span>
                  <span className="text-[9px] text-slate-500 block">Telemetry inactive</span>
                </div>
              </div>
            </div>

            {/* Navigation Tabs (WAI-ARIA Tablist) */}
            <div role="tablist" aria-label="Performance Analysis Views" className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <button
                type="button"
                role="tab"
                id="tab-overview"
                aria-selected={activeTab === 'OVERVIEW'}
                aria-controls="panel-overview"
                tabIndex={activeTab === 'OVERVIEW' ? 0 : -1}
                onKeyDown={(e) => handleTabKeyDown(e, 'OVERVIEW')}
                onClick={() => setActiveTab('OVERVIEW')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all focus-ring ${
                  activeTab === 'OVERVIEW'
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                📊 Attribution Overview
              </button>
              <button
                type="button"
                role="tab"
                id="tab-edge"
                aria-selected={activeTab === 'EDGE'}
                aria-controls="panel-edge"
                tabIndex={activeTab === 'EDGE' ? 0 : -1}
                onKeyDown={(e) => handleTabKeyDown(e, 'EDGE')}
                onClick={() => setActiveTab('EDGE')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all focus-ring ${
                  activeTab === 'EDGE'
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                🎯 Personal Edge Discovery
              </button>
              <button
                type="button"
                role="tab"
                id="tab-ledger"
                aria-selected={activeTab === 'LEDGER'}
                aria-controls="panel-ledger"
                tabIndex={activeTab === 'LEDGER' ? 0 : -1}
                onKeyDown={(e) => handleTabKeyDown(e, 'LEDGER')}
                onClick={() => setActiveTab('LEDGER')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all flex items-center gap-1.5 focus-ring ${
                  activeTab === 'LEDGER'
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                <span>📜 Realized Trades Ledger</span>
                <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-slate-300 font-bold">
                  {eligibleLiveTrades.length}
                </span>
              </button>
            </div>

            {/* Tab 1: Attribution Overview */}
            {activeTab === 'OVERVIEW' && (
              <div role="tabpanel" id="panel-overview" aria-labelledby="tab-overview" tabIndex={0} className="space-y-6 focus:outline-none">
                {/* Capital Preserved Breakdown Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-slate-400 uppercase">Drawdown Defense</span>
                      <span className="text-slate-500 font-bold">Telemetry Inactive</span>
                    </div>
                    <div className="text-2xl font-black font-mono text-white">--</div>
                    <p className="text-xs text-slate-400">
                      Losing streak risk dampeners require runtime Governor telemetry logging during execution.
                    </p>
                  </div>

                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-slate-400 uppercase">Execution Window</span>
                      <span className="text-slate-500 font-bold">Telemetry Inactive</span>
                    </div>
                    <div className="text-2xl font-black font-mono text-white">--</div>
                    <p className="text-xs text-slate-400">
                      Intraday execution window tracking requires telemetric session timestamp logging.
                    </p>
                  </div>

                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-slate-400 uppercase">Capital Floor</span>
                      <span className="text-slate-500 font-bold">Telemetry Inactive</span>
                    </div>
                    <div className="text-2xl font-black font-mono text-white">--</div>
                    <p className="text-xs text-slate-400">
                      Cash runway floor protection activates once liquid burn rate telemetry is connected.
                    </p>
                  </div>
                </div>

                {/* Realized Cumulative Equity Curve */}
                <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h3 className="text-base font-bold text-white">Realized Cumulative Equity Curve</h3>
                      <p className="text-xs text-slate-400">
                        Cumulative net profit/loss across verified closed trades (ordered strictly by closing timestamp)
                      </p>
                    </div>
                    <div className="flex items-center gap-4 text-xs font-mono">
                      <div className="flex items-center gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                        <span className="text-white font-bold">
                          Net Realized: {liveSummary.totalRealizedPnL >= 0 ? `+$${liveSummary.totalRealizedPnL.toLocaleString()}` : `-$${Math.abs(liveSummary.totalRealizedPnL).toLocaleString()}`}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Trajectory Milestone Cards or Unavailable Notice */}
                  {trajectoryResult.isAvailable ? (
                    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2 text-xs font-mono">
                      {trajectoryResult.points
                        .filter((_, i) => trajectoryResult.points.length <= 6 || i % Math.ceil(trajectoryResult.points.length / 6) === 0 || i === trajectoryResult.points.length - 1)
                        .map((pt) => (
                          <div key={pt.id} className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                            <div className="text-slate-500">#{pt.tradeIndex} ({pt.ticker ?? 'Unspecified'})</div>
                            <div className={`font-bold ${pt.cumulativePnL >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {pt.cumulativePnL >= 0 ? `+$${pt.cumulativePnL.toLocaleString()}` : `-$${Math.abs(pt.cumulativePnL).toLocaleString()}`}
                            </div>
                            <div className="text-[10px] text-slate-400">
                              Trade: {pt.realizedPnL >= 0 ? `+$${pt.realizedPnL.toLocaleString()}` : `-$${Math.abs(pt.realizedPnL).toLocaleString()}`}
                            </div>
                            <div className="text-[10px] text-slate-500">{pt.exitDate}</div>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <div className="p-6 rounded-xl border border-slate-800 bg-slate-950/60 text-center space-y-2 font-mono">
                      <div className="text-amber-400 font-bold text-xs">⚠️ Cumulative Trajectory Unavailable</div>
                      <p className="text-xs text-slate-400 max-w-xl mx-auto">
                        {trajectoryResult.unavailableReason}
                      </p>
                    </div>
                  )}

                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-400">Net Alpha Spread vs. Unclamped Naive Execution:</span>
                    <span className="text-slate-400 font-bold text-xs">
                      Unavailable — Requires real-time unclamped Governor telemetry logging
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 2: Personal Edge Discovery */}
            {activeTab === 'EDGE' && (
              <div role="tabpanel" id="panel-edge" aria-labelledby="tab-edge" tabIndex={0} className="space-y-6 focus:outline-none">
                {/* Live Setup Performance (Grouped strictly by recorded setupName) */}
                <div className="space-y-3">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400">
                    Setup Performance by Recorded Pattern ({setupGroups.length} Setup Types)
                  </h3>
                  {setupGroups.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 font-mono text-xs">
                      {setupGroups.map((setup) => (
                        <div
                          key={setup.name}
                          className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 space-y-3"
                        >
                          <div className="flex items-center justify-between">
                            <h4 className="text-sm font-bold text-white">{setup.name}</h4>
                            <span className="text-slate-400 text-[11px]">{setup.count} Trades</span>
                          </div>
                          <div className="pt-2 border-t border-slate-800/80 space-y-1.5">
                            <div className="flex justify-between">
                              <span className="text-slate-400">Win Rate:</span>
                              <span className="text-white font-bold">{setup.winRatePct}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-400">Net Realized:</span>
                              <span className={`font-bold ${setup.totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {setup.totalPnL >= 0 ? `+$${setup.totalPnL.toLocaleString()}` : `-$${Math.abs(setup.totalPnL).toLocaleString()}`}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-6 rounded-xl border border-slate-800 bg-slate-950/60 text-center font-mono text-xs text-slate-400">
                      No setup patterns recorded in completed live executions.
                    </div>
                  )}
                </div>

                {/* Window & Regime Disclaimers */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-2 font-mono text-xs">
                    <h3 className="text-slate-400 uppercase tracking-wider">Performance by Execution Window</h3>
                    <p className="text-slate-400">
                      Unavailable — Intraday execution window telemetry (e.g. Morning Prime, Midday Chop) was not logged during order execution.
                    </p>
                  </div>
                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-2 font-mono text-xs">
                    <h3 className="text-slate-400 uppercase tracking-wider">Performance by Market Regime</h3>
                    <p className="text-slate-400">
                      Unavailable — Macro regime classification (e.g. Uptrend, Distribution, Chop) was not captured at execution time.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: Realized Trades Ledger */}
            {activeTab === 'LEDGER' && (
              <div role="tabpanel" id="panel-ledger" aria-labelledby="tab-ledger" tabIndex={0} className="space-y-4 focus:outline-none">
                {/* Filter Toolbar */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div className="text-xs font-mono text-slate-400">
                    Showing {filteredLiveTrades.length} of {eligibleLiveTrades.length} verified closed executions
                  </div>

                  <div className="relative w-full sm:w-56">
                    <input
                      type="text"
                      value={searchTicker}
                      onChange={(e) => setSearchTicker(e.target.value)}
                      placeholder="Filter ticker or setup..."
                      aria-label="Filter realized trades by ticker or setup pattern"
                      className="w-full px-3 py-1.5 bg-[#0b1019] border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 font-mono focus:outline-none focus:border-cyan-500 focus-ring"
                    />
                  </div>
                </div>

                {/* Ledger Table */}
                <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
                  <table className="w-full text-left font-mono text-xs">
                    <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 text-[11px] uppercase">
                      <tr>
                        <th className="p-3">ID / Exit Date</th>
                        <th className="p-3">Ticker</th>
                        <th className="p-3">Setup</th>
                        <th className="p-3">Entry Price</th>
                        <th className="p-3">Exit Price</th>
                        <th className="p-3">Shares</th>
                        <th className="p-3">Outcome</th>
                        <th className="p-3">R-Multiple</th>
                        <th className="p-3 text-right">Realized P&L</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {filteredLiveTrades.map((entry) => (
                        <tr key={entry.id} className="hover:bg-slate-900/70 transition-colors">
                          <td className="p-3 text-slate-400">
                            <div>#{entry.id}</div>
                            <div className="text-[10px] text-slate-500">{entry.exitDate ?? entry.entryDate ?? 'Date unrecorded'}</div>
                          </td>
                          <td className="p-3 font-bold text-white">{entry.ticker ?? '--'}</td>
                          <td className="p-3 text-slate-300 text-[11px]">{entry.setupName ?? 'Unspecified'}</td>
                          <td className="p-3 text-slate-300">${entry.entryPrice.toFixed(2)}</td>
                          <td className="p-3 text-slate-300">${entry.exitPrice.toFixed(2)}</td>
                          <td className="p-3 text-slate-400">{entry.shares}</td>
                          <td className="p-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              entry.outcome === 'WIN'
                                ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                                : entry.outcome === 'LOSS'
                                ? 'bg-rose-950 text-rose-400 border border-rose-800'
                                : 'bg-slate-800 text-slate-300 border border-slate-700'
                            }`}>
                              {entry.outcome}
                            </span>
                          </td>
                          <td className="p-3 text-slate-300">
                            {entry.rAchieved !== null ? `${entry.rAchieved >= 0 ? `+${entry.rAchieved.toFixed(2)}` : entry.rAchieved.toFixed(2)}R` : '--'}
                          </td>
                          <td className={`p-3 text-right font-bold ${entry.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {entry.pnl >= 0 ? `+$${entry.pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : `-$${Math.abs(entry.pnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </TerminalShell>
  );
}
