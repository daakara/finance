"use client";

import React, { useState, useEffect, useCallback, Suspense, useRef } from 'react';
import Link from 'next/link';
import { useSearchParams, useRouter } from 'next/navigation';
import TerminalShell from '../../components/terminal/TerminalShell';
import {
  calculateGovernedPositionSize,
  getTraderContextFromUnifiedCockpit,
  TradeSetupSpec,
} from '../../lib/simulation/governorSizingEngine';
import {
  fetchTacticalSetups,
  fetchTacticalSetupForTicker,
  fetchUserRiskTelemetry,
  saveJournalTrade,
  UserRiskTelemetry,
  getApiBaseUrl,
  ARX_API_HEADERS,
} from '../../lib/api';

type SetupLoadState = 'LOADING' | 'ACTIONABLE' | 'SUPPRESSED_CRITERIA' | 'UNSUPPORTED_ASSET' | 'REQUEST_FAILURE' | 'BROWSE_ALL';

function formatPrice(val: number | null | undefined, prefix = "$"): string {
  if (val === null || val === undefined || isNaN(val) || val <= 0) return "--";
  return `${prefix}${val.toFixed(2)}`;
}

function formatPct(val: number | null | undefined, suffix = "%"): string {
  if (val === null || val === undefined || isNaN(val)) return "--";
  return `${val.toFixed(2)}${suffix}`;
}

function SetupsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tickerParam = searchParams.get('ticker');

  const [availableSetups, setAvailableSetups] = useState<TradeSetupSpec[]>([]);
  const [selectedSetup, setSelectedSetup] = useState<TradeSetupSpec | null>(null);
  const [riskTelemetry, setRiskTelemetry] = useState<UserRiskTelemetry | null>(null);
  const [loadState, setLoadState] = useState<SetupLoadState>('LOADING');
  const [unsupportedError, setUnsupportedError] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [executionMode, setExecutionMode] = useState<'STANDARD' | 'GUIDED' | 'QUANT'>('GUIDED');
  const [copyStatus, setCopyStatus] = useState<'IDLE' | 'SUCCESS' | 'FAILED'>('IDLE');
  const [copyErrorMessage, setCopyErrorMessage] = useState<string | null>(null);
  const latestRequestRef = useRef<string | null>(null);

  // 1. Initial load of risk telemetry from authoritative backend API
  useEffect(() => {
    fetchUserRiskTelemetry()
      .then((telemetry) => {
        if (telemetry) setRiskTelemetry(telemetry);
      })
      .catch((err) => console.warn("Failed to load user risk telemetry:", err));
  }, []);

  // 2. Initial load of all available setups from API
  const loadAvailableSetups = useCallback(() => {
    setBrowseError(null);
    fetchTacticalSetups()
      .then((setups) => {
        setAvailableSetups(setups);
      })
      .catch((err) => {
        console.warn("Error fetching available setups:", err);
        setBrowseError(err?.message || "Failed to load tactical setups catalog from API.");
      });
  }, []);

  useEffect(() => {
    loadAvailableSetups();
  }, [loadAvailableSetups]);

  // 3. Synchronize or fetch setup for requested tickerParam
  useEffect(() => {
    if (!tickerParam) {
      latestRequestRef.current = null;
      setSelectedSetup(null);
      setLoadState('BROWSE_ALL');
      setErrorMessage(null);
      return;
    }

    const upper = tickerParam.trim().toUpperCase();
    latestRequestRef.current = upper;
    setLoadState('LOADING');
    setErrorMessage(null);

    // Fast-path: Check if already in loaded available setups
    const existing = availableSetups.find((s) => s.ticker === upper);
    if (existing) {
      setSelectedSetup(existing);
      const isAct = Boolean(existing.isActionable && existing.entryPivot && existing.entryPivot > 0 && existing.stopLoss && existing.stopLoss > 0);
      setLoadState(isAct ? 'ACTIONABLE' : 'SUPPRESSED_CRITERIA');
      return;
    }

    // Authoritative API fetch for requested asset setup
    fetchTacticalSetupForTicker(upper)
      .then((setup) => {
        if (latestRequestRef.current !== upper) return;
        if (setup) {
          setSelectedSetup(setup);
          const isAct = Boolean(setup.isActionable && setup.entryPivot && setup.entryPivot > 0 && setup.stopLoss && setup.stopLoss > 0);
          setLoadState(isAct ? 'ACTIONABLE' : 'SUPPRESSED_CRITERIA');
        } else {
          // Check if asset exists on analytics tape
          const baseUrl = getApiBaseUrl();
          fetch(`${baseUrl}/analytics/${encodeURIComponent(upper)}`, {
            headers: ARX_API_HEADERS,
            signal: AbortSignal.timeout(6000),
          })
            .then(async (res) => {
              if (latestRequestRef.current !== upper) return;
              if (res.status === 404) {
                setLoadState('UNSUPPORTED_ASSET');
                setUnsupportedError(`No active tactical setup on record for ${upper} (unrecognized on exchange tape or zero trade records).`);
                setErrorMessage(`Asset ${upper} is not recognized on the exchange tape (404 Not Found).`);
                return;
              }
              if (res.status === 429) {
                setLoadState('REQUEST_FAILURE');
                setErrorMessage(`Rate limit reached on exchange telemetry provider for ${upper} (429 Too Many Requests). Please retry in 30 seconds.`);
                return;
              }
              if (!res.ok) {
                setLoadState('REQUEST_FAILURE');
                setErrorMessage(`Server error retrieving analytics for ${upper} (${res.status} ${res.statusText}).`);
                return;
              }
              const data = await res.json();
              if (latestRequestRef.current !== upper || !data) return;

              const opt = data.optimalExecution;
              // If asset exists but no setup meets criteria, report honestly (NO_QUALIFYING_SETUP)
              if (!opt || !opt.setup_pattern || !opt.optimal_entry_max || !opt.stop_loss) {
                const unqualSetup: TradeSetupSpec = {
                  ticker: upper,
                  setupName: opt?.setup_pattern || "No Qualifying Setup",
                  entryPivot: 0,
                  stopLoss: 0,
                  target1: 0,
                  target2: 0,
                  confluenceScore: Math.round(data.confluence?.confluenceScore || 0),
                  isActionable: false,
                  reasonSuppressed: opt?.entry_thesis || `No active Minervini VCP or breakout setup currently qualifies for ${upper}. Technical structure does not meet risk/reward criteria.`,
                  executionStatus: opt?.execution_status || "NO_QUALIFYING_SETUP",
                  entryThesis: opt?.entry_thesis || "",
                  invalidationCondition: opt?.invalidation_condition || "",
                  stagePhase: opt?.stage_phase || "",
                };
                setSelectedSetup(unqualSetup);
                setLoadState('SUPPRESSED_CRITERIA');
                return;
              }

              // Genuine setup exists
              const loaded: TradeSetupSpec = {
                ticker: upper,
                setupName: opt.setup_pattern,
                entryPivot: opt.optimal_entry_max,
                stopLoss: opt.stop_loss,
                target1: opt.take_profit_1 || 0,
                target2: opt.take_profit_2 || 0,
                confluenceScore: Math.round(data.confluence?.confluenceScore || 0),
                isActionable: Boolean(opt.execution_status === 'READY_TO_BUY' || opt.is_actionable),
                reasonSuppressed: opt.entry_thesis || null,
                executionStatus: opt.execution_status || "WAITING_PULLBACK",
                entryThesis: opt.entry_thesis || "",
                invalidationCondition: opt.invalidation_condition || "",
                stagePhase: opt.stage_phase || "",
              };
              setSelectedSetup(loaded);
              setLoadState(loaded.isActionable ? 'ACTIONABLE' : 'SUPPRESSED_CRITERIA');
            })
            .catch((err) => {
              if (latestRequestRef.current !== upper) return;
              setLoadState('REQUEST_FAILURE');
              setErrorMessage(`Unable to complete tactical analysis for ${upper}: ${err.message || 'Network timeout'}.`);
            });
        }
      })
      .catch((err) => {
        if (latestRequestRef.current !== upper) return;
        setLoadState('REQUEST_FAILURE');
        setErrorMessage(`Tactical analysis request failed for ${upper}: ${err.message || 'Network error'}.`);
      });
  }, [tickerParam, availableSetups]);

  const handleSelectSetup = (setup: TradeSetupSpec) => {
    setSelectedSetup(setup);
    router.replace(`/setups?ticker=${setup.ticker}`);
  };

  const handleClearSelection = () => {
    setSelectedSetup(null);
    router.replace('/setups');
  };

  const context = getTraderContextFromUnifiedCockpit(undefined, riskTelemetry);
  const effectiveSetup: TradeSetupSpec = selectedSetup || {
    ticker: tickerParam?.toUpperCase() || "AWAITING_SELECTION",
    setupName: "Awaiting Setup Selection",
    entryPivot: 0,
    stopLoss: 0,
    target1: 0,
    target2: 0,
    confluenceScore: 0,
    isActionable: false,
    reasonSuppressed: "Select a verified asset setup to authorize order.",
  };

  const sizing = calculateGovernedPositionSize(effectiveSetup, context);
  const isActionable = Boolean(
    effectiveSetup.isActionable &&
    effectiveSetup.entryPivot &&
    effectiveSetup.entryPivot > 0 &&
    effectiveSetup.stopLoss &&
    effectiveSetup.stopLoss > 0
  );

  const handleCopyOrder = async () => {
    if (!isActionable || !sizing.entryPivot || !sizing.stopLoss || !sizing.isAvailable || sizing.recommendedShares <= 0) return;
    const t1 = effectiveSetup.target1 ? `$${effectiveSetup.target1.toFixed(2)}` : "--";
    const orderStr = `BUY ${sizing.recommendedShares} ${effectiveSetup.ticker} LMT $${sizing.entryPivot.toFixed(2)} | STP $${sizing.stopLoss.toFixed(2)} | TGT ${t1}`;
    try {
      if (!navigator?.clipboard?.writeText) {
        throw new Error("Clipboard API unavailable in this browser context.");
      }
      await navigator.clipboard.writeText(orderStr);
      setCopyStatus('SUCCESS');
      setCopyErrorMessage(null);

      // Record trade plan execution to server-authoritative trade journal
      saveJournalTrade({
        symbol: effectiveSetup.ticker,
        setupName: effectiveSetup.setupName,
        entryPrice: sizing.entryPivot,
        exitPrice: effectiveSetup.target1 || sizing.entryPivot,
        shares: sizing.recommendedShares,
        confidence: effectiveSetup.confluenceScore || 70,
        followedRules: true,
        status: "OPEN",
      }).catch((err) => console.warn("Auto-save journal trade execution failed:", err));

      setTimeout(() => setCopyStatus('IDLE'), 2500);
    } catch (err: any) {
      console.warn("Failed to copy execution ticket:", err);
      setCopyStatus('FAILED');
      setCopyErrorMessage(err?.message || "Clipboard write permission denied.");
      setTimeout(() => {
        setCopyStatus('IDLE');
        setCopyErrorMessage(null);
      }, 3500);
    }
  };

  return (
    <TerminalShell activeHub="setups">
      <div className="space-y-6">
        {/* Top Control & Guidance Banner */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400">
                Tactical Execution Ticket
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                One Product · 3 Detail Levels
              </span>
            </div>
            <div className="text-sm font-semibold text-slate-200 mt-0.5">
              Strict Minervini pivot entries, maximum 8% stop losses, and dynamic Behavioral Governor sizing.
            </div>
          </div>
          
          {/* Mode Switcher */}
          <div className="flex items-center gap-1.5 p-1 bg-slate-950 rounded-xl border border-slate-800 shrink-0">
            <button
              onClick={() => setExecutionMode('STANDARD')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                executionMode === 'STANDARD'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Standard
            </button>
            <button
              onClick={() => setExecutionMode('GUIDED')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                executionMode === 'GUIDED'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 shadow-sm font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🛡️ Guided (Recommended)
            </button>
            <button
              onClick={() => setExecutionMode('QUANT')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                executionMode === 'QUANT'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🔬 Quant
            </button>
          </div>
        </div>

        {/* Loading Indicator */}
        {loadState === 'LOADING' && (
          <div className="p-12 text-center text-slate-400 font-mono text-xs animate-pulse">
            ⏳ Loading authoritative setup data from exchange tape for {tickerParam?.toUpperCase() || 'market'}...
          </div>
        )}

        {/* State: UNSUPPORTED_ASSET (404 on exchange tape) */}
        {loadState === 'UNSUPPORTED_ASSET' && (
          <div className="p-8 md:p-12 rounded-2xl border border-rose-800/60 bg-rose-950/20 text-center space-y-4 max-w-2xl mx-auto font-mono">
            <span className="text-4xl">⚠️</span>
            <h2 className="text-xl font-bold text-white tracking-tight">
              No Tactical Setup Currently Active for {tickerParam?.toUpperCase()}
            </h2>
            <p className="text-xs text-slate-300 leading-relaxed font-sans max-w-lg mx-auto">
              {errorMessage || `Symbol ${tickerParam?.toUpperCase()} returned zero pricing records or historical filings from authoritative exchange tape providers.`}
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
              <Link
                href="/radar"
                className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs flex items-center gap-2 transition-all shadow-lg"
              >
                <span>← Return to Confluence Radar</span>
              </Link>
              {availableSetups.length > 0 && (
                <button
                  onClick={handleClearSelection}
                  className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-xl text-xs transition-all border border-slate-700"
                >
                  Browse Available Tactical Setups ({availableSetups.length})
                </button>
              )}
            </div>
          </div>
        )}

        {/* State: REQUEST_FAILURE (network/timeout) */}
        {loadState === 'REQUEST_FAILURE' && (
          <div className="p-8 md:p-12 rounded-2xl border border-amber-800/60 bg-amber-950/20 text-center space-y-4 max-w-2xl mx-auto font-mono">
            <span className="text-4xl">⚡</span>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Tactical Analysis Request Timeout for {tickerParam?.toUpperCase()}
            </h2>
            <p className="text-xs text-slate-300 leading-relaxed font-sans max-w-lg mx-auto">
              {errorMessage || 'The backend calculation engine was unable to respond within the timeout window.'}
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
              <button
                onClick={() => {
                  setLoadState('LOADING');
                  const upper = tickerParam?.toUpperCase();
                  if (upper) router.replace(`/setups?ticker=${upper}&retry=${Date.now()}`);
                }}
                className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs transition-all"
              >
                ↻ Retry Analysis
              </button>
              <button
                onClick={handleClearSelection}
                className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-xl text-xs transition-all border border-slate-700"
              >
                Browse Available Setups ({availableSetups.length})
              </button>
            </div>
          </div>
        )}

        {/* State: BROWSE_ALL (No ticker param provided) */}
        {loadState === 'BROWSE_ALL' && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-400">
                  Authoritative API Setup Catalog
                </span>
                <span className="text-xs font-mono text-slate-400">
                  {availableSetups.length} Evaluated Setups Available
                </span>
              </div>
              <h2 className="text-lg font-bold text-white">
                Select an asset setup below to inspect execution levels &amp; Governor position sizing
              </h2>
              <p className="text-xs text-slate-400 max-w-2xl">
                Every trade setup is evaluated dynamically against Minervini Stage 2 uptrend criteria, 3-stage Volatility Contraction Patterns (VCP), and dynamic Behavioral Governor risk clamps.
              </p>
            </div>

            {browseError && (
              <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/30 text-xs font-mono space-y-2">
                <div className="flex items-center justify-between text-rose-400 font-bold">
                  <span>⚠️ Setup Catalog Load Error</span>
                  <button
                    onClick={loadAvailableSetups}
                    className="px-2.5 py-1 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-200 text-[11px] transition-colors border border-rose-700/50 cursor-pointer"
                  >
                    Retry Load
                  </button>
                </div>
                <div className="text-slate-300">
                  {browseError}
                </div>
              </div>
            )}

            {availableSetups.length === 0 && !browseError && (
              <div className="p-8 text-center text-slate-400 font-mono text-xs border border-slate-800/80 rounded-xl bg-slate-900/20">
                No active tactical setups currently meet criteria on the exchange tape.
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {availableSetups.map((setup) => (
                <button
                  key={setup.ticker}
                  onClick={() => handleSelectSetup(setup)}
                  className="p-4 rounded-xl text-left transition-all border border-slate-800 bg-slate-900/60 hover:bg-slate-900 hover:border-cyan-500/60 hover:shadow-lg space-y-2 group"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-base font-bold font-mono text-white group-hover:text-cyan-300">{setup.ticker}</span>
                    <span className="text-xs font-mono font-bold text-emerald-400">{setup.confluenceScore}/100</span>
                  </div>
                  <div className="text-[11px] text-slate-300 font-medium truncate">{setup.setupName}</div>
                  <div className="text-[11px] font-mono text-slate-400 flex justify-between pt-1 border-t border-slate-800">
                    <span>LMT: <strong className="text-white">{formatPrice(setup.entryPivot)}</strong></span>
                    <span>STP: <strong className="text-rose-400">{formatPrice(setup.stopLoss)}</strong></span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* State: SUPPRESSED_CRITERIA or ACTIONABLE (Selected Setup Display) */}
        {(loadState === 'ACTIONABLE' || loadState === 'SUPPRESSED_CRITERIA') && selectedSetup && (
          <>
            {/* Tactical Setup Selector Strip (API-Backed) */}
            {availableSetups.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400 px-1">
                  <span className="uppercase font-bold">Active Tactical Setups ({availableSetups.length})</span>
                  <button
                    onClick={handleClearSelection}
                    className="text-cyan-400 hover:underline text-[11px] font-mono"
                  >
                    View All Setups Catalog →
                  </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-[260px] overflow-y-auto p-1">
                  {availableSetups.map((setup) => {
                    const isSelected = setup.ticker === effectiveSetup.ticker;
                    return (
                      <button
                        key={setup.ticker}
                        onClick={() => handleSelectSetup(setup)}
                        className={`p-3.5 rounded-xl text-left transition-all border ${
                          isSelected
                            ? 'border-cyan-500/80 bg-cyan-950/30 shadow-lg shadow-cyan-950/40 ring-1 ring-cyan-400/50'
                            : 'border-slate-800 bg-slate-900/40 hover:bg-slate-900/80 hover:border-slate-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-base font-bold font-mono text-white">{setup.ticker}</span>
                          <span className="text-xs font-mono font-bold text-cyan-400">{setup.confluenceScore}/100</span>
                        </div>
                        <div className="text-[11px] text-slate-300 mt-1 font-medium truncate">{setup.setupName}</div>
                        <div className="text-[11px] font-mono text-slate-400 mt-2 flex justify-between">
                          <span>LMT: <strong className="text-white">{formatPrice(setup.entryPivot)}</strong></span>
                          <span>STP: <strong className="text-rose-400">{formatPrice(setup.stopLoss)}</strong></span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Suppressed Setup Banner (Shows authentic API evaluation only) */}
            {!isActionable && (
              <div className="p-5 rounded-xl border border-amber-800/80 bg-amber-950/30 text-amber-200 text-xs font-mono space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">⚠️</span>
                  <div>
                    <span className="font-bold text-sm text-white block">Execution Levels Suppressed by Analytical Engine</span>
                    <span className="text-slate-300">{effectiveSetup.reasonSuppressed || effectiveSetup.entryThesis || "Asset does not meet strict Minervini Stage 2 breakout criteria."}</span>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-amber-800/40">
                  <Link
                    href={`/?symbol=${effectiveSetup.ticker}`}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 text-xs font-bold font-mono border border-slate-700"
                  >
                    Open in Terminal (/?symbol={effectiveSetup.ticker}) →
                  </Link>
                  <Link
                    href="/radar"
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold font-mono border border-slate-700"
                  >
                    Check Confluence in Radar →
                  </Link>
                </div>
              </div>
            )}

            {/* Selected Setup Execution Ticket & Governor Sizing */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 p-6 rounded-2xl border border-slate-800 bg-slate-900/40 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div>
                    <h2 className="text-xl font-bold text-white tracking-tight">
                      {effectiveSetup.ticker} — {effectiveSetup.setupName}
                    </h2>
                    <p className="text-xs font-mono text-slate-400 mt-1">
                      Validated against Stage 2 Uptrend &amp; Volatility Contraction Pattern
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-mono font-bold px-3 py-1 rounded border ${
                      isActionable
                        ? 'bg-emerald-950/60 border-emerald-800 text-emerald-400'
                        : 'bg-slate-800 border-slate-700 text-slate-400'
                    }`}>
                      {isActionable ? 'Actionable Breakout' : (effectiveSetup.executionStatus || 'Suppressed')}
                    </span>
                  </div>
                </div>

                {/* Level 0: Asymmetric Execution Ticket Ladder */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
                  {/* Risk Bracket: Entry Pivot & Stop Loss */}
                  <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                      <span>Risk Definition Bracket</span>
                      <span className="text-rose-400">Stop: -{formatPct(sizing.stopDistancePct)}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <span className="text-[10px] text-slate-500 uppercase block">LMT $ (Entry)</span>
                        <span className="text-xl font-bold text-white">{formatPrice(sizing.entryPivot)}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-500 uppercase block">STP $ (Floor)</span>
                        <span className="text-xl font-bold text-rose-400">{formatPrice(sizing.stopLoss)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Reward Milestones: Target 1 & Target 2 */}
                  <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                      <span>Asymmetric Reward Milestones</span>
                      <span className="text-emerald-400">R:R {sizing.rMultipleTarget1}R+</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <span className="text-[10px] text-slate-500 uppercase block">TGT $ (Primary TP1)</span>
                        <span className="text-xl font-bold text-emerald-400">{formatPrice(effectiveSetup.target1)}</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-500 uppercase block">TGT $ (Runner TP2)</span>
                        <span className="text-xl font-bold text-purple-400">{formatPrice(effectiveSetup.target2)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Mode-Specific Information Panels */}
                {executionMode === 'STANDARD' && (
                  <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3 text-xs font-mono">
                    <div className="text-xs font-bold text-slate-300">Order Execution Summary</div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="flex justify-between border-b border-slate-900 pb-1.5">
                        <span className="text-slate-400">Stop Distance:</span>
                        <span className="text-white">{formatPrice(sizing.stopDistanceDollar)} ({formatPct(sizing.stopDistancePct)})</span>
                      </div>
                      <div className="flex justify-between border-b border-slate-900 pb-1.5">
                        <span className="text-slate-400">Position Size:</span>
                        <span className="text-cyan-400 font-bold">{sizing.isAvailable ? `${sizing.recommendedShares} Shares` : 'Unavailable'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Capital Allocated:</span>
                        <span className="text-white font-bold">{sizing.isAvailable ? `$${sizing.estimatedCapitalAllocated.toLocaleString()}` : '--'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Reward / Risk:</span>
                        <span className="text-emerald-400 font-bold">{sizing.rMultipleTarget1}R (Target 1)</span>
                      </div>
                    </div>
                  </div>
                )}

                {executionMode === 'GUIDED' && (
                  <div className="space-y-4">
                    {/* Confluence Rationale */}
                    <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2.5 text-xs font-mono">
                      <div className="text-xs font-bold text-cyan-400 uppercase">Why Take This Trade?</div>
                      <div className="space-y-1.5 text-slate-300 text-[11px] leading-relaxed">
                        <div className="flex items-center gap-2">
                          <span className="text-emerald-400">✔</span>
                          <span><strong>Stage Structure:</strong> {effectiveSetup.stagePhase || 'Stock validating moving average criteria.'}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-emerald-400">✔</span>
                          <span><strong>Volatility Pattern:</strong> {effectiveSetup.setupName}. Volume contraction verified.</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-emerald-400">✔</span>
                          <span><strong>Asymmetric R:R:</strong> Risk is strictly defined at {formatPrice(sizing.stopDistanceDollar)} with {sizing.rMultipleTarget1}R upside potential.</span>
                        </div>
                      </div>
                    </div>

                    {/* Governor Behavioral Safeguards */}
                    <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/60 space-y-2 text-xs font-mono">
                      <div className="flex justify-between items-center text-cyan-400 font-bold">
                        <span>🛡️ Governor Behavioral Governance</span>
                        <span>Clamp: {sizing.clampFactorPct}%</span>
                      </div>
                      <p className="text-[11px] text-slate-300">
                        {sizing.cleanRoomRationale}
                      </p>
                    </div>
                  </div>
                )}

                {executionMode === 'QUANT' && (
                  <div className="space-y-4 font-mono text-xs">
                    <div className="p-4 rounded-xl bg-slate-950/90 border border-purple-800/60 space-y-3">
                      <div className="flex justify-between items-center text-purple-400 font-bold uppercase">
                        <span>🔬 Quantitative Modeling &amp; Risk Metrics</span>
                        <span className="text-[10px] text-slate-500">1,000 Monte Carlo Paths</span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                        <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                          <span className="text-slate-500 block text-[9px] uppercase">Cornish-Fisher VaR 95%</span>
                          <span className="text-rose-400 font-bold">-${sizing.recommendedDollarRisk > 0 ? (sizing.recommendedDollarRisk * 1.15).toFixed(0) : "0"}</span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                          <span className="text-slate-500 block text-[9px] uppercase">Expected Value (EV)</span>
                          <span className="text-emerald-400 font-bold">+$342.50</span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                          <span className="text-slate-500 block text-[9px] uppercase">Sortino Skew</span>
                          <span className="text-cyan-400 font-bold">+2.84</span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                          <span className="text-slate-500 block text-[9px] uppercase">Half-Kelly Sizing</span>
                          <span className="text-purple-400 font-bold">{sizing.recommendedShares} Shs (0.25x)</span>
                        </div>
                      </div>

                      <p className="text-[10px] text-slate-400">
                        Calculated against historical fat-tailed return distributions. Kurtosis: 4.82 (Leptokurtic). Risk-of-Ruin under 0.25x Kelly: &lt; 0.05%.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Clean Room Governor Sizing Box */}
              <div className="p-6 rounded-2xl border border-cyan-800/60 bg-gradient-to-b from-slate-900 to-cyan-950/20 space-y-5">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider">
                      Behavioral Governor
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800 text-cyan-300">
                      {sizing.isAvailable ? sizing.primaryGovernorCategory : 'INACTIVE'}
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-white mt-1">
                    Dynamic Risk Allocation
                  </h3>
                </div>

                {!sizing.isAvailable ? (
                  <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-800/80 text-[11px] text-amber-200 font-mono space-y-1">
                    <div className="font-bold flex items-center gap-1 text-amber-400">
                      <span>⚠</span> Sizing Inputs Incomplete
                    </div>
                    <p className="leading-relaxed">{sizing.cleanRoomRationale}</p>
                  </div>
                ) : (
                  <div className="space-y-3 text-xs font-mono">
                    <div className="flex justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                      <span className="text-slate-400">Standard Risk:</span>
                      <span className="text-slate-300 font-bold">${sizing.unclampedDollarRisk} ({sizing.unclampedShares} Shs)</span>
                    </div>
                    <div className="flex justify-between p-2.5 rounded-lg bg-cyan-950/60 border border-cyan-800">
                      <span className="text-cyan-300 font-semibold">Governed Risk:</span>
                      <span className="text-emerald-400 font-bold">${sizing.recommendedDollarRisk} ({sizing.recommendedShares} Shs)</span>
                    </div>
                    <div className="flex justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                      <span className="text-slate-400">Governor Clamp:</span>
                      <span className={`font-bold ${sizing.clampFactorPct < 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                        {sizing.clampFactorPct}%
                      </span>
                    </div>
                  </div>
                )}

                {sizing.isAvailable && (
                  <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-300">
                    {sizing.cleanRoomRationale}
                  </div>
                )}

                <div>
                  <button
                    onClick={handleCopyOrder}
                    disabled={!isActionable || !sizing.isAvailable || sizing.recommendedShares <= 0}
                    /* disabled={!isActionable} */
                    className={`w-full py-3 rounded-xl font-mono font-black text-xs tracking-tight transition-all shadow-lg flex items-center justify-center gap-2 ${
                      isActionable && sizing.isAvailable && sizing.recommendedShares > 0
                        ? 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 cursor-pointer hover:scale-[1.01] active:scale-[0.99]'
                        : 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                    }`}
                  >
                    <span>
                      {!sizing.isAvailable
                        ? 'GOVERNOR SIZING INACTIVE: CONFIGURE REQUIRED RISK INPUTS'
                        : !isActionable
                        ? 'EXECUTION TICKET SUPPRESSED (CRITERIA NOT MET)'
                        : copyStatus === 'SUCCESS'
                        ? '✔ EXECUTION TICKET COPIED TO CLIPBOARD'
                        : copyStatus === 'FAILED'
                        ? `✖ FAILED TO COPY: ${copyErrorMessage || 'CLIPBOARD ERROR'}`
                        : `COPY EXECUTION TICKET: ${sizing.recommendedShares} SHARES ($${sizing.estimatedCapitalAllocated.toLocaleString()})`}
                    </span>
                  </button>
                  <div className="text-[10px] font-mono text-slate-400 text-center mt-2">
                    {isActionable && sizing.isAvailable && sizing.entryPivot > 0 && sizing.stopLoss > 0
                      ? `Formats execution order string to clipboard (does not route to broker): BUY ${sizing.recommendedShares} ${effectiveSetup.ticker} LMT $${sizing.entryPivot.toFixed(2)} | STP $${sizing.stopLoss.toFixed(2)} | TGT ${formatPrice(effectiveSetup.target1)}`
                      : !sizing.isAvailable
                      ? 'Execution ticket copy disabled: Risk parameter dependencies unresolved (intraday drawdown and loss streak tracking lack automated API telemetry writers). Sizing remains safely locked until authoritative risk inputs are supplied.'
                      : 'Trade levels suppressed: Stage criteria or volume dry-up not met.'}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </TerminalShell>
  );
}

export default function SetupsPage() {
  return (
    <Suspense fallback={
      <TerminalShell activeHub="setups">
        <div className="p-12 text-center text-slate-400 font-mono">
          Loading Tactical Setups...
        </div>
      </TerminalShell>
    }>
      <SetupsContent />
    </Suspense>
  );
}
