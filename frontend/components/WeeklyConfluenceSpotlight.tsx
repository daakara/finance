"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "../lib/masterCatalog";
import { addPortfolioPosition } from "../lib/portfolio";
import { SpotPriceRegistry, fetchBatchQuotes, fetchTacticalSetups, isQuoteFresh } from "../lib/api";
import type { TradeSetupSpec } from "../lib/simulation/governorSizingEngine";
import MiniSparkline from "./MiniSparkline";

// ── Canonical Decoupled Types ────────────────────────────────────────────────

export type MarketSessionStatus =
  | "REGULAR_OPEN"
  | "PRE_MARKET"
  | "AFTER_HOURS"
  | "CLOSED"
  | "HOLIDAY"
  | "UNKNOWN";

export type QuoteTelemetryStatus =
  | "LIVE_FRESH"
  | "STALE"
  | "DELAYED"
  | "MISSING"
  | "UNKNOWN";

export type SetupValidityStatus =
  | "VALID"
  | "STALE"
  | "EXPIRED"
  | "INVALID"
  | "UNKNOWN";

export type SpotlightPresentationState =
  | "LOADING"
  | "ERROR"
  | "SETUP_STALE"
  | "NO_QUALIFYING_CANDIDATES"
  | "READY_LIVE"
  | "READY_MARKET_CLOSED"
  | "READY_LIVE_TELEMETRY_DEGRADED";

export type SpotlightState =
  | "LOADING"
  | "ERROR"
  | "STALE_MARKET_DATA"
  | "NO_QUALIFYING_CANDIDATES"
  | "READY"
  | SpotlightPresentationState;

export interface DecoupledPriceResult {
  analysisPrice: number; // ROLE: RESEARCH_AND_ANALYSIS_ONLY (verified setup.analysisReferencePrice only)
  analysisDate?: string;
  marketOverlayPrice: number | null; // ROLE: PRESENTATION_ONLY_UNLESS_EXECUTION_QUALIFIED
  marketOverlayChangePct: number | null;
  displayPrice: number; // marketOverlayPrice if fresh, else analysisPrice
  displayChangePct: number;
  executionPrice: number | null; // ROLE: EXECUTION_QUALIFIED_ONLY (only present when execution is qualified)
  isLiveQuoteFresh: boolean;
  priceMode: "LIVE_OVERLAY" | "ANALYSIS_REFERENCE";
  sessionStatus: MarketSessionStatus;
  canonicalBackendSession: string | null;
  telemetryStatus: QuoteTelemetryStatus;
  canExecuteLive: boolean;
}

export interface ConfluenceCandidate {
  entry: MasterAssetEntry;
  analysisPrice: number;
  analysisDate?: string;
  marketOverlayPrice: number | null;
  marketOverlayChangePct: number | null;
  displayPrice: number;
  displayChangePct: number;
  executionPrice: number | null;
  isLiveQuoteFresh: boolean;
  priceMode: "LIVE_OVERLAY" | "ANALYSIS_REFERENCE";
  sessionStatus: MarketSessionStatus;
  canonicalBackendSession: string | null;
  telemetryStatus: QuoteTelemetryStatus;
  setupValidity: SetupValidityStatus;
  canExecuteLive: boolean;
  // Backward compatibility aliases
  livePrice: number;
  liveChangePct: number;
  convictionScore: number;
  setupBadge: string;
  setupBadgePlain: string;
  catalystSummary: string;
  catalystSummaryPlain: string;
  stopPrice: number;
  stopLossPct: string;
  target1Price: number;
  target1Pct: string;
  target2Price: number;
  target2Pct: string;
  rewardRiskRatio: string;
}

// ── Pure Domain Evaluation Helpers ───────────────────────────────────────────

/**
 * Resolves market session state for UI presentation fallback.
 * Backend `marketPriceState.marketSession` is the authoritative source.
 * Client clock derivation is strictly PRESENTATION_ONLY and CANNOT authorize execution.
 */
export function resolveMarketSession(
  backendSessionStr?: string | null,
  nowMs: number = Date.now()
): MarketSessionStatus {
  if (backendSessionStr && typeof backendSessionStr === "string") {
    const norm = backendSessionStr.toUpperCase().trim();
    if (norm === "REGULAR_SESSION" || norm === "REGULAR_OPEN" || norm === "OPEN") {
      return "REGULAR_OPEN";
    }
    if (norm === "PREMARKET" || norm === "PRE_MARKET" || norm === "PRE") {
      return "PRE_MARKET";
    }
    if (norm === "AFTER_HOURS" || norm === "POST_MARKET" || norm === "POST") {
      return "AFTER_HOURS";
    }
    if (norm === "WEEKEND" || norm === "CLOSED") {
      return "CLOSED";
    }
    if (norm === "HOLIDAY") {
      return "HOLIDAY";
    }
  }

  // Frontend deterministic presentation fallback in America/New_York
  try {
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      weekday: "short",
      hour: "numeric",
      minute: "numeric",
      hour12: false,
    });
    const parts = formatter.formatToParts(new Date(nowMs));
    let weekdayStr = "";
    let hour = 0;
    let minute = 0;
    for (const part of parts) {
      if (part.type === "weekday") weekdayStr = part.value;
      if (part.type === "hour") hour = parseInt(part.value, 10);
      if (part.type === "minute") minute = parseInt(part.value, 10);
    }

    if (weekdayStr === "Sat" || weekdayStr === "Sun") {
      return "CLOSED";
    }

    const minuteOfDay = hour * 60 + minute;
    // 04:00 is 240, 09:30 is 570, 16:00 is 960, 20:00 is 1200
    if (minuteOfDay < 240) return "CLOSED";
    if (minuteOfDay >= 240 && minuteOfDay < 570) return "PRE_MARKET";
    if (minuteOfDay >= 570 && minuteOfDay < 960) return "REGULAR_OPEN";
    if (minuteOfDay >= 960 && minuteOfDay < 1200) return "AFTER_HOURS";
    return "CLOSED";
  } catch {
    return "UNKNOWN";
  }
}

/**
 * Evaluates live quote telemetry freshness.
 * Requires finite positive price and valid observation age strictly within QUOTE_MAX_AGE_MS.
 */
export function resolveQuoteTelemetry(
  quote?: { price?: number; lastUpdated?: number; changePct?: number } | null
): QuoteTelemetryStatus {
  if (!quote) return "MISSING";
  if (typeof quote.price !== "number" || isNaN(quote.price) || quote.price <= 0) {
    return "UNKNOWN";
  }
  if (!quote.lastUpdated || typeof quote.lastUpdated !== "number" || quote.lastUpdated <= 0) {
    return "UNKNOWN";
  }

  const isFresh = isQuoteFresh(quote.lastUpdated);
  if (isFresh) return "LIVE_FRESH";

  const age = Date.now() - quote.lastUpdated;
  if (age > 24 * 60 * 60 * 1000) {
    return "DELAYED";
  }
  return "STALE";
}

/**
 * Validates setup structural integrity and verified analytical reference price authority.
 * Sourced independently from live quote feeds.
 * INVARIANT: Must strictly require verified `setup.analysisReferencePrice`.
 * Never falls back to `setup.currentPrice` (which can be a mutable realtime spot override).
 */
export function evaluateSetupValidity(setup: TradeSetupSpec): SetupValidityStatus {
  if (!setup || typeof setup !== "object") return "INVALID";

  // Check if backend marked the setup stale due to >4 calendar days historical candle age
  if (
    setup.executionStatus === "STALE_MARKET_DATA" ||
    setup.decisionState === "STALE_DATA" ||
    setup.setupName === "Stale Market Tape"
  ) {
    return "STALE";
  }

  // Check valid stop loss and target 1
  if (!setup.stopLoss || typeof setup.stopLoss !== "number" || setup.stopLoss <= 0) {
    return "INVALID";
  }
  if (!setup.target1 || typeof setup.target1 !== "number" || setup.target1 <= 0) {
    return "INVALID";
  }

  // Confluence score must be an authentic positive number
  if (typeof setup.confluenceScore !== "number" || isNaN(setup.confluenceScore) || setup.confluenceScore <= 0) {
    return "INVALID";
  }

  // Strict Invariant: Analysis reference price MUST come from verified analysisReferencePrice ONLY.
  // Never accept setup.currentPrice as fallback.
  if (
    typeof setup.analysisReferencePrice !== "number" ||
    isNaN(setup.analysisReferencePrice) ||
    setup.analysisReferencePrice <= 0
  ) {
    return "INVALID";
  }

  return "VALID";
}

/**
 * Resolves decoupled prices separating analysis reference price from market overlay and execution price.
 *
 * Invariants:
 * 1. ANALYSIS_REFERENCE_PRICE is for RESEARCH_AND_ANALYSIS_ONLY.
 * 2. MARKET_OVERLAY_PRICE is for PRESENTATION_ONLY unless execution-qualified.
 * 3. EXECUTION_PRICE requires:
 *    - Canonical backend session is REGULAR_OPEN (frontend clock fallback cannot authorize execution).
 *    - LIVE_FRESH quote telemetry (age < 5 min, price > 0).
 *    - Setup is actionable.
 *    Otherwise EXECUTION_PRICE is strictly null and paper execution is disabled.
 */
export function resolveDecoupledPrices(
  setup: TradeSetupSpec,
  quote: { price?: number; lastUpdated?: number; changePct?: number } | null | undefined,
  sessionStatus: MarketSessionStatus,
  canonicalBackendSession?: string | null
): DecoupledPriceResult | null {
  const validity = evaluateSetupValidity(setup);
  if (validity !== "VALID") {
    return null;
  }

  // Sourced strictly from verified analysisReferencePrice
  const analysisPrice = setup.analysisReferencePrice!;
  const telemetry = resolveQuoteTelemetry(quote);
  const isFresh = telemetry === "LIVE_FRESH";

  const marketOverlayPrice = (isFresh && typeof quote?.price === "number" && quote.price > 0) ? quote.price : null;
  const marketOverlayChangePct = (isFresh && typeof quote?.changePct === "number") ? quote.changePct : null;

  // Price Mode selection:
  // Live overlay is used ONLY when market is in regular session AND quote is fresh
  const isRegularOpen = sessionStatus === "REGULAR_OPEN";
  const useOverlay = isRegularOpen && isFresh && marketOverlayPrice !== null;
  const displayPrice = useOverlay ? marketOverlayPrice! : analysisPrice;
  const displayChangePct = useOverlay ? (marketOverlayChangePct ?? 0.0) : 0.0;
  const priceMode = useOverlay ? "LIVE_OVERLAY" : "ANALYSIS_REFERENCE";

  // Execution Qualification:
  // Requires:
  // 1. Authoritative CANONICAL BACKEND SESSION is REGULAR_OPEN (frontend clock fallback cannot authorize execution)
  // 2. LIVE_FRESH quote telemetry
  // 3. Setup is actionable
  const rawBackend = (canonicalBackendSession || setup.marketPriceState?.marketSession || "").toUpperCase().trim();
  const isBackendRegularOpen = rawBackend === "REGULAR_SESSION" || rawBackend === "REGULAR_OPEN" || rawBackend === "OPEN";

  const canExecuteLive = isBackendRegularOpen && isFresh && marketOverlayPrice !== null && Boolean(setup.isActionable);
  const executionPrice = canExecuteLive ? marketOverlayPrice : null;

  const analysisDate = setup.marketPriceState?.analysisReferenceDate || (setup as any).observationDate || undefined;

  return {
    analysisPrice,
    analysisDate,
    marketOverlayPrice,
    marketOverlayChangePct,
    displayPrice,
    displayChangePct,
    executionPrice,
    isLiveQuoteFresh: isFresh,
    priceMode,
    sessionStatus,
    canonicalBackendSession: canonicalBackendSession || setup.marketPriceState?.marketSession || null,
    telemetryStatus: telemetry,
    canExecuteLive,
  };
}

/**
 * Deterministically derives the component-level presentation state.
 */
export function deriveOverallSpotlightState(params: {
  isLoadingSetups: boolean;
  setupError: string | null;
  tacticalSetups: TradeSetupSpec[];
  topCandidates: ConfluenceCandidate[];
  sessionStatus: MarketSessionStatus;
}): SpotlightPresentationState {
  if (params.isLoadingSetups) return "LOADING";
  if (params.setupError) return "ERROR";
  if (!params.tacticalSetups || params.tacticalSetups.length === 0) return "NO_QUALIFYING_CANDIDATES";

  // Check if all tactical setups are stale from historical data
  const allStale = params.tacticalSetups.every(
    (s) => s.executionStatus === "STALE_MARKET_DATA" || s.decisionState === "STALE_DATA"
  );
  if (allStale) {
    return "SETUP_STALE";
  }

  if (params.topCandidates.length === 0) {
    return "NO_QUALIFYING_CANDIDATES";
  }

  // Candidates exist and are valid. Determine presentation mode based on session and telemetry.
  if (params.sessionStatus === "REGULAR_OPEN") {
    const anyFresh = params.topCandidates.some((c) => c.isLiveQuoteFresh);
    if (anyFresh) {
      return "READY_LIVE";
    }
    return "READY_LIVE_TELEMETRY_DEGRADED";
  }

  return "READY_MARKET_CLOSED";
}

// ── Component Implementation ─────────────────────────────────────────────────

interface WeeklyConfluenceSpotlightProps {
  defaultCollapsed?: boolean;
  onSelectSymbol?: (symbol: string) => void;
  selectedSymbol?: string;
}

export default function WeeklyConfluenceSpotlight({
  defaultCollapsed = false,
  onSelectSymbol,
  selectedSymbol,
}: WeeklyConfluenceSpotlightProps) {
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");
  const [userRole, setUserRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [loggedSymbol, setLoggedSymbol] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(defaultCollapsed);
  const [liveQuotes, setLiveQuotes] = useState<Record<string, { price: number; changePct: number; lastUpdated: number }>>({});
  const [tacticalSetups, setTacticalSetups] = useState<TradeSetupSpec[]>([]);
  const [isLoadingSetups, setIsLoadingSetups] = useState<boolean>(true);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [retryNonce, setRetryNonce] = useState<number>(0);

  useEffect(() => {
    setIsCollapsed(defaultCollapsed);
  }, [defaultCollapsed]);

  // Hydrate initial live quotes from registry and prune expired entries
  const refreshLocalQuotes = useCallback(() => {
    setLiveQuotes((prev) => {
      const next: Record<string, { price: number; changePct: number; lastUpdated: number }> = {};
      // 1. Keep non-expired quotes from prev
      for (const [sym, q] of Object.entries(prev)) {
        if (q && q.lastUpdated && isQuoteFresh(q.lastUpdated)) {
          next[sym] = q;
        }
      }
      // 2. Ingest fresh quotes from SpotPriceRegistry strictly for candidate setups
      const candidateList = tacticalSetups.length > 0
        ? tacticalSetups.map((s) => s.ticker)
        : Object.keys(MASTER_ASSET_CATALOG).slice(0, 5);

      for (const sym of candidateList) {
        const reg = SpotPriceRegistry.get(sym);
        if (reg && reg.price > 0 && reg.lastUpdated && isQuoteFresh(reg.lastUpdated)) {
          next[sym] = { price: reg.price, changePct: reg.changePct, lastUpdated: reg.lastUpdated };
        }
      }
      return next;
    });
  }, [tacticalSetups]);

  // Fetch setups with explicit error state propagation (never swallows failures into empty list)
  useEffect(() => {
    let isMounted = true;
    setIsLoadingSetups(true);
    setSetupError(null);

    fetchTacticalSetups(undefined, userRole)
      .then((data) => {
        if (!isMounted) return;
        setTacticalSetups(data || []);
        setIsLoadingSetups(false);
      })
      .catch((err: any) => {
        if (!isMounted) return;
        setTacticalSetups([]);
        setSetupError(err?.message || "Failed to load tactical setups from analytics engine.");
        setIsLoadingSetups(false);
      });

    return () => {
      isMounted = false;
    };
  }, [userRole, retryNonce]);

  // Derive quotes strictly from actual candidate universe
  useEffect(() => {
    refreshLocalQuotes();

    if (!tacticalSetups || tacticalSetups.length === 0) return;
    const candidateSymbols = Array.from(new Set(tacticalSetups.map((s) => s.ticker).filter(Boolean)));
    if (candidateSymbols.length === 0) return;

    fetchBatchQuotes(candidateSymbols)
      .then((batch) => {
        if (batch && Object.keys(batch).length > 0) {
          setLiveQuotes((prev) => {
            const next = { ...prev };
            for (const [sym, b] of Object.entries(batch)) {
              if (b && b.price > 0 && b.lastUpdated && isQuoteFresh(b.lastUpdated)) {
                next[sym] = b;
              }
            }
            return next;
          });
        }
      })
      .catch(() => {});

    const timer = setInterval(() => {
      refreshLocalQuotes();
    }, 15000);

    return () => clearInterval(timer);
  }, [tacticalSetups, refreshLocalQuotes]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);

      const savedRole = localStorage.getItem("FINANCE_USER_ROLE") as "DAY_TRADER" | "LONG_TERM" | null;
      if (savedRole) setUserRole(savedRole);

      const handleStorage = () => refreshLocalQuotes();
      window.addEventListener("storage", handleStorage);
      return () => {
        window.removeEventListener("storage", handleStorage);
      };
    }
  }, [refreshLocalQuotes]);

  useEffect(() => {
    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    const handleRole = (e: Event) => {
      const custom = e as CustomEvent<"DAY_TRADER" | "LONG_TERM">;
      if (custom.detail) setUserRole(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    window.addEventListener("finance:role-change", handleRole);
    return () => {
      window.removeEventListener("finance:vernacular-change", handleVernacular);
      window.removeEventListener("finance:role-change", handleRole);
    };
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";
  const isDayTrader = userRole === "DAY_TRADER";

  // Authoritative Session State
  const backendSession = useMemo(() => {
    return tacticalSetups.find((s) => s.marketPriceState?.marketSession)?.marketPriceState?.marketSession || null;
  }, [tacticalSetups]);

  const currentSession: MarketSessionStatus = useMemo(() => {
    return resolveMarketSession(backendSession);
  }, [backendSession]);

  // Dynamically compute the Top 3 High-Confluence Plays decoupled from live quote freshness
  const topCandidates: ConfluenceCandidate[] = useMemo(() => {
    if (!tacticalSetups || tacticalSetups.length === 0) return [];

    // Filter valid setups with verified analytical reference prices, valid levels, and positive confluence
    const valid = tacticalSetups
      .map((setup) => {
        const sym = setup.ticker;
        const live = liveQuotes[sym];
        const reg = SpotPriceRegistry.get(sym);
        const resolvedQuote = (live && live.price > 0) ? live : (reg && reg.price > 0 ? reg : null);

        const priceResult = resolveDecoupledPrices(setup, resolvedQuote, currentSession, backendSession);
        if (!priceResult) return null;

        const confScore = setup.confluenceScore;
        if (typeof confScore !== "number" || isNaN(confScore) || confScore <= 0) {
          return null;
        }

        return {
          setup,
          priceResult,
          confScore,
        };
      })
      .filter((item): item is { setup: TradeSetupSpec; priceResult: DecoupledPriceResult; confScore: number } => item !== null);

    // Canonical Sort: Actionable first, then highest confluenceScore descending
    // (Live quote freshness CANNOT alter weekly rank)
    const sorted = [...valid].sort((a, b) => {
      const aAct = Boolean(a.setup.isActionable);
      const bAct = Boolean(b.setup.isActionable);
      if (aAct !== bAct) {
        return aAct ? -1 : 1;
      }
      return b.confScore - a.confScore;
    });

    return sorted.slice(0, 3).map(({ setup, priceResult, confScore }) => {
      const sym = setup.ticker;
      const master = MASTER_ASSET_CATALOG[sym];
      const entry: MasterAssetEntry = master || {
        symbol: sym,
        name: sym,
        type: "Stock",
        sector: "Equities",
        category: "Trading Setup",
        roic: 0,
        grossMargin: 0,
        fwdPe: 0,
        peg: 0,
        fcfYield: 0,
        piotroski: 0,
        atr14: 0,
        rvol: 0,
        shortFloat: 0,
        beta: 1.0,
        marketCap: "-",
        growthScore: 0,
        qualityScore: 0,
        valuationScore: 0,
        momentumScore: 0,
        tailRiskScore: 0,
        compositeFactorScore: Math.round(confScore),
        verdict: setup.setupName || "High Confluence Setup",
        moatSummary: setup.entryThesis || "Verified Setup",
        upcomingCatalyst: setup.setupName || "Technical Setup",
        thesis: setup.entryThesis || "Live Confluence Setup",
      };

      const effPrice = priceResult.displayPrice;
      const stopVal = setup.stopLoss ?? 0;
      const target1Val = setup.target1 ?? 0;
      const target2Val = setup.target2 ?? target1Val;
      const stopPct = effPrice > 0 ? (((effPrice - stopVal) / effPrice) * 100).toFixed(1) : "0.0";
      const t1Pct = effPrice > 0 ? (((target1Val - effPrice) / effPrice) * 100).toFixed(1) : "0.0";
      const t2Pct = effPrice > 0 ? (((target2Val - effPrice) / effPrice) * 100).toFixed(1) : "0.0";
      const riskDelta = effPrice - stopVal;
      const rewardDelta = target1Val - effPrice;
      const rr = riskDelta > 0 && rewardDelta > 0 ? (rewardDelta / riskDelta).toFixed(1) : "N/A";

      return {
        entry,
        analysisPrice: priceResult.analysisPrice,
        analysisDate: priceResult.analysisDate,
        marketOverlayPrice: priceResult.marketOverlayPrice,
        marketOverlayChangePct: priceResult.marketOverlayChangePct,
        displayPrice: priceResult.displayPrice,
        displayChangePct: priceResult.displayChangePct,
        executionPrice: priceResult.executionPrice,
        isLiveQuoteFresh: priceResult.isLiveQuoteFresh,
        priceMode: priceResult.priceMode,
        sessionStatus: priceResult.sessionStatus,
        canonicalBackendSession: priceResult.canonicalBackendSession,
        telemetryStatus: priceResult.telemetryStatus,
        setupValidity: "VALID",
        canExecuteLive: priceResult.canExecuteLive,
        // Backward compatibility
        livePrice: priceResult.displayPrice,
        liveChangePct: priceResult.displayChangePct,
        convictionScore: Math.min(99, Math.round(confScore)),
        setupBadge: setup.setupName || (isDayTrader ? "⚡ HIGH-RVOL MOMENTUM" : "INSTITUTIONAL ACCUMULATION"),
        setupBadgePlain: isPlain ? "High Confluence Setup" : (setup.setupName || "High Confluence"),
        catalystSummary: setup.entryThesis || setup.setupName || (isDayTrader ? "Intraday Volume Expansion" : "Institutional Accumulation"),
        catalystSummaryPlain: setup.setupName || (isDayTrader ? "Surging Trading Volume" : "High Quality Accumulation"),
        stopPrice: Number(stopVal.toFixed(2)),
        stopLossPct: stopPct,
        target1Price: Number(target1Val.toFixed(2)),
        target1Pct: t1Pct,
        target2Price: Number(target2Val.toFixed(2)),
        target2Pct: t2Pct,
        rewardRiskRatio: rr,
      };
    });
  }, [tacticalSetups, liveQuotes, currentSession, backendSession, isDayTrader, isPlain]);

  // Deterministic presentation state
  const presentationState: SpotlightPresentationState = useMemo(() => {
    return deriveOverallSpotlightState({
      isLoadingSetups,
      setupError,
      tacticalSetups,
      topCandidates,
      sessionStatus: currentSession,
    });
  }, [isLoadingSetups, setupError, tacticalSetups, topCandidates, currentSession]);

  const spotlightState: SpotlightState = presentationState;

  // Live execution action: permitted ONLY when execution qualified
  const handleQuickLog = async (e: React.MouseEvent, cand: ConfluenceCandidate) => {
    e.preventDefault();
    e.stopPropagation();

    // Guard: Fail closed if execution is not qualified
    if (!cand.canExecuteLive || !cand.executionPrice) {
      alert("Paper execution is disabled: Live market session and fresh exchange tape quote required.");
      return;
    }

    const userSharesStr = window.prompt(
      `Enter quantity of ${cand.entry.symbol} shares to execute fill (Live Quote $${cand.executionPrice.toFixed(2)}):`,
      "10"
    );
    if (!userSharesStr) return;
    const parsedShares = parseFloat(userSharesStr);
    if (isNaN(parsedShares) || parsedShares <= 0) {
      alert("Invalid share quantity. Must be a positive number.");
      return;
    }

    const res = await addPortfolioPosition({
      symbol: cand.entry.symbol,
      name: cand.entry.name,
      shares: parsedShares,
      entryPrice: cand.executionPrice, // Strictly qualified live execution quote
      currentPrice: cand.executionPrice,
      targetPrice: cand.target1Price,
      stopLossPrice: cand.stopPrice,
    });

    setLoggedSymbol(`${cand.entry.symbol}: ${res.isDuplicate ? "Already in Portfolio" : res.success ? "Logged Fill!" : "Failed: " + res.message}`);
    setTimeout(() => setLoggedSymbol(null), 3500);
  };

  // Intent / Planning action: used outside live execution windows (Zero synthetic fill creation)
  const handlePlanSetup = (e: React.MouseEvent, cand: ConfluenceCandidate) => {
    e.preventDefault();
    e.stopPropagation();

    const reason = cand.sessionStatus === "CLOSED" ? "Market Closed" : "Tape Delayed";
    const userSharesStr = window.prompt(
      `Plan entry for ${cand.entry.symbol} (${reason} • reference $${cand.analysisPrice.toFixed(2)}). Enter target shares to watch:`,
      "10"
    );
    if (!userSharesStr) return;
    const parsedShares = parseFloat(userSharesStr);
    if (isNaN(parsedShares) || parsedShares <= 0) {
      alert("Invalid share quantity. Must be a positive number.");
      return;
    }

    try {
      if (typeof window !== "undefined") {
        const existingRaw = localStorage.getItem("FINANCE_PLANNED_SETUPS");
        const existing = existingRaw ? JSON.parse(existingRaw) : [];
        const updated = [
          ...existing.filter((p: any) => p.symbol !== cand.entry.symbol),
          {
            symbol: cand.entry.symbol,
            name: cand.entry.name,
            shares: parsedShares,
            referencePrice: cand.analysisPrice,
            target1: cand.target1Price,
            stopLoss: cand.stopPrice,
            plannedAt: new Date().toISOString(),
            status: "PLANNED_PENDING_MARKET_OPEN",
          },
        ];
        localStorage.setItem("FINANCE_PLANNED_SETUPS", JSON.stringify(updated));
      }
      setLoggedSymbol(`${cand.entry.symbol}: Setup Planned! (Pending Open)`);
      setTimeout(() => setLoggedSymbol(null), 3500);
    } catch {
      alert("Could not save planned setup to local storage.");
    }
  };

  const handleCardClick = (e: React.MouseEvent, symbol: string) => {
    if (onSelectSymbol) {
      e.preventDefault();
      onSelectSymbol(symbol);
    }
    setIsCollapsed(true);
    if (typeof window !== "undefined") {
      const target = document.getElementById("market-workspace-chart") || document.getElementById("main-content");
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  };

  return (
    <section className={`bg-[#0d121c] border border-[#1e293b] rounded-2xl shadow-2xl transition-all ${
      isCollapsed ? "p-3 sm:p-3.5 space-y-2 mb-4" : "p-4 sm:p-5 space-y-4 mb-6"
    }`}>
      {/* Header Bar */}
      <div className={`flex flex-wrap items-center justify-between gap-3 ${
        isCollapsed ? "" : "border-b border-[#1b2434] pb-3.5"
      }`}>
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-sm shadow-inner shrink-0 ${
            isDayTrader
              ? "bg-amber-500/10 border border-amber-500/30 text-amber-400"
              : "bg-cyan-500/10 border border-cyan-500/30 text-cyan-400"
          }`}>
            {isDayTrader ? "⚡" : "🎯"}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-sm sm:text-base font-extrabold text-white tracking-tight flex items-center gap-2">
                <span>
                  {isDayTrader
                    ? isPlain
                      ? "Top 3 Day Trader Momentum Plays"
                      : "Day Trader Confluence: Top 3 High-RVOL Setups"
                    : isPlain
                    ? "Top 3 High-Confluence Plays of the Week"
                    : "Weekly Alpha Spotlight: Top 3 High-Confluence Setups"}
                </span>
              </h2>
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border hidden sm:inline-block ${
                isDayTrader
                  ? "bg-amber-950/80 border-amber-700 text-amber-300"
                  : "bg-cyan-950/80 border-cyan-700 text-cyan-300"
              }`}>
                {isDayTrader ? "⚡ DAY TRADER SIEVE" : "🏛️ LONG-TERM SIEVE"}
              </span>

              {/* Accessible Market / Telemetry Status Indicators */}
              {spotlightState === 'READY_MARKET_CLOSED' && (
                <span
                  className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border bg-slate-900/90 border-slate-700 text-slate-300 hidden sm:inline-flex items-center gap-1"
                  aria-label="Market session closed. Analysis active based on completed session close."
                >
                  <span aria-hidden="true">🌙</span>
                  <span>SESSION CLOSED</span>
                </span>
              )}
              {spotlightState === 'READY_LIVE_TELEMETRY_DEGRADED' && (
                <span
                  className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border bg-amber-950/80 border-amber-700 text-amber-300 hidden sm:inline-flex items-center gap-1"
                  aria-label="Live quote telemetry delayed. Trade setups active on reference analysis."
                >
                  <span aria-hidden="true">⏳</span>
                  <span>TAPE DELAYED</span>
                </span>
              )}
              {spotlightState === 'READY_LIVE' && (
                <span
                  className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border bg-emerald-950/80 border-emerald-700 text-emerald-300 hidden sm:inline-flex items-center gap-1"
                  aria-label="Live exchange tape active"
                >
                  <span aria-hidden="true">⚡</span>
                  <span>LIVE TAPE</span>
                </span>
              )}
            </div>
            {!isCollapsed && (
              <p className="text-xs text-slate-400 mt-0.5">
                {isDayTrader
                  ? isPlain
                    ? "Filtered for fast-moving stocks with elevated trading volume and defined risk stops."
                    : "Intraday & swing sieve: RVOL >= 1.3 + ATR risk definition + Minervini Stage 2 + R:R >= 1.85:1."
                  : isPlain
                  ? "Filtered by multi-factor confluence (balance sheet health, institutional flow) with minimum 1.85:1 profit-to-risk ratio."
                  : "Multi-factor quantitative sieve: Minervini Stage 2 + Multi-Factor Confluence (Piotroski, Insider Flow) + Risk/Reward >= 1.85:1."}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {loggedSymbol && (
            <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-emerald-950/80 border border-emerald-700 text-emerald-300 animate-fade-in">
              💼 {loggedSymbol}
            </span>
          )}
          <button
            type="button"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-xs px-2.5 py-2 sm:py-1 min-h-[36px] sm:min-h-0 rounded-md font-mono text-slate-400 hover:text-slate-200 border border-[#243044] hover:bg-[#162030] transition-colors inline-flex items-center"
            aria-label={isCollapsed ? "Expand Weekly Spotlight" : "Collapse Weekly Spotlight"}
          >
            {isCollapsed ? "View Full Setups ▼" : "Collapse ▲"}
          </button>
        </div>
      </div>

      {/* Compact Quick-Switcher Ribbon when Collapsed */}
      {isCollapsed && (
        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[#1b2434]/60">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] font-mono text-slate-400 font-bold flex items-center gap-1">
              <span>{isDayTrader ? "⚡" : "🎯"}</span>
              <span>Top Plays:</span>
            </span>
            {topCandidates.length > 0 ? (
              topCandidates.map((cand, idx) => {
                const isSelected = selectedSymbol?.toUpperCase() === cand.entry.symbol.toUpperCase();
                const isOverlay = cand.priceMode === "LIVE_OVERLAY";
                return (
                  <button
                    key={cand.entry.symbol}
                    type="button"
                    onClick={(e) => handleCardClick(e, cand.entry.symbol)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold border transition-all flex items-center gap-1.5 active:scale-95 ${
                      isSelected
                        ? "bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                        : "bg-[#111722] border-[#243044] text-slate-300 hover:border-cyan-500/60 hover:text-white"
                    }`}
                    aria-label={`Select ${cand.entry.symbol}, ${isOverlay ? 'live price' : 'reference price'} $${cand.displayPrice.toFixed(2)}`}
                  >
                    <span className="text-[9px] text-slate-400 font-normal">#{idx + 1}</span>
                    <span className="font-extrabold">{cand.entry.symbol}</span>
                    <span className={`text-[10px] tabular-nums ${
                      isOverlay
                        ? (cand.displayChangePct >= 0 ? "text-emerald-400" : "text-rose-400")
                        : "text-slate-300"
                    }`}>
                      {!isOverlay ? "Ref: " : ""}${cand.displayPrice.toFixed(2)}
                    </span>
                  </button>
                );
              })
            ) : spotlightState === 'LOADING' ? (
              <span className="text-xs text-slate-500 font-mono animate-pulse">Scanning setups...</span>
            ) : spotlightState === 'ERROR' ? (
              <span className="text-xs text-rose-400 font-mono">⚠️ Setups telemetry error</span>
            ) : spotlightState === 'SETUP_STALE' ? (
              <span className="text-xs text-amber-400 font-mono">⏳ Tactical setups stale (&gt;4 days)</span>
            ) : (
              <span className="text-xs text-slate-500 font-mono">0 qualifying plays</span>
            )}
          </div>
          <button
            type="button"
            onClick={() => setIsCollapsed(false)}
            className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-semibold py-2 sm:py-0 min-h-[36px] sm:min-h-0"
          >
            <span>View Full Setups</span>
            <span>▼</span>
          </button>
        </div>
      )}

      {/* 3-Card Responsive Grid */}
      {!isCollapsed && (
        topCandidates.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
            {topCandidates.map((cand, idx) => {
              const isRank1 = idx === 0;
              const isOverlay = cand.priceMode === "LIVE_OVERLAY";

              return (
                <Link
                  key={cand.entry.symbol}
                  href={`/?symbol=${cand.entry.symbol}`}
                  onClick={(e) => handleCardClick(e, cand.entry.symbol)}
                  aria-label={`Analyze ${cand.entry.symbol} (${cand.entry.name}), ${isOverlay ? 'live' : 'reference'} price $${cand.displayPrice.toFixed(2)}`}
                  className={`p-4 rounded-xl border transition-all duration-150 active:scale-[0.98] active:bg-[#0e1522] bg-[#111722] space-y-3 block group cursor-pointer ${
                    isRank1
                      ? "border-cyan-500/60 shadow-[0_0_16px_rgba(6,182,212,0.12)] hover:border-cyan-400"
                      : "border-[#243044] hover:border-cyan-500/60 hover:shadow-[0_0_12px_rgba(6,182,212,0.08)]"
                  }`}
                >
                  {/* Card Header: Rank Badge, Ticker & Price + Compact Score Pill */}
                  <div className="flex items-start justify-between gap-2 min-w-0">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <span className={`w-5 h-5 rounded-md flex items-center justify-center font-mono font-black text-xs shrink-0 ${
                        isRank1 ? "bg-cyan-500 text-slate-950 font-bold" : "bg-slate-800 text-slate-300"
                      }`}>
                        #{idx + 1}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <strong className="text-base font-black text-white font-mono group-hover:text-cyan-400 transition-colors shrink-0">
                            {cand.entry.symbol}
                          </strong>
                          <span className="text-[11px] text-slate-400 truncate max-w-[80px] sm:max-w-[105px]" title={cand.entry.name}>
                            {cand.entry.name}
                          </span>
                        </div>

                        {/* Price Display with Explicit Provenance */}
                        {isOverlay ? (
                          <div
                            className="text-xs font-mono font-bold text-slate-300 tabular-nums truncate flex items-center gap-1"
                            aria-label={`Live price $${cand.displayPrice.toFixed(2)}, change ${cand.displayChangePct >= 0 ? "+" : ""}${cand.displayChangePct}%`}
                          >
                            <span>${cand.displayPrice.toFixed(2)}</span>
                            <span className={cand.displayChangePct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                              ({cand.displayChangePct >= 0 ? "+" : ""}{cand.displayChangePct}%)
                            </span>
                            <span
                              className="px-1 py-0.2 rounded text-[9px] bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 font-semibold"
                              title="Verified live quote from exchange tape"
                            >
                              LIVE
                            </span>
                          </div>
                        ) : (
                          <div
                            className="text-xs font-mono font-bold text-slate-300 tabular-nums truncate flex items-center gap-1"
                            aria-label={`Analysis reference price $${cand.displayPrice.toFixed(2)}, session closed`}
                          >
                            <span className="text-slate-400 font-normal">Ref:</span>
                            <span>${cand.displayPrice.toFixed(2)}</span>
                            <span
                              className={`px-1 py-0.2 rounded text-[9px] font-semibold border ${
                                cand.sessionStatus === 'CLOSED'
                                  ? "bg-slate-900 border-slate-700 text-slate-400"
                                  : "bg-amber-950/80 border-amber-800/60 text-amber-300"
                              }`}
                              title={
                                cand.sessionStatus === 'CLOSED'
                                  ? "Anchored to latest completed session close"
                                  : "Live tape reconnecting; using verified reference close"
                              }
                            >
                              {cand.sessionStatus === 'CLOSED' ? "CLOSED" : "DELAYED"}
                            </span>
                          </div>
                        )}
                        {!isOverlay && cand.analysisDate && (
                          <div className="text-[10px] text-slate-500 font-mono truncate" title={`Anchored to ${cand.analysisDate} close`}>
                            {cand.analysisDate} close
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Sparkline & Compact Score Pill */}
                    <div className="flex items-center gap-2 shrink-0">
                      <MiniSparkline
                        basePrice={cand.displayPrice}
                        changePct={cand.displayChangePct}
                        width={40}
                        height={18}
                        className="hidden sm:inline-block"
                      />
                      <div className="px-2 py-0.5 rounded-md bg-[#090d14] border border-cyan-800/50 text-right">
                        <span className="text-[8px] font-mono text-slate-400 block uppercase font-bold tracking-wider leading-none">
                          SCORE
                        </span>
                        <span className="text-xs font-black font-mono text-cyan-300 tabular-nums leading-none">
                          {cand.convictionScore}<span className="text-[9px] text-cyan-500/70 font-normal">/100</span>
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Setup Badge */}
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono font-extrabold min-w-0">
                    <span className="px-2 py-0.5 rounded bg-[#090d14] border border-cyan-800/50 text-cyan-300 truncate max-w-[165px]" title={isPlain ? cand.setupBadgePlain : cand.setupBadge}>
                      {isPlain ? cand.setupBadgePlain : cand.setupBadge}
                    </span>
                    <span className="text-emerald-400 shrink-0 tabular-nums">
                      {cand.rewardRiskRatio} : 1.0 R:R
                    </span>
                  </div>

                  {/* Mathematical Execution Price Ladder */}
                  <div className="bg-[#090d14] p-2.5 rounded-lg border border-[#1e293b] space-y-1.5 font-mono text-xs">
                    <div className="flex items-center justify-between gap-1 text-[11px] min-w-0">
                      <span className="text-emerald-400 font-bold truncate">{isPlain ? "Goal 1 (Sell Half):" : "Take Profit 1 (TP1):"}</span>
                      <strong className="text-white tabular-nums shrink-0">
                        ${cand.target1Price.toFixed(2)} <span className="text-emerald-500 text-[10px] font-normal">(+{cand.target1Pct}%)</span>
                      </strong>
                    </div>
                    <div className="flex items-center justify-between gap-1 text-[11px] min-w-0">
                      <span className="text-rose-400 font-bold truncate">{isPlain ? "Safety Exit Stop:" : "Hard Stop Floor:"}</span>
                      <strong className="text-rose-400 tabular-nums shrink-0">
                        ${cand.stopPrice.toFixed(2)} <span className="text-rose-500 text-[10px] font-normal">(-{cand.stopLossPct}%)</span>
                      </strong>
                    </div>
                  </div>

                  {/* Rationale / Catalyst Text */}
                  <p className="text-[11px] text-slate-300 leading-relaxed font-sans line-clamp-2">
                    {isPlain ? cand.catalystSummaryPlain : cand.catalystSummary}
                  </p>

                  {/* Footer CTAs: Explicit separation between Live Execution and Setup Planning */}
                  <div className="flex items-center justify-between pt-1 border-t border-[#1e293b] text-[11px]">
                    {cand.canExecuteLive ? (
                      <button
                        type="button"
                        onClick={(e) => handleQuickLog(e, cand)}
                        className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border bg-emerald-600/20 hover:bg-emerald-500 hover:text-slate-950 border-emerald-500/40 text-emerald-300 transition-colors flex items-center gap-1 shrink-0 active:scale-95 cursor-pointer"
                        title={`Log live execution fill into Paper Portfolio at live price $${cand.executionPrice!.toFixed(2)}`}
                      >
                        <span>💼</span>
                        <span>{isPlain ? "Quick Paper Log" : "Log Live Fill"}</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={(e) => handlePlanSetup(e, cand)}
                        className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border bg-slate-800/80 hover:bg-slate-700 hover:text-white border-slate-700 text-slate-300 transition-colors flex items-center gap-1 shrink-0 active:scale-95 cursor-pointer"
                        title={
                          cand.sessionStatus === 'CLOSED'
                            ? "Market session closed; plan setup entry (execution triggers resume at market open)"
                            : "Live tape delayed; plan setup entry (live execution triggers suspended)"
                        }
                      >
                        <span>📌</span>
                        <span>{isPlain ? "Track Setup" : "Plan Entry"}</span>
                      </button>
                    )}

                    <span className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border bg-cyan-500/10 border-cyan-500/40 text-cyan-300 group-hover:bg-cyan-500 group-hover:text-slate-950 group-hover:border-cyan-400 transition-colors flex items-center gap-1 shrink-0">
                      Analyze <span className="group-hover:translate-x-0.5 transition-transform">➔</span>
                    </span>
                  </div>
                </Link>
              );
            })}
          </div>
        ) : spotlightState === 'LOADING' ? (
          <div className="p-8 text-center text-xs font-mono text-cyan-400 bg-[#111722] rounded-xl border border-[#243044] animate-pulse flex items-center justify-center gap-3">
            <span>⏳</span>
            <span>Scanning multi-factor confluence setups &amp; reference quotes...</span>
          </div>
        ) : spotlightState === 'ERROR' ? (
          <div className="p-6 text-center space-y-3 font-mono bg-rose-950/20 rounded-xl border border-rose-800/60 text-xs">
            <div className="text-rose-400 font-bold text-sm flex items-center justify-center gap-2">
              <span>⚠️</span>
              <span>Tactical Setups Telemetry Unavailable</span>
            </div>
            <p className="text-slate-300 max-w-md mx-auto font-sans">
              {setupError || "An error occurred contacting the tactical analytics engine."}
            </p>
            <button
              type="button"
              onClick={() => setRetryNonce((n) => n + 1)}
              className="px-3.5 py-1.5 rounded-lg bg-rose-900/40 hover:bg-rose-800/60 text-rose-200 border border-rose-700/60 font-mono font-bold transition-all cursor-pointer"
            >
              🔄 Retry Sieve
            </button>
          </div>
        ) : spotlightState === 'SETUP_STALE' ? (
          <div className="p-6 text-center space-y-2 font-mono bg-amber-950/20 rounded-xl border border-amber-800/60 text-xs">
            <div className="text-amber-400 font-bold text-sm flex items-center justify-center gap-2">
              <span>⚠️</span>
              <span>Tactical Setups Telemetry Stale</span>
            </div>
            <p className="text-slate-300 max-w-lg mx-auto font-sans text-xs">
              Market history telemetry is older than 4 trading days. Tactical setups are pending fresh historical data from the exchange.
            </p>
          </div>
        ) : (
          <div className="p-6 text-center space-y-2 font-mono bg-[#111722] rounded-xl border border-[#243044] text-xs">
            <div className="text-slate-300 font-bold text-sm flex items-center justify-center gap-2">
              <span>🎯</span>
              <span>Zero Active Setups Meeting Spotlight Thresholds</span>
            </div>
            <p className="text-slate-400 max-w-md mx-auto font-sans text-xs">
              No market assets currently meet the {isDayTrader ? "RVOL >= 1.3, ATR momentum" : "Minervini Stage 2, Confluence, R:R >= 1.85:1"} criteria in this market regime.
            </p>
          </div>
        )
      )}
    </section>
  );
}
