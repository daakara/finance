"use client";

// Storage architecture: AUTHORITATIVE API PERSISTENCE with local fallback (FastAPI sync)

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";
import PageIntro from "../../components/PageIntro";
import {
  PortfolioPosition,
  PortfolioSummary,
  loadPortfolioPositions,
  savePortfolioPositions,
  calculatePortfolioSummary,
  getAnonymousUserId,
  exportPortfolioToCsv,
  syncPortfolioFromApi,
  addPortfolioPosition,
  updatePortfolioPosition,
  removePortfolioPosition,
  beginActivePortfolioEdit,
  endActivePortfolioEdit,
} from "../../lib/portfolio";
import { SHARED_FACTOR_SCORES } from "../../lib/constants";
import { fetchAssetAnalytics, SpotPriceRegistry, recordTradeExit, recordTradeClose } from "../../lib/api";
import {
  validateExitParams,
  generateIdempotencyKey,
  calculateRealizedPnL,
  calculateRealizedR,
} from "../../lib/tradeLifecycle";
import { getPersistedMarketSnapshot } from "../../lib/marketDatabase";
import { MASTER_ASSET_CATALOG, getMasterBaselinePrice } from "../../lib/masterCatalog";
import { resolveAssetAlias, getCanonicalAssetName } from "../../lib/assetRegistry";
import { trackMatomoEvent } from "../../lib/matomo";
import MacroStressTestSimulator from "../../components/MacroStressTestSimulator";

export default function PortfolioPage() {
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary>({
    totalEquity: 0,
    totalCost: 0,
    totalUnrealizedPnL: 0,
    totalUnrealizedPnLPct: 0,
    positionsCount: 0,
    isComplete: true,
    unpricedCount: 0,
  });
  const [showAddModal, setShowAddModal] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);
  const [anonId, setAnonId] = useState<string>("");
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastSyncTime, setLastSyncTime] = useState<string>("");
  const [activeSymbol, setActiveSymbol] = useState<string | null>(null);

  // Form State for Adding Position with Real-Time Auto-Population
  const [newSymbol, setNewSymbol] = useState("SEDG");
  const [newShares, setNewShares] = useState("75");
  const [newEntryPrice, setNewEntryPrice] = useState("33.51");
  const [newStopLoss, setNewStopLoss] = useState("31.16");
  const [newTarget, setNewTarget] = useState("41.89");
  const [isResolvingQuote, setIsResolvingQuote] = useState(false);
  const [resolvedAssetName, setResolvedAssetName] = useState("SolarEdge Technologies");
  const [resolvedQuotePrice, setResolvedQuotePrice] = useState<number | null>(33.51);

  // Exit / Close Position Modal State
  const [showExitModal, setShowExitModal] = useState(false);
  const [exitTargetPosition, setExitTargetPosition] = useState<PortfolioPosition | null>(null);
  const [exitMode, setExitMode] = useState<"FULL" | "PARTIAL">("FULL");
  const [exitShares, setExitShares] = useState<string>("");
  const [exitPrice, setExitPrice] = useState<string>("");
  const [exitDate, setExitDate] = useState<string>("");
  const [exitFollowedRules, setExitFollowedRules] = useState<boolean | null>(null);
  const [exitNotes, setExitNotes] = useState<string>("");
  const [exitSubmitting, setExitSubmitting] = useState(false);
  const [exitError, setExitError] = useState<string | null>(null);
  const [exitSuccess, setExitSuccess] = useState<boolean>(false);
  const [exitResultSummary, setExitResultSummary] = useState<{
    realizedPnL: number;
    returnPct: number;
    exitType: string;
  } | null>(null);

  const populateTickerData = useCallback(async (rawTicker: string) => {
    const trimmed = rawTicker.trim();
    const aliasInfo = resolveAssetAlias(trimmed);
    const symKey = (aliasInfo ? aliasInfo.canonicalTicker : trimmed).toUpperCase();
    const canonicalName = aliasInfo?.companyName || getCanonicalAssetName(symKey);
    setResolvedAssetName(canonicalName);
    setIsResolvingQuote(true);

    try {
      // 1. Fetch freshest live exchange analytics
      let price: number | null = null;
      try {
        const analytics = await fetchAssetAnalytics(symKey, "1mo", "1d");
        if (analytics?.currentPrice && !isNaN(analytics.currentPrice) && analytics.currentPrice > 0) {
          price = analytics.currentPrice;
        }
      } catch (e) {
        const reg = SpotPriceRegistry.get(symKey);
        const snap = getPersistedMarketSnapshot(symKey);
        const baseline = getMasterBaselinePrice(symKey, 0);
        if (reg?.price && reg.price > 0) price = reg.price;
        else if (snap?.currentPrice && snap.currentPrice > 0) price = snap.currentPrice;
        else if (baseline !== undefined && baseline > 0) price = baseline;
      }

      if (price && price > 0) {
        setResolvedQuotePrice(price);
        setNewEntryPrice(price.toFixed(2));
        setNewStopLoss((price * 0.93).toFixed(2));
        setNewTarget((price * 1.25).toFixed(2));
        setNewShares((prev) => (prev && Number(prev) > 0 ? prev : "10"));
      } else {
        setResolvedQuotePrice(null);
        setNewEntryPrice("");
        setNewStopLoss("");
        setNewTarget("");
        setNewShares("10");
      }
    } catch (err) {
      console.warn("Failed to auto-populate ticker data:", err);
    } finally {
      setIsResolvingQuote(false);
    }
  }, []);

  const handleOpenAddModal = (initialSymbol?: string) => {
    beginActivePortfolioEdit();
    setIsEditing(false);
    setModalError(null);
    const target = initialSymbol || "SEDG";
    setNewSymbol(target);
    setNewShares("10");
    setShowAddModal(true);
    populateTickerData(target);
  };

  const handleOpenEditModal = (pos: PortfolioPosition) => {
    beginActivePortfolioEdit();
    setIsEditing(true);
    setModalError(null);
    setNewSymbol(pos.symbol);
    setNewShares(pos.shares.toString());
    setNewEntryPrice(pos.entryPrice.toString());
    setNewStopLoss(pos.stopLossPrice ? pos.stopLossPrice.toString() : "");
    setNewTarget(pos.targetPrice ? pos.targetPrice.toString() : "");
    setResolvedAssetName(pos.name);
    setResolvedQuotePrice(pos.currentPrice);
    setShowAddModal(true);
  };

  const handleCloseModal = () => {
    endActivePortfolioEdit();
    setShowAddModal(false);
    setModalError(null);
  };

  const refreshQuotes = useCallback(async (basePositions: PortfolioPosition[]) => {
    if (basePositions.length === 0) return;
    setIsRefreshing(true);
    try {
      const updatedPromises = basePositions.map(async (pos) => {
        try {
          const res = await fetchAssetAnalytics(pos.symbol, "1mo", "1d");
          if (res && res.currentPrice && !isNaN(res.currentPrice) && res.currentPrice > 0) {
            return {
              ...pos,
              currentPrice: res.currentPrice,
            };
          }
        } catch {
          // Keep existing verified position price
        }
        return pos;
      });

      const resolved = await Promise.all(updatedPromises);
      setPositions(resolved);
      setSummary(calculatePortfolioSummary(resolved));
      savePortfolioPositions(resolved);
      setLastSyncTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    } catch (err) {
      console.warn("Failed to refresh live portfolio quotes:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    setAnonId(getAnonymousUserId());
    const loaded = loadPortfolioPositions();
    setPositions(loaded);
    setSummary(calculatePortfolioSummary(loaded));
    if (loaded.length > 0) {
      refreshQuotes(loaded);
    }

    // Sync authoritative API holdings in background
    syncPortfolioFromApi().then((synced) => {
      if (synced && synced.length > 0) {
        setPositions(synced);
        setSummary(calculatePortfolioSummary(synced));
        refreshQuotes(synced);
      }
    }).catch((err) => {
      console.warn("Portfolio API sync error:", err);
    });

    // Parse search parameters: preserve activeSymbol without opening add modal
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const sym = params.get("symbol") || params.get("ticker");
      if (sym) {
        setActiveSymbol(sym.trim().toUpperCase());
      }
      const addSym = params.get("add");
      if (addSym) {
        handleOpenAddModal(addSym);
      }
    }

    const handlePortfolioUpdate = () => {
      const refreshed = loadPortfolioPositions();
      setPositions(refreshed);
      setSummary(calculatePortfolioSummary(refreshed));
      refreshQuotes(refreshed);
    };

    const handlePurge = () => {
      refreshQuotes(loaded);
    };

    window.addEventListener("finance:cache-purge", handlePurge);
    window.addEventListener("finance:portfolio-updated", handlePortfolioUpdate);
    return () => {
      window.removeEventListener("finance:cache-purge", handlePurge);
      window.removeEventListener("finance:portfolio-updated", handlePortfolioUpdate);
    };
  }, [refreshQuotes]);

  const handleSaveHolding = async (e: React.FormEvent) => {
    e.preventDefault();
    setModalError(null);
    const trimmedSym = newSymbol.trim().toUpperCase();
    const aliasInfo = resolveAssetAlias(trimmedSym);
    const symUpper = (aliasInfo ? aliasInfo.canonicalTicker : trimmedSym).toUpperCase();

    if (!symUpper) {
      setModalError("Please specify a valid stock symbol or ticker.");
      return;
    }

    const sharesNum = parseFloat(newShares);
    if (isNaN(sharesNum) || sharesNum <= 0) {
      setModalError("Holding quantity must be a positive number greater than 0 (e.g. 0.25, 0.001, 10). Zero or negative quantities are not permitted.");
      return;
    }

    const entryNum = parseFloat(newEntryPrice);
    if (isNaN(entryNum) || entryNum <= 0) {
      setModalError("Entry price must be a positive number greater than $0.00.");
      return;
    }

    const stopNum = newStopLoss ? parseFloat(newStopLoss) : undefined;
    if (newStopLoss && (isNaN(stopNum!) || stopNum! <= 0)) {
      setModalError("Stop loss must be a positive price if specified.");
      return;
    }

    const targetNum = newTarget ? parseFloat(newTarget) : undefined;
    if (newTarget && (isNaN(targetNum!) || targetNum! <= 0)) {
      setModalError("Target price must be a positive price if specified.");
      return;
    }

    const authenticName = resolvedAssetName || getCanonicalAssetName(symUpper);
    const curPrice = (resolvedQuotePrice && !isNaN(resolvedQuotePrice) && resolvedQuotePrice > 0) ? resolvedQuotePrice : null;

    let res: { success: boolean; message: string; isDuplicate?: boolean };
    if (isEditing) {
      res = await updatePortfolioPosition({
        symbol: symUpper,
        name: authenticName,
        shares: sharesNum,
        entryPrice: entryNum,
        currentPrice: curPrice,
        targetPrice: targetNum,
        stopLossPrice: stopNum,
      });
    } else {
      res = await addPortfolioPosition({
        symbol: symUpper,
        name: authenticName,
        shares: sharesNum,
        entryPrice: entryNum,
        currentPrice: curPrice,
        targetPrice: targetNum,
        stopLossPrice: stopNum,
      });
    }

    if (!res.success) {
      setModalError(res.message);
      return;
    }

    const refreshed = loadPortfolioPositions();
    setPositions(refreshed);
    setSummary(calculatePortfolioSummary(refreshed));
    endActivePortfolioEdit();
    setShowAddModal(false);
    setModalError(null);

    trackMatomoEvent("User Journey", isEditing ? "Edit Portfolio Position" : "Add Portfolio Position", `${symUpper} (${sharesNum} shares)`);
  };

  const handleRemovePosition = async (symbol: string) => {
    const res = await removePortfolioPosition(symbol);
    if (!res.success) {
      alert(res.message);
      return;
    }
    const refreshed = loadPortfolioPositions();
    setPositions(refreshed);
    setSummary(calculatePortfolioSummary(refreshed));
    trackMatomoEvent("User Journey", "Remove Portfolio Position", symbol);
  };

  const handleOpenExitModal = (pos: PortfolioPosition) => {
    setExitTargetPosition(pos);
    setExitMode("FULL");
    setExitShares(pos.shares.toString());
    const defaultPrice = (pos.currentPrice && !isNaN(pos.currentPrice) && pos.currentPrice > 0)
      ? pos.currentPrice.toFixed(2)
      : pos.entryPrice.toFixed(2);
    setExitPrice(defaultPrice);
    setExitDate(new Date().toISOString().slice(0, 10));
    setExitFollowedRules(null);
    setExitNotes("");
    setExitError(null);
    setExitSuccess(false);
    setExitResultSummary(null);
    setShowExitModal(true);
  };

  const handleCloseExitModal = () => {
    setShowExitModal(false);
    setExitTargetPosition(null);
    setExitError(null);
    setExitSuccess(false);
    setExitResultSummary(null);
  };

  useEffect(() => {
    if (!showAddModal && !showExitModal) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (showAddModal) handleCloseModal();
        if (showExitModal) handleCloseExitModal();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [showAddModal, showExitModal]);

  const handleQuickSharesFraction = (fraction: number) => {
    if (!exitTargetPosition) return;
    const targetShares = Number((exitTargetPosition.shares * fraction).toFixed(6));
    setExitShares(targetShares.toString());
    if (fraction === 1) {
      setExitMode("FULL");
    } else {
      setExitMode("PARTIAL");
    }
  };

  const handleSubmitExit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!exitTargetPosition) return;

    setExitError(null);
    const sharesNum = parseFloat(exitShares);
    const priceNum = parseFloat(exitPrice);

    const valResult = validateExitParams(exitTargetPosition.shares, sharesNum, priceNum);
    if (!valResult.valid) {
      setExitError(valResult.error || "Invalid exit parameters.");
      return;
    }

    setExitSubmitting(true);
    const isFull = Math.abs(sharesNum - exitTargetPosition.shares) < 1e-6 || exitMode === "FULL";
    const idemKey = generateIdempotencyKey(isFull ? "close" : "exit", exitTargetPosition.symbol, anonId || "anon");

    try {
      let res;
      if (isFull) {
        res = await recordTradeClose({
          symbol: exitTargetPosition.symbol,
          exitPrice: priceNum,
          exitDate: exitDate || new Date().toISOString().slice(0, 10),
          followedRules: exitFollowedRules === null ? undefined : exitFollowedRules,
          idempotencyKey: idemKey,
          notes: exitNotes.trim() || undefined,
        }, anonId);
      } else {
        res = await recordTradeExit({
          symbol: exitTargetPosition.symbol,
          shares: sharesNum,
          exitPrice: priceNum,
          exitDate: exitDate || new Date().toISOString().slice(0, 10),
          followedRules: exitFollowedRules === null ? undefined : exitFollowedRules,
          idempotencyKey: idemKey,
          notes: exitNotes.trim() || undefined,
        }, anonId);
      }

      if (!res) {
        setExitError("Failed to record exit. Server rejected or returned an error.");
        setExitSubmitting(false);
        return;
      }

      const pnl = calculateRealizedPnL(exitTargetPosition.entryPrice, priceNum, sharesNum);
      const retPct = ((priceNum - exitTargetPosition.entryPrice) / exitTargetPosition.entryPrice) * 100;

      // Update local storage portfolio to reflect change
      if (isFull) {
        await removePortfolioPosition(exitTargetPosition.symbol);
      } else {
        const remaining = exitTargetPosition.shares - sharesNum;
        await updatePortfolioPosition({
          ...exitTargetPosition,
          shares: Number(remaining.toFixed(6)),
        });
      }

      const refreshed = loadPortfolioPositions();
      setPositions(refreshed);
      setSummary(calculatePortfolioSummary(refreshed));

      setExitSuccess(true);
      setExitResultSummary({
        realizedPnL: pnl,
        returnPct: Number(retPct.toFixed(2)),
        exitType: isFull ? "Full Close (100%)" : `Partial Scale-Out (${sharesNum} shares)`,
      });

      trackMatomoEvent(
        "User Journey",
        isFull ? "Full Close Trade" : "Partial Scale-Out Trade",
        `${exitTargetPosition.symbol} (${sharesNum} @ $${priceNum})`
      );

      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
      }
    } catch (err: any) {
      setExitError(err?.message || "An unexpected error occurred while recording exit.");
    } finally {
      setExitSubmitting(false);
    }
  };

  const [accountEquity, setAccountEquity] = useState<number>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("FINANCE_USER_ACCOUNT_SIZE");
      if (saved) {
        const parsed = Number(saved);
        if (!isNaN(parsed) && parsed > 0) return parsed;
      }
    }
    return 25000;
  });

  const handleAccountEquityChange = (val: number) => {
    const safeVal = Math.max(1, val);
    setAccountEquity(safeVal);
    if (typeof window !== "undefined") {
      localStorage.setItem("FINANCE_USER_ACCOUNT_SIZE", safeVal.toString());
      window.dispatchEvent(new Event("storage"));
    }
  };

  const handleExportCsv = () => {
    exportPortfolioToCsv(positions);
    trackMatomoEvent("User Journey", "Export Portfolio CSV", `Positions count: ${positions.length}`);
  };

  const effectiveCapital = Math.max(accountEquity, summary.totalCost);
  const investedEquity = summary.totalEquity;
  const cashReserves = Math.max(0, effectiveCapital - summary.totalCost);
  const totalNetWorth = investedEquity !== null ? cashReserves + investedEquity : null;
  const investedPct = totalNetWorth !== null && totalNetWorth > 0 && investedEquity !== null ? (investedEquity / totalNetWorth) * 100 : null;
  const cashPct = totalNetWorth !== null && totalNetWorth > 0 ? (cashReserves / totalNetWorth) * 100 : null;
  const isPositive = summary.totalUnrealizedPnL !== null ? summary.totalUnrealizedPnL >= 0 : null;

  // Level 0: Total Capital at Risk Calculation
  const totalRiskAtStop = positions.reduce((acc, p) => {
    const stop = p.stopLossPrice || p.entryPrice * 0.92;
    const currentOrEntry = p.entryPrice;
    return acc + Math.max(0, (currentOrEntry - stop) * p.shares);
  }, 0);
  const riskPctOfEquity = totalNetWorth !== null && totalNetWorth > 0 ? (totalRiskAtStop / totalNetWorth) * 100 : null;
  const stopBreaches = positions.filter((p) => p.currentPrice !== null && p.currentPrice <= (p.stopLossPrice || p.entryPrice * 0.92));
  const targetHits = positions.filter((p) => p.currentPrice !== null && !!p.targetPrice && p.currentPrice >= p.targetPrice);

  return (
    <TerminalShell activeHub="portfolio" activeSymbol={activeSymbol}>
      <main className="max-w-[1450px] mx-auto p-4 sm:p-6 space-y-6 pb-28 sm:pb-8">
        {/* Hub Guidance & Orientation (A3-AC1, A3-AC2, A3-AC6, Finding T02) */}
        <PageIntro
          hubId="portfolio"
          title="Portfolio"
          purpose="Monitor active capital at risk, protective stop floors, and current risk heat."
          badge="Live Risk Ledger"
          symbol={activeSymbol}
          primaryAction={{
            label: "Explore Setups →",
            href: "/setups",
          }}
          secondaryAction={{
            label: "Add Holding",
            onClick: () => handleOpenAddModal(),
          }}
        >
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => refreshQuotes(positions)}
              disabled={isRefreshing}
              className={`px-3 py-2 bg-[#162030] hover:bg-[#1f2d44] border border-[#243044] text-slate-200 rounded-xl text-xs font-mono font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer ${
                isRefreshing ? "opacity-60 cursor-not-allowed" : ""
              }`}
            >
              <span className={isRefreshing ? "animate-spin" : ""}>🔄</span>
              <span className="hidden sm:inline">{isRefreshing ? "Syncing Quotes..." : "Refresh Quotes"}</span>
            </button>

            <button
              type="button"
              onClick={handleExportCsv}
              className="px-3 py-2 bg-[#162030] hover:bg-[#1f2d44] border border-[#243044] text-slate-200 rounded-xl text-xs font-mono font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer"
            >
              <span>📥</span>
              <span className="hidden sm:inline">Export CSV</span>
            </button>
          </div>
        </PageIntro>

        {/* Level 0: Asymmetric Capital at Risk & Portfolio Heat Hero */}
        <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 p-5 md:p-6 shadow-2xl space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-bold bg-rose-950/80 text-rose-400 border border-rose-800/80">
                  Level 0 · Portfolio Heat
                </span>
                <span className="text-xs text-slate-400 font-sans">
                  What can hurt me if all stop floors trigger?
                </span>
              </div>
              <div className="flex items-baseline gap-3">
                <span className="text-3xl sm:text-4xl font-black font-mono text-rose-400 tabular-nums">
                  -${totalRiskAtStop.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                <span className="text-sm font-mono text-rose-300/80 font-bold">
                  ({riskPctOfEquity !== null ? `${riskPctOfEquity.toFixed(2)}% Capital at Risk` : "--"})
                </span>
              </div>
              <p className="text-xs text-slate-400 font-sans max-w-xl">
                Maximum portfolio exposure defined strictly by your stop-loss exit floors. Risk is distributed across {summary.positionsCount} active {summary.positionsCount === 1 ? 'position' : 'positions'}.
              </p>
            </div>

            {/* Right Rail: Total Capital & Deployment */}
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-2 gap-3 shrink-0 font-mono text-xs">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Total Net Worth</span>
                <span className="text-base font-bold text-white tabular-nums">
                  {totalNetWorth !== null ? `$${totalNetWorth.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "--"}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Active Holdings</span>
                <span className="text-base font-bold text-cyan-400 tabular-nums">
                  {investedEquity !== null ? `$${investedEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${investedPct?.toFixed(0)}%)` : <span className="text-amber-400">Incomplete ({summary.unpricedCount} unpriced)</span>}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Cash Buying Power</span>
                <span className="text-base font-bold text-emerald-400 tabular-nums">
                  ${cashReserves.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} {cashPct !== null ? `(${cashPct.toFixed(0)}%)` : ""}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Unrealized P&amp;L</span>
                <span className={`text-base font-bold tabular-nums ${isPositive === true ? 'text-emerald-400' : isPositive === false ? 'text-rose-400' : 'text-slate-400'}`}>
                  {summary.totalUnrealizedPnL !== null ? (isPositive ? `+$${summary.totalUnrealizedPnL.toFixed(2)}` : `-$${Math.abs(summary.totalUnrealizedPnL).toFixed(2)}`) : <span className="text-amber-400 font-normal text-xs">-- (Unpriced)</span>}
                </span>
              </div>
            </div>
          </div>

          {/* Active Exit Rule Triggers Banner */}
          {(stopBreaches.length > 0 || targetHits.length > 0) && (
            <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-800/60 flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="text-rose-400 font-bold">⚠️ ACTIVE EXIT TRIGGERS:</span>
                {stopBreaches.map((b) => (
                  <span key={b.symbol} className="px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800 font-bold text-[11px]">
                    STOP BREACH: {b.symbol} (${b.currentPrice} &le; ${b.stopLossPrice || (b.entryPrice * 0.92).toFixed(2)})
                  </span>
                ))}
                {targetHits.map((t) => (
                  <span key={t.symbol} className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold text-[11px]">
                    TARGET HIT: {t.symbol} (${t.currentPrice} &ge; ${t.targetPrice || (t.entryPrice * 1.2).toFixed(2)})
                  </span>
                ))}
              </div>
              <span className="text-[11px] text-slate-400 font-sans">Execute disciplined exit to preserve capital</span>
            </div>
          )}
        </div>

        {/* Interactive Asset & Cash Allocation Visualizer */}
        <div className="bg-[#0b101b] border border-[#1e2a3c] rounded-xl p-4 shadow-xl space-y-3 font-mono">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#182335] pb-2.5">
            <div className="flex items-center gap-2">
              <span className="text-base">💼</span>
              <span className="text-xs sm:text-sm font-bold text-white">
                Account Capital & Risk Allocation Breakdown
              </span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">Total Wallet:</span>
              <div className="flex items-center gap-1 bg-[#06090f] border border-[#24334b] rounded-lg px-2 py-1">
                <span className="text-xs text-slate-500 font-bold">$</span>
                <input
                  type="number"
                  min="1"
                  step="10"
                  value={accountEquity}
                  onChange={(e) => handleAccountEquityChange(Number(e.target.value))}
                  className="w-20 bg-transparent text-xs text-cyan-300 font-bold focus:outline-none"
                />
              </div>

              {/* Institutional Capital Presets */}
              <div className="hidden sm:flex items-center gap-1">
                {[10000, 25000, 50000, 100000, 250000].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => handleAccountEquityChange(preset)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border transition-all cursor-pointer ${
                      accountEquity === preset
                        ? "bg-cyan-600 border-cyan-400 text-white shadow-sm"
                        : "bg-[#0c121e] border-[#1f2c42] text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    ${preset / 1000}k
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Progress Stack Bar */}
          <div className="space-y-1.5">
            <div className="w-full h-3 bg-[#06090f] rounded-full overflow-hidden flex border border-[#1b2537]">
              {positions.map((pos, idx) => {
                const posVal = pos.currentPrice !== null ? pos.shares * pos.currentPrice : null;
                const pct = totalNetWorth !== null && totalNetWorth > 0 && posVal !== null ? (posVal / totalNetWorth) * 100 : 0;
                const colors = ["bg-cyan-500", "bg-emerald-500", "bg-indigo-500", "bg-amber-500", "bg-purple-500"];
                const color = colors[idx % colors.length];
                return (
                  <div
                    key={pos.symbol}
                    style={{ width: `${pct}%` }}
                    className={`${color} h-full transition-all duration-300`}
                    title={posVal !== null ? `${pos.symbol}: $${posVal.toFixed(2)} (${pct.toFixed(1)}%)` : `${pos.symbol}: Unpriced`}
                  />
                );
              })}
              <div
                style={{ width: `${cashPct ?? 0}%` }}
                className="bg-slate-700/60 h-full transition-all duration-300"
                title={`Available Cash: $${cashReserves.toFixed(2)} (${cashPct !== null ? cashPct.toFixed(1) : "--"}%)`}
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400">
              <div className="flex items-center gap-3 flex-wrap">
                {positions.map((pos, idx) => {
                  const posVal = pos.currentPrice !== null ? pos.shares * pos.currentPrice : null;
                  const pct = totalNetWorth !== null && totalNetWorth > 0 && posVal !== null ? (posVal / totalNetWorth) * 100 : 0;
                  const dotColors = ["bg-cyan-400", "bg-emerald-400", "bg-indigo-400", "bg-amber-400", "bg-purple-400"];
                  const dotColor = dotColors[idx % dotColors.length];
                  return (
                    <span key={pos.symbol} className="flex items-center gap-1">
                      <span className={`w-2 h-2 rounded-full ${dotColor}`} />
                      <strong className="text-slate-200">{pos.symbol}:</strong>
                      <span>{posVal !== null ? `$${posVal.toFixed(2)} (${pct.toFixed(1)}%)` : "Unpriced"}</span>
                    </span>
                  );
                })}
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-slate-500" />
                  <strong className="text-slate-300">Cash Reserves:</strong>
                  <span>${cashReserves.toFixed(2)} ({cashPct !== null ? `${cashPct.toFixed(1)}%` : "--"})</span>
                </span>
              </div>

              <span className="text-slate-500 hidden md:inline">
                {positions.length > 0
                  ? `Sized at 1% risk per trade. Preserves $${cashReserves.toFixed(2)} cash balance.`
                  : "Add positions from the Position Sizer to see live allocation."}
              </span>
            </div>
          </div>
        </div>

        {/* Positions Table */}
        <div className="bg-[#111722] border border-[#243044] rounded-xl shadow-xl overflow-hidden">
          <div className="p-4 border-b border-[#1b2434] flex items-center justify-between">
            <h2 className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center gap-2">
              <span>📊</span>
              <span>Open Quant Holdings</span>
            </h2>
            <span className="text-xs text-slate-400">Click any symbol to open in Terminal</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#090d14] text-slate-400 border-b border-[#1b2434] uppercase text-[10px]">
                <tr>
                  <th className="py-3 px-4">Asset</th>
                  <th className="py-3 px-4">Execution Status</th>
                  <th className="py-3 px-4">Shares</th>
                  <th className="py-3 px-4">Entry Price</th>
                  <th className="py-3 px-4">Current Price</th>
                  <th className="py-3 px-4">Market Value</th>
                  <th className="py-3 px-4">Unrealized P&L</th>
                  <th className="py-3 px-4">Risk Ladder</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1b2434] font-medium tabular-nums">
                {positions.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-12 px-4 text-center">
                      <div className="max-w-md mx-auto space-y-3 font-mono">
                        <div className="w-12 h-12 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-2xl">
                          💼
                        </div>
                        <div className="space-y-1">
                          <h3 className="text-sm font-bold text-slate-200 font-mono">No Active Portfolio Holdings Recorded</h3>
                          <p className="text-xs text-slate-400 font-sans leading-relaxed">
                            No open positions are currently recorded. Holdings appear here automatically when you record an execution fill in Setups, or you can add an existing position manually.
                          </p>
                        </div>
                        <div className="flex items-center justify-center gap-2.5 pt-2">
                          <Link
                            href="/setups"
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold font-sans transition-transform active:scale-95 shadow-lg shadow-cyan-950/50 cursor-pointer"
                          >
                            <span>Explore Setups →</span>
                          </Link>
                          <button
                            type="button"
                            onClick={() => handleOpenAddModal()}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold font-sans transition-colors border border-slate-700 cursor-pointer"
                          >
                            <span>➕ Add Manual Holding</span>
                          </button>
                        </div>
                      </div>
                    </td>
                  </tr>
                ) : (
                  positions.map((pos) => {
                  const isPriced = pos.currentPrice !== null && !isNaN(pos.currentPrice) && pos.currentPrice > 0;
                  const mktVal = isPriced ? pos.shares * pos.currentPrice! : null;
                  const cost = pos.shares * pos.entryPrice;
                  const pnl = isPriced && mktVal !== null ? mktVal - cost : null;
                  const pnlPct = isPriced && pnl !== null && cost > 0 ? (pnl / cost) * 100 : null;
                  const posUp = pnl !== null ? pnl >= 0 : null;

                  // Execution state alert
                  let statusBadge = null;
                  if (!isPriced) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800/60 font-bold text-[10px] whitespace-nowrap">
                        UNPRICED
                      </span>
                    );
                  } else if (pos.targetPrice && pos.currentPrice !== null && pos.currentPrice >= pos.targetPrice) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold text-[10px] whitespace-nowrap">
                        🎯 TP1 TARGET HIT
                      </span>
                    );
                  } else if (pos.stopLossPrice && pos.currentPrice !== null && pos.currentPrice <= pos.stopLossPrice) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800 font-bold text-[10px] whitespace-nowrap">
                        🛑 STOP LOSS HIT
                      </span>
                    );
                  } else if (pos.stopLossPrice && pos.currentPrice !== null && pos.currentPrice <= pos.stopLossPrice * 1.02) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 font-bold text-[10px] whitespace-nowrap">
                        ⚠️ NEAR STOP FLOOR
                      </span>
                    );
                  } else {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-800/60 font-bold text-[10px] whitespace-nowrap">
                        🟢 ACTIVE HOLDING
                      </span>
                    );
                  }

                  return (
                    <tr
                      key={pos.symbol}
                      className={`hover:bg-[#151e2d] transition-colors ${
                        activeSymbol && pos.symbol.toUpperCase() === activeSymbol.toUpperCase()
                          ? "bg-cyan-950/40 ring-1 ring-cyan-500/50"
                          : ""
                      }`}
                    >
                      <td className="py-3 px-4">
                        <Link href={`/?symbol=${pos.symbol}&ownership=OWNED`} className="font-bold text-cyan-400 hover:text-cyan-300 text-sm flex items-center gap-1.5">
                          <span>{pos.symbol}</span>
                          <span className="text-[10px] text-slate-500 font-normal">({pos.name})</span>
                        </Link>
                      </td>
                      <td className="py-3 px-4">{statusBadge}</td>
                      <td className="py-3 px-4 text-slate-200">{typeof pos.shares === "number" ? Number(pos.shares.toFixed(6)) : pos.shares}</td>
                      <td className="py-3 px-4 text-slate-300">${pos.entryPrice.toFixed(2)}</td>
                      <td className="py-3 px-4 text-white font-bold">
                        {isPriced ? `$${pos.currentPrice!.toFixed(2)}` : <span className="text-amber-400 font-mono text-[11px] font-bold">Unpriced</span>}
                      </td>
                      <td className="py-3 px-4 text-slate-100 font-bold">
                        {mktVal !== null ? `$${mktVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : <span className="text-slate-500 font-mono">--</span>}
                      </td>
                      <td className="py-3 px-4">
                        {pnl !== null ? (
                          <span className={`font-bold ${posUp ? "text-emerald-400" : "text-rose-400"}`}>
                            {posUp ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`} ({posUp ? `+${pnlPct!.toFixed(2)}%` : `${pnlPct!.toFixed(2)}%`})
                          </span>
                        ) : (
                          <span className="text-slate-500 font-mono">--</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-[11px]">
                        <span className="text-rose-400">Stop: ${pos.stopLossPrice?.toFixed(2) || "None"}</span>
                        <span className="text-slate-600 mx-1">|</span>
                        <span className="text-emerald-400">Target: ${pos.targetPrice?.toFixed(2) || "None"}</span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            type="button"
                            onClick={() => handleOpenExitModal(pos)}
                            className="px-2.5 py-1 text-[11px] rounded bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-800/80 transition-colors cursor-pointer font-bold"
                          >
                            Record Exit
                          </button>
                          <button
                            type="button"
                            onClick={() => handleOpenEditModal(pos)}
                            className="px-2.5 py-1 text-[11px] rounded bg-slate-800 hover:bg-cyan-600 hover:text-white text-cyan-300 border border-slate-700 transition-colors cursor-pointer"
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            onClick={() => handleRemovePosition(pos.symbol)}
                            className="px-2.5 py-1 text-[11px] rounded bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800/80 transition-colors cursor-pointer"
                          >
                            Remove
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                }))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 🌪️ MACRO STRESS-TEST & SCENARIO SIMULATOR */}
        <section aria-label="Macro Stress-Test Simulator">
          <MacroStressTestSimulator positions={positions} totalEquity={summary.totalEquity} />
        </section>

        {/* Add Position Modal (Fluid Adaptive & Real-Time Auto-Populated) */}
        {showAddModal && (
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="add-position-modal-title"
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[1200] flex items-center justify-center p-2 sm:p-4 overflow-y-auto font-mono"
          >
            <div className="bg-[#111722] border border-[#243044] rounded-2xl max-w-md w-full shadow-2xl overflow-hidden max-h-[92vh] flex flex-col my-auto text-slate-100">
              {/* Fixed Header */}
              <div className="flex items-center justify-between p-4 border-b border-[#1b2434] bg-[#0e1422] shrink-0">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">💼</span>
                  <div>
                    <h3 id="add-position-modal-title" className="text-base font-bold text-white tracking-tight">
                      {isEditing ? "Edit Portfolio Holding" : "Add Portfolio Holding"}
                    </h3>
                    <p className="text-[10px] text-slate-400">Live quantitative auto-population enabled</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleCloseModal}
                  className="focus-ring text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
                  aria-label="Close modal"
                >
                  ✕
                </button>
              </div>

              {/* Scrollable Form Body */}
              <form onSubmit={handleSaveHolding} className="p-4 sm:p-5 space-y-3.5 overflow-y-auto flex-1 text-xs">
                {/* Ticker Input & Quick Chips */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label htmlFor="add-ticker-input" className="block text-slate-300 font-bold">Ticker Symbol</label>
                    <span className="text-[10px] text-cyan-400 font-mono">
                      {isResolvingQuote ? "⏳ Syncing Quote..." : `Verified: ${resolvedAssetName}`}
                    </span>
                  </div>
                  <input
                    id="add-ticker-input"
                    type="text"
                    value={newSymbol}
                    onChange={(e) => {
                      const val = e.target.value.toUpperCase();
                      setNewSymbol(val);
                      populateTickerData(val);
                    }}
                    className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2.5 text-white font-bold tracking-wider uppercase text-sm focus:outline-none"
                    placeholder="e.g. SEDG, NVDA, AAPL, FDX"
                    required
                  />

                  {/* Quick Ticker Chips */}
                  <div className="flex flex-wrap items-center gap-1.5 mt-2">
                    <span className="text-[10px] text-slate-500 font-bold mr-0.5">Quick:</span>
                    {["SEDG", "NVDA", "AAPL", "TSLA", "MSFT", "FDX", "UPS", "DHLGY"].map((sym) => (
                      <button
                        type="button"
                        key={sym}
                        onClick={() => {
                          setNewSymbol(sym);
                          populateTickerData(sym);
                        }}
                        className={`focus-ring px-2 py-0.5 rounded text-[10px] font-bold border transition-all cursor-pointer ${
                          newSymbol.toUpperCase() === sym
                            ? "bg-cyan-600 border-cyan-400 text-white"
                            : "bg-[#090d14] border-[#1b2537] text-slate-400 hover:text-white"
                        }`}
                      >
                        {sym}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Auto-Populated Live Quote Banner */}
                <div className="bg-[#090d14] border border-cyan-900/60 p-2.5 rounded-xl flex items-center justify-between gap-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="text-cyan-400">📡</span>
                    <div>
                      <span className="text-[11px] font-bold text-white">
                        {resolvedAssetName} ({newSymbol})
                      </span>
                      <p className="text-[10px] text-slate-400">
                        Market Price: <span className="text-cyan-300 font-bold tabular-nums">${resolvedQuotePrice?.toFixed(2) || newEntryPrice}</span>
                      </p>
                    </div>
                  </div>
                  <span className="text-[9px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold uppercase">
                    Auto-Filled
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label htmlFor="add-shares-input" className="block text-slate-300 font-bold mb-1">Shares Count</label>
                    <input
                      id="add-shares-input"
                      type="number"
                      step="any"
                      min="0.000001"
                      value={newShares}
                      onChange={(e) => setNewShares(e.target.value)}
                      className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                      required
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">Supports fractional quantities (e.g. 0.25, 0.001)</span>
                  </div>
                  <div>
                    <label htmlFor="add-entry-price-input" className="block text-slate-300 font-bold mb-1">Entry Price ($)</label>
                    <input
                      id="add-entry-price-input"
                      type="number"
                      step="any"
                      value={newEntryPrice}
                      onChange={(e) => setNewEntryPrice(e.target.value)}
                      className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                      required
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">Live market execution</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label htmlFor="add-stop-loss-input" className="block text-rose-300 font-bold mb-1">Stop Loss ($)</label>
                    <input
                      id="add-stop-loss-input"
                      type="number"
                      step="any"
                      value={newStopLoss}
                      onChange={(e) => setNewStopLoss(e.target.value)}
                      className="focus-ring w-full bg-[#090d14] border border-rose-950/80 focus:border-rose-500 rounded-lg p-2 text-rose-300 font-bold focus:outline-none"
                    />
                    <span className="text-[10px] text-rose-400/80 block mt-0.5">-7% Risk Cut Floor</span>
                  </div>
                  <div>
                    <label htmlFor="add-target-input" className="block text-emerald-300 font-bold mb-1">Target Price ($)</label>
                    <input
                      id="add-target-input"
                      type="number"
                      step="any"
                      value={newTarget}
                      onChange={(e) => setNewTarget(e.target.value)}
                      className="focus-ring w-full bg-[#090d14] border border-emerald-950/80 focus:border-emerald-500 rounded-lg p-2 text-emerald-300 font-bold focus:outline-none"
                    />
                    <span className="text-[10px] text-emerald-400/80 block mt-0.5">+25% Upside Target (TP1)</span>
                  </div>
                </div>

                {/* Fixed Footer with Actions */}
                <div className="pt-3.5 flex items-center justify-end space-x-2 border-t border-[#1b2434] shrink-0">
                  <button
                    type="button"
                    onClick={handleCloseModal}
                    className="focus-ring px-3.5 py-1.5 bg-[#162030] hover:bg-[#1e2a3c] text-slate-300 rounded-lg font-bold transition-colors cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="focus-ring px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg shadow transition-all active:scale-95 cursor-pointer"
                  >
                    {isEditing ? "Update Holding" : "Save Holding"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Record Exit / Scale-Out Modal */}
        {showExitModal && exitTargetPosition && (
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="exit-modal-title"
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[1200] flex items-center justify-center p-2 sm:p-4 overflow-y-auto font-mono"
          >
            <div className="bg-[#111722] border border-[#243044] rounded-2xl max-w-lg w-full shadow-2xl overflow-hidden max-h-[92vh] flex flex-col my-auto text-slate-100">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-[#1b2434] bg-[#0e1422] shrink-0">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">🎯</span>
                  <div>
                    <h3 id="exit-modal-title" className="text-base font-bold text-white tracking-tight">Record Trade Exit / Scale-Out</h3>
                    <p className="text-[10px] text-slate-400">
                      {exitTargetPosition.symbol} · {exitTargetPosition.shares} shares @ ${exitTargetPosition.entryPrice.toFixed(2)}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleCloseExitModal}
                  className="focus-ring text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
                  aria-label="Close exit modal"
                >
                  ✕
                </button>
              </div>

              {exitSuccess && exitResultSummary ? (
                /* Success Card */
                <div className="p-6 space-y-4 text-center">
                  <div className="w-12 h-12 rounded-full bg-emerald-950 border border-emerald-800 flex items-center justify-center mx-auto text-2xl">
                    ✅
                  </div>
                  <div>
                    <h4 className="text-base font-bold text-white font-mono">Trade Exit Recorded</h4>
                    <p className="text-xs text-slate-400 mt-1">
                      {exitResultSummary.exitType} on {exitTargetPosition.symbol} has been recorded to your persistent Journal.
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-[#090d14] border border-[#1b2434] space-y-2 max-w-xs mx-auto text-left text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Realized P&amp;L:</span>
                      <span className={`font-bold ${exitResultSummary.realizedPnL >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                        {exitResultSummary.realizedPnL >= 0 ? `+$${exitResultSummary.realizedPnL.toFixed(2)}` : `-$${Math.abs(exitResultSummary.realizedPnL).toFixed(2)}`}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Return:</span>
                      <span className={`font-bold ${exitResultSummary.returnPct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                        {exitResultSummary.returnPct >= 0 ? `+${exitResultSummary.returnPct.toFixed(2)}%` : `${exitResultSummary.returnPct.toFixed(2)}%`}
                      </span>
                    </div>
                  </div>

                  <div className="pt-2 flex items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={handleCloseExitModal}
                      className="focus-ring px-4 py-2 bg-[#162030] hover:bg-[#1e2a3c] text-slate-300 rounded-lg text-xs font-bold transition-colors cursor-pointer"
                    >
                      Close
                    </button>
                    <Link
                      href="/journal"
                      className="focus-ring px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-bold transition-colors cursor-pointer"
                    >
                      View Journal →
                    </Link>
                  </div>
                </div>
              ) : (
                /* Exit Form */
                <form onSubmit={handleSubmitExit} className="p-4 sm:p-5 space-y-3.5 overflow-y-auto flex-1 text-xs">
                  {/* Mode Selector */}
                  <div className="flex items-center gap-2 p-1 bg-[#090d14] border border-[#1b2434] rounded-lg">
                    <button
                      type="button"
                      onClick={() => {
                        setExitMode("FULL");
                        setExitShares(exitTargetPosition.shares.toString());
                      }}
                      className={`focus-ring flex-1 py-1.5 rounded text-xs font-bold transition-all cursor-pointer ${
                        exitMode === "FULL"
                          ? "bg-cyan-600 text-white shadow-sm"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Full Close (100%)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setExitMode("PARTIAL");
                        setExitShares((exitTargetPosition.shares * 0.5).toFixed(4));
                      }}
                      className={`focus-ring flex-1 py-1.5 rounded text-xs font-bold transition-all cursor-pointer ${
                        exitMode === "PARTIAL"
                          ? "bg-cyan-600 text-white shadow-sm"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Partial Scale-Out
                    </button>
                  </div>

                  {/* Quick Chips for Scale-Out */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[11px] text-slate-400 font-bold">Quick Fraction:</span>
                      <span className="text-[10px] text-slate-500">Max: {exitTargetPosition.shares} shares</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1.5">
                      {[
                        { label: "25%", frac: 0.25 },
                        { label: "50%", frac: 0.5 },
                        { label: "75%", frac: 0.75 },
                        { label: "100%", frac: 1.0 },
                      ].map(({ label, frac }) => (
                        <button
                          key={label}
                          type="button"
                          onClick={() => handleQuickSharesFraction(frac)}
                          className="focus-ring px-2 py-1 rounded bg-[#090d14] border border-[#1b2537] text-slate-300 hover:border-cyan-400 hover:text-white text-xs font-bold transition-all cursor-pointer"
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Quantity and Exit Price */}
                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label htmlFor="exit-shares-input" className="block text-slate-300 font-bold mb-1">Shares to Exit</label>
                      <input
                        id="exit-shares-input"
                        type="number"
                        step="any"
                        min="0.000001"
                        max={exitTargetPosition.shares}
                        value={exitShares}
                        onChange={(e) => {
                          setExitShares(e.target.value);
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val) && Math.abs(val - exitTargetPosition.shares) < 1e-6) {
                            setExitMode("FULL");
                          } else {
                            setExitMode("PARTIAL");
                          }
                        }}
                        className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="exit-price-input" className="block text-slate-300 font-bold mb-1">Exit Price ($)</label>
                      <input
                        id="exit-price-input"
                        type="number"
                        step="any"
                        min="0.01"
                        value={exitPrice}
                        onChange={(e) => setExitPrice(e.target.value)}
                        className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                        required
                      />
                    </div>
                  </div>

                  {/* Date and Rule Adherence */}
                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label htmlFor="exit-date-input" className="block text-slate-300 font-bold mb-1">Exit Date</label>
                      <input
                        id="exit-date-input"
                        type="date"
                        value={exitDate}
                        onChange={(e) => setExitDate(e.target.value)}
                        className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-slate-300 font-bold mb-1">Followed Plan Rules?</label>
                      <div className="grid grid-cols-3 gap-1">
                        <button
                          type="button"
                          onClick={() => setExitFollowedRules(true)}
                          className={`focus-ring py-2 rounded text-[10px] font-bold border transition-all cursor-pointer ${
                            exitFollowedRules === true
                              ? "bg-emerald-950 border-emerald-500 text-emerald-300"
                              : "bg-[#090d14] border-[#1b2537] text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          onClick={() => setExitFollowedRules(false)}
                          className={`focus-ring py-2 rounded text-[10px] font-bold border transition-all cursor-pointer ${
                            exitFollowedRules === false
                              ? "bg-rose-950 border-rose-500 text-rose-300"
                              : "bg-[#090d14] border-[#1b2537] text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          No
                        </button>
                        <button
                          type="button"
                          onClick={() => setExitFollowedRules(null)}
                          className={`focus-ring py-2 rounded text-[10px] font-bold border transition-all cursor-pointer ${
                            exitFollowedRules === null
                              ? "bg-slate-800 border-slate-500 text-slate-200"
                              : "bg-[#090d14] border-[#1b2537] text-slate-400 hover:text-slate-200"
                          }`}
                          title="Preserve missing evidence (zero fabrication)"
                        >
                          Unrecorded
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Notes / Reason */}
                  <div>
                    <label htmlFor="exit-notes-input" className="block text-slate-300 font-bold mb-1">Exit Notes / Execution Thesis (Optional)</label>
                    <textarea
                      id="exit-notes-input"
                      value={exitNotes}
                      onChange={(e) => setExitNotes(e.target.value)}
                      rows={2}
                      className="focus-ring w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white text-xs focus:outline-none resize-none"
                      placeholder="e.g. Scaled 50% at Target 1, remaining shares moved to breakeven stop."
                    />
                  </div>

                  {/* Live Accounting Preview */}
                  {(() => {
                    const sharesNum = parseFloat(exitShares) || 0;
                    const priceNum = parseFloat(exitPrice) || 0;
                    const pnl = calculateRealizedPnL(exitTargetPosition.entryPrice, priceNum, sharesNum);
                    const retPct = exitTargetPosition.entryPrice > 0
                      ? ((priceNum - exitTargetPosition.entryPrice) / exitTargetPosition.entryPrice) * 100
                      : 0;
                    const rAchieved = calculateRealizedR(exitTargetPosition.entryPrice, priceNum, exitTargetPosition.stopLossPrice);
                    const remaining = Math.max(0, exitTargetPosition.shares - sharesNum);

                    return (
                      <div className="p-3 rounded-xl bg-[#090d14] border border-cyan-950/80 space-y-1.5 text-xs font-mono">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">Realized P&amp;L Preview:</span>
                          <span className={`font-bold ${pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                            {pnl >= 0 ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`} ({retPct >= 0 ? `+${retPct.toFixed(2)}%` : `${retPct.toFixed(2)}%`})
                          </span>
                        </div>
                        {rAchieved !== null && (
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-slate-400">Realized R-Multiple:</span>
                            <span className={`font-bold ${rAchieved >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                              {rAchieved >= 0 ? `+${rAchieved.toFixed(2)}R` : `${rAchieved.toFixed(2)}R`}
                            </span>
                          </div>
                        )}
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span>Remaining Shares:</span>
                          <span className="text-slate-200 font-bold">{Number(remaining.toFixed(6))}</span>
                        </div>
                      </div>
                    );
                  })()}

                  {exitError && (
                    <div className="p-2.5 rounded-lg bg-rose-950/80 border border-rose-800 text-rose-300 text-xs font-sans">
                      ⚠️ {exitError}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="pt-3 flex items-center justify-end space-x-2 border-t border-[#1b2434] shrink-0">
                    <button
                      type="button"
                      onClick={handleCloseExitModal}
                      className="focus-ring px-3.5 py-1.5 bg-[#162030] hover:bg-[#1e2a3c] text-slate-300 rounded-lg font-bold transition-colors cursor-pointer"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={exitSubmitting}
                      className={`focus-ring px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-lg shadow transition-all active:scale-95 cursor-pointer ${
                        exitSubmitting ? "opacity-60 cursor-not-allowed" : ""
                      }`}
                    >
                      {exitSubmitting ? "Recording Exit..." : "Confirm Trade Exit"}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        )}
      </main>
    </TerminalShell>
  );
}