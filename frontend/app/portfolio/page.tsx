"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import {
  PortfolioPosition,
  PortfolioSummary,
  loadPortfolioPositions,
  savePortfolioPositions,
  calculatePortfolioSummary,
  getAnonymousUserId,
  exportPortfolioToCsv,
} from "../../lib/portfolio";
import { SHARED_FACTOR_SCORES } from "../../lib/constants";
import { fetchAssetAnalytics, SpotPriceRegistry } from "../../lib/api";
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
  });
  const [showAddModal, setShowAddModal] = useState(false);
  const [anonId, setAnonId] = useState<string>("");
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastSyncTime, setLastSyncTime] = useState<string>("");

  // Form State for Adding Position with Real-Time Auto-Population
  const [newSymbol, setNewSymbol] = useState("SEDG");
  const [newShares, setNewShares] = useState("75");
  const [newEntryPrice, setNewEntryPrice] = useState("33.51");
  const [newStopLoss, setNewStopLoss] = useState("31.16");
  const [newTarget, setNewTarget] = useState("41.89");
  const [isResolvingQuote, setIsResolvingQuote] = useState(false);
  const [resolvedAssetName, setResolvedAssetName] = useState("SolarEdge Technologies");
  const [resolvedQuotePrice, setResolvedQuotePrice] = useState<number | null>(33.51);

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
        else if (baseline > 0) price = baseline;
      }

      if (price && price > 0) {
        setResolvedQuotePrice(price);
        setNewEntryPrice(price.toFixed(2));
        setNewStopLoss((price * 0.93).toFixed(2));
        setNewTarget((price * 1.25).toFixed(2));
        setNewShares(Math.max(1, Math.round(2500 / price)).toString());
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
    const target = initialSymbol || newSymbol || "SEDG";
    setNewSymbol(target);
    setShowAddModal(true);
    populateTickerData(target);
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
    refreshQuotes(loaded);

    // Auto-open add modal if ?add=SYMBOL query is present
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const addSym = params.get("add") || params.get("symbol");
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

  const handleAddPosition = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedSym = newSymbol.trim().toUpperCase();
    const aliasInfo = resolveAssetAlias(trimmedSym);
    const symUpper = (aliasInfo ? aliasInfo.canonicalTicker : trimmedSym).toUpperCase();
    const sharesNum = parseFloat(newShares);
    const entryNum = parseFloat(newEntryPrice);
    const stopNum = newStopLoss ? parseFloat(newStopLoss) : undefined;
    const targetNum = newTarget ? parseFloat(newTarget) : undefined;

    if (!symUpper || isNaN(sharesNum) || isNaN(entryNum) || sharesNum <= 0) return;

    const authenticName = getCanonicalAssetName(symUpper);
    const curPrice = resolvedQuotePrice && !isNaN(resolvedQuotePrice) ? resolvedQuotePrice : entryNum;

    const newPos: PortfolioPosition = {
      symbol: symUpper,
      name: authenticName,
      shares: sharesNum,
      entryPrice: entryNum,
      currentPrice: curPrice,
      targetPrice: targetNum,
      stopLossPrice: stopNum,
      addedAt: new Date().toISOString().split("T")[0],
      assetType: "Stock",
    };

    const updated = [newPos, ...positions.filter((p) => p.symbol !== symUpper)];
    setPositions(updated);
    setSummary(calculatePortfolioSummary(updated));
    savePortfolioPositions(updated);
    setShowAddModal(false);

    trackMatomoEvent("User Journey", "Add Portfolio Position", `${symUpper} (${sharesNum} shares)`);
  };

  const handleRemovePosition = (symbol: string) => {
    const updated = positions.filter((p) => p.symbol !== symbol);
    setPositions(updated);
    setSummary(calculatePortfolioSummary(updated));
    savePortfolioPositions(updated);

    trackMatomoEvent("User Journey", "Remove Portfolio Position", symbol);
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
  const totalNetWorth = cashReserves + investedEquity;
  const investedPct = totalNetWorth > 0 ? (investedEquity / totalNetWorth) * 100 : 0;
  const cashPct = totalNetWorth > 0 ? (cashReserves / totalNetWorth) * 100 : 0;
  const isPositive = summary.totalUnrealizedPnL >= 0;

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans selection:bg-cyan-500 selection:text-black transition-colors duration-200">
      <Navbar />

      <main className="max-w-[1450px] mx-auto p-4 sm:p-6 space-y-6 font-mono pb-28 sm:pb-8">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#243044] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-800">
                🔒 ZERO-LOGIN PRIVATE STORAGE
              </span>
              <span className="text-slate-500 text-xs hidden sm:inline">• Persistent in Client Storage</span>
              {lastSyncTime && (
                <span className="text-slate-400 text-xs hidden md:inline">
                  • Synced: <span className="text-slate-300 font-bold">{lastSyncTime}</span>
                </span>
              )}
            </div>
            <h1 className="text-xl sm:text-3xl font-extrabold text-white tracking-tight mt-1">
              My Portfolio & Holdings
            </h1>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Track your real-time holdings, profit/loss, stop-loss protection floors, and profit targets — 100% private to your browser.
            </p>
          </div>

          <div className="flex items-center space-x-2 sm:space-x-3">
            <button
              onClick={() => refreshQuotes(positions)}
              disabled={isRefreshing}
              className={`px-3 py-2 bg-[#162030] hover:bg-[#1f2d44] border border-[#243044] text-slate-200 rounded-xl text-xs font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer ${
                isRefreshing ? "opacity-60 cursor-not-allowed" : ""
              }`}
            >
              <span className={isRefreshing ? "animate-spin" : ""}>🔄</span>
              <span className="hidden sm:inline">{isRefreshing ? "Syncing Quotes..." : "Refresh Quotes"}</span>
            </button>

            <button
              onClick={handleExportCsv}
              className="px-3 py-2 bg-[#162030] hover:bg-[#1f2d44] border border-[#243044] text-slate-200 rounded-xl text-xs font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer"
            >
              <span>📥</span>
              <span className="hidden sm:inline">Export CSV</span>
            </button>

            <button
              onClick={() => handleOpenAddModal()}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer"
            >
              <span>➕</span>
              <span>Add Position</span>
            </button>
          </div>
        </div>

        {/* Portfolio Summary KPI Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Total Account Net Worth</span>
            <span className="text-xl sm:text-2xl font-extrabold text-white tabular-nums">
              ${totalNetWorth.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className="text-[10px] text-slate-400 block mt-1">
              ${investedEquity.toFixed(2)} Stock + ${cashReserves.toFixed(2)} Cash
            </span>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Active Stock Holdings</span>
            <span className="text-xl sm:text-2xl font-extrabold text-cyan-300 tabular-nums">
              ${investedEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className="text-[10px] text-cyan-500/80 block mt-1 font-bold">
              {investedPct.toFixed(1)}% Deployed ({summary.positionsCount} {summary.positionsCount === 1 ? "Holding" : "Holdings"})
            </span>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Available Cash Reserves</span>
            <span className="text-xl sm:text-2xl font-extrabold text-emerald-300 tabular-nums">
              ${cashReserves.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className="text-[10px] text-emerald-500/80 block mt-1 font-bold">
              {cashPct.toFixed(1)}% Buying Power
            </span>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Total Unrealized P&L</span>
            <span className={`text-xl sm:text-2xl font-extrabold tabular-nums flex items-center gap-1 ${isPositive ? "text-emerald-400" : "text-rose-400"}`}>
              <span>{isPositive ? `+$${summary.totalUnrealizedPnL.toFixed(2)}` : `-$${Math.abs(summary.totalUnrealizedPnL).toFixed(2)}`}</span>
            </span>
            <span className={`text-[10px] font-bold block mt-1 ${isPositive ? "text-emerald-500" : "text-rose-500"}`}>
              {isPositive ? `+${summary.totalUnrealizedPnLPct.toFixed(2)}%` : `${summary.totalUnrealizedPnLPct.toFixed(2)}%`} Total Return
            </span>
          </div>
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

              {/* Quick Presets */}
              <div className="hidden sm:flex items-center gap-1">
                {[50, 100, 500, 2500, 10000, 25000].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => handleAccountEquityChange(preset)}
                    className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border transition-all cursor-pointer ${
                      accountEquity === preset
                        ? "bg-cyan-600 border-cyan-400 text-white"
                        : "bg-[#0c121e] border-[#1f2c42] text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    ${preset >= 1000 ? `${preset / 1000}k` : preset}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Progress Stack Bar */}
          <div className="space-y-1.5">
            <div className="w-full h-3 bg-[#06090f] rounded-full overflow-hidden flex border border-[#1b2537]">
              {positions.map((pos, idx) => {
                const posVal = pos.shares * pos.currentPrice;
                const pct = totalNetWorth > 0 ? (posVal / totalNetWorth) * 100 : 0;
                const colors = ["bg-cyan-500", "bg-emerald-500", "bg-indigo-500", "bg-amber-500", "bg-purple-500"];
                const color = colors[idx % colors.length];
                return (
                  <div
                    key={pos.symbol}
                    style={{ width: `${pct}%` }}
                    className={`${color} h-full transition-all duration-300`}
                    title={`${pos.symbol}: $${posVal.toFixed(2)} (${pct.toFixed(1)}%)`}
                  />
                );
              })}
              <div
                style={{ width: `${cashPct}%` }}
                className="bg-slate-700/60 h-full transition-all duration-300"
                title={`Available Cash: $${cashReserves.toFixed(2)} (${cashPct.toFixed(1)}%)`}
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400">
              <div className="flex items-center gap-3 flex-wrap">
                {positions.map((pos, idx) => {
                  const posVal = pos.shares * pos.currentPrice;
                  const pct = totalNetWorth > 0 ? (posVal / totalNetWorth) * 100 : 0;
                  const dotColors = ["bg-cyan-400", "bg-emerald-400", "bg-indigo-400", "bg-amber-400", "bg-purple-400"];
                  const dotColor = dotColors[idx % dotColors.length];
                  return (
                    <span key={pos.symbol} className="flex items-center gap-1">
                      <span className={`w-2 h-2 rounded-full ${dotColor}`} />
                      <strong className="text-slate-200">{pos.symbol}:</strong>
                      <span>${posVal.toFixed(2)} ({pct.toFixed(1)}%)</span>
                    </span>
                  );
                })}
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-slate-500" />
                  <strong className="text-slate-300">Cash Reserves:</strong>
                  <span>${cashReserves.toFixed(2)} ({cashPct.toFixed(1)}%)</span>
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
                {positions.map((pos) => {
                  const mktVal = pos.shares * pos.currentPrice;
                  const cost = pos.shares * pos.entryPrice;
                  const pnl = mktVal - cost;
                  const pnlPct = cost > 0 ? (pnl / cost) * 100 : 0;
                  const posUp = pnl >= 0;

                  // Execution state alert
                  let statusBadge = null;
                  if (pos.targetPrice && pos.currentPrice >= pos.targetPrice) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold text-[10px] whitespace-nowrap animate-pulse">
                        🎯 TP1 TARGET HIT
                      </span>
                    );
                  } else if (pos.stopLossPrice && pos.currentPrice <= pos.stopLossPrice) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800 font-bold text-[10px] whitespace-nowrap">
                        🛑 STOP LOSS HIT
                      </span>
                    );
                  } else if (pos.stopLossPrice && pos.currentPrice <= pos.stopLossPrice * 1.02) {
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
                    <tr key={pos.symbol} className="hover:bg-[#151e2d] transition-colors">
                      <td className="py-3 px-4">
                        <Link href={`/?symbol=${pos.symbol}&ownership=OWNED`} className="font-bold text-cyan-400 hover:text-cyan-300 text-sm flex items-center gap-1.5">
                          <span>{pos.symbol}</span>
                          <span className="text-[10px] text-slate-500 font-normal">({pos.name})</span>
                        </Link>
                      </td>
                      <td className="py-3 px-4">{statusBadge}</td>
                      <td className="py-3 px-4 text-slate-200">{pos.shares}</td>
                      <td className="py-3 px-4 text-slate-300">${pos.entryPrice.toFixed(2)}</td>
                      <td className="py-3 px-4 text-white font-bold">${pos.currentPrice.toFixed(2)}</td>
                      <td className="py-3 px-4 text-slate-100 font-bold">${mktVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className="py-3 px-4">
                        <span className={`font-bold ${posUp ? "text-emerald-400" : "text-rose-400"}`}>
                          {posUp ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`} ({posUp ? `+${pnlPct.toFixed(2)}%` : `${pnlPct.toFixed(2)}%`})
                        </span>
                      </td>
                      <td className="py-3 px-4 text-[11px]">
                        <span className="text-rose-400">Stop: ${pos.stopLossPrice?.toFixed(2) || "None"}</span>
                        <span className="text-slate-600 mx-1">|</span>
                        <span className="text-emerald-400">Target: ${pos.targetPrice?.toFixed(2) || "None"}</span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => handleRemovePosition(pos.symbol)}
                          className="px-2.5 py-1 text-[11px] rounded bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800/80 transition-colors"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  );
                })}
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
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[1200] flex items-center justify-center p-2 sm:p-4 overflow-y-auto font-mono">
            <div className="bg-[#111722] border border-[#243044] rounded-2xl max-w-md w-full shadow-2xl overflow-hidden max-h-[92vh] flex flex-col my-auto text-slate-100">
              {/* Fixed Header */}
              <div className="flex items-center justify-between p-4 border-b border-[#1b2434] bg-[#0e1422] shrink-0">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">💼</span>
                  <div>
                    <h3 className="text-base font-bold text-white tracking-tight">Add Portfolio Holding</h3>
                    <p className="text-[10px] text-slate-400">Live quantitative auto-population enabled</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
                >
                  ✕
                </button>
              </div>

              {/* Scrollable Form Body */}
              <form onSubmit={handleAddPosition} className="p-4 sm:p-5 space-y-3.5 overflow-y-auto flex-1 text-xs">
                {/* Ticker Input & Quick Chips */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-slate-300 font-bold">Ticker Symbol</label>
                    <span className="text-[10px] text-cyan-400 font-mono">
                      {isResolvingQuote ? "⏳ Syncing Quote..." : `Verified: ${resolvedAssetName}`}
                    </span>
                  </div>
                  <input
                    type="text"
                    value={newSymbol}
                    onChange={(e) => {
                      const val = e.target.value.toUpperCase();
                      setNewSymbol(val);
                      populateTickerData(val);
                    }}
                    className="w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2.5 text-white font-bold tracking-wider uppercase text-sm focus:outline-none"
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
                        className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-all cursor-pointer ${
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
                    <label className="block text-slate-300 font-bold mb-1">Shares Count</label>
                    <input
                      type="number"
                      step="any"
                      min="1"
                      value={newShares}
                      onChange={(e) => setNewShares(e.target.value)}
                      className="w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                      required
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">Sized to ~$2,500</span>
                  </div>
                  <div>
                    <label className="block text-slate-300 font-bold mb-1">Entry Price ($)</label>
                    <input
                      type="number"
                      step="any"
                      value={newEntryPrice}
                      onChange={(e) => setNewEntryPrice(e.target.value)}
                      className="w-full bg-[#090d14] border border-[#243044] focus:border-cyan-400 rounded-lg p-2 text-white font-bold focus:outline-none"
                      required
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">Live market execution</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label className="block text-rose-300 font-bold mb-1">Stop Loss ($)</label>
                    <input
                      type="number"
                      step="any"
                      value={newStopLoss}
                      onChange={(e) => setNewStopLoss(e.target.value)}
                      className="w-full bg-[#090d14] border border-rose-950/80 focus:border-rose-500 rounded-lg p-2 text-rose-300 font-bold focus:outline-none"
                    />
                    <span className="text-[10px] text-rose-400/80 block mt-0.5">-7% Risk Cut Floor</span>
                  </div>
                  <div>
                    <label className="block text-emerald-300 font-bold mb-1">Target Price ($)</label>
                    <input
                      type="number"
                      step="any"
                      value={newTarget}
                      onChange={(e) => setNewTarget(e.target.value)}
                      className="w-full bg-[#090d14] border border-emerald-950/80 focus:border-emerald-500 rounded-lg p-2 text-emerald-300 font-bold focus:outline-none"
                    />
                    <span className="text-[10px] text-emerald-400/80 block mt-0.5">+25% Upside Target (TP1)</span>
                  </div>
                </div>

                {/* Fixed Footer with Actions */}
                <div className="pt-3.5 flex items-center justify-end space-x-2 border-t border-[#1b2434] shrink-0">
                  <button
                    type="button"
                    onClick={() => setShowAddModal(false)}
                    className="px-3.5 py-1.5 bg-[#162030] hover:bg-[#1e2a3c] text-slate-300 rounded-lg font-bold transition-colors cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg shadow transition-all active:scale-95 cursor-pointer"
                  >
                    Save Holding
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}