"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "../lib/masterCatalog";
import { addPortfolioPosition } from "../lib/portfolio";
import { SpotPriceRegistry, fetchBatchQuotes, fetchTacticalSetups } from "../lib/api";
import type { TradeSetupSpec } from "../lib/simulation/governorSizingEngine";
import { getPersistedMarketSnapshot, getAllPersistedMarketSnapshots } from "../lib/marketDatabase";
import MiniSparkline from "./MiniSparkline";

interface ConfluenceCandidate {
  entry: MasterAssetEntry;
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
  const [liveQuotes, setLiveQuotes] = useState<Record<string, { price: number; changePct: number }>>({});

  useEffect(() => {
    setIsCollapsed(defaultCollapsed);
  }, [defaultCollapsed]);

  // Hydrate initial live quotes from persisted client database and registry
  const refreshLocalQuotes = useCallback(() => {
    const snapshots = getAllPersistedMarketSnapshots(true);
    const initial: Record<string, { price: number; changePct: number }> = {};
    for (const [sym, snap] of Object.entries(snapshots)) {
      if (snap.currentPrice && snap.currentPrice > 0) {
        initial[sym] = { price: snap.currentPrice, changePct: snap.priceChangePct24h };
      }
    }
    for (const sym of Object.keys(MASTER_ASSET_CATALOG)) {
      const reg = SpotPriceRegistry.get(sym);
      if (reg && reg.price > 0) {
        initial[sym] = { price: reg.price, changePct: reg.changePct };
      }
    }
    setLiveQuotes((prev) => ({ ...prev, ...initial }));
  }, []);

  useEffect(() => {
    refreshLocalQuotes();

    // Fetch batch live exchange quotes for top candidate tickers on mount
    const candidateSymbols = Object.keys(MASTER_ASSET_CATALOG);
    fetchBatchQuotes(candidateSymbols).then((batch) => {
      if (batch && Object.keys(batch).length > 0) {
        setLiveQuotes((prev) => ({ ...prev, ...batch }));
      }
    }).catch(() => {});

    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);

      const savedRole = localStorage.getItem("FINANCE_USER_ROLE") as "DAY_TRADER" | "LONG_TERM" | null;
      if (savedRole) setUserRole(savedRole);

      const handleStorage = () => refreshLocalQuotes();
      window.addEventListener("storage", handleStorage);
      return () => window.removeEventListener("storage", handleStorage);
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

  const [tacticalSetups, setTacticalSetups] = useState<TradeSetupSpec[]>([]);

  useEffect(() => {
    let isMounted = true;
    fetchTacticalSetups(undefined, userRole)
      .then((data) => {
        if (isMounted) setTacticalSetups(data || []);
      })
      .catch(() => {
        if (isMounted) setTacticalSetups([]);
      });
    return () => {
      isMounted = false;
    };
  }, [userRole]);

  // Dynamically compute the Top 3 High-Confluence Plays strictly from authoritative live setups
  const topCandidates: ConfluenceCandidate[] = useMemo(() => {
    if (!tacticalSetups || tacticalSetups.length === 0) return [];

    // Filter valid setups with authentic positive prices, verified setup levels, and valid confluence score
    const valid = tacticalSetups
      .map((setup) => {
        const sym = setup.ticker;
        const live = liveQuotes[sym] || SpotPriceRegistry.get(sym);
        const snap = getPersistedMarketSnapshot(sym);
        const effectivePrice = (live?.price && live.price > 0)
          ? live.price
          : (snap?.currentPrice && snap.currentPrice > 0)
          ? snap.currentPrice
          : (setup.entryPivot && setup.entryPivot > 0 ? setup.entryPivot : null);

        if (!effectivePrice || effectivePrice <= 0) return null;
        if (!setup.stopLoss || setup.stopLoss <= 0 || !setup.target1 || setup.target1 <= 0) return null;
        const confScore = typeof setup.confluenceScore === "number" && !isNaN(setup.confluenceScore)
          ? setup.confluenceScore
          : 0;

        return {
          setup,
          effectivePrice,
          confScore,
        };
      })
      .filter((item): item is { setup: TradeSetupSpec; effectivePrice: number; confScore: number } => item !== null);

    // Sort: Actionable first, then highest confluenceScore descending
    const sorted = [...valid].sort((a, b) => {
      const aAct = Boolean(a.setup.isActionable);
      const bAct = Boolean(b.setup.isActionable);
      if (aAct !== bAct) {
        return aAct ? -1 : 1;
      }
      return b.confScore - a.confScore;
    });

    return sorted.slice(0, 3).map(({ setup, effectivePrice, confScore }) => {
      const sym = setup.ticker;
      const live = liveQuotes[sym] || SpotPriceRegistry.get(sym);
      const snap = getPersistedMarketSnapshot(sym);
      const effectiveChange = (live?.changePct !== undefined)
        ? live.changePct
        : (snap?.priceChangePct24h !== undefined)
        ? snap.priceChangePct24h
        : 0.0;

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

      const stopVal = setup.stopLoss ?? 0;
      const target1Val = setup.target1 ?? 0;
      const target2Val = setup.target2 ?? target1Val;
      const stopPct = (((effectivePrice - stopVal) / effectivePrice) * 100).toFixed(1);
      const t1Pct = (((target1Val - effectivePrice) / effectivePrice) * 100).toFixed(1);
      const t2Pct = (((target2Val - effectivePrice) / effectivePrice) * 100).toFixed(1);
      const riskDelta = effectivePrice - stopVal;
      const rewardDelta = target1Val - effectivePrice;
      const rr = riskDelta > 0 && rewardDelta > 0 ? (rewardDelta / riskDelta).toFixed(1) : "N/A";

      return {
        entry,
        livePrice: effectivePrice,
        liveChangePct: effectiveChange,
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
  }, [tacticalSetups, liveQuotes, isDayTrader, isPlain]);

  const handleQuickLog = async (e: React.MouseEvent, cand: ConfluenceCandidate) => {
    e.preventDefault();
    e.stopPropagation();

    const userSharesStr = window.prompt(`Enter quantity of ${cand.entry.symbol} shares to add:`, "10");
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
      entryPrice: cand.livePrice,
      currentPrice: cand.livePrice,
      targetPrice: cand.target1Price,
      stopLossPrice: cand.stopPrice,
    });

    setLoggedSymbol(`${cand.entry.symbol}: ${res.isDuplicate ? "Already in Portfolio" : res.success ? "Logged!" : "Failed: " + res.message}`);
    setTimeout(() => setLoggedSymbol(null), 3500);
  };

  const handleCardClick = (e: React.MouseEvent, symbol: string) => {
    if (onSelectSymbol) {
      e.preventDefault();
      onSelectSymbol(symbol);
    }
    // Auto-collapse spotlight on mobile/click so active asset details render above the fold
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
            </div>
            {!isCollapsed && (
              <p className="text-xs text-slate-400 mt-0.5">
                {isDayTrader
                  ? isPlain
                    ? "Filtered for fast-moving stocks with heavy trading volume and tight safety stops."
                    : "Intraday & swing sieve: RVOL >= 1.3 + ATR/Price >= 2.0% + Minervini Stage 2 + R:R >= 1.85:1."
                  : isPlain
                  ? "Strictly filtered by balance sheet health, insider flow, and minimum 1.85:1 profit-to-risk ratio."
                  : "Multi-factor quantitative sieve: Minervini Stage 2 + Piotroski F-Score >= 7 + SEC Form 4 Inflow + Risk/Reward >= 1.85:1."}
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
                    aria-label={`Select ${cand.entry.symbol}`}
                  >
                    <span className="text-[9px] text-slate-400 font-normal">#{idx + 1}</span>
                    <span className="font-extrabold">{cand.entry.symbol}</span>
                    <span className={`text-[10px] tabular-nums ${cand.liveChangePct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      ${cand.livePrice.toFixed(2)}
                    </span>
                  </button>
                );
              })
            ) : (
              <span className="text-xs text-slate-500 font-mono">Awaiting verified live quotes</span>
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

              return (
                <Link
                  key={cand.entry.symbol}
                  href={`/?symbol=${cand.entry.symbol}`}
                  onClick={(e) => handleCardClick(e, cand.entry.symbol)}
                  aria-label={`Analyze ${cand.entry.symbol} (${cand.entry.name})`}
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
                        <div className="text-xs font-mono font-bold text-slate-300 tabular-nums truncate">
                          ${cand.livePrice.toFixed(2)}{" "}
                          <span className={cand.liveChangePct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                            ({cand.liveChangePct >= 0 ? "+" : ""}{cand.liveChangePct}%)
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Sparkline & Compact Score Pill */}
                    <div className="flex items-center gap-2 shrink-0">
                      <MiniSparkline
                        basePrice={cand.livePrice}
                        changePct={cand.liveChangePct}
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

                  {/* Footer CTAs */}
                  <div className="flex items-center justify-between pt-1 border-t border-[#1e293b] text-[11px]">
                    <button
                      type="button"
                      onClick={(e) => handleQuickLog(e, cand)}
                      className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border bg-indigo-600/20 hover:bg-indigo-500 hover:text-slate-950 border-indigo-500/40 text-indigo-300 transition-colors flex items-center gap-1 shrink-0 active:scale-95"
                      title="Log directly into your Paper Portfolio"
                    >
                      <span>💼</span>
                      <span>{isPlain ? "Quick Paper Log" : "Log to Portfolio"}</span>
                    </button>

                    <span className="px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border bg-cyan-500/10 border-cyan-500/40 text-cyan-300 group-hover:bg-cyan-500 group-hover:text-slate-950 group-hover:border-cyan-400 transition-colors flex items-center gap-1 shrink-0">
                      Analyze <span className="group-hover:translate-x-0.5 transition-transform">➔</span>
                    </span>
                  </div>
                </Link>
              );
            })}
          </div>
        ) : (
          <div className="p-6 text-center text-sm font-mono text-slate-400 bg-[#111722] rounded-xl border border-[#243044]">
            No candidates with verified exchange pricing currently meet spotlight criteria. Awaiting market tape.
          </div>
        )
      )}
    </section>
  );
}
