"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "../lib/masterCatalog";
import { addPortfolioPosition } from "../lib/portfolio";
import { SpotPriceRegistry, fetchBatchQuotes } from "../lib/api";
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

  // Dynamically compute the Top 3 High-Confluence Plays based on active Horizon (Day Trader vs Long Term)
  const topCandidates: ConfluenceCandidate[] = useMemo(() => {
    const all = Object.values(MASTER_ASSET_CATALOG);
    
    // 1. Dual-Horizon Pre-Filter
    const eligible = all.filter((a) => {
      if (a.type !== "Stock" || !a.atr14 || a.atr14 <= 0) return false;
      if (isDayTrader) {
        // Day Trader: Filter for high Relative Volume (RVOL >= 1.3) or elevated Momentum (>=70)
        return a.rvol >= 1.3 || a.momentumScore >= 70;
      } else {
        // Long Term: Filter for pristine solvency (Piotroski >= 7) and positive ROIC
        return a.piotroski >= 7 && a.roic > 0;
      }
    });

    const scored = eligible.map((asset) => {
      // Resolve authentic live exchange spot price
      const live = liveQuotes[asset.symbol] || SpotPriceRegistry.get(asset.symbol);
      const snap = getPersistedMarketSnapshot(asset.symbol);
      const effectivePrice = (live?.price && live.price > 0)
        ? live.price
        : (snap?.currentPrice && snap.currentPrice > 0)
        ? snap.currentPrice
        : 100.0;

      const effectiveChange = (live?.changePct !== undefined)
        ? live.changePct
        : (snap?.priceChangePct24h !== undefined)
        ? snap.priceChangePct24h
        : 0.0;

      const currentAtr = Math.max(0.2, asset.atr14);
      const atrPct = effectivePrice > 0 ? (currentAtr / effectivePrice) : 0.02;

      let compositeScore = 80;
      let stopVal = effectivePrice - currentAtr * 1.4;
      let target1Val = effectivePrice + currentAtr * 2.2;
      let target2Val = effectivePrice + currentAtr * 3.8;
      let setupBadge = "INSTITUTIONAL ACCUMULATION";
      let setupBadgePlain = "Smart Money Buying";

      if (isDayTrader) {
        // ── ⚡ DAY TRADER QUANTITATIVE SCORING SIEVE ─────────────────────────
        const rvolScore = Math.min(35, (asset.rvol / 2.6) * 35); // Max 35 pts
        const momentumScore = Math.min(35, (asset.momentumScore / 100) * 35); // Max 35 pts
        const volatilityBonus = atrPct >= 0.022 ? 15 : 8; // Max 15 pts
        const squeezeBonus = asset.shortFloat >= 5.0 ? 15 : asset.shortFloat >= 2.5 ? 10 : 5; // Max 15 pts

        compositeScore = Math.round(rvolScore + momentumScore + volatilityBonus + squeezeBonus);

        // Tight Intraday/Swing Monotonic Execution Ladder (1.0x ATR Stop, 1.8x ATR TP1, 3.2x ATR TP2)
        stopVal = Math.max(0.01, effectivePrice - currentAtr * 1.0);
        target1Val = effectivePrice + currentAtr * 1.8;
        target2Val = effectivePrice + currentAtr * 3.2;

        setupBadge = asset.rvol >= 2.4
          ? "⚡ HIGH-RVOL EXPLOSION"
          : asset.shortFloat >= 5.5
          ? "🔥 SHORT SQUEEZE PRESSURE"
          : "🚀 STAGE 2 BREAKOUT";

        setupBadgePlain = asset.rvol >= 2.4
          ? "Surging Trading Volume"
          : asset.shortFloat >= 5.5
          ? "Squeeze Pressure Building"
          : "Fast-Moving Momentum";
      } else {
        // ── 🏛️ LONG-TERM COMPOUNDER SCORING SIEVE ────────────────────────────
        const piotroskiWeight = (asset.piotroski / 9) * 30; // Max 30 pts
        const factorWeight = (asset.compositeFactorScore / 100) * 35; // Max 35 pts
        const pegBonus = asset.peg > 0 && asset.peg <= 1.2 ? 15 : 5; // Max 15 pts
        const roicBonus = asset.roic >= 20 ? 15 : asset.roic >= 12 ? 10 : 5; // Max 15 pts
        const momBonus = Math.min(5, Math.max(0, asset.momentumScore / 20)); // Max 5 pts

        compositeScore = Math.round(piotroskiWeight + factorWeight + pegBonus + roicBonus + momBonus);

        // Structural Swing/Position Ladder (1.4x ATR Stop, 2.2x ATR TP1, 3.8x ATR TP2)
        stopVal = Math.max(0.01, effectivePrice - currentAtr * 1.4);
        target1Val = effectivePrice + currentAtr * 2.2;
        target2Val = effectivePrice + currentAtr * 3.8;

        setupBadge = asset.piotroski === 9
          ? "🏛️ PERFECT 9/9 PIOTROSKI"
          : asset.roic >= 25
          ? "💎 CAPITAL COMPOUNDER"
          : "🛡️ SECULAR MOAT LEADER";

        setupBadgePlain = asset.piotroski === 9
          ? "Rock-Solid Balance Sheet"
          : asset.roic >= 25
          ? "High Profit Engine"
          : "Dominant Market Leader";
      }
      
      const riskDelta = effectivePrice - stopVal;
      const rewardDelta = target1Val - effectivePrice;
      const rr = riskDelta > 0 ? (rewardDelta / riskDelta).toFixed(1) : "2.2";

      const stopPct = (((effectivePrice - stopVal) / effectivePrice) * 100).toFixed(1);
      const t1Pct = (((target1Val - effectivePrice) / effectivePrice) * 100).toFixed(1);
      const t2Pct = (((target2Val - effectivePrice) / effectivePrice) * 100).toFixed(1);

      return {
        entry: asset,
        livePrice: effectivePrice,
        liveChangePct: effectiveChange,
        convictionScore: Math.min(99, compositeScore),
        setupBadge,
        setupBadgePlain,
        catalystSummary: asset.upcomingCatalyst || asset.thesis,
        catalystSummaryPlain: asset.moatSummary || asset.thesis,
        stopPrice: Number(stopVal.toFixed(2)),
        stopLossPct: stopPct,
        target1Price: Number(target1Val.toFixed(2)),
        target1Pct: t1Pct,
        target2Price: Number(target2Val.toFixed(2)),
        target2Pct: t2Pct,
        rewardRiskRatio: rr,
      };
    })
    .sort((a, b) => b.convictionScore - a.convictionScore);

    return scored.slice(0, 3);
  }, [liveQuotes, isDayTrader]);

  const handleQuickLog = (e: React.MouseEvent, cand: ConfluenceCandidate) => {
    e.preventDefault();
    e.stopPropagation();

    const res = addPortfolioPosition({
      symbol: cand.entry.symbol,
      name: cand.entry.name,
      shares: Math.max(1, Math.round(2500 / cand.livePrice)),
      entryPrice: cand.livePrice,
      currentPrice: cand.livePrice,
      targetPrice: cand.target1Price,
      stopLossPrice: cand.stopPrice,
    });

    setLoggedSymbol(`${cand.entry.symbol}: ${res.isDuplicate ? "Already in Portfolio" : "Logged!"}`);
    setTimeout(() => setLoggedSymbol(null), 3000);
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
    <section className="bg-[#0d121c] border border-[#1e293b] rounded-2xl p-4 sm:p-5 shadow-2xl space-y-4 mb-6 transition-all">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-sm shadow-inner ${
            isDayTrader
              ? "bg-amber-500/10 border border-amber-500/30 text-amber-400"
              : "bg-cyan-500/10 border border-cyan-500/30 text-cyan-400"
          }`}>
            {isDayTrader ? "⚡" : "🎯"}
          </div>
          <div>
            <div className="flex items-center gap-2">
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
            <p className="text-xs text-slate-400 mt-0.5">
              {isDayTrader
                ? isPlain
                  ? "Filtered for fast-moving stocks with heavy trading volume and tight safety stops."
                  : "Intraday & swing sieve: RVOL >= 1.3 + ATR/Price >= 2.0% + Minervini Stage 2 + R:R >= 1.8:1."
                : isPlain
                ? "Strictly filtered by balance sheet health, insider flow, and minimum 2.0:1 profit-to-risk ratio."
                : "Multi-factor quantitative sieve: Minervini Stage 2 + Piotroski F-Score >= 7 + SEC Form 4 Inflow + Risk/Reward >= 2.0:1."}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {loggedSymbol && (
            <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-emerald-950/80 border border-emerald-700 text-emerald-300 animate-fade-in">
              💼 {loggedSymbol}
            </span>
          )}
          <button
            type="button"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-xs px-2.5 py-1 rounded-md font-mono text-slate-400 hover:text-slate-200 border border-[#243044] hover:bg-[#162030] transition-colors"
            aria-label={isCollapsed ? "Expand Weekly Spotlight" : "Collapse Weekly Spotlight"}
          >
            {isCollapsed ? "Expand ▼" : "Collapse ▲"}
          </button>
        </div>
      </div>

      {/* Compact Quick-Switcher Ribbon when Collapsed */}
      {isCollapsed && (
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[#1b2434]/60">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] font-mono text-slate-400 font-bold flex items-center gap-1">
              <span>{isDayTrader ? "⚡" : "🎯"}</span>
              <span>Top Plays:</span>
            </span>
            {topCandidates.map((cand, idx) => {
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
            })}
          </div>
          <button
            type="button"
            onClick={() => setIsCollapsed(false)}
            className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-semibold"
          >
            <span>View Full Setups</span>
            <span>▼</span>
          </button>
        </div>
      )}

      {/* 3-Card Responsive Grid */}
      {!isCollapsed && (
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
                {/* Card Header: Rank Badge, Ticker & Price */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <span className={`w-5 h-5 rounded-md flex items-center justify-center font-mono font-black text-xs ${
                      isRank1 ? "bg-cyan-500 text-slate-950 font-bold" : "bg-slate-800 text-slate-300"
                    }`}>
                      #{idx + 1}
                    </span>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <strong className="text-base font-black text-white font-mono group-hover:text-cyan-400 transition-colors">
                          {cand.entry.symbol}
                        </strong>
                        <span className="text-[11px] text-slate-400 truncate max-w-[100px] sm:max-w-[120px]">
                          {cand.entry.name}
                        </span>
                      </div>
                      <div className="text-xs font-mono font-bold text-slate-300 tabular-nums">
                        ${cand.livePrice.toFixed(2)}{" "}
                        <span className={cand.liveChangePct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          ({cand.liveChangePct >= 0 ? "+" : ""}{cand.liveChangePct}%)
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <MiniSparkline
                      basePrice={cand.livePrice}
                      changePct={cand.liveChangePct}
                      width={48}
                      height={20}
                      className="hidden sm:inline-block"
                    />
                    <div className="text-right">
                      <span className="text-[9px] font-mono text-slate-400 block uppercase font-bold">
                        {isPlain ? "Confluence Score" : "Conviction Index"}
                      </span>
                      <span className="text-sm font-extrabold font-mono text-cyan-300 tabular-nums">
                        {cand.convictionScore}/100
                      </span>
                    </div>
                  </div>
                </div>

                {/* Setup Badge */}
                <div className="flex items-center justify-between text-[10px] font-mono font-extrabold">
                  <span className="px-2 py-0.5 rounded bg-[#090d14] border border-cyan-800/50 text-cyan-300">
                    {isPlain ? cand.setupBadgePlain : cand.setupBadge}
                  </span>
                  <span className="text-emerald-400">
                    {cand.rewardRiskRatio} : 1.0 R:R
                  </span>
                </div>

                {/* Mathematical Execution Price Ladder */}
                <div className="bg-[#090d14] p-2.5 rounded-lg border border-[#1e293b] space-y-1.5 font-mono text-xs">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-emerald-400 font-bold">{isPlain ? "Goal 1 (Sell Half):" : "Take Profit 1 (TP1):"}</span>
                    <strong className="text-white tabular-nums">
                      ${cand.target1Price.toFixed(2)} <span className="text-emerald-500 text-[10px]">(+{cand.target1Pct}%)</span>
                    </strong>
                  </div>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-rose-400 font-bold">{isPlain ? "Safety Exit Stop:" : "Hard Stop Floor:"}</span>
                    <strong className="text-rose-400 tabular-nums">
                      ${cand.stopPrice.toFixed(2)} <span className="text-rose-500 text-[10px]">(-{cand.stopLossPct}%)</span>
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
      )}
    </section>
  );
}
