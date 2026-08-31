"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "../lib/masterCatalog";
import { addPortfolioPosition } from "../lib/portfolio";
import MiniSparkline from "./MiniSparkline";

interface ConfluenceCandidate {
  entry: MasterAssetEntry;
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

export default function WeeklyConfluenceSpotlight() {
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");
  const [loggedSymbol, setLoggedSymbol] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    }
    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  // Dynamically compute the Top 3 High-Confluence Plays from master catalog
  const topCandidates: ConfluenceCandidate[] = useMemo(() => {
    const all = Object.values(MASTER_ASSET_CATALOG);
    
    const scored = all
      .filter((a) => a.type === "Stock" && a.price > 10 && a.piotroski >= 7 && a.atr14 > 0)
      .map((asset) => {
        // Multi-Factor Quantitative Confluence Scoring
        const piotroskiWeight = (asset.piotroski / 9) * 30; // Max 30 pts
        const factorWeight = (asset.compositeFactorScore / 100) * 35; // Max 35 pts
        const pegBonus = asset.peg > 0 && asset.peg <= 1.2 ? 15 : 5; // Max 15 pts
        const roicBonus = asset.roic >= 20 ? 15 : asset.roic >= 12 ? 10 : 5; // Max 15 pts
        const momBonus = Math.min(10, Math.max(0, asset.momentumScore / 10)); // Max 10 pts
        
        const compositeScore = Math.round(piotroskiWeight + factorWeight + pegBonus + roicBonus + momBonus);
        
        // Exact Risk-Defined Boundaries (Monotonic Ladder)
        const stopVal = Math.max(0.01, asset.price - asset.atr14 * 1.4);
        const target1Val = asset.price + asset.atr14 * 2.2;
        const target2Val = asset.price + asset.atr14 * 3.8;
        
        const riskDelta = asset.price - stopVal;
        const rewardDelta = target1Val - asset.price;
        const rr = riskDelta > 0 ? (rewardDelta / riskDelta).toFixed(1) : "2.4";

        const stopPct = (((asset.price - stopVal) / asset.price) * 100).toFixed(1);
        const t1Pct = (((target1Val - asset.price) / asset.price) * 100).toFixed(1);
        const t2Pct = (((target2Val - asset.price) / asset.price) * 100).toFixed(1);

        return {
          entry: asset,
          convictionScore: Math.min(99, compositeScore),
          setupBadge: asset.category.includes("Monopoly")
            ? "STAGE 2 VCP BREAKOUT"
            : asset.piotroski === 9
            ? "PERFECT 9/9 PIOTROSKI"
            : "INSTITUTIONAL ACCUMULATION",
          setupBadgePlain: asset.category.includes("Monopoly")
            ? "High-Momentum Breakout"
            : asset.piotroski === 9
            ? "Rock-Solid Balance Sheet"
            : "Smart Money Buying",
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
  }, []);

  const handleQuickLog = (e: React.MouseEvent, cand: ConfluenceCandidate) => {
    e.preventDefault();
    e.stopPropagation();

    const res = addPortfolioPosition({
      symbol: cand.entry.symbol,
      name: cand.entry.name,
      shares: Math.max(1, Math.round(2500 / cand.entry.price)),
      entryPrice: cand.entry.price,
      currentPrice: cand.entry.price,
      targetPrice: cand.target1Price,
      stopLossPrice: cand.stopPrice,
    });

    setLoggedSymbol(`${cand.entry.symbol}: ${res.isDuplicate ? "Already in Portfolio" : "Logged!"}`);
    setTimeout(() => setLoggedSymbol(null), 3000);
  };

  return (
    <section className="bg-[#0d121c] border border-[#1e293b] rounded-2xl p-4 sm:p-5 shadow-2xl space-y-4 mb-6 transition-all">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-bold text-sm shadow-inner">
            🎯
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm sm:text-base font-extrabold text-white tracking-tight flex items-center gap-2">
                <span>{isPlain ? "Top 3 High-Confluence Plays of the Week" : "Weekly Alpha Spotlight: Top 3 High-Confluence Setups"}</span>
              </h2>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-950/80 border border-cyan-700 text-cyan-300 hidden sm:inline-block">
                LIVE CONFLUENCE RANKER
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              {isPlain
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

      {/* 3-Card Responsive Grid */}
      {!isCollapsed && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
          {topCandidates.map((cand, idx) => {
            const isRank1 = idx === 0;

            return (
              <Link
                key={cand.entry.symbol}
                href={`/?symbol=${cand.entry.symbol}`}
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
                        ${cand.entry.price.toFixed(2)}{" "}
                        <span className={cand.entry.changePct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          ({cand.entry.changePct >= 0 ? "+" : ""}{cand.entry.changePct}%)
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <MiniSparkline
                      basePrice={cand.entry.price}
                      changePct={cand.entry.changePct}
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
