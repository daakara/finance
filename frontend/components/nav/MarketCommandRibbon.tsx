"use client";

import React, { useEffect, useState } from "react";
import { getApiBaseUrl, ARX_API_HEADERS } from "../../lib/api";

export interface MacroRibbonPayload {
  spx?: { symbol?: string; price: number; change?: number; changePct: number };
  spy?: { price: number; changePct: number; change?: number };
  qqq?: { symbol?: string; price: number; change?: number; changePct: number };
  vix?: { level?: number; value?: number; change?: number; changePct: number; tier?: "LOW" | "NORMAL" | "ELEVATED" | "EXTREME" };
  treasury10Y?: { yield: number; changeBps: number };
  tenYearYield?: { value: number; dailyChangeBp: number };
  regime: "RISK_ON" | "NEUTRAL" | "DEFENSIVE";
  regimeSummary?: string;
  marketSession?: "OPEN" | "PRE_MARKET" | "AFTER_HOURS" | "CLOSED";
  settlementPinned?: boolean;
  isSettlementPinned?: boolean;
  dataSource?: "LIVE_FEED" | "CACHED_STORE";
  updatedAt?: string;
}

export const DEFAULT_MACRO_SNAPSHOT: MacroRibbonPayload = {
  spx: { symbol: "SPY", price: 542.10, change: 3.85, changePct: 0.72 },
  spy: { price: 542.10, change: 3.85, changePct: 0.72 },
  qqq: { symbol: "QQQ", price: 468.50, change: 4.82, changePct: 1.04 },
  vix: { level: 15.20, value: 15.20, change: -0.67, changePct: -4.22, tier: "NORMAL" },
  treasury10Y: { yield: 4.21, changeBps: -2.4 },
  tenYearYield: { value: 4.21, dailyChangeBp: -2.4 },
  regime: "RISK_ON",
  regimeSummary: "Favorable liquidity environment with low volatility tailwinds.",
  marketSession: "CLOSED",
  settlementPinned: true,
  isSettlementPinned: true,
  dataSource: "CACHED_STORE",
  updatedAt: "2026-09-07T12:00:00Z",
};

export function MarketCommandRibbonSkeleton() {
  return (
    <div
      role="region"
      aria-label="Market Command Ribbon Loading"
      data-testid="market-command-ribbon-skeleton"
      className="h-9 min-h-[36px] max-h-[36px] border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur flex items-center overflow-hidden"
    >
      <div className="max-w-[1750px] mx-auto px-2 sm:px-4 lg:px-4 xl:px-6 w-full flex items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-4 sm:space-x-6">
          <div className="h-4 w-20 bg-slate-800/80 rounded animate-pulse" />
          <div className="h-4 w-20 bg-slate-800/80 rounded animate-pulse" />
          <div className="h-4 w-16 bg-slate-800/80 rounded animate-pulse" />
          <div className="h-4 w-20 bg-slate-800/80 rounded animate-pulse hidden sm:block" />
        </div>
        <div className="flex items-center space-x-2">
          <div className="h-5 w-24 bg-slate-800/80 rounded-full animate-pulse" />
        </div>
      </div>
    </div>
  );
}

interface MarketCommandRibbonProps {
  initialData?: MacroRibbonPayload;
}

export default function MarketCommandRibbon({ initialData }: MarketCommandRibbonProps) {
  const [data, setData] = useState<MacroRibbonPayload | null>(initialData || null);
  const [isLoading, setIsLoading] = useState<boolean>(!initialData);
  const [isCachedFallback, setIsCachedFallback] = useState<boolean>(false);

  useEffect(() => {
    if (initialData) {
      setData(initialData);
      setIsLoading(false);
      return;
    }

    let isMounted = true;

    // Fast-path: read from localStorage snapshot if available to avoid any layout shift
    try {
      const cached = localStorage.getItem("FINANCE_MARKET_SNAPSHOTS_V1");
      if (cached) {
        const parsed = JSON.parse(cached);
        if (parsed && parsed.regime) {
          setData(parsed);
          setIsCachedFallback(true);
        }
      }
    } catch {}

    const fetchMacroRibbon = async () => {
      try {
        const baseUrl = getApiBaseUrl();
        const url = `${baseUrl}/macro/ribbon`;
        const res = await fetch(url, {
          headers: ARX_API_HEADERS,
          signal: AbortSignal.timeout(4000),
        });

        if (res.ok) {
          const liveData: MacroRibbonPayload = await res.json();
          if (isMounted && liveData && liveData.regime) {
            setData(liveData);
            setIsCachedFallback(liveData.dataSource === "CACHED_STORE");
            try {
              localStorage.setItem("FINANCE_MARKET_SNAPSHOTS_V1", JSON.stringify(liveData));
            } catch {}
          }
        } else {
          // Non-200 (e.g. 503 Service Unavailable) -> Fallback to cached store
          if (isMounted) {
            setIsCachedFallback(true);
            setData((prev) => prev || DEFAULT_MACRO_SNAPSHOT);
          }
        }
      } catch {
        // Network or timeout failure -> graceful degradation
        if (isMounted) {
          setIsCachedFallback(true);
          setData((prev) => prev || DEFAULT_MACRO_SNAPSHOT);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchMacroRibbon();

    return () => {
      isMounted = false;
    };
  }, [initialData]);

  if (isLoading && !data) {
    return <MarketCommandRibbonSkeleton />;
  }

  const payload = data || DEFAULT_MACRO_SNAPSHOT;

  // Normalized values
  const spyPrice = payload.spy?.price ?? payload.spx?.price ?? 542.10;
  const spyChangePct = payload.spy?.changePct ?? payload.spx?.changePct ?? 0.72;

  const qqqPrice = payload.qqq?.price ?? 468.50;
  const qqqChangePct = payload.qqq?.changePct ?? 1.04;

  const vixLevel = payload.vix?.value ?? payload.vix?.level ?? 15.20;
  const vixChangePct = payload.vix?.changePct ?? -4.22;

  const tenYearYield = payload.tenYearYield?.value ?? payload.treasury10Y?.yield ?? 4.21;
  const changeBps = payload.tenYearYield?.dailyChangeBp ?? payload.treasury10Y?.changeBps ?? -2.4;

  const regime = payload.regime || "RISK_ON";
  const isSettlementPinned = payload.settlementPinned || payload.isSettlementPinned || payload.marketSession === "CLOSED";

  // Regime visual style mappings adhering to Anti-Cyan invariant
  let regimeBadgeClass = "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  let regimeDotClass = "bg-emerald-400 animate-pulse";
  let regimeText = "RISK ON";

  if (regime === "DEFENSIVE") {
    regimeBadgeClass = "text-rose-400 bg-rose-500/10 border-rose-500/30";
    regimeDotClass = "bg-rose-400 animate-pulse";
    regimeText = "DEFENSIVE";
  } else if (regime === "NEUTRAL") {
    regimeBadgeClass = "text-amber-400 bg-amber-500/10 border-amber-500/30";
    regimeDotClass = "bg-amber-400";
    regimeText = "NEUTRAL";
  }

  return (
    <aside
      role="region"
      aria-label="Market Command Ribbon"
      data-testid="market-command-ribbon"
      className="h-9 min-h-[36px] max-h-[36px] border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur z-40 overflow-x-auto no-scrollbar flex items-center"
    >
      <div className="max-w-[1750px] mx-auto px-2 sm:px-4 lg:px-4 xl:px-6 w-full flex items-center justify-between gap-2 sm:gap-4 shrink-0">
        {/* Left: Benchmark Indexes & Yields */}
        <div className="flex items-center space-x-3 sm:space-x-5 shrink-0">
          {/* SPY */}
          <div
            className="flex items-center gap-1 sm:gap-1.5 font-mono text-xs shrink-0"
            aria-label="S&P 500"
          >
            <span className="text-slate-400 font-semibold text-[11px]">SPY</span>
            <span className="text-white font-medium text-[11px]">${spyPrice.toFixed(2)}</span>
            <span
              className={`text-[11px] font-medium ${
                spyChangePct >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {spyChangePct >= 0 ? `+${spyChangePct.toFixed(2)}%` : `${spyChangePct.toFixed(2)}%`}
            </span>
          </div>

          <span className="text-slate-700 select-none text-xs">|</span>

          {/* QQQ */}
          <div
            className="flex items-center gap-1 sm:gap-1.5 font-mono text-xs shrink-0"
            aria-label="NASDAQ 100"
          >
            <span className="text-slate-400 font-semibold text-[11px]">QQQ</span>
            <span className="text-white font-medium text-[11px]">${qqqPrice.toFixed(2)}</span>
            <span
              className={`text-[11px] font-medium ${
                qqqChangePct >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {qqqChangePct >= 0 ? `+${qqqChangePct.toFixed(2)}%` : `${qqqChangePct.toFixed(2)}%`}
            </span>
          </div>

          <span className="text-slate-700 select-none text-xs">|</span>

          {/* VIX */}
          <div
            className="flex items-center gap-1 sm:gap-1.5 font-mono text-xs shrink-0"
            aria-label="CBOE Volatility Index"
          >
            <span className="text-slate-400 font-semibold text-[11px]">VIX</span>
            <span className="text-white font-medium text-[11px]">{vixLevel.toFixed(2)}</span>
            <span
              className={`text-[11px] font-medium ${
                vixChangePct <= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {vixChangePct >= 0 ? `+${vixChangePct.toFixed(2)}%` : `${vixChangePct.toFixed(2)}%`}
            </span>
          </div>

          <span className="text-slate-700 select-none text-xs hidden sm:inline">|</span>

          {/* 10Y Yield */}
          <div
            className="hidden sm:flex items-center gap-1.5 font-mono text-xs shrink-0"
            aria-label="10-Year Treasury Yield"
          >
            <span className="text-slate-400 font-semibold text-[11px]">10Y</span>
            <span className="text-white font-medium text-[11px]">{tenYearYield.toFixed(2)}%</span>
            <span className="text-[10px] text-slate-400 font-mono">
              {changeBps >= 0 ? `+${changeBps.toFixed(1)}bp` : `${changeBps.toFixed(1)}bp`}
            </span>
          </div>
        </div>

        {/* Right: Regime Badge & Session / Cache Indicators */}
        <div className="flex items-center space-x-2 shrink-0">
          {/* Pinned Settlement Indicator */}
          {isSettlementPinned && (
            <span
              data-testid="settlement-pinned-badge"
              className="hidden md:inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700/60 text-slate-400 text-[10px] font-mono font-medium"
            >
              <span className="w-1 h-1 rounded-full bg-slate-500" aria-hidden="true" />
              Settlement Pinned
            </span>
          )}

          {/* Cached Market Snapshot Fallback Indicator */}
          {isCachedFallback && (
            <span
              data-testid="cached-snapshot-badge"
              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-950/40 border border-amber-500/40 text-amber-300 text-[10px] font-mono font-medium"
            >
              <span className="w-1 h-1 rounded-full bg-amber-400" aria-hidden="true" />
              [Cached Market Snapshot]
            </span>
          )}

          {/* Market Regime Badge */}
          <div
            role="status"
            aria-label="Market regime"
            data-testid="market-regime-badge"
            className={`flex items-center gap-1 sm:gap-1.5 px-2 py-0.5 rounded border text-[10px] sm:text-[11px] font-mono font-bold tracking-tight uppercase ${regimeBadgeClass}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${regimeDotClass}`} aria-hidden="true" />
            <span>{regimeText}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
