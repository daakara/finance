"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import HistoricalEdgeScorecard from "../../components/HistoricalEdgeScorecard";
import { fetchAssetAnalytics, AnalyticsResponse } from "../../lib/api";
import { resolveCapabilities } from "../../lib/assetTypeUtils";

function extractTickerFromPath(): string {
  if (typeof window === "undefined") return "";
  const path = window.location.pathname;
  // Match /stock/:ticker or /stock/:ticker/
  const match = path.match(/\/stock\/([A-Za-z0-9_-]+)/i);
  if (match && match[1]) {
    const raw = match[1].trim().toUpperCase();
    if (raw !== "INDEX" && raw !== "STOCK" && raw !== "PAGE") {
      return raw.replace(/-USD$/, "");
    }
  }
  return "";
}

function StockDetailLoadingSkeleton({ ticker }: { ticker?: string }) {
  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans">
      <Navbar />
      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 font-mono space-y-6">
        <div className="bg-[#0b1019] p-6 rounded-2xl border border-[#1e293b] animate-pulse space-y-4">
          <div className="h-6 w-32 bg-slate-800 rounded" />
          <div className="h-10 w-64 bg-slate-800 rounded" />
          <div className="h-4 w-96 bg-slate-800 rounded" />
        </div>
        <div className="bg-[#111722] border border-[#243044] rounded-xl p-8 text-center text-slate-400 space-y-3">
          <span className="w-3 h-3 rounded-full bg-cyan-400 animate-ping inline-block" />
          <p className="text-sm">
            Resolving authoritative Security Master classification{ticker ? ` for ${ticker}` : ""}...
          </p>
        </div>
      </main>
    </div>
  );
}

function StockDynamicContent() {
  const searchParams = useSearchParams();
  const queryTicker = searchParams.get("symbol") || searchParams.get("ticker");

  const [ticker, setTicker] = useState<string>("");
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const resolvedTicker = (queryTicker || extractTickerFromPath()).trim().toUpperCase();
    setTicker(resolvedTicker);
  }, [queryTicker]);

  useEffect(() => {
    if (!ticker) {
      setLoading(false);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchAssetAnalytics(ticker, "1y", "1d", "LONG_TERM")
      .then((res) => {
        if (isMounted) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.error(`Failed to load stock detail analytics for ${ticker}:`, err);
          setError(err instanceof Error ? err.message : "Failed to load market analytics.");
        }
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [ticker]);

  // If no ticker provided, show search & catalog prompt
  if (!ticker && !loading) {
    return (
      <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans">
        <Navbar />
        <main className="max-w-4xl mx-auto px-4 sm:px-6 py-12 sm:py-16 font-mono space-y-6">
          <div className="bg-[#0b1019] p-8 rounded-2xl border border-[#1e293b] text-center space-y-4">
            <h1 className="text-2xl font-bold text-white">Stock Intelligence Directory</h1>
            <p className="text-sm text-slate-400 max-w-md mx-auto font-sans">
              Search any equity ticker in the terminal or browse screened candidates to inspect institutional Minervini execution levels.
            </p>
            <div className="flex justify-center gap-3 pt-2">
              <Link
                href="/"
                className="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl text-xs transition-all shadow"
              >
                Go to Live Terminal →
              </Link>
              <Link
                href="/screener"
                className="px-5 py-2.5 bg-[#111722] hover:bg-[#182335] border border-[#24334a] text-cyan-300 font-bold rounded-xl text-xs transition-all"
              >
                Explore Screener
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (loading) {
    return <StockDetailLoadingSkeleton ticker={ticker} />;
  }

  const capabilities = resolveCapabilities(data, data?.optimalExecution);
  const canonicalInst = data?.canonicalInstrument || data?.instrument;
  const companyName = `${ticker} Equity`;
  const exchange = canonicalInst?.primary_exchange || "US";
  const secType = canonicalInst?.security_type || data?.securityType || "EQUITY";
  const price = data?.currentPrice;
  const changePct = data?.priceChangePct24h;

  const executionPlan = data?.optimalExecution;
  const stopLoss = executionPlan?.stop_loss;
  const entryMin = executionPlan?.optimal_entry_min;
  const entryMax = executionPlan?.optimal_entry_max;
  const target1 = executionPlan?.take_profit_1;
  const target2 = executionPlan?.take_profit_2;
  const riskReward = executionPlan?.risk_reward_ratio;

  const factorScores = data?.factorScores;
  const piotroskiScore = factorScores?.piotroskiFScore;

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans selection:bg-cyan-500 selection:text-black transition-colors duration-200">
      <Navbar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 font-mono space-y-8 pb-24 sm:pb-16">
        {/* Breadcrumb Nav */}
        <nav className="text-xs text-slate-500 flex items-center space-x-2">
          <Link href="/" className="hover:text-cyan-400">Terminal</Link>
          <span>/</span>
          <Link href="/screener" className="hover:text-cyan-400">Screener</Link>
          <span>/</span>
          <span className="text-slate-300 font-bold">{ticker}</span>
        </nav>

        {/* Hero Header */}
        <header className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-3">
                <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-xs font-bold font-mono">
                  {ticker}
                </span>
                <span className="text-slate-400 text-xs font-sans">
                  • {secType.replace(/_/g, " ")} ({exchange})
                </span>
                {capabilities.canRenderStockExecution && (
                  <span className="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800 text-[10px] font-bold">
                    VERIFIED COMMON STOCK
                  </span>
                )}
              </div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mt-1">
                {companyName} ({ticker})
              </h1>
            </div>

            <div className="text-right">
              {typeof price === "number" ? (
                <>
                  <div className="text-2xl sm:text-3xl font-bold text-white font-mono">
                    ${price.toFixed(2)}
                  </div>
                  {typeof changePct === "number" && (
                    <div className={`text-xs font-bold ${changePct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      {changePct >= 0 ? "+" : ""}{changePct.toFixed(2)}% (24h)
                    </div>
                  )}
                </>
              ) : (
                <div className="text-sm text-slate-500 font-mono">Tape Stream Pending</div>
              )}
            </div>
          </div>
        </header>

        {/* Security Master Execution Gate / Unresolved Warning */}
        {!capabilities.canRenderStockDetails ? (
          <section className="bg-[#181106] border border-amber-800/80 p-6 rounded-2xl space-y-3" role="alert">
            <div className="flex items-center space-x-2 text-amber-300 font-bold">
              <span>⚠️</span>
              <h2 className="text-sm font-mono">🎯 {ticker} Execution Unresolved / Ineligible</h2>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed font-sans">
              {capabilities.disqualificationReason ||
                `Asset classification for "${ticker}" is unverified under ARX Server Security Master integrity rules.`}
            </p>
            <div className="pt-2 flex gap-3">
              <Link
                href={`/?symbol=${ticker}`}
                className="px-4 py-2 bg-[#111722] hover:bg-[#182335] border border-[#24334a] text-cyan-300 rounded-lg text-xs font-bold transition-all"
              >
                Open in General Terminal
              </Link>
              <Link
                href="/screener"
                className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-bold transition-all"
              >
                Explore Screened Stocks
              </Link>
            </div>
          </section>
        ) : (
          <>
            {/* Minervini VCP Execution Blueprint */}
            <section className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
                <div>
                  <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                    Minervini VCP Execution Plan
                  </h2>
                  <p className="text-xs text-slate-400 font-sans mt-0.5">
                    Authoritative swing execution levels grounded in 250+ session volatility contraction analysis.
                  </p>
                </div>
                {(executionPlan?.execution_status || data?.decisionTrace?.stateLabel) && (
                  <span className="px-3 py-1 rounded-lg text-xs font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
                    {executionPlan?.execution_status || data?.decisionTrace?.stateLabel}
                  </span>
                )}
              </div>

              {executionPlan ? (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                    <span className="text-slate-400 block">Buy Zone (Entry)</span>
                    <strong className="text-white text-sm font-mono block">
                      ${entryMin?.toFixed(2) ?? "—"} - ${entryMax?.toFixed(2) ?? "—"}
                    </strong>
                    <span className="text-[10px] text-cyan-400">Optimal Accumulation</span>
                  </div>

                  <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                    <span className="text-slate-400 block">Protective Stop-Loss</span>
                    <strong className="text-rose-400 text-sm font-mono block">
                      ${stopLoss?.toFixed(2) ?? "—"}
                    </strong>
                    <span className="text-[10px] text-rose-300/80">Capital Invalidation</span>
                  </div>

                  <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                    <span className="text-slate-400 block">Profit Target 1</span>
                    <strong className="text-emerald-400 text-sm font-mono block">
                      ${target1?.toFixed(2) ?? "—"}
                    </strong>
                    <span className="text-[10px] text-emerald-300/80">First Trim Objective</span>
                  </div>

                  <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                    <span className="text-slate-400 block">Risk : Reward</span>
                    <strong className="text-cyan-300 text-sm font-mono block">
                      {typeof riskReward === "number" ? `1 : ${riskReward.toFixed(1)}` : "—"}
                    </strong>
                    <span className="text-[10px] text-slate-400">Asymmetric Edge</span>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-500 font-mono p-4 bg-[#111722] rounded-xl border border-[#1b2434]">
                  Execution levels calculating from live tape and volatility contraction filters...
                </div>
              )}

              {/* Action Banner to Live Terminal */}
              <div className="pt-2 flex flex-wrap items-center justify-between gap-3 bg-[#111722] p-4 rounded-xl border border-[#243044]">
                <div>
                  <h3 className="text-xs font-bold text-white">Execute this Setup in Live Terminal</h3>
                  <p className="text-[11px] text-slate-400 font-sans mt-0.5">
                    Open pre-flight checklists, position sizing modals, and live candle chart.
                  </p>
                </div>
                <Link
                  href={`/?symbol=${ticker}`}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg text-xs transition-transform active:scale-95 shadow"
                >
                  Launch Interactive Setup ({ticker}) →
                </Link>
              </div>
            </section>

            {/* Fundamental Factor Quality & Piotroski Radar */}
            <section className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-4">
              <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                5-Factor Institutional Quality & Solvency
              </h2>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                  <span className="text-slate-400 block">Piotroski F-Score</span>
                  <div className="flex items-center justify-between">
                    <strong className="text-white text-base font-mono">
                      {piotroskiScore !== undefined ? piotroskiScore : (data?.analytics as any)?.piotroski_f_score ?? "N/A"}
                    </strong>
                    <span className="text-[10px] text-cyan-400">
                      {piotroskiScore !== undefined && piotroskiScore >= 7 ? "/ 9 (Pristine)" : "/ 9"}
                    </span>
                  </div>
                </div>

                <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                  <span className="text-slate-400 block">Growth Score</span>
                  <div className="flex items-center justify-between">
                    <strong className="text-white text-base font-mono">{factorScores?.growthScore ?? "N/A"}</strong>
                    <span className="text-[10px] text-cyan-400">/ 100</span>
                  </div>
                </div>

                <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                  <span className="text-slate-400 block">Quality Score</span>
                  <div className="flex items-center justify-between">
                    <strong className="text-white text-base font-mono">{factorScores?.qualityScore ?? "N/A"}</strong>
                    <span className="text-[10px] text-emerald-400">/ 100</span>
                  </div>
                </div>

                <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                  <span className="text-slate-400 block">Valuation Score</span>
                  <div className="flex items-center justify-between">
                    <strong className="text-white text-base font-mono">{factorScores?.valuationScore ?? "N/A"}</strong>
                    <span className="text-[10px] text-purple-400">/ 100</span>
                  </div>
                </div>

                <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                  <span className="text-slate-400 block">Momentum Score</span>
                  <div className="flex items-center justify-between">
                    <strong className="text-white text-base font-mono">{factorScores?.momentumScore ?? "N/A"}</strong>
                    <span className="text-[10px] text-amber-400">/ 100</span>
                  </div>
                </div>

                <div className="bg-[#111722] p-3 rounded-xl border border-[#1b2434] space-y-1">
                  <span className="text-slate-400 block">Tail Risk Safety</span>
                  <div className="flex items-center justify-between">
                    <strong className="text-white text-base font-mono">{factorScores?.tailRiskScore ?? "N/A"}</strong>
                    <span className="text-[10px] text-rose-400">/ 100</span>
                  </div>
                </div>
              </div>
            </section>

            {/* Quantitative Historical Edge Scorecard */}
            <section aria-label="Quantitative Historical Edge Scorecard">
              <HistoricalEdgeScorecard strategySlug="minervini-vcp" symbol={ticker} />
            </section>
          </>
        )}

        {/* Head-to-Head Comparison Links */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Quantitative Comparisons with {ticker}
          </h2>
          <div className="flex flex-wrap gap-2 text-xs">
            <Link
              href={`/compare?a=${ticker}&b=SPY`}
              className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-cyan-300 border border-[#243044] transition-colors"
            >
              📊 {ticker} vs. S&P 500 (SPY)
            </Link>
            <Link
              href={`/compare?a=${ticker}&b=QQQ`}
              className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-amber-300 border border-[#243044] transition-colors"
            >
              📈 {ticker} vs. Nasdaq-100 (QQQ)
            </Link>
            <Link
              href="/politician/nancy-pelosi"
              className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-purple-300 border border-[#243044] transition-colors"
            >
              🏛️ Congressional Traders Tracking {ticker} →
            </Link>
          </div>
        </section>

        {/* Footer Navigation */}
        <footer className="border-t border-[#1e293b] pt-6 flex flex-wrap items-center justify-between gap-4 text-xs">
          <Link
            href="/"
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl transition-transform active:scale-95"
          >
            ← Return to Live Terminal
          </Link>
          <div className="text-slate-500 font-sans">
            Grounded in SEC EDGAR, Capitol Hill STOCK Act & Federal Reserve FRED Data
          </div>
        </footer>
      </main>
    </div>
  );
}

export default function StockDynamicPage() {
  return (
    <Suspense fallback={<StockDetailLoadingSkeleton />}>
      <StockDynamicContent />
    </Suspense>
  );
}
