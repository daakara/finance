"use client";

import { useState, useEffect } from "react";
import {
  FredMacroData,
  SecForm4Trade,
  SecFilingsResult,
  fetchFredMacroRegime,
  fetchSecForm4Insiders,
} from "../lib/institutionalFeeds";

interface InstitutionalFeedsProps {
  activeSymbol: string;
}

export default function InstitutionalFeeds({ activeSymbol }: InstitutionalFeedsProps) {
  const [macro, setMacro] = useState<FredMacroData | null>(null);
  const [filingsResult, setFilingsResult] = useState<SecFilingsResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<"FRED_MACRO" | "SEC_FORM4">("FRED_MACRO");
  const [showTechnicalEvidence, setShowTechnicalEvidence] = useState<boolean>(false);
  const [expandedAccession, setExpandedAccession] = useState<string | null>(null);

  const cleanSym = activeSymbol.toUpperCase().replace("-USD", "");

  useEffect(() => {
    let isCancelled = false;
    setIsLoading(true);

    fetchFredMacroRegime().then((macroRes) => {
      if (!isCancelled) {
        setMacro(macroRes);
      }
    });

    fetchSecForm4Insiders(activeSymbol).then((filingsRes) => {
      if (!isCancelled) {
        // Prevent late responses from displaying another ticker's data
        if (!filingsRes.symbol || filingsRes.symbol === cleanSym) {
          setFilingsResult(filingsRes);
          setIsLoading(false);
        }
      }
    });

    return () => {
      isCancelled = true;
    };
  }, [activeSymbol, cleanSym]);

  const trades: SecForm4Trade[] = filingsResult?.items ?? [];
  const latestTrade = trades[0];

  const getEventTheme = (trade: SecForm4Trade) => {
    if (trade.isMixed) {
      return {
        icon: "⚖️",
        title: "Multi-Transaction Disclosure (Mixed / Aggregates Withheld)",
        bgClass: "bg-slate-900/90 border-slate-700/90",
        textClass: "text-slate-200",
        verb: "Disclosed Multiple Mixed Transactions",
        codeLabel: "Mixed Filing",
      };
    }
    if (trade.isBuy) {
      return {
        icon: "🟢",
        title: "Corporate Insider Purchase",
        bgClass: "bg-emerald-950/30 border-emerald-800/80",
        textClass: "text-emerald-300",
        verb: "Purchased",
        codeLabel: "Code P (Open Market Purchase)",
      };
    }
    if (trade.isSell) {
      return {
        icon: "🔴",
        title: "Corporate Insider Sale",
        bgClass: "bg-rose-950/30 border-rose-800/80",
        textClass: "text-rose-300",
        verb: "Sold",
        codeLabel: "Code S (Open Market Sale)",
      };
    }
    if (trade.isGift) {
      return {
        icon: "🎁",
        title: "Bona Fide Gift Transfer",
        bgClass: "bg-sky-950/30 border-sky-800/80",
        textClass: "text-sky-300",
        verb: "Gifted / Transferred",
        codeLabel: "Code G (Bona Fide Gift)",
      };
    }
    if (trade.isExercise) {
      return {
        icon: "🟣",
        title: "Option Exercise / Conversion",
        bgClass: "bg-purple-950/30 border-purple-800/80",
        textClass: "text-purple-300",
        verb: "Exercised",
        codeLabel: "Code M (Option Exercise)",
      };
    }
    if (trade.transactionCount && trade.transactionCount > 1) {
      return {
        icon: "⚖️",
        title: `Multi-Transaction Disclosure (${trade.transactionCount} Events)`,
        bgClass: "bg-cyan-950/30 border-cyan-800/80",
        textClass: "text-cyan-300",
        verb: "Executed Multiple Transactions Totaling",
        codeLabel: "Multi-Transaction Filing",
      };
    }
    return {
      icon: "🏢",
      title: "Corporate Insider Regulatory Disclosure",
      bgClass: "bg-slate-900/80 border-slate-700/80",
      textClass: "text-slate-200",
      verb: "Disclosed",
      codeLabel: trade.transactionType || "Form 4 Disclosure",
    };
  };

  const getUnavailableCopy = (status?: string) => {
    switch (status) {
      case "UPSTREAM_RATE_LIMITED":
        return "SEC EDGAR is temporarily rate-limiting requests. Filing data is unavailable right now.";
      case "UPSTREAM_TIMEOUT":
        return "SEC EDGAR did not respond in time. Filing data is temporarily unavailable.";
      case "UPSTREAM_SERVER_ERROR":
        return "SEC EDGAR is temporarily unavailable.";
      case "UPSTREAM_NOT_FOUND":
        return `SEC Form 4 filings are unavailable for ${cleanSym} (unregistered issuer, foreign FPI, or exempt investment company).`;
      case "STRUCTURALLY_INVALID":
      case "TRANSPORT_INCOMPLETE":
        return "SEC filing data could not be verified from the upstream response.";
      case "CONFIGURATION_BLOCKED":
        return "SEC filing integration is currently unavailable.";
      case "UNKNOWN_FAIL_CLOSED":
      default:
        return "SEC EDGAR filing data is currently unavailable.";
    }
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono">
      {/* Header with Sub-Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1b2434] pb-3">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
          <h3 className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center gap-1.5">
            <span>🏛️</span>
            <span>Federal Reserve (FRED) &amp; SEC EDGAR Regulatory Disclosures</span>
          </h3>
        </div>

        <div className="flex items-center space-x-1.5 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
          <button
            type="button"
            onClick={() => setActiveTab("FRED_MACRO")}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors ${
              activeTab === "FRED_MACRO"
                ? "bg-cyan-950 text-cyan-300 border border-cyan-700"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            📊 FRED Macro Regime
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("SEC_FORM4")}
            className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors ${
              activeTab === "SEC_FORM4"
                ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            🏢 SEC Form 4 Disclosures
          </button>
        </div>
      </div>

      {/* VIEW 1: FRED MACRO REGIME */}
      {activeTab === "FRED_MACRO" && (
        isLoading ? (
          <div className="bg-[#090d14] p-4 rounded-lg text-xs text-slate-400 text-center animate-pulse border border-[#1b2434]">
            Synchronizing authentic macroeconomic feeds from St. Louis Fed (FRED)...
          </div>
        ) : !macro || macro.status === "UNAVAILABLE" ? (
          <div className="space-y-4" data-testid="fred-macro-unavailable">
            <div
              role="alert"
              className="bg-amber-950/20 border border-amber-800/60 rounded-xl p-4 text-xs space-y-1.5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center space-x-2 text-amber-400 font-bold">
                  <span>⚠️</span>
                  <span>FRED Macroeconomic Feeds Unavailable</span>
                </div>
                <span className="text-[10px] text-slate-400 font-bold bg-[#141b28] px-2 py-0.5 rounded border border-[#243044]">
                  Risk Budget Multiplier Inactive (1.0x Neutral)
                </span>
              </div>
              <p className="text-slate-300 font-sans leading-relaxed">
                {macro?.regimeSubtitle ||
                  "Authoritative macroeconomic indicators (T10Y2Y yield curve spread, high-yield credit spread, 10Y real interest rate, and fed funds rate) are currently unavailable from live Federal Reserve feeds."}
              </p>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">10Y-2Y Yield Curve</span>
                <span className="text-sm sm:text-base font-extrabold text-slate-400 tabular-nums">Unavailable</span>
                <span className="text-[9px] text-slate-500 block">Series T10Y2Y</span>
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">High Yield OAS Spread</span>
                <span className="text-sm sm:text-base font-extrabold text-slate-400 tabular-nums">Unavailable</span>
                <span className="text-[9px] text-slate-500 block">Series BAMLH0A0HYM2</span>
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">10Y Real Rate (TIPS)</span>
                <span className="text-sm sm:text-base font-extrabold text-slate-400 tabular-nums">Unavailable</span>
                <span className="text-[9px] text-slate-500 block">Series DFII10</span>
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">Effective Fed Funds</span>
                <span className="text-sm sm:text-base font-extrabold text-slate-400 tabular-nums">Unavailable</span>
                <span className="text-[9px] text-slate-500 block">Series FEDFUNDS • Monthly Average</span>
                <span className="text-[8px] text-slate-500 block">(Distinct from daily DFF)</span>
              </div>
            </div>
          </div>
        ) : (
          <div
            className="space-y-4"
            data-testid={macro.status === "STALE" ? "fred-macro-stale" : "fred-macro-available"}
          >
            {macro.status === "STALE" ? (
              <div
                role="alert"
                className="bg-amber-950/20 border border-amber-800/60 rounded-xl p-3.5 space-y-1.5"
                data-testid="fred-macro-stale-banner"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center space-x-2 text-amber-400 font-bold text-xs">
                    <span>⚠️</span>
                    <span>{macro.regimeTitle || "Macro Telemetry Stale / Multiplier Inactive"}</span>
                  </div>
                  <span className="text-[10px] text-amber-300 font-bold bg-[#141b28] px-2 py-0.5 rounded border border-amber-800/60">
                    Risk Budget Multiplier Inactive (1.0x Neutral)
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                  {macro.regimeSubtitle}
                </p>
                {macro.freshnessSummary?.staleSeries && macro.freshnessSummary.staleSeries.length > 0 && (
                  <div className="text-[10px] text-amber-400/90 font-mono pt-0.5">
                    Stale series detected: {macro.freshnessSummary.staleSeries.join(", ")}
                  </div>
                )}
                {macro.lastUpdated && (
                  <div className="text-[10px] text-slate-500 font-mono pt-0.5">
                    Conservative observation baseline: {macro.lastUpdated}
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <strong className="text-xs font-bold text-cyan-400">{macro.regimeTitle}</strong>
                  {typeof macro.macroRiskMultiplier === "number" && (
                    <span className="text-[10px] text-emerald-400 font-bold bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
                      {macro.macroRiskMultiplier.toFixed(2)}x Risk Budget Active
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                  {macro.regimeSubtitle}
                </p>
                {macro.lastUpdated && (
                  <div className="text-[10px] text-slate-500 font-mono pt-1">
                    Observed as of: {macro.lastUpdated}
                  </div>
                )}
              </div>
            )}

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">10Y-2Y Yield Curve</span>
                <span
                  className={`text-sm sm:text-base font-extrabold tabular-nums ${
                    typeof macro.yieldCurve10Y2Y === "number"
                      ? macro.yieldCurve10Y2Y >= 0
                        ? "text-emerald-400"
                        : "text-rose-400"
                      : "text-slate-400"
                  }`}
                >
                  {typeof macro.yieldCurve10Y2Y === "number"
                    ? `${macro.yieldCurve10Y2Y >= 0 ? "+" : ""}${macro.yieldCurve10Y2Y.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series T10Y2Y</span>
                {(macro.yieldCurveObservationDate || macro.seriesObservations?.["T10Y2Y"]?.date) && (
                  <span className="text-[9px] text-slate-500 font-mono block">
                    FRED: {macro.yieldCurveObservationDate || macro.seriesObservations?.["T10Y2Y"]?.date}
                  </span>
                )}
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">High Yield OAS Spread</span>
                <span
                  className={`text-sm sm:text-base font-extrabold tabular-nums ${
                    typeof macro.highYieldCreditSpread === "number"
                      ? macro.highYieldCreditSpread < 4.0
                        ? "text-emerald-400"
                        : "text-amber-400"
                      : "text-slate-400"
                  }`}
                >
                  {typeof macro.highYieldCreditSpread === "number"
                    ? `${macro.highYieldCreditSpread.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series BAMLH0A0HYM2</span>
                {(macro.creditSpreadObservationDate || macro.seriesObservations?.["BAMLH0A0HYM2"]?.date) && (
                  <span className="text-[9px] text-slate-500 font-mono block">
                    FRED: {macro.creditSpreadObservationDate || macro.seriesObservations?.["BAMLH0A0HYM2"]?.date}
                  </span>
                )}
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">10Y Real Rate (TIPS)</span>
                <span className="text-sm sm:text-base font-extrabold text-cyan-300 tabular-nums">
                  {typeof macro.realInterestRate10Y === "number"
                    ? `${macro.realInterestRate10Y.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series DFII10</span>
                {(macro.realRateObservationDate || macro.seriesObservations?.["DFII10"]?.date) && (
                  <span className="text-[9px] text-slate-500 font-mono block">
                    FRED: {macro.realRateObservationDate || macro.seriesObservations?.["DFII10"]?.date}
                  </span>
                )}
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">Effective Fed Funds</span>
                <span className="text-sm sm:text-base font-extrabold text-amber-400 tabular-nums">
                  {typeof macro.fedFundsRate === "number"
                    ? `${macro.fedFundsRate.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-400 block font-medium">Series FEDFUNDS • Monthly Average</span>
                <span className="text-[8px] text-slate-500 block">(Distinct from daily DFF)</span>
                {(macro.fedFundsObservationDate || macro.seriesObservations?.["FEDFUNDS"]?.date) && (
                  <span className="text-[9px] text-slate-500 font-mono block">
                    FRED: {macro.fedFundsObservationDate || macro.seriesObservations?.["FEDFUNDS"]?.date}
                  </span>
                )}
              </div>
            </div>
          </div>
        )
      )}

      {/* VIEW 2: SEC FORM 4 CORPORATE INSIDERS */}
      {activeTab === "SEC_FORM4" && (
        <div className="space-y-3">
          {isLoading ? (
            <div className="bg-[#090d14] p-4 rounded-lg text-xs text-slate-400 text-center animate-pulse border border-[#1b2434]">
              Loading SEC EDGAR Form 4 corporate insider disclosures...
            </div>
          ) : filingsResult === null || filingsResult.items === null ? (
            /* Unavailable / Indeterminate / Error State */
            <div
              role="alert"
              data-testid="sec-unavailable-banner"
              className="bg-amber-950/20 border border-amber-800/60 rounded-xl p-4 text-xs space-y-1.5"
            >
              <div className="flex items-center space-x-2 text-amber-400 font-bold">
                <span>⚠️</span>
                <span>SEC EDGAR Data Feed Unavailable</span>
              </div>
              <p className="text-slate-300 font-sans leading-relaxed">
                {getUnavailableCopy(filingsResult?.status)}
              </p>
            </div>
          ) : filingsResult.status === "AVAILABLE_NO_RESULTS" || trades.length === 0 ? (
            /* Genuine Empty State */
            <div
              data-testid="sec-empty-banner"
              className="bg-[#090d14] border border-[#1b2434] p-4 rounded-xl text-xs text-slate-400 text-center space-y-1"
            >
              <div className="text-slate-300 font-bold">No Qualifying Disclosures Found</div>
              <p className="text-slate-500 font-sans text-[11px]">
                No qualifying recent SEC Form 4 disclosures were found for {cleanSym} in the last 30 days.
              </p>
            </div>
          ) : (
            /* Populated Filings State */
            <>
              {/* Top Operational / Coverage Metadata Strip */}
              <div className="flex flex-wrap items-center justify-between gap-2 bg-[#090d14] px-3 py-2 rounded-lg border border-[#1b2434] text-[11px]">
                <div className="flex items-center space-x-2 flex-wrap">
                  <span className="text-slate-400 font-medium">Filing Authority:</span>
                  <span className="text-slate-200 font-bold">SEC EDGAR Form 4</span>
                  {filingsResult?.coverageStatus === "COMPLETE" ? (
                    <span className="bg-emerald-950 text-emerald-300 border border-emerald-800 px-1.5 py-0.5 rounded text-[10px] font-bold">
                      Coverage: Certified Complete
                    </span>
                  ) : filingsResult?.coverageStatus === "PARTIAL" ? (
                    <span className="bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded text-[10px] font-bold">
                      Coverage: Partial (Bounds / Gaps)
                    </span>
                  ) : (
                    <span className="bg-slate-800 text-slate-300 border border-slate-700 px-1.5 py-0.5 rounded text-[10px]">
                      Coverage: {filingsResult?.coverageStatus || "Supported SEC Issuer"}
                    </span>
                  )}
                  {filingsResult?.certifiedWatermark && (
                    <span className="text-[10px] text-cyan-400 font-mono bg-cyan-950/40 border border-cyan-800/60 px-1.5 py-0.5 rounded">
                      Certified Watermark: {filingsResult.certifiedWatermark.slice(0, 10)}
                    </span>
                  )}
                </div>
                <div className="flex items-center space-x-2 flex-wrap text-[10px] font-mono">
                  {filingsResult?.isPipelineHeld && (
                    <span className="bg-amber-950/40 border border-amber-800/60 text-amber-400 px-2 py-0.5 rounded">
                      ⏸️ HOLD_SEC_001 Standby (Stored Facts Disclosed)
                    </span>
                  )}
                  <span className="text-slate-400">
                    Evidence As-of: <span className="text-slate-300">{filingsResult?.asOf ? filingsResult.asOf.slice(0, 10) : "Pending / Unverified"}</span>
                  </span>
                  {filingsResult?.servedAt && (
                    <span className="text-slate-500">
                      Served: {filingsResult.servedAt.slice(11, 19)} UTC
                    </span>
                  )}
                </div>
              </div>

              {/* Highlight Banner: Render latest disclosed event without favoring purchases */}
              {latestTrade && latestTrade.hasParsedTransaction ? (
                (() => {
                  const theme = getEventTheme(latestTrade);
                  return (
                    <div className={`${theme.bgClass} border rounded-xl p-3.5 text-xs space-y-2`}>
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center space-x-2 flex-wrap">
                          <span className="text-base">{theme.icon}</span>
                          <strong className={`${theme.textClass} font-bold text-sm`}>
                            {latestTrade.insiderName || "Corporate Insider"} ({latestTrade.insiderRole || "Officer / Director"})
                          </strong>
                          {latestTrade.isAmendment && (
                            <span className="text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded">
                              Amendment
                            </span>
                          )}
                          {latestTrade.hasAmbiguousAmendment && (
                            <span className="text-[10px] font-bold bg-amber-900/60 text-amber-200 border border-amber-700 px-1.5 py-0.5 rounded" title="Target of ambiguous amendment (effective impact unresolved)">
                              Ambiguous Amendment
                            </span>
                          )}
                          {latestTrade.isSuperseded && (
                            <span className="text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-800 px-1.5 py-0.5 rounded">
                              Superseded
                            </span>
                          )}
                        </div>
                        <a
                          href={latestTrade.secEdgarUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[10px] font-bold text-cyan-400 hover:underline flex items-center gap-1"
                        >
                          <span>SEC EDGAR Form {latestTrade.formType || "4"}</span>
                          <span>↗</span>
                        </a>
                      </div>
                      {latestTrade.isMixed ? (
                        <div className="space-y-1">
                          <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                            {theme.verb} across {latestTrade.transactionCount || "multiple"} distinct events. Because this filing contains transactions across different owners, securities, or transaction codes, aggregate totals are withheld to avoid misleading figures.
                          </p>
                          <span className="text-[10px] text-cyan-400/90 font-mono block">
                            See individual transaction breakdown below.
                          </span>
                        </div>
                      ) : (
                        <p className="text-[11px] text-slate-200 font-sans leading-relaxed">
                          {theme.verb}{" "}
                          <strong>
                            {typeof latestTrade.sharesTraded === "number"
                              ? `${latestTrade.sharesTraded.toLocaleString()} shares`
                              : "disclosed shares"}
                          </strong>
                          {latestTrade.isGift ? (
                            latestTrade.pricePerShare === 0 ? (
                              <> via <strong>Bona Fide Gift</strong> (Reported consideration: <strong>$0.00</strong> — zero cash exchange, not zero economic value).</>
                            ) : typeof latestTrade.pricePerShare === "number" && latestTrade.pricePerShare > 0 ? (
                              <> via <strong>Bona Fide Gift</strong> (Reported consideration: <strong>${latestTrade.pricePerShare.toFixed(2)}</strong> per share).</>
                            ) : (
                              <> via <strong>Bona Fide Gift</strong> (Reported consideration: <strong>Not Reported / None</strong> — no cash consideration reported on Form 4).</>
                            )
                          ) : latestTrade.pricePerShare === 0 ? (
                            <> at <strong>$0.00</strong> (Zero Consideration / Option Exercise / Grant).</>
                          ) : typeof latestTrade.pricePerShare === "number" ? (
                            <> on the open market at <strong>${latestTrade.pricePerShare.toFixed(2)}</strong> (Total: <strong>{latestTrade.totalValueUsd !== null ? `$${(latestTrade.totalValueUsd / 1000000).toFixed(2)}M USD` : "Unavailable"}</strong>).</>
                          ) : (
                            <> (Total transaction consideration: <strong>{latestTrade.totalValueUsd !== null ? `$${(latestTrade.totalValueUsd / 1000000).toFixed(2)}M USD` : "Unavailable"}</strong>).</>
                          )}
                          {latestTrade.transactionDate && (
                            <span className="text-slate-400 block text-[10px] mt-0.5 font-mono">
                              Transaction Date: {latestTrade.transactionDate} • Filed: {latestTrade.filingDate || "recent"}
                            </span>
                          )}
                        </p>
                      )}
                    </div>
                  );
                })()
              ) : latestTrade ? (
                <div className="bg-[#0b1019] border border-[#243044] rounded-xl p-3.5 text-xs space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-base">📋</span>
                      <strong className="text-slate-200 font-bold text-sm">
                        Official Regulatory Disclosure ({cleanSym})
                      </strong>
                      {latestTrade.isAmendment && (
                        <span className="text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded">
                          Amendment
                        </span>
                      )}
                    </div>
                    {latestTrade.secEdgarUrl && (
                      <a
                        href={latestTrade.secEdgarUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] font-bold text-cyan-400 hover:underline flex items-center gap-1"
                      >
                        <span>SEC EDGAR Source</span>
                        <span>↗</span>
                      </a>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                    Recent SEC Form {latestTrade.formType || "4"} disclosure filed on <strong>{latestTrade.filingDate || "recent date"}</strong>
                    {latestTrade.accessionNumber ? (
                      <> (Accession: <span className="font-mono text-slate-400">{latestTrade.accessionNumber}</span>)</>
                    ) : null}.
                    Transaction-level breakdown (shares, price, total value) is unparsed from submission metadata; no buying or selling is inferred from filing existence.
                  </p>
                </div>
              ) : null}

              {/* Expandable Technical Verification Evidence */}
              {latestTrade && (
                <div className="bg-[#090d14] border border-[#1b2434] rounded-lg p-2.5 text-xs">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      data-testid="sec-evidence-toggle"
                      onClick={() => setShowTechnicalEvidence(!showTechnicalEvidence)}
                      className="text-[11px] font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1.5 transition-colors"
                    >
                      <span>{showTechnicalEvidence ? "▼ Hide Technical Verification Evidence" : "▶ Show Technical Verification Evidence"}</span>
                      <span className="text-slate-500 text-[10px] font-normal">
                        ({latestTrade.accessionNumber || "Filing Evidence"})
                      </span>
                    </button>
                    <span className="text-[10px] text-slate-500 font-mono">
                      Form {latestTrade.formType || "4"}
                    </span>
                  </div>

                  {showTechnicalEvidence && (
                    <div className="mt-3 pt-3 border-t border-[#1b2434] space-y-2.5 text-[11px] text-slate-300 font-mono">
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-[10px]">
                        <div className="bg-[#0e1420] p-2 rounded border border-[#1a2538]">
                          <span className="text-slate-500 block uppercase">Accession Number</span>
                          <span className="text-slate-200 select-all font-bold">{latestTrade.accessionNumber || "—"}</span>
                        </div>
                        <div className="bg-[#0e1420] p-2 rounded border border-[#1a2538]">
                          <span className="text-slate-500 block uppercase">SEC Acceptance Timestamp</span>
                          <span className="text-slate-200">{latestTrade.acceptanceDatetime || "Disclosed on filing"}</span>
                        </div>
                        <div className="bg-[#0e1420] p-2 rounded border border-[#1a2538]">
                          <span className="text-slate-500 block uppercase">XML Content SHA-256</span>
                          <span className="text-slate-200 select-all truncate block" title={latestTrade.xmlHash || "Verified store content"}>
                            {latestTrade.xmlHash ? `${latestTrade.xmlHash.slice(0, 16)}...` : "Verified store content"}
                          </span>
                        </div>
                        <div className="bg-[#0e1420] p-2 rounded border border-[#1a2538]">
                          <span className="text-slate-500 block uppercase">Public Availability Watermark</span>
                          <span className="text-slate-200">{latestTrade.publicAvailabilityAt || "Available at acceptance"}</span>
                        </div>
                      </div>

                      {latestTrade.isAmendment && (
                        <div className="p-2 bg-amber-950/20 border border-amber-800/40 rounded text-[10px] text-amber-300">
                          ⚠️ Amendment Record: Supersedes base accession <span className="font-bold">{latestTrade.baseAccession || "historical filing"}</span>
                          {typeof latestTrade.amendmentNumber === "number" && ` (Amendment #${latestTrade.amendmentNumber})`}.
                        </div>
                      )}

                      {latestTrade.transactions && latestTrade.transactions.length > 0 && (
                        <div className="mt-2 space-y-1">
                          <div className="text-[10px] uppercase font-bold text-slate-400">
                            Filing Breakdown: {latestTrade.transactions.length} Disclosed Transactions
                          </div>
                          <div className="overflow-x-auto">
                            <table className="w-full text-left text-[10px] divide-y divide-[#1b2434]">
                              <thead className="bg-[#0b1019] text-slate-400">
                                <tr>
                                  <th className="py-1 px-2">Tx ID</th>
                                  <th className="py-1 px-2">Date</th>
                                  <th className="py-1 px-2">Type / Code</th>
                                  <th className="py-1 px-2">Shares</th>
                                  <th className="py-1 px-2">Price</th>
                                  <th className="py-1 px-2">Total ($)</th>
                                  <th className="py-1 px-2">Security</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-[#1b2434] tabular-nums">
                                {latestTrade.transactions.map((tx, txIdx) => (
                                  <tr key={txIdx} className="hover:bg-[#151e2d]">
                                    <td className="py-1 px-2 text-slate-400 font-mono text-[9px]">{tx.transactionId ? tx.transactionId.slice(-12) : `#${txIdx+1}`}</td>
                                    <td className="py-1 px-2 text-slate-300">{tx.transactionDate || "—"}</td>
                                    <td className="py-1 px-2">
                                      <span className={tx.isBuy ? "text-emerald-400 font-bold" : (tx.isSell ? "text-rose-400 font-bold" : (tx.isGift ? "text-sky-400 font-bold" : "text-slate-300 font-bold"))}>
                                        {tx.transactionType}
                                      </span>
                                    </td>
                                    <td className="py-1 px-2 text-slate-200">
                                      {typeof tx.shares === "number" ? tx.shares.toLocaleString() : "Unavailable"}
                                    </td>
                                    <td className="py-1 px-2 text-slate-200">
                                      {tx.isGift ? (
                                        tx.pricePerShare === 0 ? "$0.00" : (typeof tx.pricePerShare === "number" ? `$${tx.pricePerShare.toFixed(2)}` : "None Reported")
                                      ) : tx.pricePerShare === 0
                                        ? "$0.00"
                                        : (typeof tx.pricePerShare === "number" ? `$${tx.pricePerShare.toFixed(2)}` : "Unavailable")}
                                    </td>
                                    <td className="py-1 px-2 font-bold">
                                      {tx.totalValueUsd === 0
                                        ? "$0.00"
                                        : (typeof tx.totalValueUsd === "number" ? (
                                            <span className={tx.isBuy ? "text-emerald-400" : (tx.isSell ? "text-rose-400" : "text-slate-200")}>
                                              ${tx.totalValueUsd >= 1000000 ? (tx.totalValueUsd / 1000000).toFixed(2) + "M" : tx.totalValueUsd.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                                            </span>
                                          ) : "Unavailable")}
                                    </td>
                                    <td className="py-1 px-2 text-slate-400 text-[9px]">{tx.securityTitle || (tx.isDerivative ? "Derivative Security" : "Unspecified Security")}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Filings Table */}
              <div className="overflow-x-auto pt-1">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#090d14] text-slate-400 text-[10px] uppercase border-b border-[#1b2434]">
                    <tr>
                      <th className="py-2.5 px-3">Ticker</th>
                      <th className="py-2.5 px-3">Filing / Reporting Person</th>
                      <th className="py-2.5 px-3">Shares</th>
                      <th className="py-2.5 px-3">Price ($)</th>
                      <th className="py-2.5 px-3">Total ($)</th>
                      <th className="py-2.5 px-3">Dates</th>
                      <th className="py-2.5 px-3 text-right">EDGAR Source</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1b2434] tabular-nums">
                    {trades.map((trade, idx) => {
                      const isExpanded = expandedAccession === trade.accessionNumber;
                      return (
                        <tr key={idx} className="hover:bg-[#151e2d] transition-colors">
                          <td className="py-2 px-3 font-bold text-cyan-400">{trade.ticker}</td>
                          <td className="py-2 px-3 text-slate-200">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span className="font-semibold">
                                {trade.insiderName || `Form ${trade.formType || "4"} Public Filing`}
                              </span>
                              {trade.isAmendment && (
                                <span className="text-[9px] font-bold bg-amber-950 text-amber-400 border border-amber-800 px-1 py-0.5 rounded">
                                  4/A
                                </span>
                              )}
                              {trade.hasAmbiguousAmendment && (
                                <span className="text-[9px] font-bold bg-amber-900/60 text-amber-200 border border-amber-700 px-1 py-0.5 rounded" title="Target of ambiguous amendment">
                                  Ambiguous 4/A
                                </span>
                              )}
                              {trade.isSuperseded && (
                                <span className="text-[9px] font-bold bg-rose-950 text-rose-300 border border-rose-800 px-1 py-0.5 rounded">
                                  Superseded
                                </span>
                              )}
                              {trade.isMixed && (
                                <span className="text-[9px] font-bold bg-slate-800 text-slate-300 border border-slate-700 px-1 py-0.5 rounded">
                                  Mixed
                                </span>
                              )}
                              {trade.isGift && (
                                <span className="text-[9px] font-bold bg-sky-950 text-sky-400 border border-sky-800 px-1 py-0.5 rounded">
                                  Gift
                                </span>
                              )}
                              {trade.isExercise && (
                                <span className="text-[9px] font-bold bg-purple-950 text-purple-400 border border-purple-800 px-1 py-0.5 rounded">
                                  Exercise
                                </span>
                              )}
                            </div>
                            <span className="text-[10px] text-slate-400 block font-normal font-sans">
                              {trade.insiderRole || (trade.accessionNumber ? `Accession: ${trade.accessionNumber}` : trade.transactionType)}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-slate-300">
                            {typeof trade.sharesTraded === "number" ? (
                              trade.sharesTraded.toLocaleString()
                            ) : (
                              <span className="text-slate-500 font-sans font-normal">Unavailable</span>
                            )}
                          </td>
                          <td className="py-2 px-3 text-slate-300">
                            {trade.isGift ? (
                              trade.pricePerShare === 0 ? (
                                <span className="text-slate-400 font-mono">$0.00</span>
                              ) : typeof trade.pricePerShare === "number" ? (
                                `$${trade.pricePerShare.toFixed(2)}`
                              ) : (
                                <span className="text-slate-500 font-sans font-normal">None Reported</span>
                              )
                            ) : trade.pricePerShare === 0 ? (
                              <span className="text-slate-400 font-mono">$0.00</span>
                            ) : typeof trade.pricePerShare === "number" ? (
                              `$${trade.pricePerShare.toFixed(2)}`
                            ) : (
                              <span className="text-slate-500 font-sans font-normal">Unavailable</span>
                            )}
                          </td>
                          <td className="py-2 px-3 font-bold">
                            {trade.totalValueUsd === 0 ? (
                              <span className="text-slate-400 font-mono">$0.00</span>
                            ) : typeof trade.totalValueUsd === "number" ? (
                              <span className={trade.isBuy ? "text-emerald-400" : (trade.isSell ? "text-rose-400" : "text-slate-200")}>
                                ${(trade.totalValueUsd / 1000000).toFixed(2)}M
                              </span>
                            ) : (
                              <span className="text-slate-500 font-sans font-normal">Unavailable</span>
                            )}
                          </td>
                          <td className="py-2 px-3 text-slate-400 text-[11px]">
                            <div>{trade.filingDate || "—"}</div>
                            {trade.transactionDate && (
                              <div className="text-[9px] text-slate-500 font-mono">Tx: {trade.transactionDate}</div>
                            )}
                          </td>
                          <td className="py-2 px-3 text-right">
                            <a
                              href={trade.secEdgarUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[10px] font-bold text-cyan-400 hover:text-cyan-300 underline"
                            >
                              {trade.formType ? `Form ${trade.formType} ↗` : "EDGAR ↗"}
                            </a>
                            {trade.transactions && trade.transactions.length > 1 && (
                              <button
                                type="button"
                                onClick={() => setExpandedAccession(isExpanded ? null : (trade.accessionNumber || null))}
                                className="block text-[9px] text-cyan-400 hover:underline mt-1 text-right ml-auto"
                              >
                                {isExpanded ? "Hide breakdown ▲" : `+ ${trade.transactions.length} Txs ▼`}
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
