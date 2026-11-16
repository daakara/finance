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

  useEffect(() => {
    setIsLoading(true);
    fetchFredMacroRegime().then(setMacro);
    fetchSecForm4Insiders(activeSymbol).then((res) => {
      setFilingsResult(res);
      setIsLoading(false);
    });
  }, [activeSymbol]);

  const cleanSym = activeSymbol.toUpperCase().replace("-USD", "");
  const trades: SecForm4Trade[] = filingsResult?.items ?? [];

  // Identify authentic parsed purchase transactions versus unparsed regulatory filing metadata
  const parsedPurchase = trades.find(
    (t) => t.hasParsedTransaction && t.isBuy && typeof t.sharesTraded === "number" && typeof t.pricePerShare === "number"
  );
  const primaryFiling = trades[0];

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
        ) : !macro || macro.status === "UNAVAILABLE" || !macro.isLive ? (
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
                <span className="text-[9px] text-slate-500 block">Series FEDFUNDS</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4" data-testid="fred-macro-available">
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

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">10Y-2Y Yield Curve</span>
                <span className="text-sm sm:text-base font-extrabold text-emerald-400 tabular-nums">
                  {typeof macro.yieldCurve10Y2Y === "number"
                    ? `${macro.yieldCurve10Y2Y >= 0 ? "+" : ""}${macro.yieldCurve10Y2Y.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series T10Y2Y</span>
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">High Yield OAS Spread</span>
                <span className="text-sm sm:text-base font-extrabold text-cyan-300 tabular-nums">
                  {typeof macro.highYieldCreditSpread === "number"
                    ? `${macro.highYieldCreditSpread.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series BAMLH0A0HYM2</span>
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">10Y Real Rate (TIPS)</span>
                <span className="text-sm sm:text-base font-extrabold text-slate-200 tabular-nums">
                  {typeof macro.realInterestRate10Y === "number"
                    ? `${macro.realInterestRate10Y.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series DFII10</span>
              </div>

              <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1e293b]">
                <span className="text-[10px] text-slate-400 block uppercase">Effective Fed Funds</span>
                <span className="text-sm sm:text-base font-extrabold text-amber-400 tabular-nums">
                  {typeof macro.fedFundsRate === "number"
                    ? `${macro.fedFundsRate.toFixed(2)}%`
                    : "Unavailable"}
                </span>
                <span className="text-[9px] text-slate-500 block">Series FEDFUNDS</span>
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
              {/* Highlight Banner: Render verified purchase if parsed; otherwise render filing notice */}
              {parsedPurchase && typeof parsedPurchase.sharesTraded === "number" && typeof parsedPurchase.pricePerShare === "number" && typeof parsedPurchase.totalValueUsd === "number" ? (
                <div className="bg-emerald-950/30 border border-emerald-800/80 rounded-xl p-3.5 text-xs space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-base">🟢</span>
                      <strong className="text-emerald-300 font-bold text-sm">
                        {parsedPurchase.insiderName || "Corporate Insider"} ({parsedPurchase.insiderRole || "Officer / Director"})
                      </strong>
                    </div>
                    <a
                      href={parsedPurchase.secEdgarUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[10px] font-bold text-cyan-400 hover:underline flex items-center gap-1"
                    >
                      <span>SEC EDGAR Form 4</span>
                      <span>↗</span>
                    </a>
                  </div>
                  <p className="text-[11px] text-slate-200 font-sans">
                    Purchased <strong>{parsedPurchase.sharesTraded.toLocaleString()} shares</strong> on the open market at <strong>${parsedPurchase.pricePerShare.toFixed(2)}</strong> (Total: <strong>${(parsedPurchase.totalValueUsd / 1000000).toFixed(2)}M USD</strong>).
                  </p>
                </div>
              ) : primaryFiling ? (
                <div className="bg-[#0b1019] border border-[#243044] rounded-xl p-3.5 text-xs space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-base">📋</span>
                      <strong className="text-slate-200 font-bold text-sm">
                        Official Regulatory Disclosure ({cleanSym})
                      </strong>
                      {primaryFiling.isAmendment && (
                        <span className="text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded">
                          Amendment
                        </span>
                      )}
                    </div>
                    {primaryFiling.secEdgarUrl && (
                      <a
                        href={primaryFiling.secEdgarUrl}
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
                    Recent SEC Form {primaryFiling.formType || "4"} disclosure filed on <strong>{primaryFiling.filingDate || "recent date"}</strong>
                    {primaryFiling.accessionNumber ? (
                      <> (Accession: <span className="font-mono text-slate-400">{primaryFiling.accessionNumber}</span>)</>
                    ) : null}.
                    Transaction-level breakdown (shares, price, total value) is unparsed from submission metadata; no buying or selling is inferred from filing existence.
                  </p>
                </div>
              ) : null}

              <div className="overflow-x-auto pt-1">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#090d14] text-slate-400 text-[10px] uppercase border-b border-[#1b2434]">
                    <tr>
                      <th className="py-2.5 px-3">Ticker</th>
                      <th className="py-2.5 px-3">Filing / Reporting Person</th>
                      <th className="py-2.5 px-3">Shares</th>
                      <th className="py-2.5 px-3">Total ($)</th>
                      <th className="py-2.5 px-3">Filing Date</th>
                      <th className="py-2.5 px-3 text-right">EDGAR Source</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1b2434] tabular-nums">
                    {trades.map((trade, idx) => (
                      <tr key={idx} className="hover:bg-[#151e2d] transition-colors">
                        <td className="py-2 px-3 font-bold text-cyan-400">{trade.ticker}</td>
                        <td className="py-2 px-3 text-slate-200">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold">
                              {trade.insiderName || `Form ${trade.formType || "4"} Public Filing`}
                            </span>
                            {trade.isAmendment && (
                              <span className="text-[9px] font-bold bg-amber-950 text-amber-400 border border-amber-800 px-1 py-0.5 rounded">
                                4/A
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
                        <td className="py-2 px-3 font-bold">
                          {typeof trade.totalValueUsd === "number" ? (
                            <span className={trade.isBuy ? "text-emerald-400" : (trade.isSell ? "text-rose-400" : "text-slate-200")}>
                              ${(trade.totalValueUsd / 1000000).toFixed(2)}M
                            </span>
                          ) : (
                            <span className="text-slate-500 font-sans font-normal">Unavailable</span>
                          )}
                        </td>
                        <td className="py-2 px-3 text-slate-400 text-[11px]">{trade.filingDate || "—"}</td>
                        <td className="py-2 px-3 text-right">
                          <a
                            href={trade.secEdgarUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] font-bold text-cyan-400 hover:text-cyan-300 underline"
                          >
                            {trade.formType ? `Form ${trade.formType} ↗` : "EDGAR ↗"}
                          </a>
                        </td>
                      </tr>
                    ))}
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
