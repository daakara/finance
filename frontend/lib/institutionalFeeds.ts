"use client";

import { getApiBaseUrl } from "./api";

/**
 * Institutional Data Ingestion & Synthesis Engine:
 * 1. Federal Reserve Economic Data (FRED): Real-time macroeconomic regime, 10Y-2Y curve, high yield credit spread.
 * 2. SEC EDGAR Form 4 & 13F: Regulatory filing disclosures and parsed insider purchasing.
 */

export interface FredSeriesObservation {
  value: number | null;
  date: string | null;
  timestamp?: string | null;
  series_id: string;
  name?: string;
  frequency?: "DAILY" | "MONTHLY";
  max_age_days?: number;
  age_days?: number | null;
  is_fresh?: boolean;
  status?: "FRESH" | "STALE" | "MISSING" | "FUTURE_DATED" | "UNVERIFIED";
  distinction?: string;
}

export interface FredFreshnessSummary {
  isAllFresh: boolean;
  staleSeries: string[];
  missingSeries: string[];
  oldestObservationDate: string | null;
  newestObservationDate: string | null;
}

export interface FredMacroData {
  status: "AVAILABLE" | "PARTIAL" | "UNAVAILABLE" | "STALE";
  isLive: boolean;
  regimeName:
    | "EXPANSION_GOLDILOCKS"
    | "NEUTRAL_EXPANSION"
    | "HAWKISH_TIGHTENING"
    | "RECESSIONARY_PRESSURE"
    | "MACRO_TELEMETRY_UNAVAILABLE"
    | "MACRO_TELEMETRY_STALE";
  regimeTitle: string;
  regimeSubtitle: string;
  yieldCurve10Y2Y: number | null; // e.g. +0.18% (T10Y2Y)
  realInterestRate10Y: number | null; // e.g. 1.82% (DFII10)
  highYieldCreditSpread: number | null; // e.g. 3.25% (BAMLH0A0HYM2)
  fedFundsRate: number | null; // e.g. 4.88% (FEDFUNDS)
  macroRiskMultiplier: number | null; // e.g. 1.10x scaling on position sizing (null when unverified/unavailable)
  lastUpdated: string | null;
  yieldCurveObservationDate?: string | null;
  creditSpreadObservationDate?: string | null;
  realRateObservationDate?: string | null;
  fedFundsObservationDate?: string | null;
  freshnessSummary?: FredFreshnessSummary | null;
  seriesObservations?: Record<string, FredSeriesObservation>;
}

export interface SecForm4Trade {
  ticker: string;
  companyName: string;
  insiderName: string | null;
  insiderRole: string | null;
  transactionType: string;
  sharesTraded: number | null; // null when unparsed/unavailable, NOT 0
  pricePerShare: number | null; // null when unparsed/unavailable, NOT 0
  totalValueUsd: number | null; // null when unparsed/unavailable, NOT 0
  filingDate: string;
  secEdgarUrl: string;
  isSignificantBuy: boolean; // Flagged if open-market purchase >= $100k
  isBuy?: boolean | null;
  isSell?: boolean | null;
  hasParsedTransaction?: boolean;
  accessionNumber?: string;
  formType?: string;
  isAmendment?: boolean;
}

/**
 * Unavailability contract for production FRED macro telemetry.
 * Used when live feed fails, is unverified, or returns unavailable status.
 */
export const UNAVAILABLE_FRED_MACRO_DATA: FredMacroData = {
  status: "UNAVAILABLE",
  isLive: false,
  regimeName: "MACRO_TELEMETRY_UNAVAILABLE",
  regimeTitle: "Macro Telemetry Unavailable",
  regimeSubtitle: "Authoritative macroeconomic indicators from Federal Reserve Economic Data (FRED) are currently unavailable.",
  yieldCurve10Y2Y: null,
  realInterestRate10Y: null,
  highYieldCreditSpread: null,
  fedFundsRate: null,
  macroRiskMultiplier: null,
  lastUpdated: null,
};

/**
 * Historical snapshot fixture preserved strictly for unit tests and historical regression tests.
 * Never used as a silent fallback for live production presentation.
 */
export const CURRENT_FRED_MACRO_SNAPSHOT: FredMacroData = {
  status: "AVAILABLE",
  isLive: false,
  regimeName: "NEUTRAL_EXPANSION",
  regimeTitle: "Macro Regime: Disinflationary Expansion (Yield Curve Uninverting)",
  regimeSubtitle: "10Y-2Y Spread at +0.18%, High Yield Spreads Tight at 3.25%. Macro regime is supportive of equity accumulation.",
  yieldCurve10Y2Y: 0.18,
  realInterestRate10Y: 1.82,
  highYieldCreditSpread: 3.25,
  fedFundsRate: 4.88,
  macroRiskMultiplier: 1.10,
  lastUpdated: "2026-08-26",
};

/** Alias for historical testing */
export const HISTORICAL_FRED_MACRO_FIXTURE = CURRENT_FRED_MACRO_SNAPSHOT;

export const CURATED_HISTORICAL_SEC_TRADES: SecForm4Trade[] = [
  {
    ticker: "NVDA",
    companyName: "NVIDIA Corporation",
    insiderName: "Jensen Huang",
    insiderRole: "President and CEO",
    transactionType: "P - Purchase (Open Market)",
    sharesTraded: 50000,
    pricePerShare: 210.50,
    totalValueUsd: 10525000,
    filingDate: "2026-08-20",
    secEdgarUrl: "https://www.sec.gov/edgar/browse/?CIK=0001045810",
    isSignificantBuy: true,
    isBuy: true,
    isSell: false,
    hasParsedTransaction: true,
    formType: "4",
    isAmendment: false,
  },
  {
    ticker: "AAPL",
    companyName: "Apple Inc.",
    insiderName: "Luca Maestri",
    insiderRole: "Senior VP, CFO",
    transactionType: "S - Sale (Open Market)",
    sharesTraded: 25000,
    pricePerShare: 228.40,
    totalValueUsd: 5710000,
    filingDate: "2026-08-18",
    secEdgarUrl: "https://www.sec.gov/edgar/browse/?CIK=0000320193",
    isSignificantBuy: false,
    isBuy: false,
    isSell: true,
    hasParsedTransaction: true,
    formType: "4",
    isAmendment: false,
  },
  {
    ticker: "PLTR",
    companyName: "Palantir Technologies Inc.",
    insiderName: "Alexander Karp",
    insiderRole: "CEO and Co-Founder",
    transactionType: "P - Purchase (Rule 10b5-1)",
    sharesTraded: 100000,
    pricePerShare: 32.10,
    totalValueUsd: 3210000,
    filingDate: "2026-08-15",
    secEdgarUrl: "https://www.sec.gov/edgar/browse/?CIK=0001321655",
    isSignificantBuy: true,
    isBuy: true,
    isSell: false,
    hasParsedTransaction: true,
    formType: "4",
    isAmendment: false,
  },
  {
    ticker: "TSLA",
    companyName: "Tesla, Inc.",
    insiderName: "Robyn M. Denholm",
    insiderRole: "Board Chair",
    transactionType: "S - Sale (Rule 10b5-1)",
    sharesTraded: 15000,
    pricePerShare: 215.00,
    totalValueUsd: 3225000,
    filingDate: "2026-08-10",
    secEdgarUrl: "https://www.sec.gov/edgar/browse/?CIK=0001318605",
    isSignificantBuy: false,
    isBuy: false,
    isSell: true,
    hasParsedTransaction: true,
    formType: "4",
    isAmendment: false,
  },
  {
    ticker: "MSFT",
    companyName: "Microsoft Corporation",
    insiderName: "Satya Nadella",
    insiderRole: "Chairman and CEO",
    transactionType: "P - Purchase (Open Market)",
    sharesTraded: 10000,
    pricePerShare: 405.20,
    totalValueUsd: 4052000,
    filingDate: "2026-08-22",
    secEdgarUrl: "https://www.sec.gov/edgar/browse/?CIK=0000936395",
    isSignificantBuy: true,
    isBuy: true,
    isSell: false,
    hasParsedTransaction: true,
    formType: "4",
    isAmendment: false,
  },
];

/** @deprecated Use CURATED_HISTORICAL_SEC_TRADES. Preserved for backwards compatibility with historical fixtures. */
export const LIVE_SEC_EDGAR_FORM4_TRADES = CURATED_HISTORICAL_SEC_TRADES;

export async function fetchFredMacroRegime(): Promise<FredMacroData> {
  const API_BASE = getApiBaseUrl();
  try {
    const res = await fetch(`${API_BASE}/regimes/current`);
    if (res.ok) {
      const data = await res.json();
      if (data && data.macro) {
        const m = data.macro;
        const isLive = Boolean(m.is_live);
        const status: "AVAILABLE" | "PARTIAL" | "UNAVAILABLE" | "STALE" =
          m.availability === "AVAILABLE"
            ? "AVAILABLE"
            : m.availability === "STALE"
            ? "STALE"
            : m.availability === "PARTIAL"
            ? "PARTIAL"
            : "UNAVAILABLE";

        const yc = typeof m.yield_curve_10y2y === "number" ? m.yield_curve_10y2y : null;
        const rr = typeof m.real_interest_rate === "number" ? m.real_interest_rate : (typeof m.real_interest_rate_10y === "number" ? m.real_interest_rate_10y : null);
        const cs = typeof m.credit_spread === "number" ? m.credit_spread : (typeof m.high_yield_credit_spread === "number" ? m.high_yield_credit_spread : null);
        const ff = typeof m.fed_funds_rate === "number" ? m.fed_funds_rate : null;

        const rawFresh = m.freshness_summary;
        const freshnessSummary: FredFreshnessSummary | null = rawFresh
          ? {
              isAllFresh: Boolean(rawFresh.is_all_fresh ?? rawFresh.isAllFresh),
              staleSeries: rawFresh.stale_series ?? rawFresh.staleSeries ?? [],
              missingSeries: rawFresh.missing_series ?? rawFresh.missingSeries ?? [],
              oldestObservationDate: rawFresh.oldest_observation_date ?? rawFresh.oldestObservationDate ?? null,
              newestObservationDate: rawFresh.newest_observation_date ?? rawFresh.newestObservationDate ?? null,
            }
          : null;

        if (status === "AVAILABLE" && isLive) {
          return {
            status: "AVAILABLE",
            isLive: true,
            regimeName: m.regime || "NEUTRAL_EXPANSION",
            regimeTitle: m.regime_title || "Macro Regime: Live FRED Telemetry",
            regimeSubtitle: m.regime_subtitle || "Verified macroeconomic liquidity and spread observations from FRED.",
            yieldCurve10Y2Y: yc,
            realInterestRate10Y: rr,
            highYieldCreditSpread: cs,
            fedFundsRate: ff,
            macroRiskMultiplier: typeof m.macro_multiplier === "number" ? m.macro_multiplier : null,
            lastUpdated: m.last_updated || null,
            yieldCurveObservationDate: m.yield_curve_observation_date || null,
            creditSpreadObservationDate: m.credit_spread_observation_date || null,
            realRateObservationDate: m.real_rate_observation_date || null,
            fedFundsObservationDate: m.fed_funds_observation_date || null,
            freshnessSummary: freshnessSummary,
            seriesObservations: m.series_observations || {},
          };
        }

        // Explicit unavailable or stale state: zero multiplier fabrication
        return {
          status: status,
          isLive: false,
          regimeName: status === "STALE" ? "MACRO_TELEMETRY_STALE" : "MACRO_TELEMETRY_UNAVAILABLE",
          regimeTitle: m.regime_title || (status === "STALE" ? "Macro Telemetry Stale / Multiplier Inactive" : "Macro Telemetry Unavailable"),
          regimeSubtitle: m.regime_subtitle || "Authoritative macroeconomic indicators from Federal Reserve Economic Data (FRED) are unverified or stale.",
          yieldCurve10Y2Y: yc,
          realInterestRate10Y: rr,
          highYieldCreditSpread: cs,
          fedFundsRate: ff,
          macroRiskMultiplier: null, // Strictly null when unverified/stale!
          lastUpdated: m.last_updated || null,
          yieldCurveObservationDate: m.yield_curve_observation_date || null,
          creditSpreadObservationDate: m.credit_spread_observation_date || null,
          realRateObservationDate: m.real_rate_observation_date || null,
          fedFundsObservationDate: m.fed_funds_observation_date || null,
          freshnessSummary: freshnessSummary,
          seriesObservations: m.series_observations || {},
        };
      }
    }
  } catch (err) {
    console.warn("Live FRED macro regime fetch failed:", err);
  }

  // FAIL CLOSED: Never silently use historical snapshot in production
  return UNAVAILABLE_FRED_MACRO_DATA;
}

export type SecFilingsStatus =
  | "AVAILABLE_WITH_RESULTS"
  | "AVAILABLE_NO_RESULTS"
  | "UPSTREAM_RATE_LIMITED"
  | "UPSTREAM_TIMEOUT"
  | "UPSTREAM_SERVER_ERROR"
  | "UPSTREAM_NOT_FOUND"
  | "STRUCTURALLY_INVALID"
  | "TRANSPORT_INCOMPLETE"
  | "CONFIGURATION_BLOCKED"
  | "UNKNOWN_FAIL_CLOSED";

export interface SecFilingsResult {
  status: SecFilingsStatus;
  items: SecForm4Trade[] | null;
  message?: string | null;
}

export async function fetchSecForm4Insiders(symbol?: string): Promise<SecFilingsResult> {
  if (!symbol) {
    return {
      status: "AVAILABLE_NO_RESULTS",
      items: [],
      message: "No symbol provided",
    };
  }
  const symClean = symbol.toUpperCase().replace("-USD", "");
  const API_BASE = getApiBaseUrl();
  try {
    const res = await fetch(`${API_BASE}/smart-money/sec-filings/${symClean}`);
    if (res.ok) {
      const data = await res.json();
      const status: SecFilingsStatus = (data && data.status)
        ? (data.status as SecFilingsStatus)
        : (data && Array.isArray(data.filings) && data.filings.length > 0 ? "AVAILABLE_WITH_RESULTS" : "AVAILABLE_NO_RESULTS");

      if (status === "AVAILABLE_WITH_RESULTS") {
        const rawList = Array.isArray(data.items) ? data.items : (Array.isArray(data.filings) ? data.filings : []);
        const mapped: SecForm4Trade[] = rawList.map((f: any) => {
          const hasShares = typeof f.shares_traded === "number" || typeof f.sharesTraded === "number";
          const hasPrice = typeof f.price_per_share === "number" || typeof f.pricePerShare === "number";
          const hasTotal = typeof f.total_value_usd === "number" || typeof f.totalValueUsd === "number";
          const hasParsed = hasShares || hasPrice || hasTotal || Boolean(f.has_parsed_transaction);

          const shares = hasShares ? Number(f.shares_traded ?? f.sharesTraded) : null;
          const price = hasPrice ? Number(f.price_per_share ?? f.pricePerShare) : null;
          const total = hasTotal ? Number(f.total_value_usd ?? f.totalValueUsd) : null;

          const isBuy = typeof f.is_buy === "boolean" ? f.is_buy : (typeof f.isBuy === "boolean" ? f.isBuy : null);
          const isSell = typeof f.is_sell === "boolean" ? f.is_sell : (typeof f.isSell === "boolean" ? f.isSell : null);

          const isSignif = Boolean(
            f.isSignificantBuy ||
            (isBuy && total !== null && total >= 100000)
          );

          const formType = String(f.form || f.formType || "4");
          const isAmendment = Boolean(
            f.is_amendment ?? f.isAmendment ?? formType.includes("/A")
          );
          const accNum = f.accession_number || f.accessionNumber || "";

          const insiderName = f.insider_name || f.insiderName || null;
          const insiderRole = f.role || f.insiderRole || null;

          const txType = f.transaction_type || f.transactionType || (isAmendment ? `Form ${formType} Regulatory Filing (Amendment)` : `Form ${formType} Regulatory Filing`);

          return {
            ticker: symClean,
            companyName: `${symClean} SEC Reporting Issuer`,
            insiderName,
            insiderRole,
            transactionType: txType,
            sharesTraded: shares,
            pricePerShare: price,
            totalValueUsd: total,
            filingDate: f.filing_date || f.filingDate || "",
            secEdgarUrl: f.sec_url || f.secEdgarUrl || "",
            isSignificantBuy: isSignif,
            isBuy,
            isSell,
            hasParsedTransaction: hasParsed,
            accessionNumber: accNum,
            formType,
            isAmendment,
          };
        });

        return {
          status: "AVAILABLE_WITH_RESULTS",
          items: mapped,
          message: data.message || null,
        };
      }

      if (status === "AVAILABLE_NO_RESULTS") {
        return {
          status: "AVAILABLE_NO_RESULTS",
          items: [],
          message: data.message || "0 filings match criteria",
        };
      }

      // Explicit upstream failure/unavailable states preserve items: null
      return {
        status,
        items: null,
        message: data.message || null,
      };
    } else {
      const status: SecFilingsStatus = res.status === 429
        ? "UPSTREAM_RATE_LIMITED"
        : res.status === 404
        ? "UPSTREAM_NOT_FOUND"
        : res.status >= 500
        ? "UPSTREAM_SERVER_ERROR"
        : "UNKNOWN_FAIL_CLOSED";
      return {
        status,
        items: null,
        message: `HTTP error ${res.status}`,
      };
    }
  } catch (err: any) {
    console.warn("Live SEC EDGAR filings fetch failed:", err);
    return {
      status: "TRANSPORT_INCOMPLETE",
      items: null,
      message: err?.message || "Transport connection failed",
    };
  }
}
