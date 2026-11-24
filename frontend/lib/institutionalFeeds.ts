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

export interface SecForm4Transaction {
  transactionId: string;
  accessionNumber: string;
  transactionDate: string;
  transactionCode: string;
  transactionType: string;
  shares: number | null;
  pricePerShare: number | null;
  totalValueUsd: number | null;
  reportingOwnerName: string;
  reportingOwnerRole: string;
  officerTitle?: string | null;
  isDirector: boolean;
  isOfficer: boolean;
  isTenPercent: boolean;
  ownershipType: string;
  isDerivative: boolean;
  acquiredDisposed: string;
  isBuy: boolean;
  isSell: boolean;
  isGift: boolean;
  isExercise: boolean;
  isOpenMarketPurchaseEvidence: boolean;
  isPositiveBuyingConviction: boolean;
  securityTitle?: string | null;
  publicAvailabilityAt?: string | null;
  isEffective?: boolean;
  effectiveStatus?: string;
  amendmentNote?: string | null;
}

export interface SecForm4Trade {
  ticker: string;
  companyName: string;
  insiderName: string | null;
  insiderRole: string | null;
  transactionType: string;
  sharesTraded: number | null; // null when unparsed/unavailable or mixed, NOT 0
  pricePerShare: number | null; // null when unparsed/unavailable or mixed, NOT 0
  totalValueUsd: number | null; // null when unparsed/unavailable or mixed, NOT 0
  filingDate: string;
  secEdgarUrl: string;
  isSignificantBuy: boolean; // Flagged if open-market purchase >= $100k
  isBuy?: boolean | null;
  isSell?: boolean | null;
  hasParsedTransaction?: boolean;
  accessionNumber?: string;
  formType?: string;
  isAmendment?: boolean;
  // Extended disclosure fields
  transactionCount?: number;
  transactions?: SecForm4Transaction[];
  transactionDate?: string | null;
  transactionCode?: string | null;
  xmlHash?: string | null;
  acceptanceDatetime?: string | null;
  publicAvailabilityAt?: string | null;
  amendmentNumber?: number | null;
  baseAccession?: string | null;
  isGift?: boolean;
  isExercise?: boolean;
  isDerivative?: boolean;
  isMixed?: boolean;
  isEffective?: boolean;
  effectiveStatus?: string;
  hasAmbiguousAmendment?: boolean;
  hasPartialAmendment?: boolean;
  amendmentNote?: string | null;
  isSuperseded?: boolean;
  supersededBy?: string | null;
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
  symbol?: string;
  status: SecFilingsStatus;
  items: SecForm4Trade[] | null;
  message?: string | null;
  coverageStatus?: string | null;
  certifiedWatermark?: string | null;
  asOf?: string | null;
  servedAt?: string | null;
  isPipelineHeld?: boolean;
}

function cleanNumber(val: any): number | null {
  if (val === null || val === undefined) return null;
  const n = typeof val === "number" ? val : Number(val);
  if (!Number.isFinite(n)) return null;
  return n;
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
          const rawTxs: any[] = Array.isArray(f.transactions) ? f.transactions : [];
          const accNum = f.accession_number || f.accessionNumber || "";
          const formType = String(f.form || f.formType || "4");
          const isAmendment = Boolean(
            f.is_amendment ?? f.isAmendment ?? formType.includes("/A")
          );

          const mappedTxs: SecForm4Transaction[] = rawTxs.map((tx: any) => ({
            transactionId: String(tx.transaction_id || tx.transactionId || ""),
            accessionNumber: String(tx.accession_number || tx.accessionNumber || accNum),
            transactionDate: String(tx.transaction_date || tx.transactionDate || ""),
            transactionCode: String(tx.transaction_code || tx.transactionCode || "").toUpperCase(),
            transactionType: String(tx.transaction_type || tx.transactionType || ""),
            shares: cleanNumber(tx.shares),
            pricePerShare: cleanNumber(tx.price_per_share ?? tx.pricePerShare),
            totalValueUsd: cleanNumber(
              tx.total_value_usd ??
              (tx as any).total_value ??
              tx.totalValueUsd ??
              (typeof tx.shares === "number" && typeof tx.price_per_share === "number"
                ? tx.shares * tx.price_per_share
                : null)
            ),
            reportingOwnerName: String(tx.reporting_owner_name || tx.reportingOwnerName || ""),
            reportingOwnerRole: String(tx.reporting_owner_role || tx.reportingOwnerRole || tx.officer_title || tx.officerTitle || "Corporate Insider"),
            officerTitle: tx.officer_title || tx.officerTitle || null,
            isDirector: Boolean(tx.is_director ?? tx.isDirector),
            isOfficer: Boolean(tx.is_officer ?? tx.isOfficer),
            isTenPercent: Boolean(tx.is_ten_percent ?? tx.isTenPercent),
            ownershipType: String(tx.ownership_type || tx.ownershipType || "D").toUpperCase(),
            isDerivative: Boolean(tx.is_derivative ?? tx.isDerivative),
            acquiredDisposed: String(tx.acquired_disposed || tx.acquiredDisposed || "").toUpperCase(),
            isBuy: Boolean(tx.is_buy ?? tx.isBuy),
            isSell: Boolean(tx.is_sell ?? tx.isSell),
            isGift: Boolean(tx.is_gift ?? tx.isGift ?? (String(tx.transaction_code || tx.transactionCode || "").toUpperCase() === "G")),
            isExercise: Boolean(tx.is_exercise ?? tx.isExercise ?? (["M", "X"].includes(String(tx.transaction_code || tx.transactionCode || "").toUpperCase()))),
            isOpenMarketPurchaseEvidence: Boolean(tx.is_open_market_purchase_evidence ?? tx.isOpenMarketPurchaseEvidence),
            isPositiveBuyingConviction: Boolean(tx.is_positive_buying_conviction ?? tx.isPositiveBuyingConviction),
            securityTitle: tx.security_title || tx.securityTitle || null,
            publicAvailabilityAt: tx.public_availability_at || tx.publicAvailabilityAt || null,
            isEffective: typeof tx.is_effective === "boolean" ? tx.is_effective : true,
            effectiveStatus: tx.effective_status || "EFFECTIVE",
            amendmentNote: tx.amendment_note || null,
          }));

          const isMixed = Boolean(f.is_mixed ?? f.isMixed);
          const hasShares = cleanNumber(f.shares_traded ?? f.sharesTraded) !== null;
          const hasPrice = cleanNumber(f.price_per_share ?? f.pricePerShare) !== null;
          const hasTotal = cleanNumber(f.total_value_usd ?? (f as any).total_value ?? f.totalValueUsd) !== null;
          const hasParsed = mappedTxs.length > 0 || hasShares || hasPrice || hasTotal || Boolean(f.has_parsed_transaction);

          let shares: number | null = null;
          let price: number | null = null;
          let total: number | null = null;

          if (isMixed) {
            // Mixed filing: withhold aggregates completely!
            shares = null;
            price = null;
            total = null;
          } else if (hasShares) {
            shares = cleanNumber(f.shares_traded ?? f.sharesTraded);
            price = hasPrice ? cleanNumber(f.price_per_share ?? f.pricePerShare) : (mappedTxs.length === 1 ? mappedTxs[0].pricePerShare : null);
            total = hasTotal
              ? cleanNumber(f.total_value_usd ?? (f as any).total_value ?? f.totalValueUsd)
              : (shares !== null && price !== null ? cleanNumber(shares * price) : null);
          } else if (mappedTxs.length === 1) {
            shares = mappedTxs[0].shares;
            price = mappedTxs[0].pricePerShare;
            total = mappedTxs[0].totalValueUsd;
          } else if (mappedTxs.length > 1) {
            const owners = new Set(mappedTxs.map(t => t.reportingOwnerName).filter(Boolean));
            const secTitles = new Set(mappedTxs.map(t => t.securityTitle).filter(Boolean));
            const codes = new Set(mappedTxs.map(t => t.transactionCode).filter(Boolean));
            const allSharesValid = mappedTxs.every(t => typeof t.shares === "number" && Number.isFinite(t.shares));
            const allTotalsValid = mappedTxs.every(t => typeof t.totalValueUsd === "number" && Number.isFinite(t.totalValueUsd));

            if (owners.size <= 1 && secTitles.size <= 1 && codes.size <= 1 && allSharesValid) {
              shares = mappedTxs.reduce((sum, t) => sum + (t.shares ?? 0), 0);
              total = allTotalsValid ? mappedTxs.reduce((sum, t) => sum + (t.totalValueUsd ?? 0), 0) : null;
              price = total !== null && shares > 0 ? total / shares : (mappedTxs[0].pricePerShare ?? null);
            } else {
              shares = null;
              price = null;
              total = null;
            }
          }

          const isBuy = typeof f.is_buy === "boolean" ? f.is_buy : (typeof f.isBuy === "boolean" ? f.isBuy : (mappedTxs.length > 0 ? (mappedTxs.every(t => t.isBuy) ? true : (mappedTxs.every(t => t.isSell) ? false : null)) : null));
          const isSell = typeof f.is_sell === "boolean" ? f.is_sell : (typeof f.isSell === "boolean" ? f.isSell : (mappedTxs.length > 0 ? (mappedTxs.every(t => t.isSell) ? true : (mappedTxs.every(t => t.isBuy) ? false : null)) : null));

          const isSignif = Boolean(
            f.isSignificantBuy ||
            (isBuy && total !== null && total >= 100000)
          );

          const insiderName = f.insider_name || f.insiderName || (mappedTxs[0]?.reportingOwnerName) || null;
          const insiderRole = f.role || f.insiderRole || (mappedTxs[0]?.reportingOwnerRole) || null;

          const txType = f.transaction_type || f.transactionType || (
            mappedTxs.length === 1
              ? mappedTxs[0].transactionType
              : (mappedTxs.length > 1
                  ? (isMixed ? `Multi-Transaction Filing (${mappedTxs.length} Events — Mixed)` : `Multiple Transactions (${mappedTxs.length})`)
                  : (isAmendment ? `Form ${formType} Regulatory Filing (Amendment)` : `Form ${formType} Regulatory Filing`))
          );

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
            transactionCount: mappedTxs.length,
            transactions: mappedTxs,
            transactionDate: f.transaction_date || f.transactionDate || (mappedTxs[0]?.transactionDate) || null,
            transactionCode: f.transaction_code || f.transactionCode || (mappedTxs[0]?.transactionCode) || null,
            xmlHash: f.xml_hash || f.xmlHash || null,
            acceptanceDatetime: f.acceptance_datetime || f.acceptanceDatetime || null,
            publicAvailabilityAt: f.public_availability_at || f.publicAvailabilityAt || null,
            amendmentNumber: f.amendment_number ?? f.amendmentNumber ?? null,
            baseAccession: f.base_accession || f.baseAccession || null,
            isGift: typeof f.is_gift === "boolean" ? f.is_gift : (mappedTxs.length > 0 && mappedTxs.every(t => t.isGift)),
            isExercise: typeof f.is_exercise === "boolean" ? f.is_exercise : (mappedTxs.length > 0 && mappedTxs.every(t => t.isExercise)),
            isDerivative: typeof f.is_derivative === "boolean" ? f.is_derivative : (mappedTxs.length > 0 && mappedTxs.every(t => t.isDerivative)),
            isMixed: isMixed || (f.shares_traded === null && mappedTxs.length > 1),
            isEffective: typeof f.is_effective === "boolean" ? f.is_effective : (mappedTxs.length > 0 ? mappedTxs.every(t => t.isEffective) : true),
            effectiveStatus: f.effective_status || f.effectiveStatus || "EFFECTIVE",
            hasAmbiguousAmendment: Boolean(f.has_ambiguous_amendment ?? f.hasAmbiguousAmendment),
            hasPartialAmendment: Boolean(f.has_partial_amendment ?? f.hasPartialAmendment),
            amendmentNote: f.amendment_note || f.amendmentNote || null,
            isSuperseded: Boolean(f.is_superseded ?? f.isSuperseded),
            supersededBy: f.superseded_by || f.supersededBy || null,
          };
        });

        return {
          symbol: symClean,
          status: "AVAILABLE_WITH_RESULTS",
          items: mapped,
          message: data.message || null,
          coverageStatus: data.coverage_status || null,
          certifiedWatermark: data.certified_watermark || null,
          asOf: data.as_of || null,
          servedAt: data.served_at || null,
          isPipelineHeld: typeof data.is_pipeline_held === "boolean" ? data.is_pipeline_held : true,
        };
      }

      if (status === "AVAILABLE_NO_RESULTS") {
        return {
          symbol: symClean,
          status: "AVAILABLE_NO_RESULTS",
          items: [],
          message: data.message || "0 filings match criteria",
          coverageStatus: data.coverage_status || null,
          certifiedWatermark: data.certified_watermark || null,
          asOf: data.as_of || null,
          servedAt: data.served_at || null,
          isPipelineHeld: typeof data.is_pipeline_held === "boolean" ? data.is_pipeline_held : true,
        };
      }

      // Explicit upstream failure/unavailable states preserve items: null
      return {
        symbol: symClean,
        status,
        items: null,
        message: data.message || null,
        coverageStatus: data.coverage_status || null,
        certifiedWatermark: data.certified_watermark || null,
        asOf: data.as_of || null,
        servedAt: data.served_at || null,
        isPipelineHeld: typeof data.is_pipeline_held === "boolean" ? data.is_pipeline_held : true,
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
        symbol: symClean,
        status,
        items: null,
        message: `HTTP error ${res.status}`,
        coverageStatus: null,
        isPipelineHeld: true,
      };
    }
  } catch (err: any) {
    console.warn("Live SEC EDGAR filings fetch failed:", err);
    return {
      symbol: symClean,
      status: "TRANSPORT_INCOMPLETE",
      items: null,
      message: err?.message || "Transport connection failed",
      coverageStatus: null,
      isPipelineHeld: true,
    };
  }
}
