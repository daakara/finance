/**
 * ARX Centralized Data Provenance Architecture.
 * Explicitly distinguishes live verified market tape from regulatory filings,
 * calculated metrics, insufficient history, and unverified data.
 *
 * Enforces the non-negotiable invariants:
 *   UNKNOWN ≠ FAVORABLE
 *   UNKNOWN ≠ NEGATIVE
 *   UNKNOWN ≠ ACTIONABLE
 *   PLACEHOLDER ≠ DATA
 *   SYNTHETIC ≠ MARKET DATA
 *   CURATED ≠ LIVE
 */

export type DataProvenanceStatus =
  | "VERIFIED_LIVE"
  | "VERIFIED_HISTORICAL"
  | "VERIFIED_REGULATORY"
  | "CALCULATED_FROM_VERIFIED_DATA"
  | "INSUFFICIENT_DATA"
  | "UNVERIFIED"
  | "UNAVAILABLE";

export interface DataFieldProvenance {
  status: DataProvenanceStatus;
  label: string;
  sourceDescription: string;
  badgeColor: "emerald" | "cyan" | "amber" | "rose" | "slate";
  isActionable: boolean;
}

export interface ProvenanceContext {
  hasLiveFeed?: boolean;
  candleCount?: number;
  hasSecFilings?: boolean;
  hasExecutionPlan?: boolean;
  isCataloged?: boolean;
  price?: number;
}

/**
 * Deterministically resolve data provenance for any analytical field.
 */
export function resolveDataProvenance(
  field: "price" | "candles" | "fundamentals" | "trade_levels" | "catalysts" | "insiders",
  context: ProvenanceContext
): DataFieldProvenance {
  const { hasLiveFeed, candleCount = 0, hasSecFilings, hasExecutionPlan, isCataloged, price = 0 } = context;

  switch (field) {
    case "price":
      if (hasLiveFeed && price > 0) {
        return {
          status: "VERIFIED_LIVE",
          label: "Live Market Tape",
          sourceDescription: "Real-time exchange quote verified via streaming feed.",
          badgeColor: "emerald",
          isActionable: true,
        };
      }
      if (isCataloged && price > 0) {
        return {
          status: "VERIFIED_HISTORICAL",
          label: "Verified Historical EOD",
          sourceDescription: "Official end-of-day market close price.",
          badgeColor: "cyan",
          isActionable: true,
        };
      }
      return {
        status: "UNVERIFIED",
        label: "Price Unavailable",
        sourceDescription: "No verified real-time or historical quote on exchange record.",
        badgeColor: "slate",
        isActionable: false,
      };

    case "candles":
      if (candleCount >= 50) {
        return {
          status: "VERIFIED_HISTORICAL",
          label: `${candleCount} Verified Daily Sessions`,
          sourceDescription: "Authentic exchange OHLCV daily candle history.",
          badgeColor: "emerald",
          isActionable: true,
        };
      }
      if (candleCount > 0) {
        return {
          status: "INSUFFICIENT_DATA",
          label: `Seasoning Pending (${candleCount} < 50 Sessions)`,
          sourceDescription: "Insufficient trading history to compute standard 50D trend metrics.",
          badgeColor: "amber",
          isActionable: false,
        };
      }
      return {
        status: "UNAVAILABLE",
        label: "No Exchange History",
        sourceDescription: "Zero historical candle sessions available.",
        badgeColor: "slate",
        isActionable: false,
      };

    case "fundamentals":
      if (hasSecFilings && isCataloged) {
        return {
          status: "VERIFIED_REGULATORY",
          label: "Audited SEC EDGAR 10-K/10-Q",
          sourceDescription: "Audited quarterly/annual financial statement disclosures.",
          badgeColor: "emerald",
          isActionable: true,
        };
      }
      return {
        status: "UNAVAILABLE",
        label: "Awaiting SEC Filings",
        sourceDescription: "Unassessed: audited financial disclosures not connected for this security.",
        badgeColor: "slate",
        isActionable: false,
      };

    case "trade_levels":
      if (candleCount >= 50 && hasExecutionPlan && price > 0) {
        return {
          status: "CALCULATED_FROM_VERIFIED_DATA",
          label: "Minervini Volatility Contraction Pattern",
          sourceDescription: "Systemic execution levels calculated from verified price history.",
          badgeColor: "emerald",
          isActionable: true,
        };
      }
      return {
        status: "INSUFFICIENT_DATA",
        label: "Trade Levels Suppressed",
        sourceDescription: "Execution levels withheld due to insufficient historical seasoning (< 50 sessions).",
        badgeColor: "slate",
        isActionable: false,
      };

    case "catalysts":
      if (isCataloged) {
        return {
          status: "VERIFIED_REGULATORY",
          label: "Verified Corporate Catalyst",
          sourceDescription: "Verified earnings, product cycle, or clinical runway milestone.",
          badgeColor: "cyan",
          isActionable: true,
        };
      }
      return {
        status: "UNAVAILABLE",
        label: "Awaiting Verified Disclosures",
        sourceDescription: "Uncataloged security: no fabricated catalyst events synthesized.",
        badgeColor: "slate",
        isActionable: false,
      };

    case "insiders":
      if (isCataloged) {
        return {
          status: "VERIFIED_REGULATORY",
          label: "SEC Form 4 / 13F / STOCK Act",
          sourceDescription: "Verified regulatory filings from corporate executives or lawmakers.",
          badgeColor: "emerald",
          isActionable: true,
        };
      }
      return {
        status: "UNAVAILABLE",
        label: "No Connected Filings",
        sourceDescription: "No verified insider or congressional transaction records connected.",
        badgeColor: "slate",
        isActionable: false,
      };
  }
}
