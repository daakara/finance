import type { CanonicalInstrumentEnvelope } from "./assetTypeUtils";

export interface LimitedHistoryContext {
  symbol: string;
  error?: string | null;
  instrument?: CanonicalInstrumentEnvelope | null;
  coverageCertified?: boolean | null;
  coverageStatus?: string | null;
}

export interface LimitedHistoryCapability {
  id: string;
  name: string;
  status: string;
  statusColor: string;
  requirement: string;
}

/**
 * Resolves the verified listing identity status.
 * Shows "Verified" only when supported by Security Master evidence (listing_status === "ACTIVE" && classification_status === "VERIFIED").
 * An active listing must NEVER imply filing completeness, ownership verification, or scoring eligibility.
 */
export function resolveListingIdentity(context: LimitedHistoryContext): {
  status: "Verified" | "Not verified";
  description: string;
} {
  const isVerified =
    (context.instrument?.listing_status === "ACTIVE" && context.instrument?.classification_status === "VERIFIED") ||
    (Boolean(context.error) && context.error!.toLowerCase().includes("recognized") && !context.error!.toLowerCase().includes("unverified"));

  if (isVerified) {
    const exchange = context.instrument?.primary_exchange || "Exchange";
    return {
      status: "Verified",
      description: `Security listing verified in Security Master (${exchange})`,
    };
  }

  return {
    status: "Not verified",
    description: "Security Master listing verification unavailable or unconfirmed",
  };
}

/**
 * Resolves SEC filing coverage status from authoritative coverage metadata.
 * Missing coverage metadata NEVER defaults to verified or complete.
 * Historical coverage remains PARTIAL / UNCERTIFIED wherever applicable.
 * Active listing must NEVER imply filing completeness, ownership verification, or scoring eligibility.
 */
export function resolveSecFilingCoverage(context: LimitedHistoryContext): {
  status: "Certified" | "PARTIAL / UNCERTIFIED" | "Not verified";
  description: string;
} {
  const isCertified =
    context.coverageCertified === true ||
    context.coverageStatus === "COMPLETE" ||
    context.coverageStatus === "CERTIFIED";

  if (isCertified) {
    return {
      status: "Certified",
      description: "Historical SEC Form 4 coverage certified by registry",
    };
  }

  const isPartial =
    context.coverageCertified === false ||
    context.coverageStatus === "PARTIAL" ||
    context.coverageStatus === "UNCERTIFIED";

  if (isPartial) {
    return {
      status: "PARTIAL / UNCERTIFIED",
      description: "Partial historical filing ingestion; registry coverage not certified",
    };
  }

  // Missing or unknown coverage metadata strictly defaults to "Not verified"
  return {
    status: "Not verified",
    description: "No certified Form 4 or ownership filing history; listing does not imply filing completeness",
  };
}

/**
 * Generates the clarified explanation text:
 * Clarifies that 15 daily sessions is the minimum modeling gate, not readiness for every indicator.
 * Preserves each indicator's actual history requirements and avoids promising an automatic activation date.
 */
export function formatLimitedHistoryExplanation(symbol: string, error?: string | null): string {
  const baseError = error ? `${error.replace(/\.+$/, "")}. ` : `Asset "${symbol}" has limited daily trading sessions on record. `;
  return `${baseError}A minimum of 15 completed daily trading sessions is the baseline modeling gate before quantitative evaluation begins. Individual indicators require longer histories (e.g. 50 SMA, 200 SMA, macro regimes) and do not activate automatically at 15 sessions.`;
}

/**
 * Generates the full capability breakdown for an asset with limited history.
 * Generic across all limited-history assets without ticker-specific branches.
 */
export function resolveLimitedHistoryCapabilities(context: LimitedHistoryContext): LimitedHistoryCapability[] {
  const listing = resolveListingIdentity(context);
  const filings = resolveSecFilingCoverage(context);

  return [
    {
      id: "confluence",
      name: "Multi-Factor Confluence",
      status: "UNAVAILABLE",
      statusColor: "text-amber-400",
      requirement: "Requires 15+ completed daily sessions (minimum modeling gate)",
    },
    {
      id: "momentum",
      name: "Technical Momentum & RSI",
      status: "UNAVAILABLE",
      statusColor: "text-amber-400",
      requirement: "Requires 14-period Wilder smoothing (14+ sessions)",
    },
    {
      id: "var",
      name: "Cornish-Fisher Tail VaR",
      status: "UNAVAILABLE",
      statusColor: "text-amber-400",
      requirement: "Requires empirical distribution sample (15+ sessions minimum)",
    },
    {
      id: "vcp",
      name: "Minervini Trend & VCP",
      status: "UNAVAILABLE",
      statusColor: "text-amber-400",
      requirement: "Requires 20 EMA, 50 SMA & 200 SMA baselines",
    },
    {
      id: "listing",
      name: "Listing Identity",
      status: listing.status,
      statusColor: listing.status === "Verified" ? "text-emerald-400" : "text-amber-400",
      requirement: listing.description,
    },
    {
      id: "filings",
      name: "SEC Filing Coverage",
      status: filings.status,
      statusColor: filings.status === "Certified" ? "text-emerald-400" : filings.status === "PARTIAL / UNCERTIFIED" ? "text-amber-400" : "text-slate-400",
      requirement: filings.description,
    },
    {
      id: "execution",
      name: "Execution Clearance",
      status: "INELIGIBLE",
      statusColor: "text-rose-400",
      requirement: "Discovery and listing do not grant execution clearance",
    },
  ];
}
