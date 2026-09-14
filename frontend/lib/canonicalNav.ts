/**
 * Canonical ARX Navigation & Journey Architecture (Release 1 Scope)
 *
 * Defines the authoritative 4-hub Release 1 trading journey:
 * Radar → Analysis → Trade Plan → Portfolio
 * (Find → Understand → Plan → Manage)
 *
 * Dedicated surfaces for Journal and Performance are DEFERRED TO POST-R1.
 *
 * Enforces:
 * - 100% desktop and mobile parity across the 4 core hubs
 * - First-class routing for Analysis at '/'
 * - Active asset context preservation across all hubs without synthetic fallback
 * - Query parameter classification into Global, Discovery, Route-local, and Transient
 */

export type CanonicalHubId =
  | "radar"
  | "analysis"
  | "setups"
  | "portfolio";

export type DeferredHubId =
  | "journal"
  | "performance";

export type AllHubId = CanonicalHubId | DeferredHubId;

export interface CanonicalHubMeta {
  id: CanonicalHubId;
  href: string;
  name: string;
  label: string;
  mentalModel: "Find" | "Understand" | "Plan" | "Manage";
  question: string;
  badge: string;
  icon: string;
}

export interface DeferredHubMeta {
  id: DeferredHubId;
  href: string;
  name: string;
  label: string;
  status: "DEFERRED_POST_R1";
  question: string;
  badge: string;
  icon: string;
}

export const CANONICAL_HUBS: readonly CanonicalHubMeta[] = [
  {
    id: "radar",
    href: "/radar",
    name: "Radar",
    label: "Radar",
    mentalModel: "Find",
    question: "What deserves attention today?",
    badge: "CONFLUENCE",
    icon: "📡",
  },
  {
    id: "analysis",
    href: "/",
    name: "Analysis",
    label: "Analysis",
    mentalModel: "Understand",
    question: "Is this asset worthy of capital?",
    badge: "DEEP DIVE",
    icon: "🔬",
  },
  {
    id: "setups",
    href: "/setups",
    name: "Trade Plan",
    label: "Trade Plan",
    mentalModel: "Plan",
    question: "What is actionable right now?",
    badge: "EXECUTION",
    icon: "⚡",
  },
  {
    id: "portfolio",
    href: "/portfolio",
    name: "Portfolio",
    label: "Portfolio",
    mentalModel: "Manage",
    question: "What risk am I carrying?",
    badge: "RISK HEAT",
    icon: "💼",
  },
] as const;

export const DEFERRED_POST_R1_HUBS: readonly DeferredHubMeta[] = [
  {
    id: "journal",
    href: "/journal",
    name: "Journal",
    label: "Journal",
    status: "DEFERRED_POST_R1",
    question: "Did I follow my rules?",
    badge: "DEFERRED",
    icon: "📖",
  },
  {
    id: "performance",
    href: "/performance",
    name: "Performance",
    label: "Performance",
    status: "DEFERRED_POST_R1",
    question: "Is ARX actually improving my results?",
    badge: "DEFERRED",
    icon: "📈",
  },
] as const;

/**
 * Parameter Context Matrix Classification
 */
export const QUERY_PARAM_CLASSES = {
  GLOBAL_JOURNEY_CONTEXT: ["symbol"] as const,
  DISCOVERY_SEARCH_CONTEXT: ["q"] as const,
  ROUTE_LOCAL_CONTEXT: ["tab", "filter", "sort", "ownership", "horizon"] as const,
  TRANSIENT_UI_CONTEXT: ["fromGoal", "add", "retry", "modal"] as const,
} as const;

/**
 * Helper to extract active asset symbol from various search parameter representations.
 * Never fabricates or returns a fake/default symbol.
 */
export function extractActiveSymbol(
  searchParams?: URLSearchParams | Record<string, string | string[] | undefined> | string | null
): string | null {
  if (!searchParams) return null;

  let raw: string | null = null;
  if (typeof searchParams === "string") {
    try {
      const q = new URLSearchParams(searchParams.startsWith("?") ? searchParams.slice(1) : searchParams);
      raw = q.get("symbol") || q.get("ticker");
    } catch {
      raw = null;
    }
  } else if (searchParams instanceof URLSearchParams) {
    raw = searchParams.get("symbol") || searchParams.get("ticker");
  } else if (typeof searchParams === "object") {
    const val = searchParams.symbol || searchParams.ticker;
    raw = Array.isArray(val) ? val[0] : (val ?? null);
  }

  if (!raw) return null;
  const clean = raw.trim().toUpperCase();
  // Ensure valid ticker-like identifier (1-10 alphanumeric characters / dots / hyphens)
  if (/^[A-Z0-9.\-]{1,10}$/.test(clean)) {
    return clean;
  }
  return null;
}

/**
 * Builds destination URL for a canonical hub, preserving active ticker context when present.
 * When no active symbol exists, returns the clean canonical hub route.
 */
export function buildHubHref(
  hub: CanonicalHubMeta | CanonicalHubId | string,
  activeSymbol?: string | null,
  additionalParams?: Record<string, string>
): string {
  const hubId = typeof hub === "string" ? hub : hub.id;
  const targetMeta =
    CANONICAL_HUBS.find((h) => h.id === hubId || h.href === hubId) ||
    DEFERRED_POST_R1_HUBS.find((h) => h.id === hubId || h.href === hubId);
  const baseHref = targetMeta ? targetMeta.href : (typeof hub === "string" ? hub : hub.href);

  const cleanSymbol = activeSymbol ? activeSymbol.trim().toUpperCase() : null;
  const search = new URLSearchParams();

  if (cleanSymbol && /^[A-Z0-9.\-]{1,10}$/.test(cleanSymbol)) {
    if (baseHref === "/radar") {
      search.set("q", cleanSymbol);
    } else {
      search.set("symbol", cleanSymbol);
    }
  }

  if (additionalParams) {
    for (const [key, val] of Object.entries(additionalParams)) {
      if (val !== undefined && val !== null && val !== "") {
        search.set(key, val);
      }
    }
  }

  const queryStr = search.toString();
  return queryStr ? `${baseHref}?${queryStr}` : baseHref;
}

/**
 * Determines whether a hub is currently active for the given pathname.
 * Root '/' is strictly matched to avoid active bleed.
 */
export function isHubActive(hubHref: string, pathname: string): boolean {
  if (!pathname) return false;
  const cleanPath = pathname.split("?")[0].replace(/\/$/, "") || "/";
  const cleanHub = hubHref.replace(/\/$/, "") || "/";

  if (cleanHub === "/") {
    return cleanPath === "/";
  }

  if (cleanHub === "/radar") {
    return cleanPath === "/radar" || cleanPath === "/screener";
  }

  return cleanPath === cleanHub || cleanPath.startsWith(cleanHub + "/");
}
