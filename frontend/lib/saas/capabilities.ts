/**
 * Canonical frontend capability and limit identifier collections.
 *
 * Guarantees:
 * - 1:1 parity with authoritative backend vocabulary in api/capabilities/capabilities.py.
 * - Disjoint collections (capabilities and limits never overlap).
 * - Zero commercial plan names or hard-coded pricing symbols.
 */

export const CAPABILITIES = [
  "analysis.read",
  "analysis.quant",
  "analysis.simulation",
  "radar.read",
  "radar.advanced_filters",
  "radar.custom_scan",
  "portfolio.read",
  "portfolio.manage",
  "portfolio.risk",
  "journal.read",
  "journal.write",
  "alerts.create",
  "alerts.realtime",
  "export.csv",
  "team.read",
  "team.manage",
  "api.access",
] as const;

export type Capability = (typeof CAPABILITIES)[number];

export const LIMITS = [
  "portfolio.max_holdings",
  "portfolio.max_workspaces",
  "alerts.max_active",
  "team.max_members",
  "api.requests_per_day",
] as const;

export type LimitKey = (typeof LIMITS)[number];

export const CAPABILITY_SET: ReadonlySet<string> = new Set(CAPABILITIES);
export const LIMIT_SET: ReadonlySet<string> = new Set(LIMITS);

export function isCapability(val: string): val is Capability {
  return CAPABILITY_SET.has(val);
}

export function isLimitKey(val: string): val is LimitKey {
  return LIMIT_SET.has(val);
}
