/**
 * Frontend entitlement contract types for ARX SaaS Foundation.
 *
 * Guarantees:
 * - Pure data contract representing entitlement state received from backend.
 * - Does not implement authorization policy and is not an authorization authority.
 * - Contains zero commercial plan names or hard-coded pricing symbols.
 */

export interface EntitlementSet {
  readonly capabilities: ReadonlySet<string>;
  readonly limits: Readonly<Record<string, number>>;
}

/**
 * Pure data lookup checking whether a capability identifier is present.
 * Note: UX convenience only. Authorization authority resides strictly on the backend.
 */
export function hasCapability(entitlements: EntitlementSet, capability: string): boolean {
  return entitlements.capabilities.has(capability);
}

/**
 * Pure data lookup retrieving an integer limit value or null if absent.
 */
export function getLimitValue(entitlements: EntitlementSet, limitKey: string): number | null {
  const val = entitlements.limits[limitKey];
  return typeof val === "number" && !isNaN(val) ? val : null;
}
