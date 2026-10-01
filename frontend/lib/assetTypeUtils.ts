/**
 * Asset Type Discrimination Utilities
 * Centralizes ETF / Stock / Crypto type detection for conditional UI routing.
 *
 * Invariants:
 * - Consumes authoritative Master Asset Catalog and Watchlist Registry.
 * - Zero parallel hardcoded ETF symbol lists.
 * - Missing or unresolved metadata resolves strictly to 'UNKNOWN', never silently to 'Stock'.
 */

import { getMasterAsset } from './masterCatalog';
import { SHARED_WATCHLIST_ITEMS } from './constants';
import { getCanonicalEtfSectorWeights } from './assetRegistry';

export type CanonicalAssetType = 'Stock' | 'ETF' | 'Crypto' | 'UNKNOWN';

/**
 * Resolves the canonical asset type for a given symbol using authoritative sources:
 * 1. MasterAssetCatalog (highest authority)
 * 2. SHARED_WATCHLIST_ITEMS (authoritative institutional watchlist)
 * 3. Canonical Crypto pair pattern (-USD)
 *
 * If metadata is missing or unverified, returns 'UNKNOWN'.
 */
export function resolveAssetType(symbol: string | null | undefined): CanonicalAssetType {
  if (!symbol) return 'UNKNOWN';
  const clean = symbol.trim().toUpperCase();
  if (!clean) return 'UNKNOWN';

  // 1. Authoritative Master Asset Catalog
  const master = getMasterAsset(clean);
  if (master?.type) {
    return master.type;
  }

  // 2. Authoritative Shared Watchlist Registry
  const watchItem = SHARED_WATCHLIST_ITEMS.find(
    (w) => w.symbol.toUpperCase() === clean
  );
  if (watchItem?.type) {
    return watchItem.type;
  }

  // 3. Canonical crypto pair pattern in ARX (-USD suffix)
  if (clean.endsWith('-USD')) {
    return 'Crypto';
  }

  // Unverified instrument: strictly fail-closed as UNKNOWN
  return 'UNKNOWN';
}

/** Returns true if the symbol is deterministically verified as an ETF */
export function isETF(symbol: string | null | undefined): boolean {
  return resolveAssetType(symbol) === 'ETF';
}

/** Returns true if the symbol is deterministically verified as an operating corporate Stock */
export function isStock(symbol: string | null | undefined): boolean {
  return resolveAssetType(symbol) === 'Stock';
}

/** Returns true if the symbol is deterministically verified as a Digital Asset / Crypto */
export function isCrypto(symbol: string | null | undefined): boolean {
  return resolveAssetType(symbol) === 'Crypto';
}

/** Returns true if the symbol cannot be authoritatively classified */
export function isUnknownAsset(symbol: string | null | undefined): boolean {
  return resolveAssetType(symbol) === 'UNKNOWN';
}

/** Returns true if the symbol has canonical sector allocation data */
export function hasEtfSectorData(symbol: string | null | undefined): boolean {
  if (!symbol) return false;
  const weights = getCanonicalEtfSectorWeights(symbol);
  return weights !== null && weights.length > 0;
}
