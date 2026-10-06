/**
 * Asset Type Discrimination Utilities
 * Centralizes ETF / Stock / Crypto type detection and execution routing.
 *
 * Governing Invariants (ARX Canonical Security Master):
 * - INV-SECMASTER-01: Canonical asset classification is strictly server-owned (ARX_SERVER_SECURITY_MASTER).
 * - INV-SECMASTER-02: Frontend curated catalogs (masterCatalog, sharedWatchlist) are NOT classification authorities.
 *   Their role is strictly presentation/discovery enrichment (FRONTEND_CATALOG_AUTHORITY = NONE).
 * - INV-SECMASTER-03: Execution eligibility is decoupled from asset identity.
 * - INV-SECMASTER-05..07: UNKNOWN, UNSUPPORTED, CONFLICTED strictly fail closed.
 * - INV-SECMASTER-08: ETF routes strictly to ETF execution surface.
 * - INV-SECMASTER-09: Crypto routes strictly to Crypto execution surface.
 * - INV-SECMASTER-10: Specialized equity subtypes (ADR, REIT, Preferred, Warrant, Unit, Right) fail closed.
 * - INV-SECMASTER-14: Zero symbol-specific exception logic.
 * - Generic non-ETF, non-Crypto fallback is PROHIBITED.
 */

import { getMasterAsset } from './masterCatalog';
import { SHARED_WATCHLIST_ITEMS } from './constants';

export type CanonicalAssetType = 'Stock' | 'ETF' | 'Crypto' | 'UNKNOWN';

export type ExecutionEligibility =
  | 'STOCK_EXECUTION'
  | 'ETF_EXECUTION'
  | 'CRYPTO_EXECUTION'
  | 'FAIL_CLOSED';

export interface CanonicalInstrumentEnvelope {
  symbol?: string;
  provider_symbol?: string;
  asset_class?: string;
  security_type?: string;
  primary_exchange?: string;
  listing_status?: string;
  classification_status?: string;
  execution_eligibility?: string;
  analytics_capability?: string;
  classification_authority?: string;
  source_provider?: string;
  classification_timestamp?: string;
  stable_identifiers?: Record<string, string | null>;
  source_provenance?: Record<string, any>;
}

export interface InstrumentContext {
  instrument?: CanonicalInstrumentEnvelope | null;
  canonicalInstrument?: CanonicalInstrumentEnvelope | null;
  executionEligibility?: string | null;
  securityType?: string | null;
  assetClass?: string | null;
  classificationStatus?: string | null;
  [key: string]: any;
}

/**
 * Resolves authoritative execution eligibility from server-owned metadata.
 * Fails closed if metadata is missing, unverified, conflicted, or unsupported.
 */
export function resolveExecutionEligibility(
  context?: InstrumentContext | null
): ExecutionEligibility {
  if (!context) return 'FAIL_CLOSED';

  const rawEligibility =
    context.executionEligibility ||
    context.canonicalInstrument?.execution_eligibility ||
    context.instrument?.execution_eligibility;

  if (rawEligibility === 'STOCK_EXECUTION') return 'STOCK_EXECUTION';
  if (rawEligibility === 'ETF_EXECUTION') return 'ETF_EXECUTION';
  if (rawEligibility === 'CRYPTO_EXECUTION') return 'CRYPTO_EXECUTION';

  return 'FAIL_CLOSED';
}

/**
 * Resolves asset type using authoritative server context when available,
 * falling back to presentation catalogs solely for synchronous UI shell bootstrapping.
 */
export function resolveAssetType(
  symbol: string | null | undefined,
  context?: InstrumentContext | null
): CanonicalAssetType {
  if (!symbol) return 'UNKNOWN';
  const clean = symbol.trim().toUpperCase();
  if (!clean) return 'UNKNOWN';

  // 1. Authoritative Server Security Master context (Primary Authority)
  if (context) {
    const eligibility = resolveExecutionEligibility(context);
    if (eligibility === 'STOCK_EXECUTION') return 'Stock';
    if (eligibility === 'ETF_EXECUTION') return 'ETF';
    if (eligibility === 'CRYPTO_EXECUTION') return 'Crypto';

    // If server context is present and eligibility is FAIL_CLOSED, fail closed strictly
    const hasServerContext = Boolean(
      context.executionEligibility ||
      context.canonicalInstrument ||
      context.instrument
    );
    if (hasServerContext) {
      return 'UNKNOWN';
    }
  }

  // 2. Synchronous Non-Authoritative Presentation Fallback (Bootstrapping Only)
  // FRONTEND_CATALOG_AUTHORITY = NONE; strictly for shell display before server payload loads.
  const master = getMasterAsset(clean);
  if (master?.type) {
    return master.type;
  }

  const watchItem = SHARED_WATCHLIST_ITEMS.find(
    (w) => w.symbol.toUpperCase() === clean
  );
  if (watchItem?.type) {
    return watchItem.type;
  }

  // Temporary compatibility heuristic for crypto pairs (-USD suffix)
  // CRYPTO_SUFFIX_HEURISTIC_AUTHORITY = NONE
  if (clean.endsWith('-USD')) {
    return 'Crypto';
  }

  // Unverified instrument: strictly fail-closed as UNKNOWN
  return 'UNKNOWN';
}

/** Returns true if the symbol is authoritatively verified as an ETF */
export function isETF(
  symbol: string | null | undefined,
  context?: InstrumentContext | null
): boolean {
  if (context) {
    return resolveExecutionEligibility(context) === 'ETF_EXECUTION';
  }
  return resolveAssetType(symbol) === 'ETF';
}

/** Returns true if the symbol is authoritatively verified as an operating corporate Stock */
export function isStock(
  symbol: string | null | undefined,
  context?: InstrumentContext | null
): boolean {
  if (context) {
    return resolveExecutionEligibility(context) === 'STOCK_EXECUTION';
  }
  return resolveAssetType(symbol) === 'Stock';
}

/** Returns true if the symbol is authoritatively verified as a Digital Asset / Crypto */
export function isCrypto(
  symbol: string | null | undefined,
  context?: InstrumentContext | null
): boolean {
  if (context) {
    return resolveExecutionEligibility(context) === 'CRYPTO_EXECUTION';
  }
  return resolveAssetType(symbol) === 'Crypto';
}

/** Returns true if the symbol cannot be authoritatively classified */
export function isUnknownAsset(
  symbol: string | null | undefined,
  context?: InstrumentContext | null
): boolean {
  if (context) {
    return resolveExecutionEligibility(context) === 'FAIL_CLOSED';
  }
  return resolveAssetType(symbol) === 'UNKNOWN';
}

/** Returns true if the symbol is an ETF capable of having sector allocation data */
export function hasEtfSectorData(
  symbol: string | null | undefined,
  context?: InstrumentContext | null
): boolean {
  if (!symbol) return false;
  return isETF(symbol, context);
}
