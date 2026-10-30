/**
 * Centralized Decision Context Authority & Validation Module (ARX-036-D11R-FINAL Gate 4)
 *
 * Implements strict runtime verification of decisionContextId grammar, bounds, safe encoding,
 * symbol consistency, and Analysis Hub cache isolation.
 *
 * Epistemic Invariants:
 * 1. Valid context IDs received from URLs or external navigation are classified as CONTEXT_RECEIVED_UNVERIFIED.
 * 2. CONTEXT_VERIFIED is strictly reserved for authoritative backend lookup confirmation.
 * 3. Malformed, oversized, undersized, or mismatched context IDs fail closed to null.
 * 4. Context IDs NEVER alter trade sizing, execution permissions, or financial authority.
 * 5. Analysis Hub cache keys remain strictly isolated from context IDs.
 */

export type ContextValidationStatus =
  | "CONTEXT_ABSENT"
  | "CONTEXT_MALFORMED"
  | "CONTEXT_SYMBOL_MISMATCH"
  | "CONTEXT_RECEIVED_UNVERIFIED"
  | "CONTEXT_VERIFIED";

export interface ContextValidationResult {
  isValid: boolean;
  status: ContextValidationStatus;
  contextId: string | null;
  symbol: string | null;
  timestampMs: number | null;
  error?: string;
}

// Canonical grammar: dec-radar-{SYMBOL}-{TIMESTAMP_MS}
// Minimum: "dec-radar-A-0000000000" (22 chars)
// Maximum: 64 characters (guards against payload bloat / DoS)
const CONTEXT_ID_REGEX = /^dec-radar-([A-Z0-9.\-_]{1,16})-([0-9]{10,15})$/;
export const MIN_CONTEXT_ID_LENGTH = 16;
export const MAX_CONTEXT_ID_LENGTH = 64;

/**
 * Validates a decisionContextId against canonical grammar, length bounds, safe encoding,
 * and active symbol consistency.
 *
 * @param rawContextId - Raw contextId string from query parameter or component prop
 * @param expectedSymbol - Active ticker symbol in the viewing context
 * @returns ContextValidationResult with fail-closed status and nullified contextId on any error
 */
export function validateDecisionContext(
  rawContextId: string | null | undefined,
  expectedSymbol?: string | null | undefined
): ContextValidationResult {
  if (!rawContextId || typeof rawContextId !== "string" || rawContextId.trim().length === 0) {
    return {
      isValid: false,
      status: "CONTEXT_ABSENT",
      contextId: null,
      symbol: null,
      timestampMs: null,
    };
  }

  const trimmed = rawContextId.trim();

  // 1. Length Bounds Check
  if (trimmed.length < MIN_CONTEXT_ID_LENGTH || trimmed.length > MAX_CONTEXT_ID_LENGTH) {
    return {
      isValid: false,
      status: "CONTEXT_MALFORMED",
      contextId: null,
      symbol: null,
      timestampMs: null,
      error: `Context ID length ${trimmed.length} outside valid bounds [${MIN_CONTEXT_ID_LENGTH}, ${MAX_CONTEXT_ID_LENGTH}]`,
    };
  }

  // 2. Safe URL Decoding
  let decoded: string;
  try {
    decoded = decodeURIComponent(trimmed);
  } catch {
    return {
      isValid: false,
      status: "CONTEXT_MALFORMED",
      contextId: null,
      symbol: null,
      timestampMs: null,
      error: "Context ID contains malformed URI encoding",
    };
  }

  // 3. Exact Grammar Check
  const match = CONTEXT_ID_REGEX.exec(decoded);
  if (!match) {
    return {
      isValid: false,
      status: "CONTEXT_MALFORMED",
      contextId: null,
      symbol: null,
      timestampMs: null,
      error: "Context ID does not conform to canonical grammar ^dec-radar-{SYMBOL}-{TIMESTAMP}$",
    };
  }

  const extractedSymbol = match[1].toUpperCase();
  const timestampMs = parseInt(match[2], 10);

  // 4. Timestamp Sanity Check (Must be positive integer)
  if (!Number.isFinite(timestampMs) || timestampMs <= 0) {
    return {
      isValid: false,
      status: "CONTEXT_MALFORMED",
      contextId: null,
      symbol: extractedSymbol,
      timestampMs: null,
      error: "Context ID contains non-finite or non-positive timestamp",
    };
  }

  // 5. Symbol Consistency Check (prevents cross-symbol context leakage)
  if (expectedSymbol && typeof expectedSymbol === "string" && expectedSymbol.trim().length > 0) {
    const cleanExpected = expectedSymbol.trim().toUpperCase();
    if (extractedSymbol !== cleanExpected) {
      return {
        isValid: false,
        status: "CONTEXT_SYMBOL_MISMATCH",
        contextId: null,
        symbol: extractedSymbol,
        timestampMs,
        error: `Context symbol '${extractedSymbol}' does not match expected view symbol '${cleanExpected}'`,
      };
    }
  }

  // 6. Valid Grammar and Matching Symbol -> CONTEXT_RECEIVED_UNVERIFIED
  // Note: Epistemic Invariant: Unverified until authoritative backend DecisionTrace lookup occurs.
  return {
    isValid: true,
    status: "CONTEXT_RECEIVED_UNVERIFIED",
    contextId: decoded,
    symbol: extractedSymbol,
    timestampMs,
  };
}

/**
 * Computes the authoritative Analysis Hub memory cache key.
 * Strictly decoupled from decisionContextId to prevent cache pollution and ensure deterministic TTLs.
 */
export function getAnalysisHubCacheKey(symbol: string, interval: string, userRole: string): string {
  const cleanSym = symbol ? symbol.trim().toUpperCase() : "UNKNOWN";
  return `${cleanSym}_${interval}_${userRole}`;
}
