/**
 * ARX Frontend Observability Correlation Module (Subwave 1A).
 *
 * Implements:
 * 1. RFC-4122 UUIDv4 generation for unique per-request tracing (X-Request-ID).
 * 2. In-memory logical operation tracing (X-Correlation-ID).
 * 3. Client build release tagging (X-Client-Version) from NEXT_PUBLIC_ARX_RELEASE.
 * 4. Strict privacy: Zero persistent user tracking, zero localStorage telemetry IDs.
 */

let activeCorrelationId: string | null = null;

const UUIDV4_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/**
 * Validates that a string matches RFC-4122 UUIDv4 format.
 */
export function isValidUUIDv4(val: unknown): boolean {
  if (typeof val !== "string") return false;
  return UUIDV4_REGEX.test(val.trim());
}

/**
 * Generates an RFC-4122 compliant UUIDv4 using crypto.randomUUID where available,
 * with standard cryptographic fallback.
 */
export function generateSecureUUIDv4(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  // Cryptographically secure fallback
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    // Set version 4 (bits 12-15 = 0100)
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    // Set variant (bits 6-7 = 10)
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  // Math.random fallback (e.g. ancient synthetic environments)
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Immutable correlation operation token interface.
 * Enforces explicit operation scoping across asynchronous boundaries.
 */
export interface CorrelationOperation {
  readonly correlationId: string;
  readonly createdAt: number;
  getHeaders(): Record<string, string>;
}

/**
 * Creates an immutable, isolated CorrelationOperation token.
 * Multiple requests using this token share the same correlationId.
 * Independent operations receive distinct, non-overlapping correlationIds.
 */
export function createCorrelationOperation(explicitCorrelationId?: string): CorrelationOperation {
  let cid: string;
  if (typeof explicitCorrelationId === "string" && isValidUUIDv4(explicitCorrelationId)) {
    cid = explicitCorrelationId.trim().toLowerCase();
  } else {
    cid = generateSecureUUIDv4();
  }

  const createdAt = Date.now();
  return Object.freeze({
    correlationId: cid,
    createdAt,
    getHeaders(): Record<string, string> {
      return getObservabilityHeaders(cid);
    },
  });
}

/**
 * Scoped execution helper for running asynchronous operations within a correlation context.
 */
export async function withCorrelationOperation<T>(
  operationOrId: CorrelationOperation | string | undefined,
  fn: (operation: CorrelationOperation) => Promise<T>
): Promise<T> {
  const operation =
    typeof operationOrId === "object" && operationOrId !== null && "correlationId" in operationOrId
      ? operationOrId
      : createCorrelationOperation(operationOrId);
  return await fn(operation);
}

/**
 * Constructs the canonical observability headers object for an outbound API request.
 * - X-Request-ID: unique UUIDv4 generated per individual HTTP request.
 * - X-Correlation-ID: bound from the provided CorrelationOperation or explicit UUIDv4 string.
 *   If no operation or explicit ID is provided (standalone request), a fresh, isolated
 *   correlation ID is generated for this single request. Never reuses a sticky global singleton.
 * - X-Client-Version: build commit SHA from NEXT_PUBLIC_ARX_RELEASE.
 */
export function getObservabilityHeaders(
  operationOrId?: CorrelationOperation | string
): Record<string, string> {
  const requestId = generateSecureUUIDv4();

  let correlationId: string;
  if (typeof operationOrId === "object" && operationOrId !== null && "correlationId" in operationOrId) {
    correlationId = operationOrId.correlationId;
  } else if (typeof operationOrId === "string") {
    if (isValidUUIDv4(operationOrId)) {
      correlationId = operationOrId.trim().toLowerCase();
    } else {
      correlationId = generateSecureUUIDv4();
    }
  } else {
    // Standalone request: generate an isolated correlation ID.
    // Preserves 1:1 request-correlation identity and prevents cross-request context leakage.
    correlationId = generateSecureUUIDv4();
  }

  const clientRelease = process.env.NEXT_PUBLIC_ARX_RELEASE || "";

  return {
    "X-Request-ID": requestId,
    "X-Correlation-ID": correlationId,
    "X-Client-Version": clientRelease,
  };
}

/**
 * @deprecated Legacy mutable singleton accessor retired in remediation.
 * Returns an isolated standalone UUIDv4 without persisting global state.
 * For multi-request logical operations, use `createCorrelationOperation()`.
 */
export function getCurrentCorrelationId(): string {
  return generateSecureUUIDv4();
}

/**
 * @deprecated Legacy mutable singleton setter retired in remediation.
 * For multi-request logical operations, use `createCorrelationOperation(correlationId)`.
 */
export function setExplicitCorrelationId(correlationId: string): string {
  return isValidUUIDv4(correlationId) ? correlationId.trim().toLowerCase() : generateSecureUUIDv4();
}

/**
 * @deprecated Legacy mutable singleton resetter retired in remediation.
 * No-op: global mutable singleton has been eliminated.
 */
export function resetCorrelationContext(): void {
  // No-op: immutable operation tokens manage their own lifecycle.
}
