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
 * Retrieves or initializes the active in-memory correlation ID.
 * Does not write to localStorage or cookies.
 */
export function getCurrentCorrelationId(): string {
  if (!activeCorrelationId) {
    activeCorrelationId = generateSecureUUIDv4();
  }
  return activeCorrelationId;
}

/**
 * Sets an explicit correlation ID for a scoped transaction.
 * Replaces invalid/adversarial input with a safe generated UUIDv4.
 */
export function setExplicitCorrelationId(correlationId: string): string {
  if (isValidUUIDv4(correlationId)) {
    activeCorrelationId = correlationId.trim().toLowerCase();
  } else {
    activeCorrelationId = generateSecureUUIDv4();
  }
  return activeCorrelationId;
}

/**
 * Resets the active in-memory correlation context.
 */
export function resetCorrelationContext(): void {
  activeCorrelationId = null;
}

/**
 * Constructs the canonical observability headers object for an outbound API request.
 * - X-Request-ID: unique UUIDv4 per individual request.
 * - X-Correlation-ID: preserved operation UUIDv4.
 * - X-Client-Version: build commit SHA from NEXT_PUBLIC_ARX_RELEASE.
 */
export function getObservabilityHeaders(explicitCorrelationId?: string): Record<string, string> {
  const requestId = generateSecureUUIDv4();

  let correlationId: string;
  if (explicitCorrelationId && isValidUUIDv4(explicitCorrelationId)) {
    correlationId = explicitCorrelationId.trim().toLowerCase();
  } else if (explicitCorrelationId) {
    correlationId = generateSecureUUIDv4();
  } else {
    correlationId = getCurrentCorrelationId();
  }

  const clientRelease = process.env.NEXT_PUBLIC_ARX_RELEASE || "";

  return {
    "X-Request-ID": requestId,
    "X-Correlation-ID": correlationId,
    "X-Client-Version": clientRelease,
  };
}
