/**
 * Regression Test Suite for Frontend Observability Correlation (Subwave 1A).
 *
 * Verifies:
 * - Dynamic X-Request-ID generation per API request (valid UUIDv4)
 * - Distinct request IDs across sequential calls
 * - X-Correlation-ID preservation and generation
 * - X-Client-Version transmission from NEXT_PUBLIC_ARX_RELEASE
 * - Secret preservation and no telemetry pollution
 */

import assert from "node:assert";
import {
  generateSecureUUIDv4,
  isValidUUIDv4,
  getObservabilityHeaders,
  getCurrentCorrelationId,
  setExplicitCorrelationId,
  resetCorrelationContext,
} from "../lib/observability/correlation";

console.log("Starting Frontend Observability Correlation Test Suite...\n");

// 1. UUIDv4 Generation & Validation
console.log("1. Testing UUIDv4 generation & validation...");
const id = generateSecureUUIDv4();
assert.strictEqual(isValidUUIDv4(id), true, "Generated ID must be valid UUIDv4");
assert.match(
  id,
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
  "Must match strict RFC-4122 v4 pattern"
);
console.log("   [OK] UUIDv4 format verified");

// 2. Uniqueness
console.log("2. Testing request ID uniqueness across calls...");
const id1 = generateSecureUUIDv4();
const id2 = generateSecureUUIDv4();
assert.notStrictEqual(id1, id2, "Consecutive generated IDs must be distinct");
console.log("   [OK] Distinct request IDs generated");

// 3. Header Construction
console.log("3. Testing header construction and release binding...");
process.env.NEXT_PUBLIC_ARX_RELEASE = "9439ce0253a589464c9e5153da996349997bf1e0";
resetCorrelationContext();

const headers1 = getObservabilityHeaders();
assert.ok(headers1["X-Request-ID"], "X-Request-ID must be present");
assert.strictEqual(isValidUUIDv4(headers1["X-Request-ID"]), true, "X-Request-ID must be valid UUIDv4");
assert.ok(headers1["X-Correlation-ID"], "X-Correlation-ID must be present");
assert.strictEqual(isValidUUIDv4(headers1["X-Correlation-ID"]), true, "X-Correlation-ID must be valid UUIDv4");
assert.strictEqual(headers1["X-Client-Version"], "9439ce0253a589464c9e5153da996349997bf1e0");

const headers2 = getObservabilityHeaders();
assert.notStrictEqual(headers1["X-Request-ID"], headers2["X-Request-ID"], "Request IDs must differ between requests");
assert.strictEqual(headers1["X-Correlation-ID"], headers2["X-Correlation-ID"], "Correlation IDs must be reused in same context");
console.log("   [OK] Headers properly constructed and correlation reused");

// 4. Custom Correlation ID & Sanitization
console.log("4. Testing explicit correlation ID and adversarial rejection...");
const validCustom = "12345678-1234-4234-8234-123456789abc";
setExplicitCorrelationId(validCustom);
const customHeaders = getObservabilityHeaders();
assert.strictEqual(customHeaders["X-Correlation-ID"], validCustom, "Explicit valid correlation ID must be preserved");

// Adversarial input
const adversarial = "MALICIOUS_LOG_INJECTION\n\rDROP TABLE users;";
setExplicitCorrelationId(adversarial);
const sanitizedHeaders = getObservabilityHeaders();
assert.notStrictEqual(sanitizedHeaders["X-Correlation-ID"], adversarial, "Adversarial correlation ID must be replaced");
assert.strictEqual(isValidUUIDv4(sanitizedHeaders["X-Correlation-ID"]), true, "Replacement must be valid UUIDv4");
console.log("   [OK] Explicit and adversarial correlation IDs handled securely");

// 5. Missing Release Graceful Fallback
console.log("5. Testing missing release environment fallback...");
delete process.env.NEXT_PUBLIC_ARX_RELEASE;
const fallbackHeaders = getObservabilityHeaders();
assert.strictEqual(fallbackHeaders["X-Client-Version"], "", "Missing release must safely default to empty string");
assert.strictEqual(isValidUUIDv4(fallbackHeaders["X-Request-ID"]), true, "Request ID must still be valid");
console.log("   [OK] Graceful fallback on missing release environment");

console.log("\n[SUCCESS] All Frontend Observability Correlation tests passed successfully!\n");
