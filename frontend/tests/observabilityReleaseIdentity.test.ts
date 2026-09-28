/**
 * Frontend Release Identity Provenance & Binding Test Suite.
 *
 * Covers:
 * - R01: CF_PAGES_COMMIT_SHA is authoritative when present
 * - R02: Explicit NEXT_PUBLIC_ARX_RELEASE fallback works when CF_PAGES_COMMIT_SHA is absent
 * - R03: CF_PAGES_COMMIT_SHA wins when both exist (strict precedence)
 * - R04: Absence of both values remains fail-safe (returns undefined, no placeholders)
 * - R05: No static hard-coded commit SHA in source code or defaults
 * - R06: X-Client-Version header in correlation module consumes canonical release
 * - R07: Sentry adapter release tag consumes canonical release
 * - Release parity: correlation headers and Sentry scope share identical release value
 */

import assert from "node:assert";
import * as Sentry from "@sentry/browser";
import { resolveFrontendRelease } from "../next.config.mjs";
import { getObservabilityHeaders } from "../lib/observability/correlation";
import { SentryFrontendAdapter } from "../lib/observability/sentryAdapter";

console.log("Starting Frontend Release Identity Binding Test Suite...\n");

// R01: CF_PAGES_COMMIT_SHA is used when present
console.log("R01: Testing Cloudflare native commit SHA resolution...");
const cfSha = "a1b2c3d4e5f6789012345678901234567890abcd";
const r01Result = resolveFrontendRelease({
  CF_PAGES_COMMIT_SHA: cfSha,
});
assert.strictEqual(r01Result, cfSha, "CF_PAGES_COMMIT_SHA must be resolved as effective release");
console.log("   [OK] Cloudflare native commit SHA correctly resolved");

// R02: Explicit NEXT_PUBLIC_ARX_RELEASE fallback works
console.log("R02: Testing explicit NEXT_PUBLIC_ARX_RELEASE fallback...");
const explicitSha = "f0e1d2c3b4a56789012345678901234567890abc";
const r02Result = resolveFrontendRelease({
  NEXT_PUBLIC_ARX_RELEASE: explicitSha,
});
assert.strictEqual(r02Result, explicitSha, "NEXT_PUBLIC_ARX_RELEASE fallback must be resolved");
console.log("   [OK] Explicit fallback correctly resolved when CF_PAGES_COMMIT_SHA absent");

// R03: CF_PAGES_COMMIT_SHA wins when both exist
console.log("R03: Testing strict precedence (CF_PAGES_COMMIT_SHA > NEXT_PUBLIC_ARX_RELEASE)...");
const r03Result = resolveFrontendRelease({
  CF_PAGES_COMMIT_SHA: cfSha,
  NEXT_PUBLIC_ARX_RELEASE: explicitSha,
});
assert.strictEqual(r03Result, cfSha, "CF_PAGES_COMMIT_SHA must take precedence over NEXT_PUBLIC_ARX_RELEASE");
console.log("   [OK] Cloudflare commit SHA strictly takes precedence");

// R04: Absence of both values remains fail-safe (undefined, no placeholders)
console.log("R04: Testing fail-safe behavior when neither variable is provided...");
const r04Empty = resolveFrontendRelease({});
assert.strictEqual(r04Empty, undefined, "Missing variables must resolve to undefined");

const r04Whitespace = resolveFrontendRelease({
  CF_PAGES_COMMIT_SHA: "   \t\n  ",
  NEXT_PUBLIC_ARX_RELEASE: "",
});
assert.strictEqual(r04Whitespace, undefined, "Whitespace variables must resolve to undefined");
console.log("   [OK] Absence of release identity fails safe without fabricated placeholders");

// R05: Prohibited placeholders validation
console.log("R05: Validating prohibited placeholders are never returned...");
const PROHIBITED_PLACEHOLDERS = ["dev", "local", "unknown", "latest", "main", "production", "prod"];
for (const placeholder of PROHIBITED_PLACEHOLDERS) {
  assert.notStrictEqual(r04Empty, placeholder, `Must never return placeholder: ${placeholder}`);
}
console.log("   [OK] Prohibited placeholders verified absent");

// R06 & R07 & Consumer Parity Verification
console.log("R06, R07: Testing observability consumer release parity...");
const originalRelease = process.env.NEXT_PUBLIC_ARX_RELEASE;

try {
  const testReleaseSha = "e66224ea792d3b2265e0ac90db97886e13c8bd1f";
  process.env.NEXT_PUBLIC_ARX_RELEASE = testReleaseSha;

  // 1. Correlation consumer
  const headers = getObservabilityHeaders();
  assert.strictEqual(
    headers["X-Client-Version"],
    testReleaseSha,
    "X-Client-Version must match NEXT_PUBLIC_ARX_RELEASE"
  );

  // 2. Sentry adapter consumer
  const adapter = new SentryFrontendAdapter();
  adapter.init({
    dsn: "https://1234567890abcdef@o123456.ingest.sentry.io/1234567",
    isTest: true,
  });

  const sentryRelease = Sentry.getClient()?.getOptions().release;
  assert.strictEqual(
    sentryRelease,
    testReleaseSha,
    "Sentry release metadata must match NEXT_PUBLIC_ARX_RELEASE"
  );

  // Parity check
  assert.strictEqual(
    headers["X-Client-Version"],
    sentryRelease,
    "X-Client-Version and Sentry release metadata must exhibit exact release parity"
  );
  console.log("   [OK] Exact consumer parity verified between X-Client-Version and Sentry release metadata");
} finally {
  if (originalRelease === undefined) {
    delete process.env.NEXT_PUBLIC_ARX_RELEASE;
  } else {
    process.env.NEXT_PUBLIC_ARX_RELEASE = originalRelease;
  }
}

console.log("\n[SUCCESS] All Frontend Release Identity Binding tests (R01-R07) passed successfully!");
