import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("Starting ARX CSP & Telemetry Console Integrity Invariant Test Suite...\n");

const FRONTEND_DIR = path.resolve(__dirname, "..");
const HEADERS_PATH = path.join(FRONTEND_DIR, "public", "_headers");
const LAYOUT_PATH = path.join(FRONTEND_DIR, "app", "layout.tsx");

// ---------------------------------------------------------------------------
// 1. Invariant Check: frontend/public/_headers CSP connect-src
// ---------------------------------------------------------------------------
console.log("1. Verifying Cloudflare Pages _headers CSP connect-src rules...");
const headersContent = fs.readFileSync(HEADERS_PATH, "utf-8");

// Must contain Content-Security-Policy
assert.ok(
  headersContent.includes("Content-Security-Policy:"),
  "FAIL: _headers must contain Content-Security-Policy header"
);

// Must contain exact Sentry origin
const exactSentryOrigin = "https://o4511406482522112.ingest.de.sentry.io";
assert.ok(
  headersContent.includes(exactSentryOrigin),
  `FAIL: CSP connect-src must include exact Sentry ingestion origin ${exactSentryOrigin}`
);
console.log("   [OK] Exact Sentry ingestion origin is present in connect-src");

// Must NOT contain broad wildcards for Sentry
assert.strictEqual(
  headersContent.includes("*.sentry.io") || headersContent.includes("*.ingest."),
  false,
  "FAIL: CSP connect-src must NOT contain broad wildcards for Sentry"
);
console.log("   [OK] Zero broad wildcards in connect-src (strict least-privilege origin enforced)");

// Must preserve all existing API connections
const requiredOrigins = [
  "'self'",
  "https://data.fpldna.com",
  "https://finance-backend-api-qis0.onrender.com",
  "https://web-production-e370b.up.railway.app",
  "https://query1.finance.yahoo.com",
  "https://query2.finance.yahoo.com",
];
for (const origin of requiredOrigins) {
  assert.ok(
    headersContent.includes(origin),
    `FAIL: CSP connect-src must preserve existing origin: ${origin}`
  );
}
console.log("   [OK] All existing API origins preserved in connect-src");

// ---------------------------------------------------------------------------
// 2. Invariant Check: frontend/app/layout.tsx Matomo Preload Elimination
// ---------------------------------------------------------------------------
console.log("2. Verifying Matomo preload elimination in RootLayout (layout.tsx)...");
const layoutContent = fs.readFileSync(LAYOUT_PATH, "utf-8");

// Must NOT contain the <noscript> image tag that triggers Next.js link preload
assert.strictEqual(
  layoutContent.includes("https://data.fpldna.com/matomo/matomo.php?idsite=3&rec=1"),
  false,
  "FAIL: layout.tsx must NOT contain the Matomo noscript image tag (causes unused link preload warning)"
);
console.log("   [OK] Unused Matomo image fallback eliminated from RootLayout");

// Must preserve normal Matomo Tag Manager & Analytics JavaScript tracking
assert.ok(
  layoutContent.includes("_paq.push(['setTrackerUrl', 'https://data.fpldna.com/matomo/matomo.php']);"),
  "FAIL: layout.tsx must preserve _paq.setTrackerUrl for analytics tracking"
);
assert.ok(
  layoutContent.includes("https://data.fpldna.com/matomo/js/container_tK4RnlSN.js"),
  "FAIL: layout.tsx must preserve Matomo Tag Manager container script"
);
console.log("   [OK] Normal Matomo JavaScript analytics tracking preserved");

console.log("\n[ALL CSP & TELEMETRY CONSOLE INTEGRITY INVARIANT CHECKS PASSED SUCCESSFULLY]");
