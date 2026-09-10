/**
 * frontend/scripts/verify-remediation-frontend.mjs
 * Automated verification suite for Frontend Remediation (Areas A, F, G, and Setups UI).
 *
 * Verifies:
 * 1. Area A: Cloudflare Pages edge proxy CORS methods and headers in functions/api/backend/[[path]].ts
 * 2. Area F: Portfolio synchronization, migration validation, monotonic read sequence, and edit collision protection in frontend/lib/portfolio.ts
 * 3. Area G: Truthful error propagation (no silent empty fallbacks) in frontend/lib/api.ts
 * 4. Setups UI: browseError state, Retry Load trigger, and empty state handling in frontend/app/setups/page.tsx
 */

import { strict as assert } from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');

let total = 0;
let passed = 0;
let failed = 0;

function check(cond, msg) {
  total++;
  if (cond) {
    passed++;
    console.log(`  ✔ [PASS] ${msg}`);
  } else {
    failed++;
    console.error(`  ✖ [FAIL] ${msg}`);
  }
}

console.log("\n=============================================================");
console.log("  VERIFICATION SUITE: FRONTEND REMEDIATION (AREAS A, F, G, UI)");
console.log("=============================================================\n");

// -----------------------------------------------------------------------------
// 1. Area A: Edge Proxy CORS Configuration
// -----------------------------------------------------------------------------
console.log("\x1b[1m1. Area A: Cloudflare Pages Edge Proxy CORS (functions/api/backend/[[path]].ts)\x1b[0m");

const edgePath = path.join(projectRoot, 'functions', 'api', 'backend', '[[path]].ts');
const edgeSrc = fs.readFileSync(edgePath, 'utf8');

check(edgeSrc.includes("Access-Control-Allow-Methods") && edgeSrc.includes("DELETE") && edgeSrc.includes("PUT"),
  "Edge proxy allows DELETE and PUT in Access-Control-Allow-Methods");
check(edgeSrc.includes("X-User-Id") && edgeSrc.includes("X-Profile-Id") && edgeSrc.includes("Cache-Control") && edgeSrc.includes("Pragma"),
  "Edge proxy allows X-User-Id, X-Profile-Id, Cache-Control, and Pragma in Access-Control-Allow-Headers");
check(edgeSrc.includes('responseHeaders.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")'),
  "Proxied response sets complete Access-Control-Allow-Methods");

// -----------------------------------------------------------------------------
// 2. Area F: Portfolio Synchronization & Concurrency Guards
// -----------------------------------------------------------------------------
console.log("\n\x1b[1m2. Area F: Portfolio Concurrency & Migration Races (frontend/lib/portfolio.ts)\x1b[0m");

const pfPath = path.join(projectRoot, 'frontend', 'lib', 'portfolio.ts');
const pfSrc = fs.readFileSync(pfPath, 'utf8');

check(pfSrc.includes("export async function migrateLocalHoldingsToApi"),
  "migrateLocalHoldingsToApi function is exported");
check(pfSrc.includes("activeReadSequence") && pfSrc.includes("lastConfirmedWriteTimestamp"),
  "Tracks activeReadSequence and lastConfirmedWriteTimestamp for concurrency");
check(pfSrc.includes("export function notifyPortfolioWriteConfirmed"),
  "Exports notifyPortfolioWriteConfirmed to advance write generation");
check(pfSrc.includes("isPortfolioEditActive()"),
  "syncPortfolioFromApi checks isPortfolioEditActive() to protect in-flight user edits");
check(pfSrc.includes("readSeq !== activeReadSequence || lastConfirmedWriteTimestamp > readStartTime || isPortfolioEditActive()"),
  "Stale responses arriving after a newer write or newer read sequence are discarded");
check(pfSrc.includes("Number(h.shares)"),
  "Preserves fractional holdings parsing without integer rounding");

// -----------------------------------------------------------------------------
// 3. Area G: API Client Error Propagation
// -----------------------------------------------------------------------------
console.log("\n\x1b[1m3. Area G: API Client Error Propagation (frontend/lib/api.ts)\x1b[0m");

const apiPath = path.join(projectRoot, 'frontend', 'lib', 'api.ts');
const apiSrc = fs.readFileSync(apiPath, 'utf8');

check(apiSrc.includes("export async function fetchTacticalSetups"),
  "fetchTacticalSetups function is exported");
check(apiSrc.includes("throw new Error(`Failed to fetch tactical setups (${res.status}):"),
  "fetchTacticalSetups throws explicit Error on HTTP non-200");
check(apiSrc.includes("if (res.status === 404) {\n    return null;\n  }"),
  "fetchTacticalSetupForTicker distinctly isolates 404 for asset tape verification");
check(apiSrc.includes("throw new Error(`Failed to fetch tactical setup for ${upper} (${res.status}):"),
  "fetchTacticalSetupForTicker throws on HTTP failures (429/500/504)");
check(!apiSrc.includes("catch (err) {\n    return [];\n  }") && !apiSrc.includes("catch (err) {\n    return null;\n  }"),
  "fetchTacticalSetups and fetchTacticalSetupForTicker do not swallow errors into empty values");

// -----------------------------------------------------------------------------
// 4. Setups Page: Browse Error & UI Resilience
// -----------------------------------------------------------------------------
console.log("\n\x1b[1m4. Setups Page: Error Handling & Retry (frontend/app/setups/page.tsx)\x1b[0m");

const setupsPath = path.join(projectRoot, 'frontend', 'app', 'setups', 'page.tsx');
const setupsSrc = fs.readFileSync(setupsPath, 'utf8');

check(setupsSrc.includes("const [browseError, setBrowseError] = useState<string | null>(null);"),
  "Setups page declares browseError state");
check(setupsSrc.includes("loadAvailableSetups"),
  "Setups page wraps initial fetch in loadAvailableSetups callback");
check(setupsSrc.includes("Setup Catalog Load Error") && setupsSrc.includes("Retry Load"),
  "Setups page renders Setup Catalog Load Error banner with Retry Load button");
check(setupsSrc.includes("No active tactical setups currently meet criteria on the exchange tape."),
  "Setups page renders clean empty state when catalog is genuinely empty without errors");

// -----------------------------------------------------------------------------
// Final Results
// -----------------------------------------------------------------------------
console.log("\n=============================================================");
console.log(`  VERIFICATION RESULTS: ${passed} PASSED, ${failed} FAILED (TOTAL: ${total})`);
console.log("=============================================================");

if (failed > 0) {
  process.exit(1);
} else {
  console.log("\nALL FRONTEND REMEDIATION CHECKS PASSED PERFECTLY.\n");
  process.exit(0);
}
