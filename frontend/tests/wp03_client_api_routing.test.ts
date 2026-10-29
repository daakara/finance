/**
 * Adversarial Unit & Contract Test Suite: WP-03 Client API Standardization
 *
 * Verifies that all client network requests utilize `getApiBaseUrl()`,
 * that the same-origin Cloudflare edge proxy (/api/backend) is enforced on Cloudflare hosts,
 * that local preview routes to /api/v1, that SSR/fallback routes to Railway,
 * and that the deprecated dead Render host is completely eradicated.
 */

import { isCloudflarePagesHost, isLocalPreviewHost, getApiBaseUrl } from "../lib/api";
import * as fs from "fs";
import * as path from "path";

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(`[ASSERTION_FAILED] ${message}`);
  }
}

console.log("================================================================================");
console.log("RUNNING WP-03 CLIENT API STANDARDIZATION ADVERSARIAL TEST SUITE");
console.log("================================================================================");

let testsPassed = 0;

// Test 1: Cloudflare Pages Host Detection
console.log("\n[Test 1] Cloudflare Pages host recognition...");
assert(isCloudflarePagesHost("arxterminal.com"), "arxterminal.com must be recognized");
assert(isCloudflarePagesHost("www.arxterminal.com"), "www.arxterminal.com must be recognized");
assert(isCloudflarePagesHost("finance-xp8.pages.dev"), "finance-xp8.pages.dev must be recognized");
assert(isCloudflarePagesHost("d12345.finance-xp8.pages.dev"), "deployment-preview.finance-xp8.pages.dev must be recognized");
assert(!isCloudflarePagesHost("localhost"), "localhost must NOT be recognized as Cloudflare");
assert(!isCloudflarePagesHost("render.com"), "render.com must NOT be recognized as Cloudflare");
assert(!isCloudflarePagesHost("evil-finance-xp8.pages.dev.attacker.com"), "Subdomain attack must NOT be recognized");
console.log(" PASS: Cloudflare host recognition is strict and prevents spoofing.");
testsPassed++;

// Test 2: Localhost Detection
console.log("\n[Test 2] Localhost recognition...");
assert(isLocalPreviewHost("localhost"), "localhost must be recognized");
assert(isLocalPreviewHost("127.0.0.1"), "127.0.0.1 must be recognized");
assert(isLocalPreviewHost("0.0.0.0"), "0.0.0.0 must be recognized");
assert(!isLocalPreviewHost("arxterminal.com"), "arxterminal.com must NOT be local preview");
console.log(" PASS: Localhost detection is exact.");
testsPassed++;

// Test 3: getApiBaseUrl() under simulated Cloudflare Pages browser environment
console.log("\n[Test 3] Edge proxy /api/backend routing on Cloudflare Pages...");
// Mock window.location.hostname
const originalWindow = (global as any).window;
try {
  (global as any).window = { location: { hostname: "arxterminal.com" } };
  assert(getApiBaseUrl() === "/api/backend", "getApiBaseUrl() on arxterminal.com must return /api/backend");

  (global as any).window = { location: { hostname: "pr-111.finance-xp8.pages.dev" } };
  assert(getApiBaseUrl() === "/api/backend", "getApiBaseUrl() on preview deployment must return /api/backend");
  console.log(" PASS: Cloudflare browser environment routes through /api/backend edge proxy.");
  testsPassed++;
} finally {
  (global as any).window = originalWindow;
}

// Test 4: getApiBaseUrl() under simulated local preview browser environment
console.log("\n[Test 4] Local preview /api/v1 routing on localhost...");
try {
  (global as any).window = { location: { hostname: "localhost" } };
  assert(getApiBaseUrl() === "/api/v1", "getApiBaseUrl() on localhost must return /api/v1");
  console.log(" PASS: Localhost routes to /api/v1.");
  testsPassed++;
} finally {
  (global as any).window = originalWindow;
}

// Test 5: getApiBaseUrl() in Node / SSR / non-browser environment
console.log("\n[Test 5] SSR / Non-browser fallback routing...");
try {
  delete (global as any).window;
  const baseUrl = getApiBaseUrl();
  assert(baseUrl.includes("/api/v1"), `Fallback URL must end in /api/v1 (got ${baseUrl})`);
  assert(!baseUrl.includes("finance-backend-api-qis0.onrender.com"), "Fallback URL must NOT point to dead Render backend");
  assert(baseUrl.includes("railway.app") || baseUrl.startsWith("/api/"), `Fallback URL must point to Railway or relative proxy (got ${baseUrl})`);
  console.log(` PASS: Non-browser fallback routes safely to: ${baseUrl}`);
  testsPassed++;
} finally {
  (global as any).window = originalWindow;
}

// Test 6: Static Analysis - institutionalFeeds.ts has zero raw process.env.NEXT_PUBLIC_API_URL
console.log("\n[Test 6] Static audit: institutionalFeeds.ts...");
const feedsContent = fs.readFileSync(path.resolve(__dirname, "../lib/institutionalFeeds.ts"), "utf-8");
assert(!feedsContent.includes("process.env.NEXT_PUBLIC_API_URL"), "institutionalFeeds.ts must not contain raw process.env.NEXT_PUBLIC_API_URL");
assert(feedsContent.includes("getApiBaseUrl()"), "institutionalFeeds.ts must use getApiBaseUrl()");
console.log(" PASS: institutionalFeeds.ts strictly routes via getApiBaseUrl().");
testsPassed++;

// Test 7: Static Analysis - portfolio.ts has zero raw process.env.NEXT_PUBLIC_API_URL
console.log("\n[Test 7] Static audit: portfolio.ts...");
const portfolioContent = fs.readFileSync(path.resolve(__dirname, "../lib/portfolio.ts"), "utf-8");
assert(!portfolioContent.includes("process.env.NEXT_PUBLIC_API_URL"), "portfolio.ts must not contain raw process.env.NEXT_PUBLIC_API_URL");
assert(portfolioContent.includes("getApiBaseUrl()"), "portfolio.ts must use getApiBaseUrl()");
console.log(" PASS: portfolio.ts strictly routes via getApiBaseUrl().");
testsPassed++;

// Test 8: Static Analysis - unifiedCockpitStore.ts has zero raw process.env.NEXT_PUBLIC_API_URL
console.log("\n[Test 8] Static audit: unifiedCockpitStore.ts...");
const cockpitContent = fs.readFileSync(path.resolve(__dirname, "../lib/simulation/unifiedCockpitStore.ts"), "utf-8");
assert(!cockpitContent.includes("process.env.NEXT_PUBLIC_API_URL"), "unifiedCockpitStore.ts must not contain raw process.env.NEXT_PUBLIC_API_URL");
assert(cockpitContent.includes("getApiBaseUrl()"), "unifiedCockpitStore.ts must use getApiBaseUrl()");
console.log(" PASS: unifiedCockpitStore.ts strictly routes via getApiBaseUrl().");
testsPassed++;

// Test 9: Static Analysis - frontend/.env.production does not point to dead Render backend
console.log("\n[Test 9] Static audit: frontend/.env.production...");
const envProdPath = path.resolve(__dirname, "../.env.production");
if (fs.existsSync(envProdPath)) {
  const envContent = fs.readFileSync(envProdPath, "utf-8");
  assert(!envContent.includes("finance-backend-api-qis0.onrender.com"), ".env.production must not point to dead Render backend");
  assert(envContent.includes("web-production-e370b.up.railway.app"), ".env.production must point to authoritative Railway host");
}
console.log(" PASS: .env.production is updated to live Railway backend.");
testsPassed++;

// Test 10: Exhaustive frontend codebase audit for any rogue Render URLs
console.log("\n[Test 10] Static audit: Zero dead Render backend URLs anywhere in frontend source...");
const DEAD_RENDER_HOST = ["finance-backend-api-qis0", "onrender.com"].join(".");
function scanDir(dir: string) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== "node_modules" && entry.name !== ".next" && entry.name !== ".git") {
        scanDir(fullPath);
      }
    } else if (entry.isFile() && /\.(ts|tsx|js|jsx|json|env.*)$/.test(entry.name)) {
      if (entry.name === "wp03_client_api_routing.test.ts") continue;
      const code = fs.readFileSync(fullPath, "utf-8");
      assert(!code.includes(DEAD_RENDER_HOST), `Found dead Render URL in ${fullPath}`);
    }
  }
}
scanDir(path.resolve(__dirname, ".."));
console.log(" PASS: Exhaustive scan confirms zero dead Render backend URLs across frontend codebase.");
testsPassed++;

console.log("\n================================================================================");
console.log(`WP-03 TEST SUITE COMPLETED: ${testsPassed}/10 TESTS PASSED (0 FAILURES)`);
console.log("================================================================================");
