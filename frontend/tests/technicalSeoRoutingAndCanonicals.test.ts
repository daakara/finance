import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import sitemap from "../app/sitemap";
import { getAllMasterTickers } from "../lib/masterCatalog";
import { SHARED_WATCHLIST_ITEMS } from "../lib/constants";

console.log("Starting ARX Technical SEO & Canonical Routing Invariant Test Suite...\n");

const FRONTEND_DIR = path.resolve(__dirname, "..");
const REDIRECTS_PATH = path.join(FRONTEND_DIR, "public", "_redirects");
const ROBOTS_PATH = path.join(FRONTEND_DIR, "public", "robots.txt");
const TERMINAL_SSR_PATH = path.join(FRONTEND_DIR, "components", "TerminalSsrShell.tsx");
const STOCK_TICKER_PAGE_PATH = path.join(FRONTEND_DIR, "app", "stock", "[ticker]", "page.tsx");
const STOCK_LAYOUT_PATH = path.join(FRONTEND_DIR, "app", "stock", "layout.tsx");
const SCREENER_LAYOUT_PATH = path.join(FRONTEND_DIR, "app", "screener", "layout.tsx");

// ---------------------------------------------------------------------------
// 1. Invariant Check: frontend/public/_redirects
// ---------------------------------------------------------------------------
console.log("1. Verifying Cloudflare Pages _redirects rules...");
const redirectsContent = fs.readFileSync(REDIRECTS_PATH, "utf-8");

// Must NOT contain the wildcard stock rewrite that hijacks pre-rendered static stock pages
assert.strictEqual(
  redirectsContent.includes("/stock/*  /stock/  200") || redirectsContent.includes("/stock/* /stock/ 200"),
  false,
  "FAIL: _redirects must NOT contain '/stock/* /stock/ 200' rewrite rule"
);
console.log("   [OK] Wildcard /stock/* rewrite is absent (allows static HTML to serve directly)");

// Must contain permanent 301 redirects for /screener to /radar
assert.ok(
  redirectsContent.includes("/screener     /radar  301") || redirectsContent.includes("/screener /radar 301"),
  "FAIL: _redirects must contain 301 redirect from /screener to /radar"
);
assert.ok(
  redirectsContent.includes("/screener/    /radar/  301") || redirectsContent.includes("/screener/ /radar/ 301"),
  "FAIL: _redirects must contain 301 redirect from /screener/ to /radar/"
);
console.log("   [OK] Permanent 301 redirects from /screener to /radar are present");

// ---------------------------------------------------------------------------
// 2. Invariant Check: frontend/app/sitemap.ts
// ---------------------------------------------------------------------------
console.log("2. Verifying sitemap.ts route inventory & canonical URL generation...");
const routes = sitemap();
const urls = routes.map((r) => r.url);

// /screener must NOT be in sitemap
assert.strictEqual(
  urls.some((u) => u.includes("/screener")),
  false,
  "FAIL: sitemap.ts must NOT include redirecting /screener/"
);
console.log("   [OK] /screener is excluded from sitemap");

// /radar/ must be in sitemap
assert.ok(
  urls.includes("https://www.arxterminal.com/radar/"),
  "FAIL: sitemap.ts must include canonical /radar/"
);
console.log("   [OK] Canonical /radar/ is present in sitemap");

// Every URL must have https://www.arxterminal.com and trailing slash
for (const url of urls) {
  assert.ok(
    url.startsWith("https://www.arxterminal.com/"),
    `FAIL: URL ${url} must start with https://www.arxterminal.com/`
  );
  assert.ok(
    url.endsWith("/"),
    `FAIL: URL ${url} must terminate with canonical trailing slash`
  );
}
console.log(`   [OK] All ${urls.length} sitemap URLs enforce https://www.arxterminal.com and trailing slash`);

// Check stock route inventory in sitemap
const masterTickers = getAllMasterTickers().map((t) => t.toLowerCase());
const watchlistTickers = SHARED_WATCHLIST_ITEMS.map((item) => item.symbol.toLowerCase());
const uniqueTickers = Array.from(new Set([...masterTickers, ...watchlistTickers])).sort();

assert.strictEqual(
  uniqueTickers.length,
  49,
  `FAIL: Expected 49 unique stock tickers, got ${uniqueTickers.length}`
);

for (const ticker of uniqueTickers) {
  const expectedUrl = `https://www.arxterminal.com/stock/${ticker}/`;
  assert.ok(
    urls.includes(expectedUrl),
    `FAIL: sitemap must include programmatic stock route: ${expectedUrl}`
  );
}
console.log(`   [OK] All 49 programmatic stock routes are present in sitemap`);

// ---------------------------------------------------------------------------
// 3. Invariant Check: frontend/public/robots.txt
// ---------------------------------------------------------------------------
console.log("3. Verifying robots.txt crawl directives & query hygiene...");
const robotsContent = fs.readFileSync(ROBOTS_PATH, "utf-8");

assert.ok(
  robotsContent.includes("Disallow: /*?*symbol=*"),
  "FAIL: robots.txt must disallow crawler query parameter /*?*symbol=*"
);
assert.ok(
  robotsContent.includes("Disallow: /*?*fromGoal=*"),
  "FAIL: robots.txt must disallow crawler query parameter /*?*fromGoal=*"
);
assert.ok(
  robotsContent.includes("Disallow: /*?*fromCount=*"),
  "FAIL: robots.txt must disallow crawler query parameter /*?*fromCount=*"
);
assert.ok(
  robotsContent.includes("Sitemap: https://www.arxterminal.com/sitemap.xml"),
  "FAIL: robots.txt must declare sitemap location"
);
console.log("   [OK] robots.txt contains query parameter crawl hygiene and sitemap reference");

// ---------------------------------------------------------------------------
// 4. Invariant Check: TerminalSsrShell.tsx H1 & Brand Identity
// ---------------------------------------------------------------------------
console.log("4. Verifying homepage H1 semantic hierarchy in TerminalSsrShell.tsx...");
const ssrContent = fs.readFileSync(TERMINAL_SSR_PATH, "utf-8");

// Must NOT have AAPL price inside h1
const h1Match = ssrContent.match(/<h1[^>]*>([\s\S]*?)<\/h1>/i);
assert.ok(h1Match, "FAIL: TerminalSsrShell.tsx must contain an <h1> tag");
const h1Inner = h1Match[1];
assert.strictEqual(
  h1Inner.includes("AAPL") && h1Inner.includes("$"),
  false,
  "FAIL: TerminalSsrShell <h1> must NOT hardcode ticker quote AAPL $"
);
assert.ok(
  h1Inner.includes("ARX Terminal"),
  "FAIL: TerminalSsrShell <h1> must identify ARX Terminal platform"
);
assert.strictEqual(
  ssrContent.includes('href="/screener"'),
  false,
  "FAIL: TerminalSsrShell must NOT link to legacy /screener"
);
console.log("   [OK] TerminalSsrShell renders authoritative ARX Terminal H1 and links to /radar");

// ---------------------------------------------------------------------------
// 5. Invariant Check: stock/[ticker]/page.tsx Schema & Breadcrumbs
// ---------------------------------------------------------------------------
console.log("5. Verifying stock detail schema.org and breadcrumbs in stock/[ticker]/page.tsx...");
const stockPageContent = fs.readFileSync(STOCK_TICKER_PAGE_PATH, "utf-8");

assert.strictEqual(
  stockPageContent.includes('"@type": "FinancialProduct"'),
  false,
  "FAIL: stock/[ticker]/page.tsx must NOT claim @type: 'FinancialProduct'"
);
assert.ok(
  stockPageContent.includes('"@type": "WebPage"'),
  "FAIL: stock/[ticker]/page.tsx must use @type: 'WebPage'"
);
assert.ok(
  stockPageContent.includes('"@type": "Corporation"'),
  "FAIL: stock/[ticker]/page.tsx must identify subject company as Corporation"
);
assert.strictEqual(
  stockPageContent.includes('href="/screener"'),
  false,
  "FAIL: stock/[ticker]/page.tsx must not link to legacy /screener"
);
assert.ok(
  stockPageContent.includes('https://www.arxterminal.com/radar/'),
  "FAIL: stock/[ticker]/page.tsx breadcrumbs must link to /radar/"
);
console.log("   [OK] stock/[ticker]/page.tsx uses WebPage/Corporation schema and radar breadcrumbs");

// ---------------------------------------------------------------------------
// 6. Invariant Check: stock/layout.tsx
// ---------------------------------------------------------------------------
console.log("6. Verifying stock/layout.tsx self-referential canonical...");
assert.ok(fs.existsSync(STOCK_LAYOUT_PATH), "FAIL: frontend/app/stock/layout.tsx must exist");
const stockLayoutContent = fs.readFileSync(STOCK_LAYOUT_PATH, "utf-8");
assert.ok(
  stockLayoutContent.includes("https://www.arxterminal.com/stock/"),
  "FAIL: stock/layout.tsx must emit canonical https://www.arxterminal.com/stock/"
);
console.log("   [OK] stock/layout.tsx emits self-referencing canonical for /stock/");

// ---------------------------------------------------------------------------
// 7. Invariant Check: screener/layout.tsx
// ---------------------------------------------------------------------------
console.log("7. Verifying screener/layout.tsx noindex directive...");
const screenerLayoutContent = fs.readFileSync(SCREENER_LAYOUT_PATH, "utf-8");
assert.ok(
  screenerLayoutContent.includes("noindex") || screenerLayoutContent.includes("index: false"),
  "FAIL: screener/layout.tsx must declare noindex"
);
assert.ok(
  screenerLayoutContent.includes("https://www.arxterminal.com/radar/"),
  "FAIL: screener/layout.tsx canonical must point to /radar/"
);
console.log("   [OK] screener/layout.tsx declares index: false and canonical /radar/");

console.log("\n[ALL 7 TECHNICAL SEO INVARIANT CHECKS PASSED SUCCESSFULLY]");
