import fs from "fs";
import path from "path";

let failures = 0;
function assert(condition, message) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    failures++;
  } else {
    console.log(`PASS: ${message}`);
  }
}

console.log("=================================================");
console.log("ARX TERMINAL — SEO PHASE 1 COMPREHENSIVE AUDIT");
console.log("=================================================");

// 1. SITEMAP AUDIT
console.log("\n[1. SITEMAP VERIFICATION]");
assert(fs.existsSync("out/sitemap.xml"), "out/sitemap.xml must exist");
assert(!fs.existsSync("public/sitemap.xml"), "public/sitemap.xml must be absent to eliminate dual authority");

const sitemapContent = fs.readFileSync("out/sitemap.xml", "utf8");
const sitemapUrls = [...sitemapContent.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1]);

assert(sitemapUrls.length === 99, `Total sitemap URLs must be exactly 99 (got ${sitemapUrls.length})`);
assert(new Set(sitemapUrls).size === 99, `Unique sitemap URLs must be exactly 99 (got ${new Set(sitemapUrls).size})`);

const privatePatterns = [
  "/api/", "/action-center", "/adoption-center", "/audit-explorer",
  "/autonomous-governance", "/coaching-intelligence", "/committee-intelligence",
  "/committee-network", "/decision-explorer", "/decision-inbox", "/dissent-explorer",
  "/evaluation", "/executive-sandbox", "/executive-workspace", "/governance-center",
  "/graph-explorer", "/intelligence-center", "/learning-intelligence", "/oos",
  "/optimization-intelligence", "/release-dashboard", "/resilience-intelligence",
  "/risks-and-groupthink", "/simulation-intelligence", "/strategy-laboratory",
  "/strategy-orchestrator", "/workspace", "/design-system-preview", "/me/",
  "/cockpit", "/future", "/household", "/progress", "/today", "/journal",
  "/performance", "/workbench/", "/research"
];

let invalidUrls = 0;
for (const u of sitemapUrls) {
  if (!u.startsWith("https://www.arxterminal.com/")) {
    console.error(`Invalid protocol/domain: ${u}`);
    invalidUrls++;
  }
  if (!u.endsWith("/")) {
    console.error(`Missing trailing slash: ${u}`);
    invalidUrls++;
  }
  if (u.includes("?") || u.includes("#")) {
    console.error(`URL contains query or fragment: ${u}`);
    invalidUrls++;
  }
  if (privatePatterns.some(p => u.includes(p))) {
    console.error(`Private/noindex URL in sitemap: ${u}`);
    invalidUrls++;
  }
  if (u === "https://www.arxterminal.com/committee/" || u === "https://www.arxterminal.com/politician/" || u === "https://www.arxterminal.com/strategy/") {
    console.error(`Redirect stub in sitemap: ${u}`);
    invalidUrls++;
  }
  if (u.includes("404")) {
    console.error(`404 route in sitemap: ${u}`);
    invalidUrls++;
  }
}
assert(invalidUrls === 0, `INVALID_SITEMAP_URLS must be 0 (got ${invalidUrls})`);

// 1B. INDEPENDENT AUTHORITY RECONCILIATION
const staticHubs = ["/", "/radar/", "/setups/", "/smart-money/", "/smart-money/late-filers/", "/screener/", "/portfolio/", "/guide/", "/compare/", "/glossary/", "/vs/"];
const masterCatalogContent = fs.readFileSync("lib/masterCatalog.ts", "utf8");
const masterKeys = [...masterCatalogContent.matchAll(/["']?([A-Za-z0-9_-]+)["']?:\s*\{[\r\n\s]+symbol:/g)].map(m => m[1].toLowerCase());
const constantsContent = fs.readFileSync("lib/constants.ts", "utf8");
const watchlistSymbols = [...constantsContent.matchAll(/symbol:\s*["']([^"']+)["']/g)].map(m => m[1].toLowerCase());
const uniqueTickers = Array.from(new Set([...masterKeys, ...watchlistSymbols])).sort();
const stockRoutes = uniqueTickers.map(t => `/stock/${t}/`);
const strategyRoutes = ["minervini-vcp", "magic-formula", "peter-lynch-garp", "short-squeeze", "rule-breakers"].map(s => `/strategy/${s}/`);
const committeeRoutes = ["armed-services", "energy-commerce", "intelligence", "foreign-affairs", "financial-services"].map(c => `/committee/${c}/`);
const politicianRoutes = ["nancy-pelosi", "dan-crenshaw", "tommy-tuberville", "michael-mccaul", "mark-green", "ro-khanna", "josh-gottheimer", "sheldon-whitehouse"].map(p => `/politician/${p}/`);
const glossaryContent = fs.readFileSync("lib/glossaryCatalog.ts", "utf8");
const glossaryRoutes = [...glossaryContent.matchAll(/slug:\s*["']([^"']+)["']/g)].map(m => `/glossary/${m[1].toLowerCase()}/`);
const competitorContent = fs.readFileSync("lib/competitorCatalog.ts", "utf8");
const competitorRoutes = [...competitorContent.matchAll(/slug:\s*["']([^"']+)["']/g)].map(m => `/vs/${m[1].toLowerCase()}/`);
const compareRoutes = ["nvo-vs-lly", "spy-vs-qqq", "nvda-vs-aapl", "tsla-vs-pltr", "amd-vs-nvda", "msft-vs-aapl", "cprx-vs-powi"].map(p => `/compare/${p}/`);

const expectedPaths = [...staticHubs, ...stockRoutes, ...strategyRoutes, ...committeeRoutes, ...politicianRoutes, ...glossaryRoutes, ...competitorRoutes, ...compareRoutes];
const expectedUrls = new Set(expectedPaths.map(p => `https://www.arxterminal.com${p}`));
const sitemapUrlSet = new Set(sitemapUrls);

const missingFromSitemap = [...expectedUrls].filter(u => !sitemapUrlSet.has(u));
const extraInSitemap = [...sitemapUrlSet].filter(u => !expectedUrls.has(u));

assert(expectedUrls.size === 99, `EXPECTED_INDEXABLE_URLS must be 99 (got ${expectedUrls.size})`);
assert(missingFromSitemap.length === 0, `MISSING_SITEMAP_URLS must be 0 (got ${missingFromSitemap.length}: ${missingFromSitemap.join(", ")})`);
assert(extraInSitemap.length === 0, `EXTRA_SITEMAP_URLS must be 0 (got ${extraInSitemap.length}: ${extraInSitemap.join(", ")})`);

// 2. ROBOTS.TXT AUDIT
console.log("\n[2. ROBOTS.TXT POLICY VERIFICATION]");
const robots = fs.readFileSync("out/robots.txt", "utf8");
assert(robots.includes("User-agent: *\nAllow: /"), "General search crawlers must be allowed root access");
assert(robots.includes("User-agent: OAI-SearchBot"), "OAI-SearchBot block must be explicitly declared");
assert(robots.includes("User-agent: GPTBot\nDisallow: /"), "GPTBot must be globally disallowed");

// Verify OAI-SearchBot has private disallows and Allow: /
const oaiSection = robots.split("User-agent: OAI-SearchBot")[1]?.split("User-agent:")[0] || "";
assert(oaiSection.includes("Disallow: /workbench/"), "OAI-SearchBot must disallow /workbench/");
assert(oaiSection.includes("Disallow: /research"), "OAI-SearchBot must disallow /research");
assert(oaiSection.includes("Disallow: /me"), "OAI-SearchBot must disallow /me");
assert(oaiSection.includes("Disallow: /cockpit"), "OAI-SearchBot must disallow /cockpit");
assert(oaiSection.includes("Allow: /"), "OAI-SearchBot must allow public content");

const sitemapDirectives = [...robots.matchAll(/Sitemap:\s*https:\/\/www\.arxterminal\.com\/sitemap\.xml/g)];
assert(sitemapDirectives.length === 1, `Sitemap directive must exist exactly once in robots.txt (found ${sitemapDirectives.length})`);

// 3. STOCK DETAIL PRICE SEMANTICS
console.log("\n[3. STOCK DETAIL PRICE SEMANTICS AUDIT]");
const stockTickers = fs.readdirSync("out/stock");
assert(stockTickers.length === 49, `Stock routes count must be 49 (got ${stockTickers.length})`);

let baselineRefPresent = 0;
let noRefAvailable = 0;
let falseLiveClaims = 0;

const problematicLivePhrases = [
  "live price", "live market price", "current market price",
  "realtime price", "real-time price"
];

for (const ticker of stockTickers) {
  const html = fs.readFileSync(`out/stock/${ticker}/index.html`, "utf8");
  if (html.includes("Baseline Reference Price")) {
    baselineRefPresent++;
  } else {
    noRefAvailable++;
  }

  const lowerHtml = html.toLowerCase();
  for (const phrase of problematicLivePhrases) {
    if (lowerHtml.includes(phrase)) {
      console.error(`FALSE LIVE CLAIM in stock/${ticker}: "${phrase}"`);
      falseLiveClaims++;
    }
  }
}

assert(baselineRefPresent === 44, `BASELINE_REFERENCE_PRICE_PRESENT must be 44 (got ${baselineRefPresent})`);
assert(noRefAvailable === 5, `NO_REFERENCE_AVAILABLE must be 5 (got ${noRefAvailable})`);
assert(falseLiveClaims === 0, `FALSE_LIVE_PRICE_CLAIMS must be 0 (got ${falseLiveClaims})`);

// 4. SOCIAL METADATA FULL SCAN
console.log("\n[4. SOCIAL METADATA FULL SCAN]");
const targetDirs = [
  { prefix: "stock", dirs: stockTickers },
  { prefix: "strategy", dirs: fs.readdirSync("out/strategy").filter(d => !d.endsWith(".html")) },
  { prefix: "committee", dirs: fs.readdirSync("out/committee").filter(d => !d.endsWith(".html")) },
  { prefix: "politician", dirs: fs.readdirSync("out/politician").filter(d => !d.endsWith(".html")) },
  { prefix: "compare", dirs: fs.readdirSync("out/compare").filter(d => !d.endsWith(".html")) },
];

let rootMetadataLeaks = 0;
let pelosiLeaksOutsideRoute = 0;
let missingRouteSpecificTwitter = 0;

for (const group of targetDirs) {
  for (const sub of group.dirs) {
    const filePath = `out/${group.prefix}/${sub}/index.html`;
    if (!fs.existsSync(filePath)) continue;
    const html = fs.readFileSync(filePath, "utf8");

    // twitter:card
    const hasTwitterCard = html.includes('name="twitter:card" content="summary_large_image"');
    if (!hasTwitterCard) {
      console.error(`Missing twitter:card in ${filePath}`);
      missingRouteSpecificTwitter++;
    }

    // twitter:title
    const titleMatch = html.match(/name="twitter:title" content="([^"]+)"/);
    if (!titleMatch || !titleMatch[1].trim()) {
      console.error(`Missing twitter:title in ${filePath}`);
      missingRouteSpecificTwitter++;
    } else {
      const t = titleMatch[1];
      if (t === "ARX Terminal" || t === "ARX Terminal • Autonomous Quantitative Trading Terminal") {
        console.error(`Root generic twitter:title leak in ${filePath}`);
        rootMetadataLeaks++;
      }
      if (t.toLowerCase().includes("pelosi") && sub !== "nancy-pelosi") {
        console.error(`Pelosi metadata leak in ${filePath}: "${t}"`);
        pelosiLeaksOutsideRoute++;
      }
    }

    // twitter:description
    const descMatch = html.match(/name="twitter:description" content="([^"]+)"/);
    if (!descMatch || !descMatch[1].trim()) {
      console.error(`Missing twitter:description in ${filePath}`);
      missingRouteSpecificTwitter++;
    }

    // og:title & og:description
    const ogTitleMatch = html.match(/property="og:title" content="([^"]+)"/);
    if (!ogTitleMatch || !ogTitleMatch[1].trim()) {
      console.error(`Missing og:title in ${filePath}`);
      missingRouteSpecificTwitter++;
    }
  }
}

assert(rootMetadataLeaks === 0, `ROOT_METADATA_LEAKS must be 0 (got ${rootMetadataLeaks})`);
assert(pelosiLeaksOutsideRoute === 0, `PELOSI_METADATA_LEAKS_OUTSIDE_RELEVANT_ROUTE must be 0 (got ${pelosiLeaksOutsideRoute})`);
assert(missingRouteSpecificTwitter === 0, `MISSING_ROUTE_SPECIFIC_TWITTER_METADATA must be 0 (got ${missingRouteSpecificTwitter})`);

// 5. CANONICAL & DOUBLE-BRAND AUDIT
console.log("\n[5. CANONICAL FULL-OUTPUT SCAN]");
let publicRoutesWithoutCanonical = 0;
let mismatchedCanonicals = 0;
let doubleBrandedTitles = 0;

for (const sitemapUrl of sitemapUrls) {
  const relPath = sitemapUrl.replace("https://www.arxterminal.com/", "");
  const htmlPath = relPath === "" ? "out/index.html" : `out/${relPath}index.html`;

  if (!fs.existsSync(htmlPath)) {
    console.error(`Missing HTML file for indexable URL: ${htmlPath}`);
    publicRoutesWithoutCanonical++;
    continue;
  }

  const html = fs.readFileSync(htmlPath, "utf8");
  const canonicalMatch = html.match(/<link rel="canonical" href="([^"]+)"/);
  if (!canonicalMatch) {
    console.error(`Public route missing canonical: ${htmlPath}`);
    publicRoutesWithoutCanonical++;
  } else {
    const canonicalHref = canonicalMatch[1];
    if (canonicalHref !== sitemapUrl) {
      console.error(`Mismatched canonical in ${htmlPath}: expected ${sitemapUrl} got ${canonicalHref}`);
      mismatchedCanonicals++;
    }
  }

  const titleMatch = html.match(/<title>([^<]+)<\/title>/);
  if (titleMatch) {
    const t = titleMatch[1];
    if (t.includes("ARX Terminal | ARX Terminal") || t.includes("ARX Terminal - ARX Terminal")) {
      console.error(`Double-branded title in ${htmlPath}: "${t}"`);
      doubleBrandedTitles++;
    }
  }
}

assert(publicRoutesWithoutCanonical === 0, `PUBLIC_ROUTES_WITHOUT_CANONICAL must be 0 (got ${publicRoutesWithoutCanonical})`);
assert(mismatchedCanonicals === 0, `MISMATCHED_CANONICALS must be 0 (got ${mismatchedCanonicals})`);
assert(doubleBrandedTitles === 0, `DOUBLE_BRANDED_TITLES must be 0 (got ${doubleBrandedTitles})`);

// 6. NOINDEX EXCLUSION AUDIT
console.log("\n[6. NOINDEX EXCLUSION AUDIT]");
const perfHtml = fs.readFileSync("out/performance/index.html", "utf8");
assert(perfHtml.includes('name="robots" content="noindex'), "/performance/ must have robots noindex meta");
assert(!sitemapUrls.includes("https://www.arxterminal.com/performance/"), "/performance/ must NOT be in sitemap");

const journalHtml = fs.readFileSync("out/journal/index.html", "utf8");
assert(journalHtml.includes('name="robots" content="noindex'), "/journal/ must have robots noindex meta");
assert(!sitemapUrls.includes("https://www.arxterminal.com/journal/"), "/journal/ must NOT be in sitemap");

// 7. REDIRECT TARGETS AUDIT
console.log("\n[7. REDIRECT TARGETS AUDIT]");
const redirectsContent = fs.readFileSync("public/_redirects", "utf8");
const redirectLines = redirectsContent
  .split("\n")
  .map(l => l.trim())
  .filter(l => l && !l.startsWith("#") && l.includes("301!") && l.startsWith("/"));

let brokenRedirects = 0;
let redirectLoops = 0;

for (const line of redirectLines) {
  const parts = line.split(/\s+/);
  if (parts.length >= 3) {
    const source = parts[0];
    const target = parts[1];
    // Check loop
    if (source === target || (source.replace("/*", "") === target.replace("/*", ""))) {
      console.error(`Redirect loop detected: ${source} -> ${target}`);
      redirectLoops++;
    }

    // Verify target exists in out
    const targetRel = target.replace(/^\//, "");
    const targetFile = targetRel === "" ? "out/index.html" : `out/${targetRel}/index.html`;
    if (!fs.existsSync(targetFile)) {
      console.error(`Broken redirect target: ${source} -> ${target} (file ${targetFile} not found)`);
      brokenRedirects++;
    } else {
      console.log(`PASS: Valid redirect: ${source} -> ${target} (resolves to ${targetFile})`);
    }
  }
}

assert(brokenRedirects === 0, `BROKEN_REDIRECTS must be 0 (got ${brokenRedirects})`);
assert(redirectLoops === 0, `REDIRECT_LOOPS must be 0 (got ${redirectLoops})`);

// 8. JSON-LD STRUCTURED DATA AUDIT
console.log("\n[8. JSON-LD STRUCTURED DATA AUDIT]");
function auditJsonLd(filePath) {
  const html = fs.readFileSync(filePath, "utf8");
  const jsonLdMatches = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  assert(jsonLdMatches.length > 0, `${filePath} must contain application/ld+json script`);

  let types = [];
  for (const m of jsonLdMatches) {
    try {
      const parsed = JSON.parse(m[1]);
      if (Array.isArray(parsed)) {
        types.push(...parsed.map(item => item["@type"]));
      } else {
        types.push(parsed["@type"]);
      }
    } catch (err) {
      console.error(`JSON-LD parse error in ${filePath}:`, err.message);
      failures++;
    }
  }
  return types;
}

const radarTypes = auditJsonLd("out/radar/index.html");
assert(radarTypes.includes("WebApplication"), "/radar/ must declare WebApplication schema");
assert(radarTypes.includes("BreadcrumbList"), "/radar/ must declare BreadcrumbList schema");

const setupsTypes = auditJsonLd("out/setups/index.html");
assert(setupsTypes.includes("WebApplication"), "/setups/ must declare WebApplication schema");
assert(setupsTypes.includes("BreadcrumbList"), "/setups/ must declare BreadcrumbList schema");

// FINAL SUMMARY & EXIT CODE
console.log("\n=================================================");
if (failures === 0) {
  console.log("ALL ARX SEO PHASE 1 AUDIT INVARIANTS PASSED! (0 failures)");
  console.log("=================================================");
  process.exit(0);
} else {
  console.error(`AUDIT FAILED with ${failures} assertion failure(s)!`);
  console.log("=================================================");
  process.exit(1);
}
