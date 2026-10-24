import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendOut = path.resolve(__dirname, "..", "out");

console.log("=== ARX Production Build Artifact Verification ===\n");

// 1. Homepage
const homeHtml = fs.readFileSync(path.join(frontendOut, "index.html"), "utf-8");
const h1Matches = [...homeHtml.matchAll(/<h1[^>]*>([\s\S]*?)<\/h1>/gi)].map(m => m[1].trim());
console.log("1. Homepage H1 Tags:", h1Matches);
const homeCanon = homeHtml.match(/<link[^>]*rel="canonical"[^>]*href="([^"]+)"/);
console.log("   Homepage Canonical:", homeCanon ? homeCanon[1] : "NONE");

// 2. Stock AAPL
const aaplHtml = fs.readFileSync(path.join(frontendOut, "stock", "aapl", "index.html"), "utf-8");
const aaplCanon = aaplHtml.match(/<link[^>]*rel="canonical"[^>]*href="([^"]+)"/);
const aaplTitle = aaplHtml.match(/<title[^>]*>([\s\S]*?)<\/title>/);
console.log("\n2. Stock /stock/aapl/:");
console.log("   Canonical:", aaplCanon ? aaplCanon[1] : "NONE");
console.log("   Title:", aaplTitle ? aaplTitle[1] : "NONE");
console.log("   FinancialProduct schema in HTML?:", aaplHtml.includes("FinancialProduct"));
console.log("   WebPage schema in HTML?:", aaplHtml.includes("WebPage"));
console.log("   Corporation about schema in HTML?:", aaplHtml.includes("Corporation"));
console.log("   Radar breadcrumb in HTML?:", aaplHtml.includes("https://www.arxterminal.com/radar/"));

// 3. Generic Stock Directory
const stockHtml = fs.readFileSync(path.join(frontendOut, "stock", "index.html"), "utf-8");
const stockCanon = stockHtml.match(/<link[^>]*rel="canonical"[^>]*href="([^"]+)"/);
const stockTitle = stockHtml.match(/<title[^>]*>([\s\S]*?)<\/title>/);
console.log("\n3. Generic Directory /stock/:");
console.log("   Canonical:", stockCanon ? stockCanon[1] : "NONE");
console.log("   Title:", stockTitle ? stockTitle[1] : "NONE");

// 4. Screener
const screenerHtml = fs.readFileSync(path.join(frontendOut, "screener", "index.html"), "utf-8");
const screenerCanon = screenerHtml.match(/<link[^>]*rel="canonical"[^>]*href="([^"]+)"/);
const screenerRobots = screenerHtml.match(/<meta[^>]*name="robots"[^>]*content="([^"]+)"/);
console.log("\n4. Redirect Stub /screener/:");
console.log("   Canonical:", screenerCanon ? screenerCanon[1] : "NONE");
console.log("   Robots meta:", screenerRobots ? screenerRobots[1] : "NONE");

// 5. Out _redirects
const redirects = fs.readFileSync(path.join(frontendOut, "_redirects"), "utf-8");
console.log("\n5. Out _redirects Rules:\n" + redirects.trim());

// 6. Out sitemap.xml
const sitemapXml = fs.readFileSync(path.join(frontendOut, "sitemap.xml"), "utf-8");
const locs = [...sitemapXml.matchAll(/<loc>(.*?)<\/loc>/g)].map(m => m[1]);
console.log("\n6. Out sitemap.xml Summary:");
console.log("   Total Submitted URLs:", locs.length);
console.log("   Contains /screener?:", locs.some(u => u.includes("/screener")));
console.log("   Contains /radar/:", locs.includes("https://www.arxterminal.com/radar/"));
console.log("   Stock URLs count:", locs.filter(u => u.includes("/stock/")).length);

// 7. Check Other Sampled Stocks (nvda, tsla, msft, cprx)
for (const sym of ["nvda", "tsla", "msft", "cprx"]) {
  const stockPath = path.join(frontendOut, "stock", sym, "index.html");
  if (fs.existsSync(stockPath)) {
    const content = fs.readFileSync(stockPath, "utf-8");
    const c = content.match(/<link[^>]*rel="canonical"[^>]*href="([^"]+)"/);
    console.log(`   Sample /stock/${sym}/ Canonical:`, c ? c[1] : "NONE");
  }
}

console.log("\n=== Verification Finished Successfully ===");
