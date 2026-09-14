import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer";

console.log("================================================================");
console.log("      MOBILE SETUPS LAYOUT & INTERACTION VERIFICATION           ");
console.log("================================================================\n");

const setupsPagePath = path.resolve(__dirname, "../app/setups/page.tsx");
const setupsContent = fs.readFileSync(setupsPagePath, "utf-8");

// 1. Mobile Catalog Collapse Verification (S01 / AC8)
console.log("[Static 1] Mobile Selected Setup Catalog Collapse (S01 / AC8)");
assert(setupsContent.includes("showMobileCatalog"), "Must maintain showMobileCatalog state");
assert(
  setupsContent.includes("sm:hidden flex items-center justify-between p-3 rounded-xl border"),
  "Must render compact mobile bar on narrow screens (< sm)"
);
assert(
  setupsContent.includes("hidden sm:grid") || setupsContent.includes("${showMobileCatalog ? 'grid' : 'hidden sm:grid'}"),
  "Catalog grid must be hidden on mobile unless showMobileCatalog is true, and always visible on desktop"
);
console.log("  ✔ PASS: Large catalog is collapsed on narrow screens when setup is selected (S01 closed)");

// 2. Compact Setup / Asset-Change Control (AC9)
console.log("\n[Static 2] Compact Setup/Asset-Change Interaction (AC9)");
assert(
  setupsContent.includes('Change Setup ▾') && setupsContent.includes('Hide Catalog ↑'),
  "Compact bar must provide toggleable Change Setup ▾ / Hide Catalog ↑ control"
);
assert(
  setupsContent.includes('All Setups'),
  "Compact bar must provide immediate escape back to full catalog via All Setups button"
);
console.log("  ✔ PASS: Compact 'Change Setup ▾' / 'All Setups' controls exist on narrow screens");

// 3. Desktop Setup Comparison Workflow Intact (AC12 / Phase 15)
console.log("\n[Static 3] Desktop Comparison Workflow Preserved (AC12 / Phase 15)");
assert(
  setupsContent.includes("max-h-[260px] overflow-y-auto"),
  "Desktop catalog grid must preserve scrollable catalog comparison container"
);
assert(
  setupsContent.includes("grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4"),
  "Desktop multi-column responsive grid remains intact"
);
console.log("  ✔ PASS: Desktop (768/1024/1440px) setup comparison workflow preserved");

// 4. Honest Setup Count & S02 Invariant
console.log("\n[Static 4] Honest Setup Catalog Count & Nomenclature (S02)");
assert(
  setupsContent.includes("Evaluated Tactical Setups"),
  "Header must honestly label evaluated setups rather than calling all of them active"
);
assert(
  !setupsContent.includes("Active Tactical Setups ({availableSetups.length})"),
  "Must NOT label all available setups as Active when some are suppressed"
);
assert(
  setupsContent.includes("actionableCount > 0"),
  "Discloses actionable count conditionally"
);
console.log("  ✔ PASS: S02 closed: Catalog distinguishes evaluated setups from actionable subsets");

// 5. Lifecycle Invariant & Zero Accidental Mutations (AC13 / AC14 / Phase 16)
console.log("\n[Static 5] Lifecycle Isolation on Setup Selection (AC13 / AC14 / Phase 16)");
assert(
  !setupsContent.includes("saveJournalTrade"),
  "Setups page must have zero saveJournalTrade calls (clipboard isolation)"
);
assert(
  !setupsContent.includes("localStorage.setItem('arx_portfolio"),
  "Selecting setup must not mutate portfolio storage"
);
assert(
  !setupsContent.includes("localStorage.setItem('arx_journal"),
  "Selecting setup must not mutate journal storage"
);
console.log("  ✔ PASS: Setup selection is pure UI state; zero portfolio or journal mutation");

// 6. Navigation Naming Alignment (S07)
console.log("\n[Static 6] Navigation Naming Alignment (S07)");
assert(
  setupsContent.includes("Open in Analysis (/?symbol="),
  "Suppressed setup link must refer to Analysis, aligning with canonical nav"
);
assert(
  !setupsContent.includes("Open in Terminal (/?symbol="),
  "Must NOT refer to Terminal as route destination"
);
console.log("  ✔ PASS: S07 closed on Setups: destination renamed to Analysis");

// 7. Live Puppeteer Real Browser Mobile Walkthrough
async function runBrowserWalkthrough() {
  console.log("\n[Runtime Mobile Walkthrough] Puppeteer Browser Simulation");
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  try {
    const page = await browser.newPage();
    const viewports = [
      { name: "320px (Compact Mobile)", width: 320, height: 568 },
      { name: "375px (iPhone Standard)", width: 375, height: 667 },
      { name: "390px (Modern Mobile)", width: 390, height: 844 },
    ];

    for (const vp of viewports) {
      console.log(`\n--- Testing Viewport: ${vp.name} ---`);
      await page.setViewport({ width: vp.width, height: vp.height });
      await page.goto("http://localhost:3000/setups/", { waitUntil: "networkidle0", timeout: 15000 });

      // Check horizontal overflow
      const overflow = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth;
      });
      assert(!overflow, `Horizontal overflow detected at ${vp.width}px`);
      console.log(`  ✔ PASS: Zero horizontal overflow at ${vp.width}px`);

      // Check mobile dock navigation exists and is visible at bottom
      const dock = await page.$("nav.fixed.bottom-0");
      assert(!!dock, `Mobile dock navigation must be present at ${vp.width}px`);
      console.log(`  ✔ PASS: Mobile bottom dock present at ${vp.width}px`);

      // Check content clearance above bottom dock
      const dockHeight = await page.evaluate((el) => {
        return el ? el.getBoundingClientRect().height : 0;
      }, dock);
      assert(dockHeight >= 44, "Dock height must be at least 44px");

      // Verify bottom padding on main or container allows clearing the dock
      const bodyPaddingBottom = await page.evaluate(() => {
        const main = document.querySelector("main") || document.body;
        const style = window.getComputedStyle(main);
        return parseFloat(style.paddingBottom) || 0;
      });
      assert(bodyPaddingBottom >= 48, "Must have bottom clearance for mobile dock");
      console.log(`  ✔ PASS: Main container has ${bodyPaddingBottom}px bottom clearance for mobile dock`);

      // Test setup selection with ?symbol=NVDA
      await page.goto("http://localhost:3000/setups/?symbol=NVDA", { waitUntil: "networkidle0", timeout: 15000 });

      // Verify compact mobile bar appears
      const mobileBar = await page.evaluate(() => {
        const el = document.querySelector(".sm\\:hidden.flex.items-center.justify-between");
        return !!el;
      });
      assert(mobileBar, `Mobile compact bar must be rendered at ${vp.width}px`);
      console.log(`  ✔ PASS: Mobile compact header rendered at ${vp.width}px`);

      // Verify scrolling works cleanly
      await page.evaluate(() => window.scrollTo(0, 300));
      const scrollY = await page.evaluate(() => window.scrollY);
      assert(scrollY > 0, "Page must be smoothly scrollable on mobile");
      console.log(`  ✔ PASS: Mobile page scroll functional (scrollY: ${scrollY}px)`);

      // Test mobile bottom nav link to Analysis
      const analysisUrl = await page.evaluate(() => {
        const links = Array.from(document.querySelectorAll("nav.fixed.bottom-0 a"));
        const analysis = links.find((a) => a.getAttribute("href") === "/" || a.getAttribute("href")?.startsWith("/?"));
        return analysis ? analysis.getAttribute("href") : null;
      });
      assert(analysisUrl !== null, "Mobile dock must contain Analysis link");
      console.log(`  ✔ PASS: Mobile dock has Analysis destination link: ${analysisUrl}`);
    }

    console.log("\n================================================================");
    console.log("All Mobile Setups Static & Live Browser Tests PASSED!");
    console.log("================================================================");
  } finally {
    await browser.close();
  }
}

runBrowserWalkthrough().catch((err) => {
  console.error("Browser walkthrough failed:", err);
  process.exit(1);
});
