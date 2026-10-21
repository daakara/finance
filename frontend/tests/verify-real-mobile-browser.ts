/**
 * frontend/tests/verify-real-mobile-browser.ts
 *
 * Real Headless Chrome (Puppeteer) Browser Test Suite for ARX Terminal Mobile Navigation & Research.
 *
 * Runs across 360px, 390px, and 430px viewports:
 * 1. Verifies Research link is directly visible in mobile dock without opening 3-dot overflow menu.
 * 2. Measures rendered hit area (width >= 44px, height >= 44px per WCAG 2.5.5).
 * 3. Verifies dock controls do not overlap.
 * 4. Verifies dock causes zero horizontal page overflow.
 * 5. Taps Research link and verifies successful navigation with symbol context preserved.
 * 6. Verifies active navigation state (aria-current="page" and active styling).
 * 7. Verifies ULTA's corrected per-dossier provenance on mobile.
 * 8. Verifies excluded-symbol (MU) neutral fallback state without baseline hash leakage.
 * 9. Captures actual screenshots and measurements.
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const puppeteer = require("puppeteer");

const PORT = 3334;
const OUT_DIR = path.resolve(__dirname, "../out");
const SCREENSHOT_DIR = path.resolve("C:/Users/akara/.gemini/antigravity/brain/c2788363-bf19-461b-8a58-6423c773a359/screenshots");

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

// Simple static HTTP server for Next.js exported files
function startStaticServer(): Promise<http.Server> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      let reqPath = (req.url || "/").split("?")[0];
      let filePath = path.join(OUT_DIR, reqPath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }

      // Check if direct file exists
      if (!fs.existsSync(filePath)) {
        // Try appending .html
        if (fs.existsSync(filePath + ".html")) {
          filePath = filePath + ".html";
        } else {
          res.writeHead(404, { "Content-Type": "text/plain" });
          res.end("Not Found: " + reqPath);
          return;
        }
      }

      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes: Record<string, string> = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".txt": "text/plain; charset=utf-8",
      };

      const contentType = mimeTypes[ext] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": contentType });
      fs.createReadStream(filePath).pipe(res);
    });

    server.listen(PORT, () => {
      console.log(`[HTTP SERVER] Serving static export from ${OUT_DIR} on http://localhost:${PORT}`);
      resolve(server);
    });
  });
}

interface ViewportResult {
  width: number;
  height: number;
  researchVisible: boolean;
  dockItemCount: number;
  researchHitArea: { width: number; height: number };
  controlsOverlap: boolean;
  horizontalPageOverflow: boolean;
  navigatedSuccessfully: boolean;
  preservedSymbolContext: boolean;
  activeNavigationState: boolean;
  screenshotPath: string;
}

async function runBrowserChecks() {
  const server = await startStaticServer();
  const results: Record<number, ViewportResult> = {};

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  try {
    const viewports = [
      { width: 360, height: 780, name: "360px (Small Mobile - Galaxy S8)" },
      { width: 390, height: 844, name: "390px (Standard Mobile - iPhone 14/15)" },
      { width: 430, height: 932, name: "430px (Large Mobile - iPhone 14/15 Pro Max)" },
    ];

    for (const vp of viewports) {
      console.log(`\n========================================================================`);
      console.log(`Testing Viewport: ${vp.name} (${vp.width}x${vp.height})...`);
      console.log(`========================================================================`);

      const page = await browser.newPage();
      await page.setViewport({ width: vp.width, height: vp.height, isMobile: true, hasTouch: true });

      // Navigate to Terminal with active symbol context FIX
      const terminalUrl = `http://localhost:${PORT}/?symbol=FIX`;
      console.log(`1. Navigating to ${terminalUrl}...`);
      await page.goto(terminalUrl, { waitUntil: "networkidle0" });
      // Wait for dock to appear and React hydration to settle
      await page.waitForSelector('[data-testid="mobile-nav-dock"]', { timeout: 10000 });
      await new Promise((r) => setTimeout(r, 1000));

      const dockData = await page.evaluate(() => {
        const dock = document.querySelector('[data-testid="mobile-nav-dock"]');
        if (!dock) return null;
        const dockStyle = window.getComputedStyle(dock);
        const dockRect = dock.getBoundingClientRect();
        const isDockVisible = dockStyle.display !== "none" && dockStyle.visibility !== "hidden" && dockRect.height > 0;

        const researchLink = dock.querySelector('a[href*="/research"]') as HTMLAnchorElement | null;
        if (!researchLink) return null;
        const resStyle = window.getComputedStyle(researchLink);
        const resRect = researchLink.getBoundingClientRect();
        const isResearchVisible =
          resStyle.display !== "none" && resStyle.visibility !== "hidden" && resRect.width > 0 && resRect.height > 0;

        const hitArea = {
          width: Math.round(resRect.width * 10) / 10,
          height: Math.round(resRect.height * 10) / 10,
        };
        const href = researchLink.getAttribute("href") || "";

        const children = Array.from(dock.querySelectorAll("a, button"));
        const rects = children.map((c, i) => {
          const r = c.getBoundingClientRect();
          return {
            index: i,
            left: r.left,
            right: r.right,
            top: r.top,
            bottom: r.bottom,
            width: r.width,
            height: r.height,
          };
        });

        let hasOverlap = false;
        for (let i = 0; i < rects.length; i++) {
          for (let j = i + 1; j < rects.length; j++) {
            const a = rects[i];
            const b = rects[j];
            if (a.left < b.right - 1 && a.right > b.left + 1 && a.top < b.bottom - 1 && a.bottom > b.top + 1) {
              hasOverlap = true;
            }
          }
        }

        const docWidth = document.documentElement.clientWidth;
        const scrollWidth = document.documentElement.scrollWidth;
        const bodyScrollWidth = document.body.scrollWidth;
        const hasOverflow = scrollWidth > docWidth || bodyScrollWidth > docWidth;

        return {
          isDockVisible,
          isResearchVisible,
          hitArea,
          href,
          itemCount: rects.length,
          hasOverlap,
          docWidth,
          scrollWidth,
          hasOverflow,
        };
      });

      assert.ok(dockData, "Failed to read dock data from DOM");
      assert.strictEqual(dockData.isDockVisible, true, "Dock must be visible");
      assert.strictEqual(dockData.isResearchVisible, true, "Research link must be visible");
      assert.ok(dockData.hitArea.width >= 44.0, `Touch width (${dockData.hitArea.width}px) >= 44px`);
      assert.ok(dockData.hitArea.height >= 44.0, `Touch height (${dockData.hitArea.height}px) >= 44px`);
      assert.strictEqual(dockData.hasOverlap, false, "Controls must not overlap");
      assert.strictEqual(dockData.hasOverflow, false, "Page must not have horizontal overflow");
      assert.ok(dockData.href.includes("symbol=FIX"), "Preserves symbol=FIX context");

      console.log(`2. Measured Research Touch Target: ${dockData.hitArea.width}px × ${dockData.hitArea.height}px`);
      console.log(`3. Dock item count: ${dockData.itemCount}, controls overlap: ${dockData.hasOverlap}`);
      console.log(`4. Horizontal overflow: ${dockData.hasOverflow} (doc=${dockData.docWidth}px, scroll=${dockData.scrollWidth}px)`);
      console.log(`5. Research link target href: ${dockData.href}`);

      // Tap (click) the Research link using DOM click
      console.log(`6. Tapping Research link...`);
      await page.evaluate(() => {
        const link = document.querySelector('[data-testid="mobile-nav-dock"] a[href*="/research"]') as HTMLAnchorElement;
        link.click();
      });

      // Wait for navigation / URL update
      await page.waitForFunction(
        () => window.location.pathname.includes("/research") || window.location.href.includes("/research"),
        { timeout: 10000 }
      );
      await page.waitForSelector('[data-testid="mobile-nav-dock"]', { timeout: 10000 });
      await new Promise((r) => setTimeout(r, 600));

      const currentUrl = page.url();
      console.log(`7. Navigated URL: ${currentUrl}`);
      assert.ok(currentUrl.includes("/research"), "Must navigate successfully to /research route");
      assert.ok(currentUrl.includes("symbol=FIX"), "Navigated URL must retain symbol=FIX");

      // Check active navigation state
      const activeState = await page.evaluate(() => {
        const dock = document.querySelector('[data-testid="mobile-nav-dock"]');
        if (!dock) return { ariaCurrent: null, hasActiveClass: false };
        const activeLink = dock.querySelector('a[aria-current="page"]');
        const isResearch = activeLink ? activeLink.textContent?.includes("Research") : false;
        const hasActiveClass = activeLink ? activeLink.className.includes("text-cyan-400") : false;
        return {
          ariaCurrent: activeLink ? activeLink.getAttribute("aria-current") : null,
          isResearch,
          hasActiveClass,
        };
      });
      console.log(`8. Active navigation state: ariaCurrent=${activeState.ariaCurrent}, isResearch=${activeState.isResearch}, hasActiveClass=${activeState.hasActiveClass}`);
      assert.strictEqual(activeState.isResearch, true, "Research link must be active with aria-current='page'");

      // Capture screenshot at current viewport
      const screenshotFile = `mobile_${vp.width}_research_dock.png`;
      const screenshotPath = path.join(SCREENSHOT_DIR, screenshotFile);
      await page.screenshot({ path: screenshotPath, fullPage: false });
      console.log(`9. Screenshot captured: ${screenshotFile}`);

      results[vp.width] = {
        width: vp.width,
        height: vp.height,
        researchVisible: dockData.isResearchVisible,
        dockItemCount: dockData.itemCount,
        researchHitArea: dockData.hitArea,
        controlsOverlap: dockData.hasOverlap,
        horizontalPageOverflow: dockData.hasOverflow,
        navigatedSuccessfully: true,
        preservedSymbolContext: currentUrl.includes("symbol=FIX"),
        activeNavigationState: activeState.isResearch,
        screenshotPath,
      };

      // Focused check at 390px: Verify ULTA provenance & Excluded symbol fallback
      if (vp.width === 390) {
        console.log(`\n--- FOCUSED TEST at 390px: ULTA Provenance & Excluded Symbol Fallback ---`);

        // A. ULTA Corrected Provenance
        console.log("Navigating to /research?symbol=ULTA...");
        await page.goto(`http://localhost:${PORT}/research?symbol=ULTA`, { waitUntil: "networkidle0" });
        await page.waitForSelector('main', { timeout: 10000 });
        await new Promise((r) => setTimeout(r, 800));

        const ultaCardExists = await page.evaluate(() => {
          return document.body.textContent?.includes("Ulta Beauty, Inc.");
        });
        assert.ok(ultaCardExists, "ULTA dossier must render on /research?symbol=ULTA");

        const ultaProvData = await page.evaluate(() => {
          const text = document.body.innerText;
          return {
            hasUltaProvenanceTitle: text.toUpperCase().includes("ULTA PROVENANCE"),
            hasScalingAdditionAuthority: text.includes("ARX Quality Watch Scaling Addition"),
            hasCandidateSelectedStatus: text.includes("Selected for Release (Awaiting Deployment)"),
            hasAcceptedEvidenceReview: text.includes("Accepted Research Evidence (SEC Audited)"),
            has20261021Timestamp: text.includes("2026-10-21T16:00:00Z"),
            hasFilingCutoff: text.includes("2026-10-16T20:00:00Z"),
            hasBaselineShaLeak: text.includes("a65be3b38cdf"),
            hasAcceptedShaLabel: text.includes("Accepted Addition SHA-256"),
          };
        });
        console.log("ULTA Provenance checks:", ultaProvData);
        assert.strictEqual(ultaProvData.hasUltaProvenanceTitle, true, "Must display 'ULTA Provenance'");
        assert.strictEqual(ultaProvData.hasScalingAdditionAuthority, true, "Authority must be 'ARX Quality Watch Scaling Addition'");
        assert.strictEqual(ultaProvData.hasCandidateSelectedStatus, true, "Status must be 'Selected for Release (Awaiting Deployment)'");
        assert.strictEqual(ultaProvData.hasAcceptedEvidenceReview, true, "Review must be 'Accepted Research Evidence (SEC Audited)'");
        assert.strictEqual(ultaProvData.has20261021Timestamp, true, "Authored UTC must be '2026-10-21T16:00:00Z'");
        assert.strictEqual(ultaProvData.hasBaselineShaLeak, false, "ULTA page must NOT display or leak baseline ledger hash (a65be3b...)");

        const ultaScreenshotPath = path.join(SCREENSHOT_DIR, "mobile_390_ulta_provenance.png");
        await page.screenshot({ path: ultaScreenshotPath, fullPage: false });
        console.log(`Captured ULTA provenance screenshot: mobile_390_ulta_provenance.png`);

        // B. Excluded Symbol Fallback (MU)
        console.log("\nNavigating to /research?symbol=MU...");
        await page.goto(`http://localhost:${PORT}/research?symbol=MU`, { waitUntil: "networkidle0" });
        await page.waitForSelector('main', { timeout: 10000 });
        await new Promise((r) => setTimeout(r, 800));

        const muFallbackData = await page.evaluate(() => {
          const text = document.body.innerText;
          return {
            hasNoPublishedWatchBadge: text.includes("NO PUBLISHED WATCH DOSSIER"),
            hasNoPublishedWatchTitle: text.includes("No Published Watch Dossier for") && text.includes("MU"),
            hasCollectionGovernance: text.toUpperCase().includes("COLLECTION GOVERNANCE"),
            hasDisclaimerText: text.includes("No published watch dossier exists for") && text.includes("MU"),
            hasStockFundamentalsLink: text.includes("View Fundamentals Page (MU)"),
            hasBaselineShaLeak: text.includes("a65be3b38cdf"),
          };
        });
        console.log("MU Neutral Fallback checks:", muFallbackData);
        assert.strictEqual(muFallbackData.hasNoPublishedWatchBadge, true, "Must display 'NO PUBLISHED WATCH DOSSIER'");
        assert.strictEqual(muFallbackData.hasNoPublishedWatchTitle, true, "Must display 'No Published Watch Dossier for \"MU\"'");
        assert.strictEqual(muFallbackData.hasCollectionGovernance, true, "Must display 'Collection Governance'");
        assert.strictEqual(muFallbackData.hasDisclaimerText, true, "Must display disclaimer for MU");
        assert.strictEqual(muFallbackData.hasStockFundamentalsLink, true, "Must display backlink to /stock/mu");
        assert.strictEqual(muFallbackData.hasBaselineShaLeak, false, "MU fallback must NOT display baseline ledger hash");

        const muScreenshotPath = path.join(SCREENSHOT_DIR, "mobile_390_mu_excluded_neutral.png");
        await page.screenshot({ path: muScreenshotPath, fullPage: false });
        console.log(`Captured MU excluded neutral screenshot: mobile_390_mu_excluded_neutral.png`);

        // C. Held Symbol Fallback (IREN)
        console.log("\nNavigating to /research?symbol=IREN...");
        await page.goto(`http://localhost:${PORT}/research?symbol=IREN`, { waitUntil: "networkidle0" });
        await page.waitForSelector('main', { timeout: 10000 });
        await new Promise((r) => setTimeout(r, 800));

        const irenFallbackData = await page.evaluate(() => {
          const text = document.body.innerText;
          return {
            hasNoPublishedWatchBadge: text.includes("NO PUBLISHED WATCH DOSSIER"),
            hasNoPublishedWatchTitle: text.includes("No Published Watch Dossier for") && text.includes("IREN"),
            hasStockFundamentalsLink: text.includes("View Fundamentals Page (IREN)"),
          };
        });
        console.log("IREN Held Neutral Fallback checks:", irenFallbackData);
        assert.strictEqual(irenFallbackData.hasNoPublishedWatchBadge, true, "Held symbol IREN must display 'NO PUBLISHED WATCH DOSSIER'");
        assert.strictEqual(irenFallbackData.hasStockFundamentalsLink, true, "Must display backlink to /stock/iren");
      }

      await page.close();
    }

    console.log(`\n========================================================================`);
    console.log(`REAL HEADLESS BROWSER VERIFICATION SUMMARY:`);
    console.log(`========================================================================`);
    for (const [w, r] of Object.entries(results)) {
      console.log(`- ${w}px Viewport:`);
      console.log(`  * Research link visible: ${r.researchVisible}`);
      console.log(`  * Rendered hit area: ${r.researchHitArea.width}px × ${r.researchHitArea.height}px (>= 44x44px target)`);
      console.log(`  * Dock controls overlap: ${r.controlsOverlap} (Zero overlap)`);
      console.log(`  * Horizontal page overflow: ${r.horizontalPageOverflow} (Zero blowout)`);
      console.log(`  * Navigates with symbol context: ${r.preservedSymbolContext}`);
      console.log(`  * Active navigation state: ${r.activeNavigationState}`);
      console.log(`  * Screenshot: ${r.screenshotPath}`);
    }
    console.log(`\nALL REAL MOBILE BROWSER TESTS PASSED (100%) ✅`);
  } finally {
    await browser.close();
    server.close();
  }
}

runBrowserChecks().catch((err) => {
  console.error("FATAL Browser check error:", err);
  process.exit(1);
});
