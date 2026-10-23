/**
 * frontend/scripts/verify-deck-prototype.mjs
 *
 * Automated Browser & DOM Verification for DECK Decision-First Research Prototype.
 * Governed by ARX Closure Model v2.
 *
 * Serves the static export (frontend/out) and drives Puppeteer across viewports:
 * - Desktop: 1280x800
 * - Mobile: 390x844 (iPhone 12/13/14)
 * - Compact Mobile: 360x740 (Standard Android)
 *
 * Enforces:
 * 1. Zero horizontal page overflow across all viewports.
 * 2. Immediate identifiability of WATCH verdict and decisive reason.
 * 3. Prominent one-sentence thesis and Reverse-DCF priced-in expectations ($85.72, -2.34% CAGR).
 * 4. Strict Bear -> Base -> Bull scenario cards ordering and verified values ($73.22, $127.49, $209.87).
 * 5. Presence of 3 decisive drivers and 3 invalidation conditions.
 * 6. Explicit expectation gap statement ("An expectation gap has not yet been established...").
 * 7. Progressive disclosure behavior: expand/collapse works, aria-expanded updates, touch targets >= 44px.
 * 8. Preservation of standard presentation on non-DECK published assets (FIX, ULTA).
 * 9. Neutral fail-closed fallback on unpublished symbols (IREN).
 * 10. Guaranteed screenshot capture across all viewports.
 * 11. Fail-closed: exits with code 1 if any condition fails.
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3337;
const OUT_DIR = path.resolve(__dirname, "../out");

// Screenshot directories (persisted artifact directory + local worktree directory)
const ARTIFACT_SCREENSHOT_DIR = "C:/Users/akara/.gemini/antigravity/brain/c2788363-bf19-461b-8a58-6423c773a359/screenshots";
const LOCAL_SCREENSHOT_DIR = path.resolve(__dirname, "../../artifacts/screenshots");

for (const dir of [ARTIFACT_SCREENSHOT_DIR, LOCAL_SCREENSHOT_DIR]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function startServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      let reqPath = (req.url || "/").split("?")[0];
      let filePath = path.join(OUT_DIR, reqPath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }

      if (!fs.existsSync(filePath)) {
        if (fs.existsSync(filePath + ".html")) {
          filePath = filePath + ".html";
        } else {
          res.writeHead(404, { "Content-Type": "text/plain" });
          res.end("Not Found: " + reqPath);
          return;
        }
      }

      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
      };

      res.writeHead(200, {
        "Content-Type": mimeTypes[ext] || "application/octet-stream",
        "Cache-Control": "no-cache",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.on("error", reject);
    server.listen(PORT, () => {
      console.log(`[SERVER] Serving static export on http://localhost:${PORT}`);
      resolve(server);
    });
  });
}

function stopServer(server) {
  return new Promise((resolve) => {
    if (!server) return resolve();
    server.close(() => resolve());
  });
}

async function runVerification() {
  let server = null;
  let browser = null;
  const failures = [];
  let totalChecks = 0;

  function recordCheck(symbol, viewport, checkName, passed, errorMsg = "") {
    totalChecks++;
    if (!passed) {
      failures.push({ symbol, viewport, checkName, errorMsg });
      console.error(`  ❌ [FAIL] ${symbol} [${viewport}]: ${checkName} - ${errorMsg}`);
    } else {
      console.log(`  ✓ [PASS] ${symbol} [${viewport}]: ${checkName}`);
    }
  }

  const viewports = [
    { name: "desktop_1280", width: 1280, height: 800, isMobile: false },
    { name: "mobile_390", width: 390, height: 844, isMobile: true, hasTouch: true },
    { name: "mobile_360", width: 360, height: 740, isMobile: true, hasTouch: true },
  ];

  try {
    server = await startServer();
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
    });

    const page = await browser.newPage();

    // =========================================================================
    // A. DECK Decision-First Prototype Verification
    // =========================================================================
    console.log("\n========================================================================");
    console.log("Testing DECK Decision-First Prototype Across Viewports");
    console.log("========================================================================");

    for (const vp of viewports) {
      await page.setViewport(vp);
      const url = `http://localhost:${PORT}/research?symbol=DECK`;
      await page.goto(url, { waitUntil: "networkidle0" });

      // 1. Zero Horizontal Overflow
      const overflow = await page.evaluate(() => {
        return {
          docWidth: document.documentElement.scrollWidth,
          winWidth: window.innerWidth,
        };
      });
      const hasOverflow = overflow.docWidth > overflow.winWidth;
      recordCheck(
        "DECK",
        vp.name,
        "Zero Horizontal Overflow",
        !hasOverflow,
        `Scroll width (${overflow.docWidth}px) exceeds viewport width (${overflow.winWidth}px)`
      );

      // 2. Verdict & Decisive Reason Check
      const verdictCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasWatchBadge = text.includes("WATCH");
        const hasVerdictHeading = text.includes("A. VERDICT & DECISIVE REASON") || text.includes("VERDICT & DECISIVE REASON");
        const hasNonBuyNotice = text.includes("STATUS: WATCH") && text.includes("NOT A BUY RECOMMENDATION");
        const hasDecisiveReason = text.includes("DECKERS OUTDOOR DISPLAYS EXCEPTIONAL BALANCE SHEET STRENGTH") ||
          text.includes("BALANCE SHEET STRENGTH");
        return { hasWatchBadge, hasVerdictHeading, hasNonBuyNotice, hasDecisiveReason };
      });

      recordCheck(
        "DECK",
        vp.name,
        "WATCH Verdict & Non-BUY Notice Present",
        verdictCheck.hasWatchBadge && verdictCheck.hasNonBuyNotice,
        "WATCH badge or explicit non-BUY notice missing"
      );
      recordCheck(
        "DECK",
        vp.name,
        "Decisive Reason Present",
        verdictCheck.hasVerdictHeading && verdictCheck.hasDecisiveReason,
        "Decisive reason text missing"
      );

      // 3. One-Sentence Thesis Check
      const thesisCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        return text.includes("B. ONE-SENTENCE THESIS") &&
          text.includes("CAN HOKA AND UGG SUSTAIN DIRECT-TO-CONSUMER MOMENTUM AND OPERATING MARGINS ABOVE 20%");
      });
      recordCheck(
        "DECK",
        vp.name,
        "One-Sentence Thesis Rendered",
        thesisCheck,
        "One-sentence thesis section or exact text missing"
      );

      // 4. What Is Priced In? (Reverse-DCF & Scenario Values)
      const pricedInCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasHeader = text.includes("C. WHAT IS PRICED IN?");
        const hasCagr = text.includes("-2.34%");
        const hasRefPrice = text.includes("$85.72");
        const hasExplanation = text.includes("THE IMPLIED GROWTH RATE DEPENDS ON THE MODEL ASSUMPTIONS");
        const hasBear = text.includes("BEAR CASE") && text.includes("$73.22") && text.includes("-14.6%");
        const hasBase = text.includes("BASE CASE") && text.includes("$127.49") && text.includes("+48.7%");
        const hasBull = text.includes("BULL CASE") && text.includes("$209.87") && text.includes("+144.8%");
        return { hasHeader, hasCagr, hasRefPrice, hasExplanation, hasBear, hasBase, hasBull };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Reverse-DCF Solve & Assumption Explanation",
        pricedInCheck.hasHeader && pricedInCheck.hasCagr && pricedInCheck.hasRefPrice && pricedInCheck.hasExplanation,
        "Reverse-DCF parameters or required assumption explanation missing"
      );
      recordCheck(
        "DECK",
        vp.name,
        "Bear / Base / Bull Scenario Values & Safe Percentages",
        pricedInCheck.hasBear && pricedInCheck.hasBase && pricedInCheck.hasBull,
        "Scenario values ($73.22, $127.49, $209.87) or diff percentages missing/incorrect"
      );

      // 5. Drivers & Invalidation Check
      const driversCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasDrivers = text.includes("D. WHAT MUST BE TRUE?") &&
          text.includes("1. DTC VELOCITY") &&
          text.includes("2. GROSS MARGIN") &&
          text.includes("3. ZERO-DEBT BALANCE SHEET STRENGTH");
        const hasKillers = text.includes("E. WHAT BREAKS THE THESIS?") &&
          text.includes("1. GROSS MARGIN BREACH BELOW 50.0%") &&
          text.includes("2. HOKA DTC DECELERATION BELOW 10.0%") &&
          text.includes("3. WHOLESALE CHANNEL CONTRACTION");
        return { hasDrivers, hasKillers };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Three Decisive Drivers Present",
        driversCheck.hasDrivers,
        "Drivers section or required driver cards missing"
      );
      recordCheck(
        "DECK",
        vp.name,
        "Three Invalidation Conditions Present",
        driversCheck.hasKillers,
        "Thesis killers section or invalidation triggers missing"
      );

      // 6. Expectation Gap & Review Trigger Check
      const gapAndTriggerCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasGap = text.includes("F. POTENTIAL EXPECTATION GAP") &&
          text.includes("AN EXPECTATION GAP HAS NOT YET BEEN ESTABLISHED BEYOND OBSERVED BRAND CHANNEL DIVERGENCE");
        const hasTrigger = text.includes("G. WHAT WOULD CHANGE THE DECISION?") &&
          text.includes("NEXT REVIEW TRIGGER:") &&
          text.includes("DECISIVE UNKNOWN:");
        return { hasGap, hasTrigger };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Expectation Gap Declaration (No Fake Consensus)",
        gapAndTriggerCheck.hasGap,
        "Unestablished expectation gap declaration missing"
      );
      recordCheck(
        "DECK",
        vp.name,
        "Review Triggers & Decisive Unknown",
        gapAndTriggerCheck.hasTrigger,
        "Next review trigger or decisive unknown missing"
      );

      // 7. Progressive Disclosure Interactive Behavior
      const interactiveCheck = await page.evaluate(async () => {
        const btnValuation = document.querySelector('button[aria-controls="disclosure-valuation-deck"]');
        if (!btnValuation) return { exists: false };

        const initialAria = btnValuation.getAttribute("aria-expanded");
        const height = btnValuation.clientHeight;

        // Click to expand
        btnValuation.click();
        return {
          exists: true,
          initialAria,
          heightValid: height >= 44,
        };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Progressive Disclosure Controls & Touch Target (>= 44px)",
        interactiveCheck.exists && interactiveCheck.initialAria === "false" && interactiveCheck.heightValid,
        "Progressive disclosure toggle missing, initial aria-expanded incorrect, or touch target < 44px"
      );

      // Verify that after click, disclosure content rendered
      await page.waitForTimeout ? page.waitForTimeout(100) : new Promise((r) => setTimeout(r, 100));
      const expandedCheck = await page.evaluate(() => {
        const btnValuation = document.querySelector('button[aria-controls="disclosure-valuation-deck"]');
        const container = document.getElementById("disclosure-valuation-deck");
        const updatedAria = btnValuation?.getAttribute("aria-expanded");
        const hasTable = container?.querySelector("table") !== null;
        return { updatedAria, hasTable };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Progressive Disclosure Expands with Table",
        expandedCheck.updatedAria === "true" && expandedCheck.hasTable,
        "Disclosure failed to expand or table missing"
      );

      // Capture screenshot
      const shotName = `deck_prototype_${vp.name}.png`;
      for (const dir of [ARTIFACT_SCREENSHOT_DIR, LOCAL_SCREENSHOT_DIR]) {
        await page.screenshot({
          path: path.join(dir, shotName),
          fullPage: true,
        });
      }
      console.log(`  📸 Captured screenshot: ${shotName}`);

      // On desktop, also capture screenshot with disclosure expanded
      if (vp.name === "desktop_1280") {
        // Expand financial resilience disclosure too
        await page.evaluate(() => {
          const btnFin = document.querySelector('button[aria-controls="disclosure-financials-deck"]');
          if (btnFin) btnFin.click();
        });
        await new Promise((r) => setTimeout(r, 100));

        const expandedShot = "deck_prototype_expanded_desktop_1280.png";
        for (const dir of [ARTIFACT_SCREENSHOT_DIR, LOCAL_SCREENSHOT_DIR]) {
          await page.screenshot({
            path: path.join(dir, expandedShot),
            fullPage: true,
          });
        }
        console.log(`  📸 Captured screenshot: ${expandedShot}`);
      }
    }

    // =========================================================================
    // B. Non-DECK Preservation (FIX & ULTA retain standard card)
    // =========================================================================
    console.log("\n========================================================================");
    console.log("Testing Non-DECK Published Assets (Preservation Verification)");
    console.log("========================================================================");

    for (const sym of ["FIX", "ULTA"]) {
      await page.setViewport(viewports[0]); // desktop_1280
      const url = `http://localhost:${PORT}/research?symbol=${sym}`;
      await page.goto(url, { waitUntil: "networkidle0" });

      const check = await page.evaluate((s) => {
        const text = document.body.innerText.toUpperCase();
        const hasSummarySection = document.querySelector('section[aria-label="Investment Research Summary"]') !== null;
        const hasStandardHeader = text.includes("INVESTMENT RESEARCH SUMMARY") && text.includes("WHY THIS BUSINESS DESERVES ATTENTION");
        const isNotDeckPrototype = !text.includes("DECISION-FIRST RESEARCH PROTOTYPE (DECK)");
        return { hasSummarySection, hasStandardHeader, isNotDeckPrototype };
      }, sym);

      recordCheck(
        sym,
        "desktop_1280",
        "Retains Accepted Standard Presentation",
        check.hasSummarySection && check.hasStandardHeader && check.isNotDeckPrototype,
        "Standard layout altered or prototype leaked to non-DECK asset"
      );

      if (sym === "FIX") {
        for (const dir of [ARTIFACT_SCREENSHOT_DIR, LOCAL_SCREENSHOT_DIR]) {
          await page.screenshot({
            path: path.join(dir, "fix_preserved_desktop_1280.png"),
            fullPage: true,
          });
        }
        console.log(`  📸 Captured screenshot: fix_preserved_desktop_1280.png`);
      }
    }

    // =========================================================================
    // C. Unpublished Neutral Fallback (IREN)
    // =========================================================================
    console.log("\n========================================================================");
    console.log("Testing Unpublished Fallback (IREN)");
    console.log("========================================================================");

    await page.setViewport(viewports[0]);
    await page.goto(`http://localhost:${PORT}/research?symbol=IREN`, { waitUntil: "networkidle0" });

    const fallbackCheck = await page.evaluate(() => {
      const text = document.body.innerText;
      const hasFallbackBadge = text.includes("NO PUBLISHED WATCH DOSSIER");
      const hasSummary = document.querySelector('section[aria-label="Investment Research Summary"]') !== null;
      const hasBearCard = document.querySelector('div[aria-label="Bear case scenario card"]') !== null;
      const hasDeckView = text.includes("Decision-First Research Prototype (DECK)");
      return { hasFallbackBadge, hasSummary, hasBearCard, hasDeckView };
    });

    recordCheck(
      "IREN",
      "desktop_1280",
      "Unpublished Fallback Cleanly Rendered (No Leaks)",
      fallbackCheck.hasFallbackBadge && !fallbackCheck.hasSummary && !fallbackCheck.hasBearCard && !fallbackCheck.hasDeckView,
      "Fallback missing or published card/prototype leaked"
    );

    for (const dir of [ARTIFACT_SCREENSHOT_DIR, LOCAL_SCREENSHOT_DIR]) {
      await page.screenshot({
        path: path.join(dir, "iren_fallback_desktop_1280.png"),
        fullPage: true,
      });
    }
    console.log(`  📸 Captured screenshot: iren_fallback_desktop_1280.png`);

  } catch (err) {
    console.error("[FATAL] Error running browser verification:", err);
    failures.push({ symbol: "ALL", viewport: "ALL", checkName: "Runner Exception", errorMsg: err.message });
  } finally {
    if (browser) await browser.close();
    if (server) await stopServer(server);
  }

  console.log("\n========================================================================");
  console.log(`VERIFICATION SUMMARY: ${totalChecks - failures.length}/${totalChecks} CHECKS PASSED`);
  console.log("========================================================================");

  if (failures.length > 0) {
    console.error(`❌ [FAIL-CLOSED] ${failures.length} check(s) failed:`);
    for (const f of failures) {
      console.error(`  - ${f.symbol} [${f.viewport}]: ${f.checkName} (${f.errorMsg})`);
    }
    process.exit(1);
  } else {
    console.log("✅ ALL BROWSER & DOM VERIFICATION CHECKS PASSED SUCCESSFULLY!");
    process.exit(0);
  }
}

runVerification();
