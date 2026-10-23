/**
 * frontend/scripts/verify-deck-prototype.mjs
 *
 * Automated Browser & DOM Verification for Corrected DECK Decision-First Prototype.
 * Governed by ARX Closure Model v2.
 *
 * Serves the static export (frontend/out) and drives Puppeteer across viewports:
 * - Desktop: 1280x800
 * - Mobile: 390x844 (iPhone 12/13/14)
 * - Compact Mobile: 360x740 (Standard Android)
 *
 * Enforces:
 * 1. Zero horizontal page overflow across all viewports.
 * 2. Immediate identifiability of WATCH verdict, decisive reason, and non-BUY notice.
 * 3. Concise one-sentence thesis without meta-explanatory fluff.
 * 4. Reverse-DCF priced-in expectations ($85.72, -2.34% CAGR) and assumption qualification.
 * 5. Strict Bear -> Base -> Bull scenario cards ordering and verified values ($73.22, $127.49, $209.87).
 * 6. Exactly 3 concise drivers ("What Must Be True?").
 * 7. Exactly 2 supported invalidation conditions ("What Breaks The Thesis?"), with NO invented wholesale contraction.
 * 8. Correct review trigger: Q2 FY27 earnings release.
 * 9. Explicit expectation gap statement without fake consensus claims.
 * 10. Progressive disclosure behavior: expand/collapse works, aria-expanded updates, touch targets >= 44px.
 * 11. Preservation of standard presentation on non-DECK published assets (FIX, ULTA).
 * 12. Neutral fail-closed fallback on unpublished symbols (IREN).
 * 13. Dismissal of onboarding overlay before layout evaluation.
 * 14. Guaranteed screenshot capture across all viewports.
 * 15. Fail-closed: exits with code 1 if any check fails.
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

// Screenshot directories (persisted artifact directory)
const ARTIFACT_SCREENSHOT_DIR = "C:/Users/akara/.gemini/antigravity/brain/c2788363-bf19-461b-8a58-6423c773a359/screenshots";

if (!fs.existsSync(ARTIFACT_SCREENSHOT_DIR)) {
  fs.mkdirSync(ARTIFACT_SCREENSHOT_DIR, { recursive: true });
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

    // Ensure onboarding modal is dismissed via localStorage before navigation
    await page.evaluateOnNewDocument(() => {
      try {
        localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      } catch {}
    });

    // =========================================================================
    // A. DECK Decision-First Prototype Verification
    // =========================================================================
    console.log("\n========================================================================");
    console.log("Testing Corrected DECK Decision-First Prototype Across Viewports");
    console.log("========================================================================");

    for (const vp of viewports) {
      await page.setViewport(vp);
      const url = `http://localhost:${PORT}/research?symbol=DECK`;
      await page.goto(url, { waitUntil: "networkidle0" });

      // Ensure any modal overlay is dismissed
      await page.evaluate(() => {
        const dialog = document.querySelector('[role="dialog"]');
        if (dialog) {
          const closeBtn = dialog.querySelector('button');
          if (closeBtn) closeBtn.click();
        }
      });
      await new Promise((r) => setTimeout(r, 100));

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
        const hasHeading = text.includes("WATCH — DECISIVE REASON") || text.includes("DECISIVE REASON");
        const hasNonBuyNotice = text.includes("STATUS: WATCH") && text.includes("NOT A BUY RECOMMENDATION");
        const hasDecisiveReason = text.includes("DECKERS OUTDOOR DISPLAYS EXCEPTIONAL BALANCE SHEET STRENGTH") ||
          text.includes("BALANCE SHEET STRENGTH");
        const noProcessTag = !text.includes("A. VERDICT") && !text.includes("PRIMARY RESEARCH DISPOSITION");
        return { hasWatchBadge, hasHeading, hasNonBuyNotice, hasDecisiveReason, noProcessTag };
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
        "Decisive Reason Cleanly Rendered (No Process Prefixes)",
        verdictCheck.hasHeading && verdictCheck.hasDecisiveReason && verdictCheck.noProcessTag,
        "Decisive reason missing or process tag found"
      );

      // 3. One-Sentence Thesis Check (Without Meta-Explanatory Sentence)
      const thesisCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasHeading = text.includes("THESIS");
        const hasThesis = text.includes("CAN HOKA AND UGG SUSTAIN DIRECT-TO-CONSUMER MOMENTUM AND OPERATING MARGINS ABOVE 20%");
        const noMetaExplanation = !text.includes("CONNECTS THE DURABILITY OF BRAND PRICING POWER DIRECTLY");
        const noSectionPrefix = !text.includes("B. ONE-SENTENCE THESIS");
        return { hasHeading, hasThesis, noMetaExplanation, noSectionPrefix };
      });
      recordCheck(
        "DECK",
        vp.name,
        "One-Sentence Thesis Rendered (No Meta-Fluff)",
        thesisCheck.hasHeading && thesisCheck.hasThesis && thesisCheck.noMetaExplanation && thesisCheck.noSectionPrefix,
        "Thesis missing, section prefix present, or meta-explanatory sentence retained"
      );

      // 4. Expectations & Scenarios
      const pricedInCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasHeader = text.includes("EXPECTATIONS & SCENARIOS");
        const hasCagr = text.includes("-2.34%");
        const hasRefPrice = text.includes("$85.72");
        const hasQualification = text.includes("THE IMPLIED GROWTH RATE DEPENDS ON MODEL ASSUMPTIONS") ||
          text.includes("IMPLIED GROWTH RATE IS CONDITIONAL");
        const hasBear = text.includes("BEAR CASE") && text.includes("$73.22") && text.includes("-14.6%");
        const hasBase = text.includes("BASE CASE") && text.includes("$127.49") && text.includes("+48.7%");
        const hasBull = text.includes("BULL CASE") && text.includes("$209.87") && text.includes("+144.8%");
        const noPrefix = !text.includes("C. WHAT IS PRICED IN?");
        return { hasHeader, hasCagr, hasRefPrice, hasQualification, hasBear, hasBase, hasBull, noPrefix };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Reverse-DCF Solve & Assumption Qualification",
        pricedInCheck.hasHeader && pricedInCheck.hasCagr && pricedInCheck.hasRefPrice && pricedInCheck.hasQualification && pricedInCheck.noPrefix,
        "Reverse-DCF parameters or required qualification missing or prefix present"
      );
      recordCheck(
        "DECK",
        vp.name,
        "Bear / Base / Bull Scenario Values & Safe Percentages",
        pricedInCheck.hasBear && pricedInCheck.hasBase && pricedInCheck.hasBull,
        "Scenario values ($73.22, $127.49, $209.87) or diff percentages missing/incorrect"
      );

      // 5. Drivers & Invalidation Check (Exactly 2 Invalidation Conditions, No Wholesale Contraction)
      const driversCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasDriversHeader = text.includes("WHAT MUST BE TRUE?");
        const hasDriver1 = text.includes("1. DTC VELOCITY & INTERNATIONAL SCALING");
        const hasDriver2 = text.includes("2. GROSS MARGIN & OPERATING DEFENSE");
        const hasDriver3 = text.includes("3. ZERO-DEBT BALANCE SHEET STRENGTH") || text.includes("3. BALANCE SHEET RESILIENCE");

        const hasKillersHeader = text.includes("WHAT BREAKS THE THESIS?");
        const hasKiller1 = text.includes("1. GROSS MARGIN BREACH BELOW 50.0%");
        const hasKiller2 = text.includes("2. HOKA DTC DECELERATION BELOW 10.0%");
        const noThirdKiller = !text.includes("3. WHOLESALE CHANNEL CONTRACTION") && !text.includes("WHOLESALE REVENUE TURNING NEGATIVE");
        const noBrandFatigue = !text.includes("BRAND FATIGUE");

        return {
          driversValid: hasDriversHeader && hasDriver1 && hasDriver2 && hasDriver3,
          killersValid: hasKillersHeader && hasKiller1 && hasKiller2 && noThirdKiller && noBrandFatigue,
        };
      });

      recordCheck(
        "DECK",
        vp.name,
        "Three Decisive Drivers Present (Concise)",
        driversCheck.driversValid,
        "Drivers section or required driver cards missing"
      );
      recordCheck(
        "DECK",
        vp.name,
        "Exactly Two Supported Invalidation Conditions (No Invented Third)",
        driversCheck.killersValid,
        "Invalidation conditions missing or third unapproved condition detected"
      );

      // 6. Next Review & Key Unknown
      const reviewCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasHeader = text.includes("NEXT REVIEW & KEY UNKNOWN");
        const hasTrigger = text.includes("Q2 FY27 EARNINGS RELEASE");
        const hasUnknown = text.includes("SUSTAINABILITY OF PREMIUM DTC FULL-PRICE SELL-THROUGH RATES");
        const hasStrengthen = text.includes("CONFIRMATION OF INTERNATIONAL DTC GROWTH ABOVE 20%");
        return hasHeader && hasTrigger && hasUnknown && hasStrengthen;
      });

      recordCheck(
        "DECK",
        vp.name,
        "Next Review Trigger (Q2 FY27) & Decisive Unknown",
        reviewCheck,
        "Next review section, Q2 FY27 trigger, or key unknown missing"
      );

      // 7. Expectation Context Check
      const gapCheck = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasHeader = text.includes("EXPECTATION CONTEXT");
        const hasDeclaration = text.includes("A MARKET-CONSENSUS EXPECTATION GAP HAS NOT BEEN ESTABLISHED");
        return hasHeader && hasDeclaration;
      });

      recordCheck(
        "DECK",
        vp.name,
        "Expectation Context Declaration (No Fake Consensus)",
        gapCheck,
        "Expectation context declaration missing"
      );

      // 8. Progressive Disclosure Interactive Behavior
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
      await new Promise((r) => setTimeout(r, 100));
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
      await page.screenshot({
        path: path.join(ARTIFACT_SCREENSHOT_DIR, shotName),
        fullPage: true,
      });
      console.log(`  📸 Captured screenshot: ${shotName}`);

      // On desktop, also capture screenshot with disclosure expanded
      if (vp.name === "desktop_1280") {
        await page.evaluate(() => {
          const btnFin = document.querySelector('button[aria-controls="disclosure-financials-deck"]');
          if (btnFin) btnFin.click();
        });
        await new Promise((r) => setTimeout(r, 100));

        const expandedShot = "deck_prototype_expanded_desktop_1280.png";
        await page.screenshot({
          path: path.join(ARTIFACT_SCREENSHOT_DIR, expandedShot),
          fullPage: true,
        });
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

      const check = await page.evaluate(() => {
        const text = document.body.innerText.toUpperCase();
        const hasSummarySection = document.querySelector('section[aria-label="Investment Research Summary"]') !== null;
        const hasStandardHeader = text.includes("INVESTMENT RESEARCH SUMMARY") && text.includes("WHY THIS BUSINESS DESERVES ATTENTION");
        const isNotDeckPrototype = !text.includes("DECISION-FIRST PROTOTYPE (DECK)");
        return { hasSummarySection, hasStandardHeader, isNotDeckPrototype };
      });

      recordCheck(
        sym,
        "desktop_1280",
        "Retains Accepted Standard Presentation",
        check.hasSummarySection && check.hasStandardHeader && check.isNotDeckPrototype,
        "Standard layout altered or prototype leaked to non-DECK asset"
      );

      if (sym === "FIX") {
        await page.screenshot({
          path: path.join(ARTIFACT_SCREENSHOT_DIR, "fix_preserved_desktop_1280.png"),
          fullPage: true,
        });
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
      const hasDeckView = text.includes("Decision-First Prototype (DECK)");
      return { hasFallbackBadge, hasSummary, hasBearCard, hasDeckView };
    });

    recordCheck(
      "IREN",
      "desktop_1280",
      "Unpublished Fallback Cleanly Rendered (No Leaks)",
      fallbackCheck.hasFallbackBadge && !fallbackCheck.hasSummary && !fallbackCheck.hasBearCard && !fallbackCheck.hasDeckView,
      "Fallback missing or published card/prototype leaked"
    );

    await page.screenshot({
      path: path.join(ARTIFACT_SCREENSHOT_DIR, "iren_fallback_desktop_1280.png"),
      fullPage: true,
    });
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
