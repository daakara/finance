/**
 * frontend/scripts/verify-research-presentation.mjs
 *
 * Automated Visual & DOM Verification for Research Summary & Scenario Presentation.
 * Serves the static export (frontend/out) and drives Puppeteer across viewports:
 * - Desktop: 1280x800
 * - Mobile: 390x844 (iPhone 12/13/14)
 * - Mobile Compact: 360x740 (Standard Android)
 *
 * Enforces:
 * 1. Intended published dossier and symbol rendering.
 * 2. Complete, accurate 4-quadrant Investment Research Summary using authoritative dossier content.
 * 3. Strict Bear -> Base -> Bull scenario card DOM ordering.
 * 4. Exact authoritative modeled values ($/share) matching snapshot data.
 * 5. Exact independent arithmetic calculation of percentage differences ((modeled/ref) - 1) * 100.
 * 6. Correct dated reference price display.
 * 7. Selective rendering of descriptive scenario names (present for FIX, suppressed for ULTA/DECK).
 * 8. Neutral shared scenario interpretation copy (no unsupported generic company narratives).
 * 9. Zero horizontal page overflow across all tested viewports.
 * 10. Clean fail-closed neutral fallback for unpublished symbols (without published cards/scenarios).
 * 11. Fail-closed error collection: exits with code 1 if ANY check fails.
 * 12. Portable screenshot directory and guaranteed server/browser cleanup.
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3335;
const OUT_DIR = path.resolve(__dirname, "../out");

// Configurable portable screenshot directory
const SCREENSHOT_DIR = process.env.SCREENSHOT_DIR
  ? path.resolve(process.env.SCREENSHOT_DIR)
  : path.resolve(__dirname, "../../artifacts/screenshots");

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

// Load authoritative snapshot and manifest data for independent verification
const snapshotPath = path.resolve(__dirname, "../lib/opportunityDossierSnapshot.json");
const manifestPath = path.resolve(__dirname, "../lib/publishedWatchManifest.json");

if (!fs.existsSync(snapshotPath)) {
  console.error(`[FATAL] Snapshot file missing: ${snapshotPath}`);
  process.exit(1);
}
const snapshotData = JSON.parse(fs.readFileSync(snapshotPath, "utf-8"));
const manifestData = fs.existsSync(manifestPath)
  ? JSON.parse(fs.readFileSync(manifestPath, "utf-8"))
  : { accepted_additions: {} };

function getAuthoritativeDossier(symbol) {
  if (snapshotData.dossiers && snapshotData.dossiers[symbol]) {
    return snapshotData.dossiers[symbol];
  }
  if (manifestData.accepted_additions && manifestData.accepted_additions[symbol]) {
    const item = manifestData.accepted_additions[symbol];
    return item.dossier_payload || item;
  }
  return null;
}

// Optional negative control for negative-path proof
const IS_NEGATIVE_CONTROL = process.env.SIMULATE_NEGATIVE_CONTROL === "true" || process.argv.includes("--negative-control");

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

  const testCases = [
    { symbol: "FIX", viewports: ["desktop_1280", "mobile_390", "mobile_360"], isPublished: true },
    { symbol: "ULTA", viewports: ["desktop_1280", "mobile_390"], isPublished: true },
    { symbol: "DECK", viewports: ["desktop_1280", "mobile_390"], isPublished: true },
    { symbol: "IREN", viewports: ["desktop_1280"], isPublished: false },
  ];

  try {
    server = await startServer();
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
    });

    const page = await browser.newPage();

    for (const tc of testCases) {
      console.log(`\n========================================================================`);
      console.log(`Testing Symbol: ${tc.symbol} (${tc.isPublished ? "Published Quality Watch Asset" : "Unpublished Neutral Fallback"})`);
      console.log(`========================================================================`);

      const dossier = tc.isPublished ? getAuthoritativeDossier(tc.symbol) : null;

      for (const vpName of tc.viewports) {
        const vp = viewports.find((v) => v.name === vpName);
        await page.setViewport(vp);

        const url = `http://localhost:${PORT}/research?symbol=${tc.symbol}`;
        await page.goto(url, { waitUntil: "networkidle0" });

        // 1. Horizontal Overflow Check
        const overflow = await page.evaluate(() => {
          return {
            docWidth: document.documentElement.scrollWidth,
            winWidth: window.innerWidth,
          };
        });
        const hasOverflow = overflow.docWidth > overflow.winWidth;
        recordCheck(
          tc.symbol,
          vpName,
          "Zero Horizontal Overflow",
          !hasOverflow,
          `Scroll width (${overflow.docWidth}px) exceeds viewport width (${overflow.winWidth}px)`
        );

        if (!tc.isPublished) {
          // 2. Unpublished Fallback Checks (Distinguished from valuation suppression)
          const fallbackData = await page.evaluate(() => {
            const bodyText = document.body.innerText;
            const hasFallbackBadge = bodyText.includes("NO PUBLISHED WATCH DOSSIER") || bodyText.includes("No published Quality Watch dossier");
            const hasSummarySection = document.querySelector('section[aria-label="Investment Research Summary"]') !== null;
            const hasBearCard = document.querySelector('div[aria-label="Bear case scenario card"]') !== null;
            return { hasFallbackBadge, hasSummarySection, hasBearCard };
          });

          recordCheck(
            tc.symbol,
            vpName,
            "Unpublished Fallback State Rendered",
            fallbackData.hasFallbackBadge,
            "Fallback container / badge missing for unpublished symbol"
          );
          recordCheck(
            tc.symbol,
            vpName,
            "Published Summary Excluded in Fallback",
            !fallbackData.hasSummarySection,
            "Investment Research Summary illegally rendered on unpublished symbol fallback"
          );
          recordCheck(
            tc.symbol,
            vpName,
            "Scenario Cards Excluded in Fallback",
            !fallbackData.hasBearCard,
            "Scenario cards illegally rendered on unpublished symbol fallback"
          );
        } else {
          // 3. Published Dossier Rendered Check
          const cardExists = await page.evaluate((sym) => {
            const heading = document.querySelector("h1, h2");
            return document.body.innerText.includes(sym);
          }, tc.symbol);
          recordCheck(tc.symbol, vpName, "Dossier Rendered For Symbol", cardExists, `Dossier content for ${tc.symbol} not found`);

          // 4. Investment Research Summary Structure & Content
          const summaryCheck = await page.evaluate(() => {
            const section = document.querySelector('section[aria-label="Investment Research Summary"]');
            if (!section) return { exists: false };
            const text = section.innerText.toLowerCase();
            return {
              exists: true,
              hasHeader: text.includes("investment research summary"),
              hasWhy: text.includes("why this business deserves attention"),
              hasDebate: text.includes("central investment debate"),
              hasRemainsWatch: text.includes("why it remains watch"),
              hasEvidence: text.includes("observable evidence that could change assessment"),
              hasWatchStatus: text.includes("status: watch") && text.includes("not a buy recommendation"),
              fullText: section.innerText,
            };
          });

          recordCheck(
            tc.symbol,
            vpName,
            "Investment Research Summary Container Present",
            summaryCheck.exists,
            "section[aria-label='Investment Research Summary'] missing from DOM"
          );

          if (summaryCheck.exists) {
            recordCheck(
              tc.symbol,
              vpName,
              "Summary 4-Quadrant Headers Present",
              summaryCheck.hasHeader && summaryCheck.hasWhy && summaryCheck.hasDebate && summaryCheck.hasRemainsWatch && summaryCheck.hasEvidence,
              "One or more required summary headers missing"
            );
            recordCheck(
              tc.symbol,
              vpName,
              "WATCH Non-BUY Status Explicitly Declared",
              summaryCheck.hasWatchStatus,
              "Explicit 'Status: WATCH (Conditions required for entry • Not a BUY recommendation)' notice missing"
            );

            // Check authoritative content snippet match
            const whyText = (dossier.selection_rationale?.why_this_business || dossier.central_assessment?.presentation_summary || "").slice(0, 40);
            const contentMatched = summaryCheck.fullText.includes(whyText);
            recordCheck(
              tc.symbol,
              vpName,
              "Authoritative Summary Content Grounding",
              contentMatched,
              `Summary text did not contain authoritative snippet: "${whyText}"`
            );
          }

          // 5. Scenario Comparison Ordering & Values
          const scenarioData = await page.evaluate(() => {
            const cards = Array.from(document.querySelectorAll('div[aria-label$="scenario card"]'));
            const cardLabels = cards.map((c) => c.getAttribute("aria-label"));

            const bearCard = document.querySelector('div[aria-label="Bear case scenario card"]');
            const baseCard = document.querySelector('div[aria-label="Base case scenario card"]');
            const bullCard = document.querySelector('div[aria-label="Bull case scenario card"]');

            const sharedNotes = document.querySelector('.space-y-1\\.5.font-sans, .space-y-1.font-sans');

            return {
              cardLabels,
              bearText: bearCard ? bearCard.innerText : null,
              baseText: baseCard ? baseCard.innerText : null,
              bullText: bullCard ? bullCard.innerText : null,
              bodyText: document.body.innerText,
            };
          });

          // Verify strict Bear -> Base -> Bull DOM ordering
          const expectedLabels = [
            "Bear case scenario card",
            "Base case scenario card",
            "Bull case scenario card",
          ];
          const orderMatches = JSON.stringify(scenarioData.cardLabels) === JSON.stringify(expectedLabels);
          recordCheck(
            tc.symbol,
            vpName,
            "Scenario Card Strict Sequence (Bear -> Base -> Bull)",
            orderMatches,
            `Rendered order was [${scenarioData.cardLabels.join(", ")}], expected [${expectedLabels.join(", ")}]`
          );

          // Authoritative values
          const refPrice = dossier.valuation.reference_price;
          const refDate = dossier.valuation.reference_date;
          const bearVal = dossier.valuation.scenarios.conservative.per_share_value;
          const baseVal = dossier.valuation.scenarios.base.per_share_value;
          const bullVal = dossier.valuation.scenarios.optimistic.per_share_value;

          // Arithmetic checks
          const expectedBearDiff = ((bearVal / refPrice) - 1) * 100;
          const expectedBearDiffStr = `${expectedBearDiff > 0 ? "+" : ""}${expectedBearDiff.toFixed(1)}%`;

          const expectedBaseDiff = ((baseVal / refPrice) - 1) * 100;
          const expectedBaseDiffStr = `${expectedBaseDiff > 0 ? "+" : ""}${expectedBaseDiff.toFixed(1)}%`;

          const expectedBullDiff = ((bullVal / refPrice) - 1) * 100;
          const expectedBullDiffStr = `${expectedBullDiff > 0 ? "+" : ""}${expectedBullDiff.toFixed(1)}%`;

          // Deliberate mutation if negative control mode is requested
          const targetBearVal = IS_NEGATIVE_CONTROL && tc.symbol === "FIX" ? 999999.99 : bearVal;

          const bearValMatches = scenarioData.bearText && scenarioData.bearText.includes(`$${targetBearVal.toFixed(2)}`);
          recordCheck(
            tc.symbol,
            vpName,
            "Bear Case Modeled Value Parity",
            bearValMatches,
            `Expected $${targetBearVal.toFixed(2)} in Bear card`
          );

          const bearDiffMatches = scenarioData.bearText && scenarioData.bearText.includes(expectedBearDiffStr);
          recordCheck(
            tc.symbol,
            vpName,
            "Bear Case Diff Percentage Arithmetic",
            bearDiffMatches,
            `Expected difference ${expectedBearDiffStr} in Bear card`
          );

          const baseValMatches = scenarioData.baseText && scenarioData.baseText.includes(`$${baseVal.toFixed(2)}`);
          recordCheck(
            tc.symbol,
            vpName,
            "Base Case Modeled Value Parity",
            baseValMatches,
            `Expected $${baseVal.toFixed(2)} in Base card`
          );

          const baseDiffMatches = scenarioData.baseText && scenarioData.baseText.includes(expectedBaseDiffStr);
          recordCheck(
            tc.symbol,
            vpName,
            "Base Case Diff Percentage Arithmetic",
            baseDiffMatches,
            `Expected difference ${expectedBaseDiffStr} in Base card`
          );

          const baseIsCentral = scenarioData.baseText && scenarioData.baseText.includes("Central Case");
          recordCheck(
            tc.symbol,
            vpName,
            "Base Case Designated Neutrally As Central Case",
            baseIsCentral,
            "Base case missing 'Central Case' subtitle"
          );

          const bullValMatches = scenarioData.bullText && scenarioData.bullText.includes(`$${bullVal.toFixed(2)}`);
          recordCheck(
            tc.symbol,
            vpName,
            "Bull Case Modeled Value Parity",
            bullValMatches,
            `Expected $${bullVal.toFixed(2)} in Bull card`
          );

          const bullDiffMatches = scenarioData.bullText && scenarioData.bullText.includes(expectedBullDiffStr);
          recordCheck(
            tc.symbol,
            vpName,
            "Bull Case Diff Percentage Arithmetic",
            bullDiffMatches,
            `Expected difference ${expectedBullDiffStr} in Bull card`
          );

          // 6. Reference Date and Anchor Display Check
          const refDateMatches = scenarioData.bearText && scenarioData.bearText.includes(`Reference price as of ${refDate}: $${refPrice.toFixed(2)}`);
          recordCheck(
            tc.symbol,
            vpName,
            "Dated Reference Price Anchor Display",
            refDateMatches,
            `Expected 'Reference price as of ${refDate}: $${refPrice.toFixed(2)}'`
          );

          // 7. Neutral Shared Interpretation Block
          const hasNeutralSharedCopy = scenarioData.bodyText.includes(
            "Bear, base, and bull cases use the assumptions shown below. Their modeled values are conditional estimates, not assigned probabilities or future price targets."
          );
          recordCheck(
            tc.symbol,
            vpName,
            "Neutral Shared Scenario Interpretation Copy",
            hasNeutralSharedCopy,
            "Shared interpretation block missing neutral explanatory copy"
          );

          // Ensure unsupported generic definition bullets are absent
          const hasGenericCapexBullet = scenarioData.bodyText.includes("Downside stress case assuming customer capex moderation");
          recordCheck(
            tc.symbol,
            vpName,
            "Unsupported Generic Bullet Elimination",
            !hasGenericCapexBullet,
            "Unsupported generic narrative 'Downside stress case assuming customer capex moderation' illegally present in shared copy"
          );

          // 8. Company-Specific Descriptive Scenario Name Rules
          if (tc.symbol === "FIX") {
            const fixDescriptiveName = dossier.valuation.scenarios.conservative.name;
            const fixHasNarrative = scenarioData.bearText && scenarioData.bearText.includes(fixDescriptiveName);
            recordCheck(
              tc.symbol,
              vpName,
              "Company-Specific Descriptive Scenario Name Preserved (FIX)",
              fixHasNarrative,
              `Descriptive scenario narrative "${fixDescriptiveName}" missing from FIX Bear card`
            );
          } else {
            // For ULTA/DECK, no invented business paragraph should exist
            const ultaHasFakeParagraph = scenarioData.bearText && (
              scenarioData.bearText.includes("moderation") ||
              scenarioData.bearText.includes("operating margin normalizes")
            );
            recordCheck(
              tc.symbol,
              vpName,
              "Generic Scenario Excludes Invented Business Explanation",
              !ultaHasFakeParagraph,
              `Invented business narrative found on generic scenario for ${tc.symbol}`
            );
          }
        }

        // Capture full-page screenshot into portable artifacts directory
        const screenshotFileName = `presentation_${tc.symbol.toLowerCase()}_${vpName}.png`;
        const screenshotPath = path.join(SCREENSHOT_DIR, screenshotFileName);
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.log(`  📸 Full-page screenshot captured: ${screenshotFileName}`);
      }
    }
  } finally {
    if (browser) {
      await browser.close();
    }
    if (server) {
      await stopServer(server);
    }
  }

  console.log(`\n========================================================================`);
  console.log(`VERIFICATION SUMMARY: ${totalChecks} total checks executed.`);
  if (failures.length > 0) {
    console.error(`❌ VERIFICATION FAILED: ${failures.length} check(s) failed:`);
    for (const f of failures) {
      console.error(`   - [${f.symbol}] [${f.viewport}] ${f.checkName}: ${f.errorMsg}`);
    }
    process.exit(1);
  } else {
    console.log(`✅ ALL ${totalChecks} BROWSER VERIFICATION CHECKS PASSED CLEANLY!`);
    process.exit(0);
  }
}

runVerification().catch((err) => {
  console.error("Unhandled verification exception:", err);
  process.exit(1);
});
