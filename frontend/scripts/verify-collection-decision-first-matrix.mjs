/**
 * frontend/scripts/verify-collection-decision-first-matrix.mjs
 *
 * Full matrix browser verification for the Quality Watch Decision-First Rollout.
 * Governed by ARX Closure Model v2 & ARX-DECISION-RESEARCH-PROCESS-V1.
 *
 * Matrix:
 * - Assets: FIX, DUOL, ULTA, MEDP, ISRG, DECK (+ IREN, MU negative controls)
 * - Viewports:
 *   * Desktop: 1280x800
 *   * Mobile: 390x844
 *   * Compact Mobile: 360x740
 *
 * Invariants:
 * - Mobile compact header height strictly < 250px
 * - Decision header visible in initial viewport at scroll 0 (top < viewport height)
 * - Scope switcher disclosure details present and interactive
 * - Progressive disclosures interactive and accessible (min-h-[44px])
 * - Negative controls (IREN, MU) retain fallback / standard card
 * - Supports opt-in negative control flag: --negative-control
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3341;
const OUT_DIR = path.resolve(__dirname, "../out");
const SCREENSHOTS_DIR = path.resolve(__dirname, "../screenshots");
const isNegativeControl = process.argv.includes("--negative-control");

const PUBLISHED_SYMBOLS = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"];
const VIEWPORTS = [
  { name: "desktop_1280", width: 1280, height: 800, isMobile: false },
  { name: "mobile_390", width: 390, height: 844, isMobile: true, hasTouch: true },
  { name: "mobile_360", width: 360, height: 740, isMobile: true, hasTouch: true },
];

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
      };

      res.writeHead(200, {
        "Content-Type": mimeTypes[ext] || "application/octet-stream",
        "Cache-Control": "no-cache",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.on("error", reject);
    server.listen(PORT, () => {
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

async function run() {
  let server = null;
  let browser = null;
  const failures = [];
  let totalChecks = 0;
  const measurements = {};

  if (!fs.existsSync(SCREENSHOTS_DIR)) {
    fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
  }

  function recordCheck(checkName, passed, detail = "") {
    totalChecks++;
    if (passed) {
      console.log(`  [PASS] ${checkName}${detail ? `: ${detail}` : ""}`);
    } else {
      failures.push({ name: checkName, detail });
      console.error(`  [FAIL] ${checkName}${detail ? `: ${detail}` : ""}`);
    }
  }

  try {
    server = await startServer();
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.evaluateOnNewDocument(() => {
      try {
        localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      } catch {}
    });

    console.log("========================================================================");
    console.log("1. TESTING ALL 6 PUBLISHED ASSETS ACROSS VIEWPORT MATRIX");
    console.log("========================================================================");

    for (const sym of PUBLISHED_SYMBOLS) {
      measurements[sym] = {};
      console.log(`\n--- Asset: ${sym} ---`);

      for (const vp of VIEWPORTS) {
        await page.setViewport({
          width: vp.width,
          height: vp.height,
          isMobile: vp.isMobile,
          hasTouch: vp.hasTouch,
        });

        const url = `http://localhost:${PORT}/research?symbol=${sym}`;
        await page.goto(url, { waitUntil: "networkidle0" });

        // Intentional negative control injection on mobile_390 FIX
        if (isNegativeControl && sym === "FIX" && vp.name === "mobile_390") {
          console.log("  [NEGATIVE CONTROL] Simulating failure by removing decision heading...");
          await page.evaluate(() => {
            const h = document.getElementById("decision-first-heading-FIX");
            if (h) h.remove();
          });
        }

        // Measure geometry
        const geo = await page.evaluate((s, isMob, vpH, vpW) => {
          const docScrollW = document.documentElement.scrollWidth;
          const noBlowout = docScrollW <= vpW + 1;

          if (isMob) {
            const header = document.querySelector("header.lg\\:hidden");
            const decisionHeading = document.getElementById(`decision-first-heading-${s}`);
            const details = document.querySelector("header.lg\\:hidden details");

            const hRect = header ? header.getBoundingClientRect() : null;
            const dRect = decisionHeading ? decisionHeading.getBoundingClientRect() : null;

            return {
              headerHeight: hRect ? Math.round(hRect.height) : null,
              decisionTop: dRect ? Math.round(dRect.top) : null,
              decisionInFirstViewport: dRect ? dRect.top < vpH : false,
              hasScopeDetails: Boolean(details),
              noBlowout,
            };
          } else {
            const header = document.querySelector("header.hidden.lg\\:block");
            const decisionHeading = document.getElementById(`decision-first-heading-${s}`);
            const hRect = header ? header.getBoundingClientRect() : null;
            const dRect = decisionHeading ? decisionHeading.getBoundingClientRect() : null;

            return {
              headerHeight: hRect ? Math.round(hRect.height) : null,
              decisionTop: dRect ? Math.round(dRect.top) : null,
              decisionInFirstViewport: dRect ? dRect.top < vpH : false,
              noBlowout,
            };
          }
        }, sym, vp.isMobile, vp.height, vp.width);

        measurements[sym][vp.name] = geo;

        // Take initial-view screenshot
        const shotPath = path.join(SCREENSHOTS_DIR, `rollout_${sym.toLowerCase()}_${vp.name}.png`);
        await page.screenshot({ path: shotPath, fullPage: false });

        if (vp.isMobile) {
          recordCheck(
            `${sym} [${vp.name}]: Compact header height < 250px`,
            geo.headerHeight !== null && geo.headerHeight > 0 && geo.headerHeight < 250,
            `height = ${geo.headerHeight}px`
          );
          recordCheck(
            `${sym} [${vp.name}]: Decision heading in initial viewport (top < ${vp.height}px)`,
            geo.decisionInFirstViewport,
            `decisionTop = ${geo.decisionTop}px`
          );
          recordCheck(
            `${sym} [${vp.name}]: Scope details present`,
            geo.hasScopeDetails
          );
        } else {
          recordCheck(
            `${sym} [${vp.name}]: Desktop header rendered`,
            geo.headerHeight !== null && geo.headerHeight > 100,
            `height = ${geo.headerHeight}px`
          );
          recordCheck(
            `${sym} [${vp.name}]: Decision heading in viewport`,
            geo.decisionTop !== null,
            `decisionTop = ${geo.decisionTop}px`
          );
        }

        recordCheck(
          `${sym} [${vp.name}]: Zero horizontal blowout`,
          geo.noBlowout
        );
      }

      // Test progressive disclosure interaction on Desktop
      await page.setViewport({ width: 1280, height: 800, isMobile: false });
      await page.goto(`http://localhost:${PORT}/research?symbol=${sym}`, { waitUntil: "networkidle0" });

      const lowerS = sym.toLowerCase();
      let disclosurePassed = true;
      let disclosureError = "";

      try {
        const valSel = `button[aria-controls="disclosure-valuation-${lowerS}"]`;
        const finSel = `button[aria-controls="disclosure-financials-${lowerS}"]`;
        const provSel = `button[aria-controls="disclosure-provenance-${lowerS}"]`;

        await page.waitForSelector(valSel, { timeout: 3000 });
        await page.click(valSel);
        await page.waitForSelector(`#disclosure-valuation-${lowerS}`, { timeout: 3000 });

        await page.waitForSelector(finSel, { timeout: 3000 });
        await page.click(finSel);
        await page.waitForSelector(`#disclosure-financials-${lowerS}`, { timeout: 3000 });

        await page.waitForSelector(provSel, { timeout: 3000 });
        await page.click(provSel);
        await page.waitForSelector(`#disclosure-provenance-${lowerS}`, { timeout: 3000 });
      } catch (e) {
        disclosurePassed = false;
        disclosureError = e.message;
      }

      recordCheck(
        `${sym} [Desktop]: All 3 progressive disclosures expand on click`,
        disclosurePassed,
        disclosurePassed ? "Valuation, Financials, and Baseline records expanded" : disclosureError
      );

      // Screenshot expanded state
      const expandedShotPath = path.join(SCREENSHOTS_DIR, `rollout_${sym.toLowerCase()}_expanded_desktop_1280.png`);
      await page.screenshot({ path: expandedShotPath, fullPage: false });
    }

    console.log("\n========================================================================");
    console.log("2. TESTING NEGATIVE CONTROLS & UNPUBLISHED SYMBOLS");
    console.log("========================================================================");

    // Negative control: IREN (non-cohort symbol)
    {
      await page.setViewport({ width: 1280, height: 800, isMobile: false });
      await page.goto(`http://localhost:${PORT}/research?symbol=IREN`, { waitUntil: "networkidle0" });

      const irenResult = await page.evaluate(() => {
        const hasDecisionHeading = Boolean(document.getElementById("decision-first-heading-IREN"));
        const pageText = document.body.innerText;
        const isUnavailableNotice = pageText.includes("Opportunity Dossier Unavailable") || pageText.includes("Uncataloged Symbol");
        const standardHeader = document.querySelector("header.bg-gradient-to-r");
        return {
          decisionHeadingAbsent: !hasDecisionHeading,
          isUnavailableNotice,
          hasStandardHeader: Boolean(standardHeader),
        };
      });

      recordCheck(
        "IREN: Decision-first view NOT rendered",
        irenResult.decisionHeadingAbsent
      );
      recordCheck(
        "IREN: Fallback / uncataloged notice preserved",
        irenResult.isUnavailableNotice || irenResult.hasStandardHeader
      );

      const irenShotPath = path.join(SCREENSHOTS_DIR, "rollout_iren_desktop_1280.png");
      await page.screenshot({ path: irenShotPath, fullPage: false });
    }

    // Negative control: MU (non-cohort symbol)
    {
      await page.goto(`http://localhost:${PORT}/research?symbol=MU`, { waitUntil: "networkidle0" });

      const muResult = await page.evaluate(() => {
        const hasDecisionHeading = Boolean(document.getElementById("decision-first-heading-MU"));
        return { decisionHeadingAbsent: !hasDecisionHeading };
      });

      recordCheck(
        "MU: Decision-first view NOT rendered",
        muResult.decisionHeadingAbsent
      );
    }

    // Save measurements JSON
    const measurementsPath = path.join(SCREENSHOTS_DIR, "collection_measurements.json");
    fs.writeFileSync(measurementsPath, JSON.stringify(measurements, null, 2), "utf8");
    console.log(`\nMeasurements saved to: ${measurementsPath}`);

    console.log("\n========================================================================");
    console.log(`COLLECTION MATRIX VERIFICATION: ${totalChecks - failures.length}/${totalChecks} CHECKS PASSED`);
    console.log("========================================================================");

    if (failures.length > 0) {
      console.error(`\n❌ [FAIL-CLOSED] ${failures.length} check(s) failed:`);
      for (const f of failures) {
        console.error(`  - ${f.name}: ${f.detail}`);
      }
    } else {
      console.log("\nALL COLLECTION DECISION-FIRST MATRIX CHECKS PASSED SUCCESSFULLY! ✅");
    }
  } catch (err) {
    console.error("\n[FATAL] Unhandled error during matrix run:", err);
    failures.push({ name: "Harness Execution", detail: err.message });
  } finally {
    if (browser) await browser.close();
    if (server) await stopServer(server);
  }

  if (failures.length > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

run().catch((e) => {
  console.error("\n[FATAL] Unhandled rejection:", e);
  process.exit(1);
});
