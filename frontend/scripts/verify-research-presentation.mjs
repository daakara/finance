/**
 * frontend/scripts/verify-research-presentation.mjs
 *
 * Automated Visual & DOM Verification for Research Summary & Scenario Presentation.
 * Serves the static export (frontend/out) and drives Puppeteer across viewports:
 * - Desktop: 1280x800
 * - Mobile: 390x844 (iPhone 12/13/14)
 * - Mobile Compact: 360x740 (Standard Android)
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3335;
const OUT_DIR = path.resolve(__dirname, "../out");
const SCREENSHOT_DIR = path.resolve("C:/Users/akara/.gemini/antigravity/brain/c2788363-bf19-461b-8a58-6423c773a359/screenshots");

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

function startServer() {
  return new Promise((resolve) => {
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

    server.listen(PORT, () => {
      console.log(`[SERVER] Serving static export on http://localhost:${PORT}`);
      resolve(server);
    });
  });
}

async function runVerification() {
  const server = await startServer();
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  const viewports = [
    { name: "desktop_1280", width: 1280, height: 800, isMobile: false },
    { name: "mobile_390", width: 390, height: 844, isMobile: true, hasTouch: true },
    { name: "mobile_360", width: 360, height: 740, isMobile: true, hasTouch: true },
  ];

  const testCases = [
    { symbol: "FIX", viewports: ["desktop_1280", "mobile_390", "mobile_360"], expectSuppressed: false },
    { symbol: "ULTA", viewports: ["desktop_1280", "mobile_390"], expectSuppressed: false },
    { symbol: "DECK", viewports: ["desktop_1280", "mobile_390"], expectSuppressed: false },
    { symbol: "IREN", viewports: ["desktop_1280"], expectSuppressed: true },
  ];

  const report = {
    timestamp: new Date().toISOString(),
    results: [],
  };

  try {
    const page = await browser.newPage();

    for (const tc of testCases) {
      console.log(`\nTesting symbol: ${tc.symbol}...`);
      for (const vpName of tc.viewports) {
        const vp = viewports.find((v) => v.name === vpName);
        await page.setViewport(vp);

        const url = `http://localhost:${PORT}/research?symbol=${tc.symbol}`;
        await page.goto(url, { waitUntil: "networkidle0" });

        // Check horizontal overflow
        const overflowCheck = await page.evaluate(() => {
          const docWidth = document.documentElement.scrollWidth;
          const winWidth = window.innerWidth;
          return {
            docWidth,
            winWidth,
            hasOverflow: docWidth > winWidth,
          };
        });

        console.log(`  [${vpName}] Horizontal overflow: ${overflowCheck.hasOverflow ? "FAIL" : "PASS"} (${overflowCheck.docWidth}px / ${overflowCheck.winWidth}px)`);

        if (tc.expectSuppressed) {
          // Check suppressed / unavailable state
          const pageText = await page.evaluate(() => document.body.innerText);
          console.log(`  [${vpName}] Suppressed text snippet:`, pageText.slice(0, 300).replace(/\s+/g, " "));
          const hasFallback = pageText.includes("not currently available") || pageText.includes("Opportunity Dossier Unavailable") || pageText.includes("Select a Company") || pageText.includes("NO PUBLISHED WATCH DOSSIER") || pageText.includes("No published Quality Watch dossier");
          console.log(`  [${vpName}] Suppressed symbol fallback state: ${hasFallback ? "PASS" : "FAIL"}`);
        } else {
          const headings = await page.evaluate(() => Array.from(document.querySelectorAll('h1, h2, h3')).map(el => el.innerText));
          console.log(`  [${vpName}] Rendered Headings:`, headings);

          // Check Investment Research Summary (case-insensitive for CSS uppercase transform)
          const summaryDetails = await page.evaluate(() => {
            const text = document.body.innerText.toLowerCase();
            return {
              hasHeader: text.includes("investment research summary"),
              hasWhy: text.includes("why this business deserves attention"),
              hasDebate: text.includes("central investment debate"),
              hasRemainsWatch: text.includes("why it remains watch"),
              hasEvidence: text.includes("observable evidence that could change assessment"),
            };
          });
          const hasSummary = Object.values(summaryDetails).every(Boolean);
          console.log(`  [${vpName}] Investment Research Summary visible: ${hasSummary ? "PASS" : "FAIL"} (${JSON.stringify(summaryDetails)})`);

          // Check Scenario Comparison
          const scenarioCheck = await page.evaluate(() => {
            const text = document.body.innerText;
            const hasBear = text.includes("Bear case");
            const hasBase = text.includes("Base case");
            const hasBull = text.includes("Bull case");
            const hasCentral = text.includes("Central Case");
            const hasDiff = text.includes("Difference from reference price");
            const hasDisclaimer = text.includes("Scenarios reflect different assumptions, not assigned probabilities.");
            return { hasBear, hasBase, hasBull, hasCentral, hasDiff, hasDisclaimer };
          });

          console.log(`  [${vpName}] Standardized scenarios: Bear=${scenarioCheck.hasBear}, Base=${scenarioCheck.hasBase}, Bull=${scenarioCheck.hasBull}, Diff=${scenarioCheck.hasDiff}`);
        }

        // Capture screenshot
        const screenshotFileName = `presentation_${tc.symbol.toLowerCase()}_${vpName}.png`;
        const screenshotPath = path.join(SCREENSHOT_DIR, screenshotFileName);
        await page.screenshot({ path: screenshotPath, fullPage: false });
        console.log(`  [${vpName}] Screenshot saved: ${screenshotFileName}`);

        report.results.push({
          symbol: tc.symbol,
          viewport: vpName,
          overflow: overflowCheck,
        });
      }
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log("\nVisual & DOM verification completed successfully!");
}

runVerification().catch((err) => {
  console.error("Verification error:", err);
  process.exit(1);
});
