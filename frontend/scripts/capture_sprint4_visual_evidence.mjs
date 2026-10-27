/**
 * frontend/scripts/capture_sprint4_visual_evidence.mjs
 *
 * ARX Sprint 4 PR #98 Rendered Visual Evidence Generator
 * Covers:
 * - ARX-S3-004: Radar Error vs. Zero States (Connection Error vs. Zero Matching Search)
 * - ARX-S3-005: Radar Candidate Truth and Provenance (Discovery Badges, Zero Execution Badges)
 * - ARX-S3-006: Research Workstation Decision-First Layout (DECK, ULTA, FIX across Desktop/Tablet/Mobile)
 * - ARX-S3-016: Synthesis E V3 Parity across Workstation, Trade Plan, Portfolio, and Design System
 * - Decision Modals: Pre-Flight Checklist Modal & Position Sizer Modal
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3590;
const OUT_DIR = path.resolve(__dirname, "../out");

const OUTPUT_DIR_1 = path.resolve(__dirname, "../../audit-output/sprint4/ws42/screenshots");
const OUTPUT_DIR_2 = path.resolve("C:/Users/akara/.gemini/antigravity/brain/da608944-9152-4928-aad6-8a68379e1238/screenshots");

fs.mkdirSync(OUTPUT_DIR_1, { recursive: true });
fs.mkdirSync(OUTPUT_DIR_2, { recursive: true });

const MIME_TYPES = {
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
  ".ttf": "font/ttf",
  ".txt": "text/plain; charset=utf-8",
};

function startServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "*");

      if (req.method === "OPTIONS") {
        res.writeHead(200);
        res.end();
        return;
      }

      let reqPath = (req.url || "/").split("?")[0];
      let filePath = path.join(OUT_DIR, reqPath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }

      if (!fs.existsSync(filePath)) {
        if (fs.existsSync(filePath + ".html")) {
          filePath = filePath + ".html";
        } else {
          const tryIndex = path.join(OUT_DIR, reqPath, "index.html");
          if (fs.existsSync(tryIndex)) {
            filePath = tryIndex;
          } else {
            res.writeHead(404, { "Content-Type": "text/plain" });
            res.end("Not Found: " + reqPath);
            return;
          }
        }
      }

      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, {
        "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
        "Cache-Control": "no-cache",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.on("error", reject);
    server.listen(PORT, () => {
      console.log(`[HTTP SERVER] Serving static export on http://localhost:${PORT}`);
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

const MOCK_RADAR_RESPONSE = {
  total_candidates: 3,
  gems_found: 3,
  status: "SUCCESS",
  error: null,
  capabilities: {
    VALUE_GARP: { status: "AVAILABLE", universeScreening: "AVAILABLE" },
    VCP: { status: "AVAILABLE", universeScreening: "AVAILABLE" },
    SMART_MONEY: { status: "AVAILABLE", universeScreening: "AVAILABLE" },
  },
  results: [
    {
      ticker: "NVDA",
      confluenceScore: 88,
      gemScore: 92,
      companyName: "NVIDIA Corporation",
      expertArchetype: "Growth Momentum",
      categories: ["VCP", "SMART_MONEY"],
      categoryEvidence: {
        vcpStage: "Stage 2 Contraction",
        insiderSignal: "Heavy Net Accumulation",
      },
      peg_ratio: 1.45,
      observedAt: 1740000000000,
    },
    {
      ticker: "DECK",
      confluenceScore: 84,
      gemScore: 87,
      companyName: "Deckers Outdoor Corp",
      expertArchetype: "Quality Compounder",
      categories: ["VALUE_GARP", "VCP"],
      categoryEvidence: {
        garpScore: "Top Decile ROIC",
        vcpStage: "Stage 1 Base Breakout",
      },
      peg_ratio: 1.12,
      observedAt: 1740000000000,
    },
    {
      ticker: "ULTA",
      confluenceScore: 79,
      gemScore: 82,
      companyName: "Ulta Beauty Inc",
      expertArchetype: "Value Quality",
      categories: ["VALUE_GARP"],
      categoryEvidence: {
        garpScore: "Historical P/E Floor",
      },
      peg_ratio: 0.95,
      observedAt: 1740000000000,
    },
  ],
};

async function safeGoto(page, url, delayMs = 1200) {
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15000 });
  } catch (err) {
    console.warn(`[WARN] Navigation warning for ${url}:`, err.message);
  }
  await new Promise((r) => setTimeout(r, delayMs));
}

async function saveScreenshot(page, filename, description) {
  const file1 = path.join(OUTPUT_DIR_1, filename);
  const file2 = path.join(OUTPUT_DIR_2, filename);
  await page.screenshot({ path: file1, fullPage: false });
  fs.copyFileSync(file1, file2);
  const stat = fs.statSync(file1);
  console.log(`[CAPTURED] ${filename} (${(stat.size / 1024).toFixed(1)} KB) - ${description}`);
  return { filename, sizeKb: (stat.size / 1024).toFixed(1), description };
}

async function run() {
  const server = await startServer();
  const manifest = [];

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  try {
    const page = await browser.newPage();
    await page.setRequestInterception(true);

    let simulateRadarError = false;

    page.on("request", (req) => {
      const url = req.url();
      if (url.includes("matomo") || url.includes("analytics.php")) {
        return req.abort();
      }

      if (url.includes("/screener/run")) {
        if (simulateRadarError) {
          return req.respond({
            status: 503,
            contentType: "application/json",
            body: JSON.stringify({
              status: "ERROR",
              error: "Radar Screener Service Offline (503 Service Unavailable)",
              total_candidates: 0,
              results: [],
            }),
          });
        } else {
          return req.respond({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(MOCK_RADAR_RESPONSE),
          });
        }
      }

      req.continue();
    });

    // ──────────────────────────────────────────────────────────────────────────
    // 1. Radar with Loaded Candidates (Candidate Truth & Provenance)
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Radar (Candidate Truth & Provenance) ---");
    simulateRadarError = false;
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/radar/?mode=standard`, 1500);
    manifest.push(await saveScreenshot(page, "01_radar_desktop_1440.png", "Radar Desktop (1440px) showing loaded candidates, discovery badge, observation timestamp"));

    await page.setViewport({ width: 375, height: 812 });
    await new Promise((r) => setTimeout(r, 600));
    manifest.push(await saveScreenshot(page, "02_radar_mobile_375.png", "Radar Mobile (375px) responsive attention card & candidate list"));

    // ──────────────────────────────────────────────────────────────────────────
    // 2. Radar Zero Semantics (Search with 0 matches)
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Radar Zero Semantics (Search Filter) ---");
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/radar/?mode=standard`, 1000);
    const searchInput = await page.$('input[placeholder*="Search candidate"]');
    if (searchInput) {
      await searchInput.type("NONEXISTENT_TICKER");
      await new Promise((r) => setTimeout(r, 800));
    }
    manifest.push(await saveScreenshot(page, "03_radar_search_empty_zero_semantics.png", "Radar Zero Semantics showing truthful 0 matching candidates banner on search miss"));

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Radar Explicit Connection Error State (ARX-S3-004 Error vs Zero)
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Radar Error State (Connection Error vs Zero) ---");
    simulateRadarError = true;
    await safeGoto(page, `http://localhost:${PORT}/radar/?mode=standard`, 1200);
    manifest.push(await saveScreenshot(page, "04_radar_connection_error_state.png", "Radar Connection Error State showing explicit tape connection error banner & retry CTA"));

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Research Workstation: DECK (Decision-First Prototype ARX-S3-006)
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Research Workstation: DECK (Decision-First) ---");
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/research/?symbol=DECK&mode=standard`, 1500);
    manifest.push(await saveScreenshot(page, "05_research_deck_desktop_1440.png", "Research Workstation DECK Desktop (1440px) decision verdict preceding metrics"));

    await page.setViewport({ width: 768, height: 1024 });
    await new Promise((r) => setTimeout(r, 600));
    manifest.push(await saveScreenshot(page, "06_research_deck_tablet_768.png", "Research Workstation DECK Tablet (768px) responsive multi-column layout"));

    await page.setViewport({ width: 375, height: 812 });
    await new Promise((r) => setTimeout(r, 600));
    manifest.push(await saveScreenshot(page, "07_research_deck_mobile_375.png", "Research Workstation DECK Mobile (375px) responsive tab navigation & verdict priority"));

    // ──────────────────────────────────────────────────────────────────────────
    // 5. Research Workstation: ULTA & FIX
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Research Workstation: ULTA & FIX ---");
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/research/?symbol=ULTA&mode=standard`, 1500);
    manifest.push(await saveScreenshot(page, "08_research_ulta_desktop_1440.png", "Research Workstation ULTA Desktop (1440px) decision assessment card"));

    await safeGoto(page, `http://localhost:${PORT}/research/?symbol=FIX&mode=standard`, 1500);
    manifest.push(await saveScreenshot(page, "09_research_fix_desktop_1440.png", "Research Workstation FIX Desktop (1440px) decision assessment card"));

    // ──────────────────────────────────────────────────────────────────────────
    // 6. Main Workstation / Analysis (NVDA & AAPL)
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Main Workstation (NVDA & AAPL) ---");
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/?symbol=NVDA&mode=standard`, 2000);
    manifest.push(await saveScreenshot(page, "10_workstation_nvda_desktop_1440.png", "Main Analysis Workstation NVDA Desktop (1440px) showing 4 pillars & execution ladder"));

    await page.setViewport({ width: 375, height: 812 });
    await new Promise((r) => setTimeout(r, 600));
    manifest.push(await saveScreenshot(page, "11_workstation_nvda_mobile_375.png", "Main Analysis Workstation NVDA Mobile (375px) responsive layout"));

    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/?symbol=AAPL&mode=standard`, 2000);
    manifest.push(await saveScreenshot(page, "12_workstation_aapl_desktop_1440.png", "Main Analysis Workstation AAPL Desktop (1440px)"));

    // ──────────────────────────────────────────────────────────────────────────
    // 7. Trade Plan / Setups
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Trade Plan / Setups ---");
    await safeGoto(page, `http://localhost:${PORT}/setups/?ticker=NVDA&mode=standard`, 1500);
    manifest.push(await saveScreenshot(page, "13_setups_nvda_desktop_1440.png", "Trade Plan & Tactical Setups NVDA Desktop (1440px)"));

    await page.setViewport({ width: 375, height: 812 });
    await new Promise((r) => setTimeout(r, 600));
    manifest.push(await saveScreenshot(page, "14_setups_nvda_mobile_375.png", "Trade Plan & Tactical Setups NVDA Mobile (375px)"));

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Portfolio / Risk Manager
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Portfolio ---");
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/portfolio/?mode=standard`, 1500);
    manifest.push(await saveScreenshot(page, "15_portfolio_desktop_1440.png", "Portfolio & Risk Management Desktop (1440px)"));

    await page.setViewport({ width: 375, height: 812 });
    await new Promise((r) => setTimeout(r, 600));
    manifest.push(await saveScreenshot(page, "16_portfolio_mobile_375.png", "Portfolio & Risk Management Mobile (375px)"));

    // ──────────────────────────────────────────────────────────────────────────
    // 9. Modals: Pre-Flight Checklist & Position Sizer
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Modals (Pre-Flight & Sizer) ---");
    await page.setViewport({ width: 1440, height: 900 });
    await safeGoto(page, `http://localhost:${PORT}/?symbol=NVDA&mode=standard`, 1800);

    // Open Pre-Flight modal
    const preflightButton = await page.evaluateHandle(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      return btns.find((b) => b.textContent && b.textContent.includes("Pre-Flight Checklist")) || null;
    });

    if (preflightButton && preflightButton.asElement()) {
      await preflightButton.asElement().click();
      await new Promise((r) => setTimeout(r, 800));
      manifest.push(await saveScreenshot(page, "17_modal_preflight_checklist.png", "Pre-Flight Checklist Modal opened, showing 5 mandatory clearance checks"));
      // Close modal
      const closeBtn = await page.$('button[aria-label="Close modal"]');
      if (closeBtn) await closeBtn.click();
      await new Promise((r) => setTimeout(r, 500));
    }

    // Open Position Sizer modal
    const sizerButton = await page.evaluateHandle(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      return btns.find((b) => b.textContent && (b.textContent.includes("Position Sizer") || b.textContent.includes("Sizer"))) || null;
    });

    if (sizerButton && sizerButton.asElement()) {
      await sizerButton.asElement().click();
      await new Promise((r) => setTimeout(r, 800));
      manifest.push(await saveScreenshot(page, "18_modal_position_sizer.png", "Position Sizer Modal opened, showing capital risk and size calculations"));
      // Close modal
      const closeBtn = await page.$('button[aria-label="Close modal"]');
      if (closeBtn) await closeBtn.click();
      await new Promise((r) => setTimeout(r, 500));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 10. Design System & Release Dashboard
    // ──────────────────────────────────────────────────────────────────────────
    console.log("\n--- Capturing Design System & Release Dashboard ---");
    await safeGoto(page, `http://localhost:${PORT}/design-system-preview`, 1200);
    manifest.push(await saveScreenshot(page, "19_design_system_preview_1440.png", "Design System Preview Desktop (1440px) showing Synthesis E V3 typography & tokens"));

    await safeGoto(page, `http://localhost:${PORT}/release-dashboard`, 1200);
    manifest.push(await saveScreenshot(page, "20_release_dashboard_1440.png", "Release Dashboard Desktop (1440px) showing system telemetry"));

    // Write manifest JSON
    const manifestPath = path.resolve(OUTPUT_DIR_1, "visual_manifest.json");
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    console.log(`\n[SUCCESS] Captured ${manifest.length} rendered browser screenshots!`);
    console.log(`Manifest written to: ${manifestPath}`);

  } finally {
    await browser.close();
    await stopServer(server);
  }
}

run().catch((err) => {
  console.error("[FATAL ERROR] Visual capture failed:", err);
  process.exit(1);
});
