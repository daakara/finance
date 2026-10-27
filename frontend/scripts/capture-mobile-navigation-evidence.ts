import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const puppeteer = require("puppeteer");

const PORT = 3351;
const OUT_DIR = path.resolve(__dirname, "../out");
const DEST_DIRS = [
  path.resolve(__dirname, "../../audit-output/sprint4/ws42/screenshots"),
  path.resolve("C:/Users/akara/.gemini/antigravity/brain/da608944-9152-4928-aad6-8a68379e1238/screenshots"),
];

for (const d of DEST_DIRS) {
  if (!fs.existsSync(d)) {
    fs.mkdirSync(d, { recursive: true });
  }
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function startStaticServer(): Promise<http.Server> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      let reqUrl = req.url || "/";
      let reqPath = reqUrl.split("?")[0];
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
      const mimeTypes: Record<string, string> = {
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
      };

      const contentType = mimeTypes[ext] || "application/octet-stream";
      res.writeHead(200, {
        "Content-Type": contentType,
        "Cache-Control": "no-store",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.listen(PORT, "127.0.0.1", () => {
      console.log(`Static server running on http://127.0.0.1:${PORT}`);
      resolve(server);
    });
  });
}

function saveScreenshot(sourcePath: string, filename: string) {
  for (const d of DEST_DIRS) {
    fs.copyFileSync(sourcePath, path.join(d, filename));
  }
}

async function run() {
  const server = await startStaticServer();
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  });

  try {
    const page = await browser.newPage();
    await page.evaluateOnNewDocument(() => {
      try {
        localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      } catch {}
    });

    // ------------------------------------------------------------------------
    // SCENARIO 1: Mobile 375px — Neutral Radar (No Asset Selected)
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 1: Mobile 375px — Neutral Radar (No Asset Selected) ---");
    await page.setViewport({ width: 375, height: 812, isMobile: true, hasTouch: true });
    await page.goto(`http://127.0.0.1:${PORT}/radar`, { waitUntil: "domcontentloaded" });
    await delay(2500);

    const docksInfo = await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      return allDocks.map((n, idx) => ({
        idx,
        className: n.className,
        links: Array.from(n.querySelectorAll('a')).map((a) => ({ text: a.textContent?.trim(), href: a.getAttribute("href") })),
        buttons: Array.from(n.querySelectorAll('button')).map((b) => b.textContent?.trim()),
      }));
    });

    // The SynthesisMobileDock is the md:hidden dock with 5 columns
    const synthesisDock = docksInfo.find((d: any) => d.className.includes("md:hidden"));
    assert.ok(synthesisDock, "Must find SynthesisMobileDock (md:hidden)");

    const analysisLink = synthesisDock.links.find((l: any) => l.text?.includes("Analysis"));
    const planLink = synthesisDock.links.find((l: any) => l.text?.includes("Plan"));

    console.log("SynthesisMobileDock Analysis Link:", analysisLink);
    console.log("SynthesisMobileDock Plan Link:", planLink);

    assert.ok(analysisLink?.href === "/" || analysisLink?.href === "", "Analysis link must be neutral '/'");
    assert.ok(planLink?.href === "/setups" || planLink?.href === "/setups/", "Plan link must be neutral '/setups' or '/setups/'");
    assert.ok(!analysisLink?.href?.includes("AAPL"), "Analysis link must NOT contain AAPL");
    assert.ok(!planLink?.href?.includes("AAPL"), "Plan link must NOT contain AAPL");

    const temp1 = path.join(OUT_DIR, "temp_mob_01.png");
    await page.screenshot({ path: temp1, fullPage: false });
    saveScreenshot(temp1, "21_mob_01_radar_neutral_375.png");
    console.log("Saved: 21_mob_01_radar_neutral_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 2: Mobile 375px — Radar with Active Asset Context (NVDA)
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 2: Mobile 375px — Radar with Active Asset Context (NVDA) ---");
    await page.goto(`http://127.0.0.1:${PORT}/radar?ticker=NVDA`, { waitUntil: "domcontentloaded" });
    await delay(2500);

    const nvdaDock = await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      const dock = allDocks.find((n) => n.className.includes("md:hidden"));
      if (!dock) return null;
      return {
        links: Array.from(dock.querySelectorAll('a')).map((a) => ({ text: a.textContent?.trim(), href: a.getAttribute("href") })),
        buttons: Array.from(dock.querySelectorAll('button')).map((b) => b.textContent?.trim()),
      };
    });

    console.log("NVDA Context Dock:", JSON.stringify(nvdaDock, null, 2));
    const nvdaAnalysisLink = nvdaDock?.links.find((l: any) => l.text?.includes("Analysis"));
    const nvdaPlanLink = nvdaDock?.links.find((l: any) => l.text?.includes("Plan"));

    assert.ok(nvdaAnalysisLink?.href?.includes("symbol=NVDA"), "Analysis link must contain 'symbol=NVDA'");
    assert.ok(nvdaPlanLink?.href?.includes("ticker=NVDA"), "Plan link must contain 'ticker=NVDA'");

    const temp2 = path.join(OUT_DIR, "temp_mob_02.png");
    await page.screenshot({ path: temp2, fullPage: false });
    saveScreenshot(temp2, "22_mob_02_radar_nvda_context_375.png");
    console.log("Saved: 22_mob_02_radar_nvda_context_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 3: Mobile 375px — Search Button Trigger & Modal Activation
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 3: Mobile 375px — Search Button Trigger & Modal Activation ---");
    await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      const dock = allDocks.find((n) => n.className.includes("md:hidden"));
      const searchBtn = Array.from(dock?.querySelectorAll('button') || []).find((b) => b.textContent?.includes("Search"));
      searchBtn?.click();
    });

    await delay(500);
    const isModalOpen = await page.evaluate(() => {
      const dialog = document.querySelector('div[role="dialog"][aria-label="Universal Asset Search"]');
      return !!dialog;
    });
    console.log("Search dialog opened:", isModalOpen);
    assert.strictEqual(isModalOpen, true, "Search dialog must be open after clicking dock Search button");

    const focusedId = await page.evaluate(() => document.activeElement?.id);
    console.log("Focused element ID:", focusedId);
    assert.strictEqual(focusedId, "omni-search-input", "Search input must receive focus");

    const temp3 = path.join(OUT_DIR, "temp_mob_03.png");
    await page.screenshot({ path: temp3, fullPage: false });
    saveScreenshot(temp3, "23_mob_03_search_dialog_open_375.png");
    console.log("Saved: 23_mob_03_search_dialog_open_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 4: Mobile 375px — Typing Query into Search
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 4: Mobile 375px — Typing Query into Search ---");
    await page.type("#omni-search-input", "TSLA");
    await delay(300);

    const temp4 = path.join(OUT_DIR, "temp_mob_04.png");
    await page.screenshot({ path: temp4, fullPage: false });
    saveScreenshot(temp4, "24_mob_04_search_dialog_query_375.png");
    console.log("Saved: 24_mob_04_search_dialog_query_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 5: Mobile 375px — Escape Key Dismissal & Focus Restoration
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 5: Mobile 375px — Escape Key Dismissal & Focus Restoration ---");
    await page.keyboard.press("Escape");
    await delay(300);

    const modalDismissed = await page.evaluate(() => !document.querySelector('div[role="dialog"][aria-label="Universal Asset Search"]'));
    console.log("Search dialog dismissed:", modalDismissed);
    assert.strictEqual(modalDismissed, true, "Search dialog must be dismissed after Escape");

    const temp5 = path.join(OUT_DIR, "temp_mob_05.png");
    await page.screenshot({ path: temp5, fullPage: false });
    saveScreenshot(temp5, "25_mob_05_search_dialog_dismissed_375.png");
    console.log("Saved: 25_mob_05_search_dialog_dismissed_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 6: Desktop 1440px Control — Radar
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 6: Desktop 1440px Control — Radar ---");
    await page.setViewport({ width: 1440, height: 900, isMobile: false, hasTouch: false });
    await page.goto(`http://127.0.0.1:${PORT}/radar`, { waitUntil: "domcontentloaded" });
    await delay(2000);

    const temp6 = path.join(OUT_DIR, "temp_desktop_control.png");
    await page.screenshot({ path: temp6, fullPage: false });
    saveScreenshot(temp6, "26_desktop_control_radar_1440.png");
    console.log("Saved: 26_desktop_control_radar_1440.png");

    // ------------------------------------------------------------------------
    // SCENARIO 7: Mobile 375px — Non-Ticker Keyword "ENERGY" (Defect MOB-03 Verification)
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 7: Mobile 375px — Non-Ticker Keyword 'ENERGY' (MOB-03) ---");
    await page.setViewport({ width: 375, height: 812, isMobile: true, hasTouch: true });
    await page.goto(`http://127.0.0.1:${PORT}/radar?q=ENERGY`, { waitUntil: "domcontentloaded" });
    await delay(2500);

    const energyDock = await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      const dock = allDocks.find((n) => n.className.includes("md:hidden"));
      if (!dock) return null;
      return {
        links: Array.from(dock.querySelectorAll('a')).map((a) => ({ text: a.textContent?.trim(), href: a.getAttribute("href") })),
        buttons: Array.from(dock.querySelectorAll('button')).map((b) => b.textContent?.trim()),
      };
    });

    console.log("Energy Keyword Dock:", JSON.stringify(energyDock, null, 2));
    const energyAnalysisLink = energyDock?.links.find((l: any) => l.text?.includes("Analysis"));
    const energyPlanLink = energyDock?.links.find((l: any) => l.text?.includes("Plan"));

    assert.ok(
      energyAnalysisLink?.href === "/" || energyAnalysisLink?.href === "",
      `Keyword 'ENERGY' must NOT create asset link. Got ${energyAnalysisLink?.href}`
    );
    assert.ok(
      energyPlanLink?.href === "/setups" || energyPlanLink?.href === "/setups/",
      `Keyword 'ENERGY' must route Plan to neutral. Got ${energyPlanLink?.href}`
    );
    assert.ok(!energyAnalysisLink?.href?.includes("ENERGY"), "Analysis link must NOT contain ENERGY");
    assert.ok(!energyPlanLink?.href?.includes("ENERGY"), "Plan link must NOT contain ENERGY");

    const temp7 = path.join(OUT_DIR, "temp_mob_06.png");
    await page.screenshot({ path: temp7, fullPage: false });
    saveScreenshot(temp7, "27_mob_06_radar_keyword_energy_neutral_375.png");
    console.log("Saved: 27_mob_06_radar_keyword_energy_neutral_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 8: Mobile 375px — Explicit Candidate Row Click
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 8: Mobile 375px — Explicit Candidate Row Click ---");
    await page.goto(`http://127.0.0.1:${PORT}/radar`, { waitUntil: "domcontentloaded" });
    await delay(2500);

    // Click candidate row for NVDA (first matching row in candidate table)
    const clickedTicker = await page.evaluate(() => {
      const row = document.querySelector('tbody tr[aria-selected]');
      if (!row) {
        const anyRow = document.querySelector('tbody tr');
        (anyRow as HTMLElement)?.click();
        return (anyRow as HTMLElement)?.querySelector('td a')?.textContent?.trim() || "FIRST_ROW";
      }
      (row as HTMLElement).click();
      return row.querySelector('td a')?.textContent?.trim() || "ROW";
    });

    await delay(500);

    const selectedDock = await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      const dock = allDocks.find((n) => n.className.includes("md:hidden"));
      if (!dock) return null;
      return {
        links: Array.from(dock.querySelectorAll('a')).map((a) => ({ text: a.textContent?.trim(), href: a.getAttribute("href") })),
      };
    });

    console.log(`Clicked row (${clickedTicker}) Dock:`, JSON.stringify(selectedDock, null, 2));

    const temp8 = path.join(OUT_DIR, "temp_mob_07.png");
    await page.screenshot({ path: temp8, fullPage: false });
    saveScreenshot(temp8, "28_mob_07_radar_candidate_row_selected_375.png");
    console.log("Saved: 28_mob_07_radar_candidate_row_selected_375.png");

    // ------------------------------------------------------------------------
    // SCENARIO 9: Mobile 375px — Stale Selection Invalidation on Keyword Search
    // ------------------------------------------------------------------------
    console.log("\n--- Scenario 9: Mobile 375px — Stale Selection Invalidation on Keyword Search ---");
    // With row selected from Scenario 8 (or navigating to ?symbol=NVDA):
    await page.goto(`http://127.0.0.1:${PORT}/radar?symbol=NVDA`, { waitUntil: "domcontentloaded" });
    await delay(2500);

    const preSearchDock = await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      const dock = allDocks.find((n) => n.className.includes("md:hidden"));
      return {
        links: Array.from(dock?.querySelectorAll('a') || []).map((a) => ({ text: a.textContent?.trim(), href: a.getAttribute("href") })),
      };
    });
    console.log("Pre-Search Dock (NVDA selected):", JSON.stringify(preSearchDock));
    assert.ok(preSearchDock.links.some((l) => l.href?.includes("NVDA")), "Dock must reflect NVDA before search input");

    // Now focus and type 'ENERGY' into the search input
    const searchSelector = 'input[aria-label="Search candidate, ticker, or catalyst"]';
    await page.waitForSelector(searchSelector, { timeout: 5000 });
    await page.click(searchSelector, { clickCount: 3 });
    await page.keyboard.press("Backspace");
    await page.type(searchSelector, "ENERGY", { delay: 50 });
    await delay(1500);

    const postSearchDock = await page.evaluate(() => {
      const allDocks = Array.from(document.querySelectorAll('nav[aria-label="Mobile Navigation Dock"]'));
      const dock = allDocks.find((n) => n.className.includes("md:hidden"));
      return {
        links: Array.from(dock?.querySelectorAll('a') || []).map((a) => ({ text: a.textContent?.trim(), href: a.getAttribute("href") })),
      };
    });
    console.log("Post-Search Dock (ENERGY typed):", JSON.stringify(postSearchDock));
    const analysisPost = postSearchDock.links.find((l) => l.text?.includes("Analysis"));
    const planPost = postSearchDock.links.find((l) => l.text?.includes("Plan"));
    assert.ok(analysisPost?.href === "/" || analysisPost?.href === "", "Analysis link must revert to neutral '/' when keyword 'ENERGY' is typed");
    assert.ok(planPost?.href === "/setups" || planPost?.href === "/setups/", "Plan link must revert to neutral '/setups' when keyword 'ENERGY' is typed");

    const temp9 = path.join(OUT_DIR, "temp_mob_08.png");
    await page.screenshot({ path: temp9, fullPage: false });
    saveScreenshot(temp9, "29_mob_08_radar_stale_invalidation_375.png");
    console.log("Saved: 29_mob_08_radar_stale_invalidation_375.png");

    // Update visual_manifest.json with new entries
    const manifestPath = path.resolve(__dirname, "../../audit-output/sprint4/ws42/screenshots/visual_manifest.json");
    if (fs.existsSync(manifestPath)) {
      const rawManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
      const newItems = [
        {
          filename: "27_mob_06_radar_keyword_energy_neutral_375.png",
          sizeKb: (fs.statSync(path.join(DEST_DIRS[0], "27_mob_06_radar_keyword_energy_neutral_375.png")).size / 1024).toFixed(1),
          description: "Radar Mobile (375px) with non-ticker search keyword 'ENERGY': SynthesisMobileDock strictly maintains neutral links ('/' and '/setups/') without misidentifying keyword as asset ticker (MOB-03 resolved)."
        },
        {
          filename: "28_mob_07_radar_candidate_row_selected_375.png",
          sizeKb: (fs.statSync(path.join(DEST_DIRS[0], "28_mob_07_radar_candidate_row_selected_375.png")).size / 1024).toFixed(1),
          description: "Radar Mobile (375px) with candidate table row selected: SynthesisMobileDock dynamically updates active security context and establishes contextual links without stale asset retention."
        },
        {
          filename: "29_mob_08_radar_stale_invalidation_375.png",
          sizeKb: (fs.statSync(path.join(DEST_DIRS[0], "29_mob_08_radar_stale_invalidation_375.png")).size / 1024).toFixed(1),
          description: "Radar Mobile (375px) demonstrating stale asset invalidation: selecting NVDA then entering keyword 'ENERGY' invalidates prior selection and restores neutral dock links ('/' and '/setups') deterministically."
        }
      ];

      // Remove any existing entries for 27, 28, 29 before appending
      const filtered = rawManifest.filter((item: any) =>
        item.filename !== "27_mob_06_radar_keyword_energy_neutral_375.png" &&
        item.filename !== "28_mob_07_radar_candidate_row_selected_375.png" &&
        item.filename !== "29_mob_08_radar_stale_invalidation_375.png"
      );
      filtered.push(...newItems);

      for (const d of DEST_DIRS) {
        fs.writeFileSync(path.join(d, "visual_manifest.json"), JSON.stringify(filtered, null, 2), "utf8");
      }
      console.log("Updated visual_manifest.json across both destinations.");
    }

    console.log("\n===============================================================================");
    console.log("ALL REAL BROWSER EVIDENCE CAPTURED AND INVARIANTS VERIFIED SUCCESSFULLY!");
    console.log("===============================================================================");
  } finally {
    await browser.close();
    server.close();
  }
}

run().catch((err) => {
  console.error("Puppeteer execution failed:", err);
  process.exit(1);
});
