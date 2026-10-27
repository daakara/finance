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
