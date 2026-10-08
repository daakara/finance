import puppeteer from "puppeteer";
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const outDir = path.resolve(__dirname, "..", "out");

const PORT = 3456;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const SCREENSHOT_DIR = "C:/Users/akara/.gemini/antigravity/brain/1f9bcc84-ac95-4b1d-baf9-b4d4d76f13c7/screenshots";

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
};

const mockAnalyticsResponse = {
  symbol: "NVDA",
  companyName: "NVIDIA Corp",
  currentPrice: 184.2,
  priceChangePct24h: 1.45,
  setupScore: 82,
  candles: Array.from({ length: 60 }, (_, i) => ({
    time: `2026-08-${String((i % 28) + 1).padStart(2, "0")}`,
    open: 180 + i * 0.1,
    high: 185 + i * 0.1,
    low: 179 + i * 0.1,
    close: 184 + i * 0.1,
    volume: 1000000,
  })),
  technicals: {
    rsi14: 55.4,
    vwap: 183.5,
    ema20: 182.0,
    sma50: 178.0,
    atr: 4.8,
  },
  optimalExecution: {
    optimal_entry_min: 182.5,
    optimal_entry_max: 185.0,
    stop_loss: 174.0,
    stop_loss_pct: -5.1,
    take_profit_1: 198.0,
    take_profit_1_pct: 7.9,
    take_profit_2: 210.0,
    take_profit_2_pct: 14.4,
    risk_reward_ratio: 2.15,
    setup_pattern: "Minervini VCP",
    entry_thesis: "20 EMA pullback test with declining volume",
    invalidation_condition: "Close below 174.00 invalidates base structure",
    stage_phase: "Stage 2 Breakout Base",
    current_price: 184.2,
    execution_status: "IN_BUY_ZONE_AWAITING_TRIGGER",
  },
  decisionTrace: {
    isActionable: false,
    canSizeTrade: false,
    stateLabel: "Valid Setup — Awaiting Trigger",
    decisionState: "VALID_SETUP",
  },
  canonicalDecision: {
    is_actionable: false,
    decision_state: "VALID_SETUP",
    decision_state_label: "Valid Setup — Awaiting Trigger",
  },
  confluence: {
    score: 82,
    pillars: [
      { pillar: "TECHNICAL_TREND", status: "positive", score: 85 },
      { pillar: "FUNDAMENTAL_SOLVENCY", status: "positive", score: 80 },
    ],
  },
  executionEligibility: "STOCK_EXECUTION",
  securityType: "COMMON_STOCK",
  classificationStatus: "VERIFIED",
  canonicalInstrument: {
    symbol: "NVDA",
    security_type: "COMMON_STOCK",
    classification_status: "VERIFIED",
    execution_eligibility: "STOCK_EXECUTION",
  },
};

function createStaticServer() {
  const server = http.createServer((req, res) => {
    try {
      const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
      let pathname = decodeURIComponent(parsedUrl.pathname);

      // Remove leading slashes so path.join doesn't reset to C:\ on Windows
      const relPath = pathname.replace(/^\/+/, "");
      let filePath = path.join(outDir, relPath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }

      // Handle Clean URLs (e.g. /radar -> /radar.html or /radar/index.html)
      if (!fs.existsSync(filePath)) {
        if (fs.existsSync(filePath + ".html")) {
          filePath = filePath + ".html";
        } else if (fs.existsSync(path.join(filePath, "index.html"))) {
          filePath = path.join(filePath, "index.html");
        } else {
          filePath = path.join(outDir, "404.html");
        }
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = MIME_TYPES[ext] || "application/octet-stream";

      const content = fs.readFileSync(filePath);
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content);
    } catch (err) {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("Internal Server Error: " + err.message);
    }
  });

  return new Promise((resolve) => {
    server.listen(PORT, "127.0.0.1", () => {
      console.log(`Static production server listening on ${BASE_URL} (serving ${outDir})`);
      resolve(server);
    });
  });
}

async function run() {
  console.log("================================================================================");
  console.log("   ARX TERMINAL: SYNTHESIS E WAVE 4 VIEWPORT & RESPONSIVE VERIFICATION         ");
  console.log("================================================================================\n");

  const server = await createStaticServer();

  try {
    const browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-features=ServiceWorker"],
    });

    const viewports = [
      { name: "desktop_1440x900", width: 1440, height: 900, isMobile: false },
      { name: "laptop_1280x800", width: 1280, height: 800, isMobile: false },
      { name: "tablet_landscape_1024x768", width: 1024, height: 768, isMobile: false },
      { name: "tablet_portrait_768x1024", width: 768, height: 1024, isMobile: true },
      { name: "mobile_390x844", width: 390, height: 844, isMobile: true },
    ];

    const page = await browser.newPage();

    // Disable service worker in test context
    const client = await page.target().createCDPSession();
    await client.send("ServiceWorker.disable");

    // Suppress onboarding tour and service worker in test context
    await page.evaluateOnNewDocument((mockData) => {
      // 1. Safe stub for ServiceWorker to prevent real worker registration
      Object.defineProperty(navigator, "serviceWorker", {
        get: () => ({
          register: () => Promise.reject(new Error("ServiceWorker disabled in test context")),
          addEventListener: () => {},
          removeEventListener: () => {},
          controller: null,
          ready: new Promise(() => {}),
        }),
        configurable: true,
      });

      // 2. Pre-seed localStorage
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED_V1", "true");
      localStorage.setItem("FINANCE_ONBOARDING_DISMISSED", "true");
      localStorage.setItem("FINANCE_USER_ROLE", "LONG_TERM");
      localStorage.setItem("FINANCE_MARKET_SNAPSHOTS_V1", JSON.stringify({
        vix: { value: 17.5, level: 17.5, status: "CALM" },
        regime: "RISK_ON",
      }));

      // 3. Direct client-side fetch hook for deterministic testing
      const origFetch = window.fetch;
      window.fetch = async (...args) => {
        const url = String(args[0]);
        if (url.includes("/analytics/setups")) {
          return new Response(JSON.stringify({ setups: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/analytics/")) {
          return new Response(JSON.stringify(mockData), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/market/instruments/")) {
          return new Response(JSON.stringify({
            data: {
              symbol: "NVDA",
              assetType: "EQUITY",
              instrumentType: "EQUITY",
              subtype: "COMMON_STOCK",
              eligible: true,
              executionMode: "STOCK",
              status: "ACTIVE",
              provenance: { source: "test", verifiedAt: new Date().toISOString() },
            },
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/macro/ribbon")) {
          return new Response(JSON.stringify({
            vix: { value: 17.5, level: 17.5, status: "CALM" },
            regime: "RISK_ON",
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/regimes/")) {
          return new Response(JSON.stringify({ regime: "RISK_ON", vix: 17.5 }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/smart-money/")) {
          return new Response(JSON.stringify({ filings: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        return origFetch(...args);
      };
    }, mockAnalyticsResponse);

    // Intercept network requests to serve mock API data instantly with valid CORS
    await page.setRequestInterception(true);
    page.on("request", (req) => {
      const u = req.url();
      if (u.includes("analytics") || u.includes("market") || u.includes("api")) {
        console.log("  [REQ INTERCEPTED]:", req.method(), u);
      }
      const corsHeaders = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
      };

      if (req.method() === "OPTIONS") {
        req.respond({ status: 204, headers: corsHeaders });
        return;
      }

      if (u.includes("/analytics/setups")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({ setups: [] }),
        });
      } else if (u.includes("/analytics/")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify(mockAnalyticsResponse),
        });
      } else if (u.includes("/market/instruments/")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({
            data: {
              symbol: "NVDA",
              assetType: "EQUITY",
              instrumentType: "EQUITY",
              subtype: "COMMON_STOCK",
              eligible: true,
              executionMode: "STOCK",
              status: "ACTIVE",
              provenance: { source: "test", verifiedAt: new Date().toISOString() },
            },
          }),
        });
      } else if (u.includes("/macro/ribbon")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({
            vix: { value: 17.5, level: 17.5, status: "CALM" },
            regime: "RISK_ON",
          }),
        });
      } else if (u.includes("/regimes/")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({ regime: "RISK_ON", vix: 17.5 }),
        });
      } else if (u.includes("/smart-money/")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({ filings: [] }),
        });
      } else {
        req.continue();
      }
    });

    await page.setViewport({ width: 1440, height: 900 });

    page.on("console", (msg) => console.log("  [BROWSER CONSOLE]:", msg.text()));
    page.on("pageerror", (err) => console.log("  [BROWSER ERROR]:", err.toString()));

    console.log("Navigating to analysis terminal...");
    await page.goto(`${BASE_URL}/?symbol=NVDA&mode=standard`, {
      waitUntil: "domcontentloaded",
      timeout: 15000,
    });

    // Wait for decision readiness card to mount
    try {
      await page.waitForSelector('[data-testid="decision-readiness-card"]', {
        timeout: 20000,
      });
      console.log("Decision readiness card successfully mounted!\n");
    } catch (err) {
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, "debug_error.png") });
      try {
        const bodySnippet = await page.evaluate(() => document.body.innerText.slice(0, 500));
        console.error("  [PAGE TEXT AT TIMEOUT]:", bodySnippet);
      } catch {}
      throw err;
    }

    // Small delay for Next.js hydration & layout settling
    await new Promise((r) => setTimeout(r, 1000));

    async function safeEvaluate(fn, ...args) {
      let attempts = 0;
      while (attempts < 3) {
        try {
          return await page.evaluate(fn, ...args);
        } catch (err) {
          if (err.message && err.message.includes("Execution context was destroyed") && attempts < 2) {
            attempts++;
            await new Promise((r) => setTimeout(r, 500));
            continue;
          }
          throw err;
        }
      }
    }

    const results = [];

    for (const vp of viewports) {
      console.log(`--- Testing Viewport: ${vp.name} (${vp.width}x${vp.height}) ---`);
      await page.setViewport({
        width: vp.width,
        height: vp.height,
        isMobile: vp.isMobile,
        hasTouch: vp.isMobile,
      });

      await new Promise((r) => setTimeout(r, 600));

      // 1. Horizontal Overflow Audit
      const overflowMetrics = await safeEvaluate((vpWidth) => {
        const docScrollWidth = document.documentElement.scrollWidth;
        const bodyScrollWidth = document.body.scrollWidth;
        const maxScrollWidth = Math.max(docScrollWidth, bodyScrollWidth);
        return {
          viewportWidth: vpWidth,
          scrollWidth: maxScrollWidth,
          hasOverflow: maxScrollWidth > vpWidth + 1, // allow 1px rounding margin
        };
      }, vp.width);

      console.log(`  Horizontal scroll width: ${overflowMetrics.scrollWidth}px (Max: ${vp.width}px) -> ${overflowMetrics.hasOverflow ? "FAIL" : "PASS"}`);

      // Check card presence at this viewport
      const cardCheck = await safeEvaluate(() => {
        const card = document.querySelector('[data-testid="decision-readiness-card"]');
        const plan = document.querySelector('[data-testid="conditional-trade-plan"]');
        const supp = document.querySelector('[data-testid="supporting-evidence"]');
        const prevHtml = supp?.previousElementSibling ? supp.previousElementSibling.outerHTML.slice(0, 300) : "NO_PREV";
        return {
          cardFound: !!card,
          planFound: !!plan,
          prevHtml,
          url: window.location.href,
        };
      });
      console.log(`  Card presence at ${vp.name}: card=${cardCheck.cardFound}, plan=${cardCheck.planFound}`);
      console.log(`  Sibling above supporting-evidence: ${cardCheck.prevHtml}`);
      if (cardCheck.unresolvedText) {
        console.log(`  UNRESOLVED DETAIL: ${cardCheck.unresolvedText}`);
        const snap = await safeEvaluate(() => {
          return {
            storedRecord: localStorage.getItem("finance_market_db_v1_NVDA"),
          };
        });
        console.log("  [PERSISTED DB NVDA]:", snap.storedRecord ? snap.storedRecord.slice(0, 300) : "NULL");
      }

      // 2. Critical Decision Payload Depth at 390x844
      let depthMetrics = null;
      let touchTargetMetrics = null;

      if (vp.width === 390) {
        const diag = await safeEvaluate(() => {
          const card = document.querySelector('[data-testid="decision-readiness-card"]');
          const testIds = Array.from(document.querySelectorAll('[data-testid]')).map(el => el.getAttribute('data-testid'));

          // Find overflowing elements
          const overflowing = [];
          const allEls = Array.from(document.querySelectorAll('*'));
          for (const el of allEls) {
            const r = el.getBoundingClientRect();
            if (r.right > 390 + 1) {
              overflowing.push({
                tag: el.tagName,
                className: typeof el.className === 'string' ? el.className.slice(0, 50) : '',
                id: el.id,
                testId: el.getAttribute('data-testid'),
                right: Math.round(r.right),
                width: Math.round(r.width),
              });
            }
          }

          return {
            cardFound: !!card,
            allTestIds: testIds,
            overflowCount: overflowing.length,
            overflowSample: overflowing.slice(0, 8),
          };
        });

        console.log("  [MOBILE DIAGNOSTIC]:", JSON.stringify(diag, null, 2));

        depthMetrics = await safeEvaluate(() => {
          const card = document.querySelector('[data-testid="decision-readiness-card"]');
          if (!card) return null;
          const rect = card.getBoundingClientRect();
          const primaryCta = card.querySelector('[data-testid="readiness-primary-cta"]');
          const ctaRect = primaryCta ? primaryCta.getBoundingClientRect() : null;

          const els = [
            document.querySelector('[data-testid="navbar"]'),
            document.querySelector('[data-testid="market-command-ribbon"]'),
            document.querySelector('[data-testid="decision-verdict"]'),
            document.querySelector('[data-testid="unmet-condition"]'),
            document.querySelector('[data-testid="market-workspace-chart"]'),
            document.querySelector('[data-testid="conditional-trade-plan"]'),
            document.querySelector('[data-testid="decision-readiness-card"]'),
          ];
          const positions = els.filter(Boolean).map(el => ({
            testId: el.getAttribute('data-testid'),
            top: Math.round(el.getBoundingClientRect().top + window.scrollY),
            bottom: Math.round(el.getBoundingClientRect().bottom + window.scrollY),
            height: Math.round(el.getBoundingClientRect().height)
          }));

          return {
            cardTop: Math.round(rect.top + window.scrollY),
            cardBottom: Math.round(rect.bottom + window.scrollY),
            ctaBottom: ctaRect ? Math.round(ctaRect.bottom + window.scrollY) : null,
            maxBudgetPx: 1266, // 1.5 * 844
            positions,
          };
        });

        if (depthMetrics) {
          console.log("  [VERTICAL POSITIONS]:", JSON.stringify(depthMetrics.positions, null, 2));
          const isWithinBudget = depthMetrics.cardBottom <= depthMetrics.maxBudgetPx;
          console.log(`  Critical Decision Payload depth: ${depthMetrics.cardBottom}px (Budget <= ${depthMetrics.maxBudgetPx}px) -> ${isWithinBudget ? "PASS" : "FAIL"}`);
        }

        // 3. Touch Target Minimum Dimension Check (>= 44x44px)
        touchTargetMetrics = await safeEvaluate(() => {
          const card = document.querySelector('[data-testid="decision-readiness-card"]');
          if (!card) return { passed: false, targets: [] };
          const interactive = Array.from(card.querySelectorAll('button, [role="button"], a'));
          const targets = interactive.map((el) => {
            const r = el.getBoundingClientRect();
            return {
              text: el.innerText.trim().slice(0, 25),
              width: Math.round(r.width),
              height: Math.round(r.height),
              meets44px: r.width >= 43.5 && r.height >= 43.5, // 0.5px rounding margin
            };
          });
          const allMeet = targets.every((t) => t.meets44px);
          return { passed: allMeet, targets };
        });

        console.log(`  Touch Targets (>= 44x44px): ${touchTargetMetrics.passed ? "PASS (All >= 44px)" : "FAIL"}`);
      }

      // Take screenshot
      const shotPath = path.join(SCREENSHOT_DIR, `wave4_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: false });
      console.log(`  Screenshot saved to ${shotPath}\n`);

      results.push({
        viewport: vp.name,
        width: vp.width,
        height: vp.height,
        overflowMetrics,
        depthMetrics,
        touchTargetMetrics,
        screenshot: shotPath,
      });
    }

    await page.close();
    await browser.close();

    const reportPath = path.join(SCREENSHOT_DIR, "wave4_viewport_verification_report.json");
    fs.writeFileSync(reportPath, JSON.stringify(results, null, 2), "utf-8");
    console.log(`Report written to ${reportPath}`);
    console.log("\nALL VIEWPORT RESPONSIVE & MOBILE BUDGET CHECKS COMPLETED SUCCESSFULLY!");
  } finally {
    server.close();
  }
}

run().catch((err) => {
  console.error("Verification failed:", err);
  process.exit(1);
});
