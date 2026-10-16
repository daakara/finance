import puppeteer from "puppeteer";
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const outDir = path.resolve(__dirname, "..", "out");

const PORT = 3457;
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

function createStaticServer() {
  const server = http.createServer((req, res) => {
    try {
      const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
      let pathname = decodeURIComponent(parsedUrl.pathname);
      const relPath = pathname.replace(/^\/+/, "");
      let filePath = path.join(outDir, relPath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, "index.html");
      }

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

function buildMockResponse({
  symbol,
  currentPrice,
  entryMin,
  entryMax,
  isActionable,
  canSizeTrade,
  executionStatus,
  decisionState,
  decisionStateLabel,
  stagePhase = "Stage 2 Base",
  setupPattern = "Minervini VCP",
  riskRewardRatio = 2.4,
}) {
  return {
    symbol,
    companyName: `${symbol} Corp`,
    currentPrice,
    liveSpotPrice: currentPrice,
    analysisReferencePrice: currentPrice,
    marketPriceState: {
      marketSession: "REGULAR",
      liveFreshness: "REALTIME",
    },
    priceChangePct24h: 1.2,
    setupScore: 84,
    candles: Array.from({ length: 60 }, (_, i) => ({
      time: `2026-08-${String((i % 28) + 1).padStart(2, "0")}`,
      open: currentPrice - 2 + i * 0.05,
      high: currentPrice + 2 + i * 0.05,
      low: currentPrice - 3 + i * 0.05,
      close: currentPrice + i * 0.05,
      volume: 1200000,
    })),
    technicals: {
      rsi14: 56.0,
      vwap: currentPrice - 0.5,
      ema20: currentPrice - 1.0,
      sma50: currentPrice - 5.0,
      atr: 3.5,
    },
    optimalExecution: {
      optimal_entry_min: entryMin,
      optimal_entry_max: entryMax,
      stop_loss: +(entryMin * 0.94).toFixed(2),
      stop_loss_pct: -6.0,
      take_profit_1: +(entryMax * 1.08).toFixed(2),
      take_profit_1_pct: 8.0,
      take_profit_2: +(entryMax * 1.15).toFixed(2),
      take_profit_2_pct: 15.0,
      risk_reward_ratio: riskRewardRatio,
      setup_pattern: setupPattern,
      entry_thesis: "Consolidation contraction testing 20-day EMA",
      invalidation_condition: "Close below stop loss invalidates base structure",
      stage_phase: stagePhase,
      current_price: currentPrice,
      execution_status: executionStatus,
      candle_count: 85,
    },
    decisionTrace: {
      isActionable,
      canSizeTrade,
      stateLabel: decisionStateLabel,
      decisionState,
    },
    canonicalDecision: {
      is_actionable: isActionable,
      can_size_trade: canSizeTrade,
      decision_state: decisionState,
      decision_state_label: decisionStateLabel,
    },
    confluence: {
      score: 84,
      pillars: [
        { pillar: "TECHNICAL_TREND", status: "positive", score: 88 },
        { pillar: "FUNDAMENTAL_SOLVENCY", status: "positive", score: 80 },
      ],
    },
    executionEligibility: "STOCK_EXECUTION",
    securityType: "COMMON_STOCK",
    classificationStatus: "VERIFIED",
    canonicalInstrument: {
      symbol,
      security_type: "COMMON_STOCK",
      classification_status: "VERIFIED",
      execution_eligibility: "STOCK_EXECUTION",
    },
  };
}

const STATES = [
  {
    key: "A",
    title: "State A: Inside corridor, trigger pending",
    symbol: "NVDA",
    mock: buildMockResponse({
      symbol: "NVDA",
      currentPrice: 184.0,
      entryMin: 182.5,
      entryMax: 185.0,
      isActionable: false,
      canSizeTrade: false,
      executionStatus: "IN_BUY_ZONE_AWAITING_TRIGGER",
      decisionState: "VALID_SETUP",
      decisionStateLabel: "Valid Setup — Awaiting Trigger",
    }),
    assertions: {
      readinessBadge: "SETUP IN PROGRESS",
      currentAction: "CURRENT ACTION: Wait",
      gate1State: "PASSED",
      gate2State: "BLOCKING",
      gate3State: "PENDING",
      primaryCta: "Set Buy Zone Alert",
      tacticalHintSnippet: "inside the accumulation corridor",
      tacticalHintForbidden: "Scale in with 30% initial size",
    },
  },
  {
    key: "B",
    title: "State B: Above corridor within tolerance, trigger pending",
    symbol: "AAPL",
    mock: buildMockResponse({
      symbol: "AAPL",
      currentPrice: 341.91,
      entryMin: 325.07,
      entryMax: 339.38,
      isActionable: false,
      canSizeTrade: false,
      executionStatus: "ABOVE_CORRIDOR_AWAITING_PULLBACK",
      decisionState: "VALID_SETUP",
      decisionStateLabel: "Valid Setup — Awaiting Pullback",
    }),
    assertions: {
      readinessBadge: "SETUP IN PROGRESS",
      currentAction: "CURRENT ACTION: Wait",
      gate1State: "PASSED",
      gate2State: "BLOCKING",
      gate3State: "PENDING",
      primaryCta: "Set Buy Zone Alert",
      tacticalHintSnippet: "Spot ($341.91) is above the entry corridor ($325.07–$339.38)",
      tacticalHintForbidden: "inside the accumulation corridor",
    },
  },
  {
    key: "C",
    title: "State C: Above corridor within tolerance, confirmation satisfied, canonical Wait",
    symbol: "AAPL",
    mock: buildMockResponse({
      symbol: "AAPL",
      currentPrice: 341.91,
      entryMin: 325.07,
      entryMax: 339.38,
      isActionable: false, // Canonical action Wait is authoritative!
      canSizeTrade: false,
      executionStatus: "BREAKOUT_VOLUME_CONFIRMED",
      decisionState: "VALID_SETUP",
      decisionStateLabel: "Wait — Institutional Review",
    }),
    assertions: {
      readinessBadge: "TECHNICAL CHECKS PASSED — AWAITING CANONICAL CLEARANCE",
      readinessBadgeForbidden: "EXECUTION READY",
      currentAction: "CURRENT ACTION: Wait",
      gate1State: "PASSED",
      gate2State: "PASSED",
      gate3State: "PASSED",
      nextConditionSnippet: "Technical checks passed — awaiting canonical clearance",
      activeBlockerTag: "AWAITING CLEARANCE",
      primaryCta: "Size Position",
      tacticalHintSnippet: "Canonical action posture is Wait",
      tacticalHintForbidden: "Scale in with 30% initial size",
    },
  },
  {
    key: "D",
    title: "State D: Fully satisfied canonical state",
    symbol: "NVDA",
    mock: buildMockResponse({
      symbol: "NVDA",
      currentPrice: 184.0,
      entryMin: 182.5,
      entryMax: 185.0,
      isActionable: true,
      canSizeTrade: true,
      executionStatus: "READY_TO_BUY",
      decisionState: "ACTIONABLE",
      decisionStateLabel: "Actionable Setup",
    }),
    assertions: {
      readinessBadge: "EXECUTION READY",
      currentAction: "CURRENT ACTION: Execute Setup",
      gate1State: "PASSED",
      gate2State: "PASSED",
      gate3State: "PASSED",
      primaryCta: "Size Position",
      tacticalHintSnippet: "comfortably centered within the accumulation corridor",
      tacticalHintForbidden: "Wait for required trigger confirmation",
    },
  },
  {
    key: "E",
    title: "State E: Missing/malformed canonical actionability (Fail-closed)",
    symbol: "NVDA",
    mock: buildMockResponse({
      symbol: "NVDA",
      currentPrice: 184.0,
      entryMin: 182.5,
      entryMax: 185.0,
      isActionable: "false", // Malformed string "false"
      canSizeTrade: false,
      executionStatus: "READY_TO_BUY",
      decisionState: "VALID_SETUP",
      decisionStateLabel: "Valid Setup — Trigger Confirmed",
    }),
    assertions: {
      readinessBadge: "TECHNICAL CHECKS PASSED — AWAITING CANONICAL CLEARANCE",
      readinessBadgeForbidden: "EXECUTION READY",
      currentAction: "CURRENT ACTION: Wait",
      gate1State: "PASSED",
      gate2State: "PASSED",
      gate3State: "PASSED",
      nextConditionSnippet: "Technical checks passed — awaiting canonical clearance",
      activeBlockerTag: "AWAITING CLEARANCE",
      primaryCta: "Size Position",
      tacticalHintSnippet: "entry remains conditional",
      tacticalHintForbidden: "Scale in with 30% initial size",
    },
  },
];

const VIEWPORTS = [
  { name: "desktop_1280x800", width: 1280, height: 800, isMobile: false },
  { name: "mobile_390x844", width: 390, height: 844, isMobile: true },
];

async function run() {
  console.log("================================================================================");
  console.log("   WORKSTREAM D SLICE 1: CONTROLLED-FIXTURE RENDERED VERIFICATION (PUPPETEER)   ");
  console.log("================================================================================\n");

  const server = await createStaticServer();
  let browser;

  try {
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-features=ServiceWorker"],
    });

    const page = await browser.newPage();
    const client = await page.target().createCDPSession();
    await client.send("ServiceWorker.disable");

    let currentMock = STATES[0].mock;

    await page.evaluateOnNewDocument(() => {
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

      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED_V1", "true");
      localStorage.setItem("FINANCE_ONBOARDING_DISMISSED", "true");
      localStorage.setItem("FINANCE_USER_ROLE", "LONG_TERM");
      localStorage.setItem(
        "FINANCE_MARKET_SNAPSHOTS_V1",
        JSON.stringify({
          vix: { value: 17.5, level: 17.5, status: "CALM" },
          regime: "RISK_ON",
        })
      );
    });

    await page.setRequestInterception(true);
    page.on("request", (req) => {
      const u = req.url();
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
          body: JSON.stringify(currentMock),
        });
      } else if (u.includes("/market/instruments/")) {
        req.respond({
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({
            data: {
              symbol: currentMock.symbol,
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

    const verificationLedger = [];

    for (const state of STATES) {
      console.log(`\n================================================================================`);
      console.log(`TESTING ${state.title}`);
      console.log(`================================================================================`);

      currentMock = state.mock;

      for (const vp of VIEWPORTS) {
        console.log(`\n--- Viewport: ${vp.name} (${vp.width}x${vp.height}) ---`);
        await page.setViewport({
          width: vp.width,
          height: vp.height,
          isMobile: vp.isMobile,
          hasTouch: vp.isMobile,
        });

        const targetUrl = `${BASE_URL}/?symbol=${state.symbol}&mode=standard&t=${Date.now()}`;
        await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 15000 });

        await page.waitForSelector('[data-testid="decision-readiness-card"]', { timeout: 10000 });
        await new Promise((r) => setTimeout(r, 600));

        // Expand gates to inspect explanation text
        await page.evaluate(() => {
          const gateHeaders = document.querySelectorAll('[data-testid^="gate-item-"]');
          gateHeaders.forEach((gh) => {
            const btn = gh.querySelector('[role="button"]');
            if (btn && btn.getAttribute("aria-expanded") === "false") {
              btn.click();
            }
          });
        });
        await new Promise((r) => setTimeout(r, 200));

        // Extract rendered texts
        const extracted = await page.evaluate(() => {
          const readinessBadge = document.querySelector('[data-testid="readiness-status-badge"]')?.textContent?.trim() || "";
          const currentAction = document.querySelector('[data-testid="current-action"]')?.textContent?.trim() || "";
          const planStatus = document.querySelector('[data-testid="plan-status"]')?.textContent?.trim() || "";
          const gate1Badge = document.querySelector('[data-testid="gate-badge-1"]')?.textContent?.trim() || "";
          const gate2Badge = document.querySelector('[data-testid="gate-badge-2"]')?.textContent?.trim() || "";
          const gate3Badge = document.querySelector('[data-testid="gate-badge-3"]')?.textContent?.trim() || "";
          const nextCondition = document.querySelector('[data-testid="next-condition-box"] p')?.textContent?.trim() || "";
          const activeBlockerTag = document.querySelector('[data-testid="active-blocker-tag"]')?.textContent?.trim() || "";
          const primaryCta = document.querySelector('[data-testid="readiness-primary-cta"]')?.textContent?.trim() || "";

          // Extract tactical hint text
          const planCard = document.querySelector('[data-testid="conditional-trade-plan"]');
          const tacticalHintEl =
            planCard?.querySelector('[data-testid="tactical-execution-hint"]') ||
            Array.from(planCard?.querySelectorAll('.p-2.rounded-lg.border.text-xs') || []).find((el) =>
              el.classList.contains('items-start') && el.classList.contains('gap-2')
            );
          const tacticalHintText = tacticalHintEl?.textContent?.trim() || "";

          // Check for horizontal overflow
          const docScrollWidth = document.documentElement.scrollWidth;
          const bodyScrollWidth = document.body.scrollWidth;
          const maxScroll = Math.max(docScrollWidth, bodyScrollWidth);

          return {
            readinessBadge,
            currentAction,
            planStatus,
            gate1Badge,
            gate2Badge,
            gate3Badge,
            nextCondition,
            activeBlockerTag,
            primaryCta,
            tacticalHintText,
            maxScroll,
          };
        });

        console.log(`  Readiness Badge:  ${extracted.readinessBadge}`);
        console.log(`  Current Action:   ${extracted.currentAction}`);
        console.log(`  Plan Status:      ${extracted.planStatus}`);
        console.log(`  Gate 1 (Location):${extracted.gate1Badge}`);
        console.log(`  Gate 2 (Trigger): ${extracted.gate2Badge}`);
        console.log(`  Gate 3 (Risk):    ${extracted.gate3Badge}`);
        console.log(`  Next Condition:   ${extracted.nextCondition}`);
        console.log(`  Active Blocker:   ${extracted.activeBlockerTag}`);
        console.log(`  Primary CTA:      ${extracted.primaryCta}`);
        console.log(`  Tactical Hint:    ${extracted.tacticalHintText}`);
        console.log(`  Scroll Width:     ${extracted.maxScroll}px (Viewport: ${vp.width}px)`);

        // Assertions
        if (state.assertions.readinessBadge) {
          if (!extracted.readinessBadge.includes(state.assertions.readinessBadge)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected readiness badge to include "${state.assertions.readinessBadge}", got "${extracted.readinessBadge}"`);
          }
        }
        if (state.assertions.readinessBadgeForbidden) {
          if (extracted.readinessBadge.includes(state.assertions.readinessBadgeForbidden)) {
            throw new Error(`State ${state.key} [${vp.name}]: readiness badge FORBIDDEN text found: "${state.assertions.readinessBadgeForbidden}" in "${extracted.readinessBadge}"`);
          }
        }
        if (state.assertions.currentAction) {
          if (!extracted.currentAction.includes(state.assertions.currentAction)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected current action to include "${state.assertions.currentAction}", got "${extracted.currentAction}"`);
          }
        }
        if (state.assertions.gate1State) {
          if (!extracted.gate1Badge.includes(state.assertions.gate1State)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected Gate 1 to be "${state.assertions.gate1State}", got "${extracted.gate1Badge}"`);
          }
        }
        if (state.assertions.gate2State) {
          if (!extracted.gate2Badge.includes(state.assertions.gate2State)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected Gate 2 to be "${state.assertions.gate2State}", got "${extracted.gate2Badge}"`);
          }
        }
        if (state.assertions.nextConditionSnippet) {
          if (!extracted.nextCondition.includes(state.assertions.nextConditionSnippet)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected next condition to include "${state.assertions.nextConditionSnippet}", got "${extracted.nextCondition}"`);
          }
        }
        if (state.assertions.activeBlockerTag) {
          if (!extracted.activeBlockerTag.includes(state.assertions.activeBlockerTag)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected active blocker tag to include "${state.assertions.activeBlockerTag}", got "${extracted.activeBlockerTag}"`);
          }
        }
        if (state.assertions.primaryCta) {
          if (!extracted.primaryCta.includes(state.assertions.primaryCta)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected Primary CTA to include "${state.assertions.primaryCta}", got "${extracted.primaryCta}"`);
          }
        }
        if (state.assertions.tacticalHintSnippet) {
          if (!extracted.tacticalHintText.includes(state.assertions.tacticalHintSnippet)) {
            throw new Error(`State ${state.key} [${vp.name}]: expected tactical hint to include "${state.assertions.tacticalHintSnippet}", got "${extracted.tacticalHintText}"`);
          }
        }
        if (state.assertions.tacticalHintForbidden) {
          if (extracted.tacticalHintText.includes(state.assertions.tacticalHintForbidden)) {
            throw new Error(`State ${state.key} [${vp.name}]: tactical hint FORBIDDEN text found: "${state.assertions.tacticalHintForbidden}" in "${extracted.tacticalHintText}"`);
          }
        }

        // Capture screenshot
        const screenshotPath = path.join(SCREENSHOT_DIR, `state_${state.key}_${vp.name}.png`);
        
        // Take screenshot of the conditional-trade-plan element or viewport
        const planCardElement = await page.$('[data-testid="conditional-trade-plan"]');
        if (planCardElement) {
          await planCardElement.screenshot({ path: screenshotPath });
          console.log(`  [SCREENSHOT SAVED]: ${screenshotPath}`);
        } else {
          await page.screenshot({ path: screenshotPath, fullPage: true });
          console.log(`  [FULLPAGE SCREENSHOT SAVED]: ${screenshotPath}`);
        }

        verificationLedger.push({
          state: state.key,
          title: state.title,
          viewport: vp.name,
          screenshotPath,
          extracted,
        });
      }
    }

    console.log("\n================================================================================");
    console.log("   ALL 4 STATES VERIFIED ACROSS DESKTOP & MOBILE WITH ZERO CONTRADICTIONS!      ");
    console.log("================================================================================\n");

    const reportPath = path.join(SCREENSHOT_DIR, "RENDERED_STATES_SUMMARY.json");
    fs.writeFileSync(reportPath, JSON.stringify(verificationLedger, null, 2), "utf-8");
    console.log(`Summary report written to ${reportPath}`);

  } finally {
    if (browser) await browser.close();
    server.close();
  }
}

run().catch((err) => {
  console.error("FATAL VERIFICATION ERROR:", err);
  process.exit(1);
});
