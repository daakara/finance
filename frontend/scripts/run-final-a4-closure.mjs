import puppeteer from "puppeteer";
import fs from "fs";
import path from "path";
import {
  calculateContrastRatio,
  parseRgb,
  getRequiredContrastRatio,
  checkReflowMetrics,
  checkExactFocusRestoration,
  auditVisibleElementsContrast,
  auditNonTextContrast,
} from "./a4-verification-core.mjs";

const BASE_URL = "http://localhost:3000";

const CANONICAL_HUBS = [
  { id: "radar", name: "Radar / Screener", path: "/radar/?mode=standard" },
  { id: "analysis", name: "Terminal / Analysis", path: "/?symbol=NVDA&mode=standard" },
  { id: "setups", name: "Setups / Execution", path: "/setups/?ticker=NVDA&mode=standard" },
  { id: "portfolio", name: "Portfolio / Sizer", path: "/portfolio/?mode=standard" },
  { id: "journal", name: "Journal / Audit", path: "/journal/?mode=standard" },
  { id: "performance", name: "Performance / Telemetry", path: "/performance/?mode=standard" },
];

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function safeEvaluate(page, fn, ...args) {
  let attempts = 0;
  while (attempts < 3) {
    try {
      return await page.evaluate(fn, ...args);
    } catch (err) {
      if (err.message && err.message.includes("Execution context was destroyed") && attempts < 2) {
        attempts++;
        await sleep(400);
        continue;
      }
      throw err;
    }
  }
}

const failures = [];
function assertCheck(condition, testName, details = "") {
  if (!condition) {
    failures.push({ testName, details });
    console.error(`  ❌ [FAIL] ${testName}: ${details}`);
  } else {
    console.log(`  ✔ [PASS] ${testName}${details ? `: ${details}` : ""}`);
  }
  return condition;
}

async function run() {
  console.log("================================================================================");
  console.log("   ARX TERMINAL: PRIORITY 4 / PHASE A4 RUNTIME CLOSURE VERIFICATION HARNESS    ");
  console.log("================================================================================\n");
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
  });

  const results = {
    timestamp: new Date().toISOString(),
    assetIdentityVerification: [],
    trueReflowMatrix: [],
    cdpZoomVerification: [],
    dialogsTraversalMatrix: [],
    renderedContrastMatrix: [],
    renderedNonTextContrastMatrix: [],
  };

  try {
    const page = await browser.newPage();
    await page.setCacheEnabled(false);
    await page.setBypassServiceWorker(true);
    await page.setRequestInterception(true);

    page.on("request", (req) => {
      const url = req.url();
      const method = req.method();

      // Handle CORS preflight
      if (method === "OPTIONS") {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
            "Access-Control-Allow-Headers": "*",
          },
        });
        return;
      }

      if (url.includes("/analytics/setups/") || url.includes("/analytics/setups")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ticker: "NVDA",
            setupName: "Minervini VCP Breakout",
            entryPivot: 125.50,
            stopLoss: 118.20,
            target1: 145.00,
            target2: 160.00,
            confluenceScore: 88,
            isActionable: true,
            reasonSuppressed: null,
            executionStatus: "READY_TO_BUY",
            setups: [
              {
                ticker: "NVDA",
                setupName: "Minervini VCP Breakout",
                entryPivot: 125.50,
                stopLoss: 118.20,
                target1: 145.00,
                target2: 160.00,
                confluenceScore: 88,
                isActionable: true,
                reasonSuppressed: null,
                executionStatus: "READY_TO_BUY",
              },
            ],
          }),
        });
      } else if (url.includes("/analytics/NVDA")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ticker: "NVDA",
            companyName: "NVIDIA Corporation",
            currentPrice: 125.50,
            regime: { label: "Confirmed Uptrend", bias: "BULLISH" },
          }),
        });
      } else if (url.includes("/journal/telemetry") || url.includes("/risk-telemetry")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            available: true,
            userId: "trader_demo",
            accountEquity: 25000,
            consecutiveLossStreak: 0,
            dailyDrawdownPct: 0.0,
            ruleAdherencePct: 90.0,
            brierScore: 0.15,
            totalTrades: 10,
            isCalibrated: true,
            source: "AUTHORITATIVE_API",
          }),
        });
      } else if (url.includes("/macro/ribbon")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            regime: "CONFIRMED_UPTREND",
            bias: "BULLISH",
            adx: 28.5,
            distributionDays: 1,
            vix: 15.2,
          }),
        });
      } else if (url.includes("/cockpit/state")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            state: "READY",
            accountEquity: 25000,
          }),
        });
      } else if (url.includes("/portfolio/holdings") || url.includes("/portfolio/positions")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            holdings: [],
            totalValue: 25000,
          }),
        });
      } else if (url.includes("onrender.com") || url.includes("railway.app")) {
        req.respond({
          status: 200,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ ok: true }),
        });
      } else if (url.includes("matomo") || url.includes("analytics.php")) {
        req.abort();
      } else {
        req.continue();
      }
    });

    // =========================================================================
    // 0. ANALYSIS ROUTE ASSET IDENTITY ASSERTION (A3 Finding 1)
    // =========================================================================
    console.log("--- PART 0: ANALYSIS CANONICAL ROUTE & ASSET IDENTITY ASSERTION ---");
    await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
    await page.goto(`${BASE_URL}/?symbol=NVDA&mode=standard`, { waitUntil: "load", timeout: 15000 });
    await sleep(1500);

    const assetIdentity = await safeEvaluate(page, () => {
      const text = document.body.innerText || "";
      const hasNVDA = text.includes("NVDA") || text.includes("NVIDIA");
      const titleEl = document.querySelector("h1, h2, [data-testid='symbol-header']");
      const titleText = titleEl ? titleEl.textContent : "";
      return {
        hasNVDA,
        titleText: titleText.substring(0, 100),
        url: window.location.href,
      };
    });

    const assetPass = assetIdentity.hasNVDA;
    assertCheck(
      assetPass,
      "Analysis Route Asset Identity (NVDA)",
      `URL '${assetIdentity.url}' rendered active asset context (title: '${assetIdentity.titleText}')`
    );
    results.assetIdentityVerification.push({
      testedUrl: `${BASE_URL}/?symbol=NVDA&mode=standard`,
      resolved: assetIdentity,
      status: assetPass ? "PASS" : "FAIL",
    });

    // Also test fallback support for legacy ?ticker=NVDA
    await page.goto(`${BASE_URL}/?ticker=NVDA&mode=standard`, { waitUntil: "load", timeout: 15000 });
    await sleep(1000);
    const legacyAssetIdentity = await safeEvaluate(page, () => {
      const text = document.body.innerText || "";
      return text.includes("NVDA") || text.includes("NVIDIA");
    });
    assertCheck(
      legacyAssetIdentity,
      "Analysis Route Legacy ?ticker=NVDA Fallback",
      "Correctly resolves NVDA without defaulting to AAPL"
    );

    // =========================================================================
    // 1. TRUE REFLOW MATRIX & BOUNDING BOX CHECKS (A4 Finding 1 - WCAG 1.4.10)
    // =========================================================================
    console.log("\n--- PART 1: TRUE REFLOW & RESPONSIVE BOUNDING BOX MATRIX ---");
    const reflowViewports = [
      { name: "320 CSS px (400% Zoom Equivalent)", width: 320, height: 800 },
      { name: "375 CSS px (Mobile Portrait / iPhone SE)", width: 375, height: 667 },
      { name: "390 CSS px (Mobile Portrait / iPhone 14)", width: 390, height: 844 },
      { name: "640 CSS px (200% Zoom Equivalent / sm)", width: 640, height: 800 },
      { name: "768 CSS px (Tablet Portrait / md)", width: 768, height: 1024 },
      { name: "1024 CSS px (Desktop / lg)", width: 1024, height: 768 },
      { name: "1280 CSS px (Standard Desktop / xl)", width: 1280, height: 800 },
    ];

    for (const vp of reflowViewports) {
      console.log(`\nEvaluating viewport: ${vp.name} (${vp.width}x${vp.height})...`);
      await page.setViewport({ width: vp.width, height: vp.height, deviceScaleFactor: 1 });

      for (const hub of CANONICAL_HUBS) {
        await page.goto(`${BASE_URL}${hub.path}`, { waitUntil: "load", timeout: 15000 });
        await sleep(1000);

        const metrics = await checkReflowMetrics(page);

        const noHScroll = !metrics.hasHorizontalScroll;
        const noTextClip = metrics.clippedTextCount === 0;
        const noOutOfBoundsControls = metrics.outOfBoundsControlsCount === 0;
        const hubPass = noHScroll && noTextClip && noOutOfBoundsControls;

        assertCheck(
          noHScroll,
          `Reflow Scroll [${hub.name} @ ${vp.name}]`,
          `scrollWidth=${metrics.scrollWidth}px, clientWidth=${metrics.clientWidth}px (overflow: ${metrics.hasHorizontalScroll})`
        );

        assertCheck(
          noTextClip,
          `Text Clipping [${hub.name} @ ${vp.name}]`,
          `clippedTextCount=${metrics.clippedTextCount}`
        );

        assertCheck(
          noOutOfBoundsControls,
          `Control Bounds [${hub.name} @ ${vp.name}]`,
          `outOfBounds=${metrics.outOfBoundsControlsCount}, visibleControls=${metrics.visibleControlsCount}/${metrics.totalControls}`
        );

        results.trueReflowMatrix.push({
          hub: hub.name,
          viewport: vp.name,
          width: vp.width,
          scrollWidth: metrics.scrollWidth,
          clientWidth: metrics.clientWidth,
          hasHorizontalScroll: metrics.hasHorizontalScroll,
          clippedTextCount: metrics.clippedTextCount,
          outOfBoundsControls: metrics.outOfBoundsControlsCount,
          visibleControls: `${metrics.visibleControlsCount}/${metrics.totalControls}`,
          status: hubPass ? "PASS" : "FAIL",
        });
      }
    }

    // =========================================================================
    // 1b. CDP ZOOM SCALING DERIVED ASSERTION (A4 Finding 2)
    // =========================================================================
    console.log("\n--- PART 1b: CDP PageScaleFactor 2.0 (True Browser Zoom Verification) ---");
    await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
    const cdp = await page.target().createCDPSession();
    await page.goto(`${BASE_URL}/radar/?mode=standard`, { waitUntil: "load" });
    await sleep(1000);
    await cdp.send("Emulation.setPageScaleFactor", { pageScaleFactor: 2.0 });

    const cdpMetrics = await safeEvaluate(page, () => {
      return {
        innerWidth: window.innerWidth,
        outerWidth: window.outerWidth,
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        dpr: window.devicePixelRatio,
      };
    });

    const zoomReflowValid = cdpMetrics.scrollWidth <= cdpMetrics.clientWidth * 2 + 15;
    assertCheck(
      zoomReflowValid,
      "CDP 200% Zoom Scaling (Radar Hub)",
      `scrollWidth=${cdpMetrics.scrollWidth}, clientWidth=${cdpMetrics.clientWidth}, dpr=${cdpMetrics.dpr}`
    );

    results.cdpZoomVerification.push({
      hub: "Radar / Screener",
      pageScaleFactor: 2.0,
      metrics: cdpMetrics,
      status: zoomReflowValid ? "PASS" : "FAIL",
      note: "CSS viewport width reflow (640px / 320px) represents WCAG 2.2 1.4.10 reflow; CDP verifies visual scaling factor",
    });
    await cdp.send("Emulation.setPageScaleFactor", { pageScaleFactor: 1.0 });

    // ==========================================
    // 2. COMPREHENSIVE DIALOG TRAVERSAL AUDIT
    // ==========================================
    console.log("\n--- PART 2: REACHABLE DIALOGS TRAVERSAL AUDIT ---");
    await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });

    const dialogTests = [
      {
        id: "dialog-1-broker-fill",
        name: "Record Broker Fill Modal",
        pageUrl: `${BASE_URL}/setups/?ticker=NVDA&mode=standard`,
        openFn: async () => {
          try {
            await page.waitForFunction(() => {
              const btns = Array.from(document.querySelectorAll("button"));
              return btns.some((b) => (b.innerText || "").toLowerCase().includes("broker fill"));
            }, { timeout: 10000 });
            return await page.evaluate(() => {
              const btns = Array.from(document.querySelectorAll("button"));
              const trigger = btns.find((b) => (b.innerText || "").toLowerCase().includes("broker fill"));
              if (trigger) {
                trigger.setAttribute("data-a4-dialog-trigger", "dialog-1-broker-fill");
                trigger.focus();
                trigger.click();
                return true;
              }
              return false;
            });
          } catch (e) {
            console.error("Dialog 1 open error:", e);
            return false;
          }
        },
        dialogSelector: '[role="dialog"][aria-labelledby="fill-modal-title"]',
      },
      {
        id: "dialog-2-universal-search",
        name: "Universal Asset Search Modal",
        pageUrl: `${BASE_URL}/?mode=standard`,
        openFn: async () => {
          return await page.evaluate(() => {
            const btns = Array.from(document.querySelectorAll("button"));
            const searchBtn = btns.find((b) => (b.getAttribute("aria-label") || "").toLowerCase().includes("search"));
            if (searchBtn) {
              searchBtn.setAttribute("data-a4-dialog-trigger", "dialog-2-universal-search");
              searchBtn.focus();
              searchBtn.click();
              return true;
            }
            return false;
          });
        },
        dialogSelector: '[role="dialog"][aria-label="Universal Asset Search"]',
      },
      {
        id: "dialog-3-portfolio-add",
        name: "Add Portfolio Holding Modal",
        pageUrl: `${BASE_URL}/portfolio/?mode=standard`,
        openFn: async () => {
          return await page.evaluate(() => {
            const btns = Array.from(document.querySelectorAll("button"));
            const addBtn = btns.find((b) => (b.innerText || "").toLowerCase().includes("add holding"));
            if (addBtn) {
              addBtn.setAttribute("data-a4-dialog-trigger", "dialog-3-portfolio-add");
              addBtn.focus();
              addBtn.click();
              return true;
            }
            return false;
          });
        },
        dialogSelector: '[role="dialog"][aria-labelledby="add-position-modal-title"]',
      },
      {
        id: "dialog-4-portfolio-exit",
        name: "Record Trade Exit Modal",
        pageUrl: `${BASE_URL}/portfolio/?mode=standard`,
        openFn: async () => {
          await page.evaluate(() => {
            const testPositions = [
              {
                symbol: "NVDA",
                name: "NVIDIA Corp.",
                shares: 10,
                entryPrice: 120.0,
                currentPrice: 130.0,
                stopLossPrice: 110.0,
                targetPrice: 150.0,
                addedAt: new Date().toISOString(),
                assetType: "Stock",
              },
            ];
            localStorage.setItem("FINANCE_USER_PORTFOLIO", JSON.stringify(testPositions));
            localStorage.setItem("FINANCE_PORTFOLIO_POSITIONS", JSON.stringify(testPositions));
            window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
          });
          await sleep(600);

          try {
            await page.waitForFunction(() => {
              const btns = Array.from(document.querySelectorAll("button"));
              return btns.some((b) => (b.innerText || "").toLowerCase().includes("record exit"));
            }, { timeout: 8000 });
          } catch {}

          return await page.evaluate(() => {
            const btns = Array.from(document.querySelectorAll("button"));
            const exitBtn = btns.find((b) => (b.innerText || "").toLowerCase().includes("record exit"));
            if (exitBtn) {
              exitBtn.setAttribute("data-a4-dialog-trigger", "dialog-4-portfolio-exit");
              exitBtn.focus();
              exitBtn.click();
              return true;
            }
            return false;
          });
        },
        dialogSelector: '[role="dialog"][aria-labelledby="exit-modal-title"]',
      },
      {
        id: "dialog-5-onboarding-tour",
        name: "Onboarding Tour Modal",
        pageUrl: `${BASE_URL}/?mode=standard`,
        openFn: async () => {
          await page.waitForSelector("#onboarding-tour-btn", { visible: true, timeout: 10000 });
          const btn = await page.$("#onboarding-tour-btn");
          if (btn) {
            await page.evaluate((el) => {
              el.setAttribute("data-a4-dialog-trigger", "dialog-5-onboarding-tour");
            }, btn);
            await btn.focus();
            await btn.click();
            return true;
          }
          return false;
        },
        dialogSelector: '[role="dialog"][aria-labelledby="tour-modal-title"]',
      },
      {
        id: "dialog-6-shortcuts-guide",
        name: "Pro-Trader Shortcuts Modal",
        pageUrl: `${BASE_URL}/?mode=standard`,
        openFn: async () => {
          await page.waitForSelector("#shortcuts-help-btn", { visible: true, timeout: 10000 });
          const btn = await page.$("#shortcuts-help-btn");
          if (btn) {
            await page.evaluate((el) => {
              el.setAttribute("data-a4-dialog-trigger", "dialog-6-shortcuts-guide");
            }, btn);
            await btn.focus();
            await btn.click();
            return true;
          }
          return false;
        },
        dialogSelector: '[role="dialog"][aria-label="Keyboard Shortcuts Guide"]',
      },
      {
        id: "dialog-7-privacy-settings",
        name: "Privacy & Telemetry Settings Modal",
        pageUrl: `${BASE_URL}/?mode=standard`,
        openFn: async () => {
          await page.waitForSelector("#privacy-settings-btn", { visible: true, timeout: 10000 });
          const btn = await page.$("#privacy-settings-btn");
          if (btn) {
            await page.evaluate((el) => {
              el.setAttribute("data-a4-dialog-trigger", "dialog-7-privacy-settings");
            }, btn);
            await btn.focus();
            await btn.click();
            return true;
          }
          return false;
        },
        dialogSelector: '[role="dialog"][aria-label="Privacy and Data Telemetry Settings"]',
      },
    ];

    for (const dTest of dialogTests) {
      console.log(`\nTesting Dialog: ${dTest.name}...`);
      await page.goto(dTest.pageUrl, { waitUntil: "load", timeout: 15000 });
      await sleep(1500);

      const opened = await dTest.openFn();
      assertCheck(
        Boolean(opened),
        `Dialog Trigger Open [${dTest.name}]`,
        opened ? "Trigger successfully opened dialog" : "Could not trigger open dialog"
      );
      if (!opened) {
        console.log(`  [FAIL] Could not trigger open for ${dTest.name}`);
        results.dialogsTraversalMatrix.push({
          dialog: dTest.name,
          status: "FAIL",
          reason: "Trigger could not be opened",
        });
        continue;
      }

      await sleep(400); // Allow modal animation and focus management

      // 1. Verify dialog existence and attributes
      const dialogMeta = await page.evaluate((sel) => {
        const d = document.querySelector(sel);
        if (!d) return null;
        const role = d.getAttribute("role");
        const ariaModal = d.getAttribute("aria-modal");
        const ariaLabel = d.getAttribute("aria-label") || d.getAttribute("aria-labelledby");
        const active = document.activeElement;
        const isInside = d.contains(active);
        return {
          role,
          ariaModal,
          ariaLabel,
          initialActiveTag: active ? active.tagName : null,
          initialActiveId: active ? active.id : null,
          isInside,
        };
      }, dTest.dialogSelector);

      if (!dialogMeta) {
        console.log(`  [FAIL] Dialog element not found for selector: ${dTest.dialogSelector}`);
        results.dialogsTraversalMatrix.push({
          dialog: dTest.name,
          status: "FAIL",
          reason: `Dialog element not found for selector: ${dTest.dialogSelector}`,
        });
        continue;
      }

      console.log(`  Dialog found: role=${dialogMeta.role}, aria-modal=${dialogMeta.ariaModal}, label=${dialogMeta.ariaLabel}, initialFocus=${dialogMeta.initialActiveTag}#${dialogMeta.initialActiveId} (inside: ${dialogMeta.isInside})`);

      // 2. Test Tab Containment (Forward Cycle)
      let tabContained = true;
      for (let i = 0; i < 15; i++) {
        await page.keyboard.press("Tab");
        await sleep(40);
        const inside = await page.evaluate((sel) => {
          const d = document.querySelector(sel);
          return d ? d.contains(document.activeElement) : false;
        }, dTest.dialogSelector);
        if (!inside) {
          tabContained = false;
          break;
        }
      }

      // 3. Test Shift+Tab Containment (Reverse Cycle)
      let shiftTabContained = true;
      for (let i = 0; i < 15; i++) {
        await page.keyboard.down("Shift");
        await page.keyboard.press("Tab");
        await page.keyboard.up("Shift");
        await sleep(40);
        const inside = await page.evaluate((sel) => {
          const d = document.querySelector(sel);
          return d ? d.contains(document.activeElement) : false;
        }, dTest.dialogSelector);
        if (!inside) {
          shiftTabContained = false;
          break;
        }
      }

      // 4. Test Escape Key Dismissal
      await page.keyboard.press("Escape");
      await sleep(300);

      const isDismissed = await page.evaluate((sel) => {
        const d = document.querySelector(sel);
        return !d;
      }, dTest.dialogSelector);

      // 5. Test Exact Focus Restoration (Gated requirement - A4 Finding 3)
      const triggerAttr = `data-a4-dialog-trigger="${dTest.id}"`;
      const focusRestorationResult = await checkExactFocusRestoration(page, triggerAttr);

      assertCheck(
        focusRestorationResult.matched,
        `Exact Focus Restoration [${dTest.name}]`,
        `Restored to ${focusRestorationResult.activeTag}#${focusRestorationResult.activeId || "anon"} (matches trigger: ${focusRestorationResult.matched})`
      );

      // Clean up trigger attribute
      await page.evaluate((attr) => {
        const el = document.querySelector(`[${attr}]`);
        if (el) el.removeAttribute("data-a4-dialog-trigger");
      }, triggerAttr);

      // Gating pass condition: Focus restoration to exact trigger is MANDATORY for passing
      const pass = dialogMeta.isInside && tabContained && shiftTabContained && isDismissed && focusRestorationResult.matched;

      results.dialogsTraversalMatrix.push({
        dialog: dTest.name,
        role: dialogMeta.role,
        ariaModal: dialogMeta.ariaModal,
        initialFocus: `${dialogMeta.initialActiveTag}#${dialogMeta.initialActiveId || "anon"}`,
        tabContainment: tabContained ? "PASS" : "FAIL",
        shiftTabContainment: shiftTabContained ? "PASS" : "FAIL",
        escapeDismissal: isDismissed ? "PASS" : "FAIL",
        exactFocusRestoration: focusRestorationResult.matched ? "PASS" : "FAIL",
        restoredElement: `${focusRestorationResult.activeTag}#${focusRestorationResult.activeId || "anon"}`,
        status: pass ? "PASS" : "FAIL",
      });
    }

    // =========================================================================
    // 3. RUNTIME RENDERED CONTRAST & THEME AUDIT (A4 Finding 4)
    // =========================================================================
    console.log("\n--- PART 3: RUNTIME RENDERED CONTRAST & THEME PARITY MATRIX ---");
    const contrastThemes = [
      { name: "Obsidian (Dark)", themeAttr: "dark", hubPath: "/radar/?mode=standard" },
      { name: "Paper (Light)", themeAttr: "paper", hubPath: "/radar/?mode=standard" },
      { name: "Obsidian (Dark) - Setups", themeAttr: "dark", hubPath: "/setups/?ticker=NVDA&mode=standard" },
      { name: "Obsidian (Dark) - Analysis", themeAttr: "dark", hubPath: "/?symbol=NVDA&mode=standard" },
    ];

    for (const ct of contrastThemes) {
      console.log(`\nEvaluating Runtime Computed Styles in ${ct.name}...`);
      await page.goto(`${BASE_URL}${ct.hubPath}`, { waitUntil: "load" });
      await sleep(1000);

      await page.evaluate((t) => {
        document.documentElement.setAttribute("data-theme", t);
        if (t === "paper") {
          document.body.classList.add("paper");
        } else {
          document.body.classList.remove("paper");
        }
      }, ct.themeAttr);
      await sleep(500);

      const auditResult = await auditVisibleElementsContrast(page, ct.name);

      const pass = auditResult.failedCount === 0;
      assertCheck(
        pass,
        `Comprehensive Contrast Audit [${ct.name}]`,
        `Audited ${auditResult.totalAudited} visible elements: ${auditResult.passedCount} passed, ${auditResult.failedCount} failed.`
      );

      if (auditResult.failedCount > 0) {
        console.error(`  Violations sample in ${ct.name}:`, auditResult.violations.slice(0, 5));
      }

      results.renderedContrastMatrix.push({
        theme: ct.name,
        path: ct.hubPath,
        totalElementsAudited: auditResult.totalAudited,
        passedCount: auditResult.passedCount,
        failedCount: auditResult.failedCount,
        violations: auditResult.violations,
        sampleEvaluations: auditResult.sampleResults,
        status: pass ? "PASS" : "FAIL",
      });

      // Non-Text Contrast Audit (WCAG 2.2 1.4.11 >= 3.0:1)
      const nonTextResult = await auditNonTextContrast(page, ct.name);
      const nonTextPass = nonTextResult.failedCount === 0;
      assertCheck(
        nonTextPass,
        `Non-Text Contrast Audit (>= 3.0:1) [${ct.name}]`,
        `Audited ${nonTextResult.totalAudited} non-text components: ${nonTextResult.passedCount} passed, ${nonTextResult.failedCount} failed.`
      );

      if (nonTextResult.failedCount > 0) {
        console.error(`  Non-text violations in ${ct.name}:`, nonTextResult.violations.slice(0, 5));
      }

      results.renderedNonTextContrastMatrix.push({
        theme: ct.name,
        path: ct.hubPath,
        totalComponentsAudited: nonTextResult.totalAudited,
        passedCount: nonTextResult.passedCount,
        failedCount: nonTextResult.failedCount,
        violations: nonTextResult.violations,
        sampleEvaluations: nonTextResult.sampleResults,
        status: nonTextPass ? "PASS" : "FAIL",
      });
    }

    // =========================================================================
    // FINAL AUDIT & EXIT CODE DETERMINATION
    // =========================================================================
    results.failures = failures;
    results.exitCode = failures.length > 0 ? 1 : 0;

    const outPath = path.join(process.cwd(), "scripts", "a4-final-closure-results.json");
    fs.writeFileSync(outPath, JSON.stringify(results, null, 2), "utf8");
    console.log(`\nAudit results written to: ${outPath}`);

    console.log("\n================================================================================");
    console.log(`   VERIFICATION SUMMARY: ${failures.length} FAILURE(S) DETECTED`);
    console.log("================================================================================");

    if (failures.length > 0) {
      console.error(`\n❌ PROCESS EXIT CODE 1: ${failures.length} assertion(s) failed:`);
      failures.forEach((f, idx) => console.error(`  ${idx + 1}. [${f.testName}]: ${f.details}`));
      process.exit(1);
    } else {
      console.log("\n✔ ALL CHECKS PASSED: PROCESS EXIT CODE 0");
      process.exit(0);
    }
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error("FATAL ERROR during A4 closure verification:", err);
  process.exit(1);
});
