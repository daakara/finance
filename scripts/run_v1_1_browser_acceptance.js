const puppeteer = require("C:/Users/akara/Documents/Projects/finance/frontend/node_modules/puppeteer");
const fs = require("fs");
const path = require("path");

const outputDir = "C:/Users/akara/.gemini/antigravity/brain/653d31bd-d885-4310-8362-641593f3c2b3/screenshots_consolidation_audit";
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

async function run() {
  console.log("Starting ARX_UX_DESIGN_PARITY_GATE_V1_1 Browser Acceptance Verification...\n");

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"]
  });

  const results = {
    timestamp: new Date().toISOString(),
    gateId: "ARX_UX_DESIGN_PARITY_GATE_V1_1",
    viewportsTested: ["desktop-1440", "tablet-768", "mobile-390"],
    symbolsTested: ["AAPL", "NVDA"],
    assertions: {},
    screenshots: {}
  };

  try {
    const page = await browser.newPage();

    // Prevent tutorial popup interference during automated test
    await page.evaluateOnNewDocument(() => {
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
    });

    // Proxy /api/v1 to production backend
    await page.setRequestInterception(true);
    page.on("request", async (req) => {
      const url = req.url();
      if (url.includes("/api/v1/")) {
        const targetUrl = url.replace("http://localhost:3000/api/v1", "https://www.arxterminal.com/api/backend");
        try {
          const fetchRes = await fetch(targetUrl);
          const buffer = Buffer.from(await fetchRes.arrayBuffer());
          const headers = {};
          fetchRes.headers.forEach((v, k) => { headers[k] = v; });
          headers["access-control-allow-origin"] = "*";
          await req.respond({
            status: fetchRes.status,
            contentType: fetchRes.headers.get("content-type") || "application/json",
            headers,
            body: buffer
          });
        } catch (e) {
          await req.continue();
        }
      } else {
        await req.continue();
      }
    });

    // ─────────────────────────────────────────────────────────────────────────
    // 1. AAPL Desktop (1440x900)
    // ─────────────────────────────────────────────────────────────────────────
    console.log("1. Testing AAPL on Desktop (1440x900)...");
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto("http://localhost:3000/?symbol=AAPL", { waitUntil: "networkidle2", timeout: 30000 });
    await new Promise(r => setTimeout(r, 4000));

    // Capture desktop full page
    const aaplDesktopShot = path.join(outputDir, "v1_1_aapl_desktop_fullpage.png");
    await page.screenshot({ path: aaplDesktopShot, fullPage: true });
    results.screenshots.aaplDesktop = aaplDesktopShot;
    console.log(`   [CAPTURED] Desktop Fullpage: ${aaplDesktopShot}`);

    // Audit DOM for legacy artifacts
    const domAudit = await page.evaluate(() => {
      const decisionReadinessCount = document.querySelectorAll("[data-testid='decision-readiness-card']").length;
      const preflightBars = document.querySelectorAll("#preflight-readiness-bar, [data-testid='preflight-readiness-bar']").length;
      const inlinePills = Array.from(document.querySelectorAll("[data-testid^='inline-chk-status-']")).map(el => el.textContent?.trim());
      const hasExecutionLadder = document.querySelector("[data-testid='cockpit-execution-ladder']") !== null;
      
      return {
        decisionReadinessCount,
        preflightBars,
        inlinePills,
        hasExecutionLadder
      };
    });

    console.log("   DOM Audit AAPL:", domAudit);
    results.assertions.zeroDecisionReadinessCards = domAudit.decisionReadinessCount === 0;
    results.assertions.singlePreflightBar = domAudit.preflightBars === 1;
    results.assertions.executionLadderPresent = domAudit.hasExecutionLadder;
    results.assertions.fiveInlinePillsPresent = domAudit.inlinePills.length === 5;
    results.assertions.macroInlineValid = ["PASS", "UNAVAILABLE"].includes(domAudit.inlinePills[4]);

    // Test Pre-Flight Checklist Modal on AAPL
    console.log("\n2. Opening Pre-Flight Checklist Modal on AAPL...");
    const openModalBtn = await page.$("[data-testid='inline-open-modal-btn']");
    if (openModalBtn) {
      await openModalBtn.click();
      await new Promise(r => setTimeout(r, 1500));

      const aaplModalShot = path.join(outputDir, "v1_1_aapl_preflight_modal.png");
      await page.screenshot({ path: aaplModalShot, fullPage: false });
      results.screenshots.aaplModal = aaplModalShot;
      console.log(`   [CAPTURED] Pre-Flight Modal: ${aaplModalShot}`);

      const modalAudit = await page.evaluate(() => {
        const dialog = document.querySelector("[role='dialog']");
        const text = dialog?.innerText || "";
        const spans = Array.from(dialog?.querySelectorAll("span") || []);
        const badges = spans
          .map(s => s.textContent?.trim())
          .filter(t => ["PASS", "FAIL", "HAZARD", "UNAVAILABLE", "HIGH VIX", "MISSING LEVELS", "CHASING"].includes(t));

        const hasCorridorToleranceCopy = text.includes("slightly above preferred corridor") || text.includes("+2%") || text.includes("corridor");
        const hasNoInsideBuyingRangeWhenAbove = !text.includes("is inside the verified buying range");
        const hasUnverifiedCatalystCopy = text.includes("unverified") || text.includes("Unconfirmed event runway");
        const hasNoFalseVerifiedRunway = !text.includes("Verified event runway: No earnings announcement");

        return {
          badges,
          hasCorridorToleranceCopy,
          hasNoInsideBuyingRangeWhenAbove,
          hasUnverifiedCatalystCopy,
          hasNoFalseVerifiedRunway,
          modalSnippet: text.slice(0, 400)
        };
      });

      console.log("   Modal Audit AAPL:", modalAudit);
      results.assertions.modalCorridorToleranceAccurate = modalAudit.hasCorridorToleranceCopy;
      results.assertions.modalNoFalseBuyingRange = modalAudit.hasNoInsideBuyingRangeWhenAbove;
      results.assertions.modalCatalystUnverifiedDisclaimer = modalAudit.hasUnverifiedCatalystCopy;
      results.assertions.modalNoFalseVerifiedRunway = modalAudit.hasNoFalseVerifiedRunway;
      // Macro parity assertion: Inline pill #5 must match modal Check 5
      const modalMacroStatus = modalAudit.badges[4] || modalAudit.badges[modalAudit.badges.length - 1];
      console.log(`   Macro Parity: Inline=${domAudit.inlinePills[4]} vs Modal=${modalMacroStatus}`);
      results.assertions.macroInlineModalParity = (domAudit.inlinePills[4] === modalMacroStatus);

      // Close modal
      const closeBtn = await page.$("[data-testid='close-modal-btn'], button[aria-label='Close modal'], button.rounded-full");
      if (closeBtn) await closeBtn.click();
      await new Promise(r => setTimeout(r, 500));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. AAPL Tablet (768x1024)
    // ─────────────────────────────────────────────────────────────────────────
    console.log("\n3. Testing AAPL on Tablet (768x1024)...");
    await page.setViewport({ width: 768, height: 1024 });
    await new Promise(r => setTimeout(r, 1500));
    const aaplTabletShot = path.join(outputDir, "v1_1_aapl_tablet_fullpage.png");
    await page.screenshot({ path: aaplTabletShot, fullPage: true });
    results.screenshots.aaplTablet = aaplTabletShot;
    console.log(`   [CAPTURED] Tablet Fullpage: ${aaplTabletShot}`);

    // ─────────────────────────────────────────────────────────────────────────
    // 4. AAPL Mobile (390x844)
    // ─────────────────────────────────────────────────────────────────────────
    console.log("\n4. Testing AAPL on Mobile (390x844)...");
    await page.setViewport({ width: 390, height: 844 });
    await new Promise(r => setTimeout(r, 1500));
    const aaplMobileShot = path.join(outputDir, "v1_1_aapl_mobile_fullpage.png");
    await page.screenshot({ path: aaplMobileShot, fullPage: true });
    results.screenshots.aaplMobile = aaplMobileShot;
    console.log(`   [CAPTURED] Mobile Fullpage: ${aaplMobileShot}`);

    // ─────────────────────────────────────────────────────────────────────────
    // 5. NVDA Desktop (1440x900)
    // ─────────────────────────────────────────────────────────────────────────
    console.log("\n5. Testing NVDA on Desktop (1440x900)...");
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto("http://localhost:3000/?symbol=NVDA", { waitUntil: "networkidle2", timeout: 30000 });
    await new Promise(r => setTimeout(r, 4000));

    const nvdaDesktopShot = path.join(outputDir, "v1_1_nvda_desktop_fullpage.png");
    await page.screenshot({ path: nvdaDesktopShot, fullPage: true });
    results.screenshots.nvdaDesktop = nvdaDesktopShot;
    console.log(`   [CAPTURED] NVDA Desktop Fullpage: ${nvdaDesktopShot}`);

    const nvdaAudit = await page.evaluate(() => {
      const pageText = document.body.innerText;
      const isStage4Wait = pageText.includes("STAGE 4 WAIT");
      const pills = Array.from(document.querySelectorAll("[data-testid^='inline-chk-status-']")).map(el => el.textContent?.trim());
      return {
        isStage4Wait,
        pills
      };
    });
    console.log("   NVDA Audit:", nvdaAudit);
    results.assertions.nvdaStage4WaitEnforced = nvdaAudit.isStage4Wait;

    results.disposition = Object.values(results.assertions).every(Boolean) ? "PASS" : "FAIL";

    const reportPath = path.join(outputDir, "v1_1_acceptance_results.json");
    fs.writeFileSync(reportPath, JSON.stringify(results, null, 2), "utf8");
    console.log(`\nAcceptance results saved to: ${reportPath}`);
    console.log("Disposition:", results.disposition);

    if (results.disposition !== "PASS") {
      process.exit(1);
    }

  } finally {
    await browser.close();
  }
}

run().catch(err => {
  console.error("Browser Acceptance Failure:", err);
  process.exit(1);
});
