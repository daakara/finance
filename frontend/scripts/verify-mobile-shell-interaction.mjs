/**
 * frontend/scripts/verify-mobile-shell-interaction.mjs
 *
 * Verifies mobile shell interaction, switching into DECK, out to FIX, and back,
 * accessible disclosure toggle, and desktop preservation.
 * Governed by ARX Closure Model v2.
 *
 * Supports opt-in intentional negative control:
 *   node scripts/verify-mobile-shell-interaction.mjs --negative-control
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3340;
const OUT_DIR = path.resolve(__dirname, "../out");
const isNegativeControl = process.argv.includes("--negative-control");

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

  function recordCheck(checkName, passed, detail = "") {
    totalChecks++;
    if (passed) {
      console.log(`[PASS] ${checkName}${detail ? `: ${detail}` : ""}`);
    } else {
      failures.push({ name: checkName, detail });
      console.error(`[FAIL] ${checkName}${detail ? `: ${detail}` : ""}`);
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

    // 1. Mobile 390x844 on DECK
    await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
    await page.goto(`http://localhost:${PORT}/research?symbol=DECK`, { waitUntil: "networkidle0" });

    // Negative control: temporarily mutate test browser DOM before visibility check
    if (isNegativeControl) {
      console.log("[NEGATIVE CONTROL] Mutating test browser DOM: hiding compact header to verify failure detection...");
      await page.evaluate(() => {
        const el = document.querySelector("header.lg\\:hidden");
        if (el) el.style.display = "none";
      });
    }

    // Check 1: Assert compact mobile header is visible and height < 250px
    const compactHeaderResult = await page.evaluate(() => {
      const el = document.querySelector("header.lg\\:hidden");
      if (!el) return { passed: false, error: "Compact header element 'header.lg:hidden' not found" };
      const r = el.getBoundingClientRect();
      const passed = r.height > 0 && r.height < 250;
      return {
        passed,
        height: r.height,
        error: passed ? null : `unexpected height ${Math.round(r.height)}px (expected 0 < height < 250)`,
      };
    });

    recordCheck(
      "Mobile DECK: Compact header rendered (height < 250px)",
      compactHeaderResult.passed,
      compactHeaderResult.passed ? `height: ${Math.round(compactHeaderResult.height)}px` : compactHeaderResult.error
    );

    // Negative control: restore mutation so subsequent checks operate on valid DOM
    if (isNegativeControl) {
      console.log("[NEGATIVE CONTROL] Restoring test browser DOM: unhiding compact header...");
      await page.evaluate(() => {
        const el = document.querySelector("header.lg\\:hidden");
        if (el) el.style.display = "";
      });
    }

    // Check 2: Assert details disclosure control exists and can open
    const scopeDisclosureResult = await page.evaluate(() => {
      const details = document.querySelector("header.lg\\:hidden details");
      if (!details) return { passed: false, error: "Disclosure element 'header.lg:hidden details' not found" };
      const summary = details.querySelector("summary");
      if (!summary) return { passed: false, error: "Summary toggle not found inside details" };
      summary.click();
      const open = details.hasAttribute("open");
      return {
        passed: open,
        error: open ? null : "Details element did not open after clicking summary",
      };
    });

    recordCheck(
      "Mobile DECK: Collection scope disclosure opens on click",
      scopeDisclosureResult.passed,
      scopeDisclosureResult.passed ? "expanded successfully" : scopeDisclosureResult.error
    );

    // Check 3: Switch to FIX
    console.log("Navigating via link click to FIX...");
    const fixLinkResult = await page.evaluate(() => {
      const links = Array.from(document.querySelectorAll(".lg\\:hidden a"));
      const fixLink = links.find((a) => a.href.includes("symbol=FIX"));
      if (!fixLink) return { found: false, error: "Navigation link with symbol=FIX not found in mobile header" };
      fixLink.click();
      return { found: true };
    });

    if (!fixLinkResult.found) {
      recordCheck("Switch to FIX: Navigation link click", false, fixLinkResult.error);
    } else {
      await page.waitForFunction(() => window.location.search.includes("symbol=FIX"), { timeout: 3000 });
      console.log(`Current URL after click FIX: ${page.url()}`);

      const fixResult = await page.evaluate(() => {
        const isFix = window.location.search.includes("symbol=FIX");
        if (!isFix) return { passed: false, error: `URL does not include symbol=FIX (search: ${window.location.search})` };
        const compactHeader = document.querySelector("header.lg\\:hidden");
        if (!compactHeader) return { passed: false, error: "Compact header 'header.lg:hidden' not found on FIX" };
        const r = compactHeader.getBoundingClientRect();
        if (r.height <= 0 || r.height >= 250) {
          return { passed: false, error: `Compact header height ${Math.round(r.height)}px expected 0 < height < 250` };
        }
        const decisionHeading = document.getElementById("decision-first-heading-FIX");
        if (!decisionHeading) return { passed: false, error: "Decision-first heading not found on FIX" };
        return { passed: true, height: r.height };
      });

      recordCheck(
        "Switch to FIX: Compact header active (< 250px) & Decision-First view loaded",
        fixResult.passed,
        fixResult.passed ? `height: ${Math.round(fixResult.height)}px < 250px` : fixResult.error
      );
    }

    // Check 4: Switch back to DECK
    console.log("Navigating via link click back to DECK...");
    const deckLinkResult = await page.evaluate(() => {
      const links = Array.from(document.querySelectorAll(".lg\\:hidden a"));
      const deckLink = links.find((a) => a.href.includes("symbol=DECK"));
      if (!deckLink) return { found: false, error: "Navigation link with symbol=DECK not found in mobile header" };
      deckLink.click();
      return { found: true };
    });

    if (!deckLinkResult.found) {
      recordCheck("Switch back to DECK: Navigation link click", false, deckLinkResult.error);
    } else {
      await page.waitForFunction(() => window.location.search.includes("symbol=DECK"), { timeout: 3000 });
      await new Promise((r) => setTimeout(r, 300));
      console.log(`Current URL after click DECK: ${page.url()}`);

      const deckRestoredResult = await page.evaluate(() => {
        const isDeck = window.location.search.includes("symbol=DECK");
        if (!isDeck) return { passed: false, error: `URL does not include symbol=DECK (search: ${window.location.search})` };
        const compactHeader = document.querySelector("header.lg\\:hidden");
        if (!compactHeader) return { passed: false, error: "Compact header 'header.lg:hidden' not found on DECK" };
        const r = compactHeader.getBoundingClientRect();
        if (r.height <= 0 || r.height >= 250) {
          return { passed: false, error: `Compact header height ${Math.round(r.height)}px expected 0 < height < 250` };
        }
        return { passed: true, height: r.height };
      });

      recordCheck(
        "Switch back to DECK: Compact header restored",
        deckRestoredResult.passed,
        deckRestoredResult.passed ? `height: ${Math.round(deckRestoredResult.height)}px < 250px` : deckRestoredResult.error
      );
    }

    // Check 5: Desktop preservation check (1280x800)
    await page.setViewport({ width: 1280, height: 800, isMobile: false });
    await page.goto(`http://localhost:${PORT}/research?symbol=DECK`, { waitUntil: "networkidle0" });

    const desktopResult = await page.evaluate(() => {
      const desktopHeader = document.querySelector("header.hidden.lg\\:block");
      if (!desktopHeader) return { passed: false, error: "Desktop header 'header.hidden.lg:block' not found" };
      const r = desktopHeader.getBoundingClientRect();
      if (!r || r.height <= 150 || r.top < 100) {
        return {
          passed: false,
          error: `Desktop header expected height > 150 and top >= 100, got height=${Math.round(r?.height ?? 0)}px, top=${Math.round(r?.top ?? 0)}px`,
        };
      }
      return { passed: true, height: r.height, top: r.top };
    });

    recordCheck(
      "Desktop DECK: Standard desktop header preserved",
      desktopResult.passed,
      desktopResult.passed ? `height: ${Math.round(desktopResult.height)}px, top: ${Math.round(desktopResult.top)}px` : desktopResult.error
    );

    console.log("\n========================================================================");
    console.log(`INTERACTION VERIFICATION SUMMARY: ${totalChecks - failures.length}/${totalChecks} CHECKS PASSED`);
    console.log("========================================================================");

    if (failures.length > 0) {
      console.error(`\n❌ [FAIL-CLOSED] ${failures.length} check(s) failed:`);
      for (const f of failures) {
        console.error(`  - ${f.name}: ${f.detail}`);
      }
    } else {
      console.log("\nALL INTERACTION AND PRESERVATION TESTS PASSED SUCCESSFULLY! ✅");
    }
  } catch (err) {
    console.error("\n[FATAL] Unhandled error during verification run:", err);
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
