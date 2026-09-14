import puppeteer from "puppeteer";
import { computeAccessibleName } from "./accessible-name-evaluator.mjs";

const BASE_URL = "http://localhost:3000";
const PAGES = [
  { id: "root", name: "Analysis / Workstation", path: "/?symbol=NVDA&mode=standard" },
  { id: "radar", name: "Radar / Screener", path: "/radar/?mode=standard" },
  { id: "setups", name: "Trade Plan / Setups", path: "/setups/?ticker=NVDA&mode=standard" },
  { id: "portfolio", name: "Portfolio / Risk Manager", path: "/portfolio/?mode=standard" },
  { id: "guide", name: "Platform Guide", path: "/guide/?mode=standard" },
  { id: "journal", name: "Journal (Deferred R1)", path: "/journal/?mode=standard" },
  { id: "performance", name: "Performance (Deferred R1)", path: "/performance/?mode=standard" },
];

const VIEWPORTS = [
  { name: "320px", width: 320, height: 568 },
  { name: "375px", width: 375, height: 667 },
  { name: "390px", width: 390, height: 844 },
];

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function auditPage(page, p) {
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto(`${BASE_URL}${p.path}`, { waitUntil: "domcontentloaded", timeout: 15000 });
  await sleep(1000);

  // 1. Heading structure extraction
  const headingTree = await page.evaluate(() => {
    const headings = Array.from(document.querySelectorAll("h1, h2, h3, h4, h5, h6"));
    return headings.map((h) => ({
      level: parseInt(h.tagName.substring(1), 10),
      tag: h.tagName,
      text: (h.textContent || "").trim().replace(/\s+/g, " ").substring(0, 60),
    }));
  });

  const headingSkips = [];
  let prevLevel = 0;
  for (const h of headingTree) {
    if (prevLevel > 0 && h.level > prevLevel + 1) {
      headingSkips.push({
        from: `H${prevLevel}`,
        to: `H${h.level}`,
        text: h.text,
      });
    }
    prevLevel = h.level;
  }

  const h1s = headingTree.filter((h) => h.level === 1);

  // 2. Accessible Names using W3C algorithm
  const missingLabels = await page.evaluate((fnSource) => {
    const computeAccName = new Function("return " + fnSource)();
    const controls = Array.from(document.querySelectorAll("button, a[href], input, select, textarea, [role='button'], [role='tab'], [role='radio'], [role='menuitem']"));
    const missing = [];
    for (const c of controls) {
      if (!c.offsetParent && c.offsetWidth === 0 && c.offsetHeight === 0) continue;
      const accName = computeAccName(c, document);
      if (!accName || accName.trim().length === 0) {
        missing.push({
          tag: c.tagName,
          id: c.id || "",
          className: (c.className || "").toString().substring(0, 40),
          role: c.getAttribute("role") || "",
          text: (c.textContent || "").trim().substring(0, 30),
        });
      }
    }
    return {
      auditedCount: controls.length,
      missingCount: missing.length,
      missingLabels: missing,
    };
  }, computeAccessibleName.toString());

  // 3. Multi-viewport touch target evaluation
  const touchResults = {};
  for (const vp of VIEWPORTS) {
    await page.setViewport({ width: vp.width, height: vp.height, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await sleep(400);

    const vpMetrics = await page.evaluate(() => {
      const controls = Array.from(document.querySelectorAll("button, a[href], input, select, textarea, [role='button'], [role='tab'], [role='radio']"));
      const wcagFailures = [];
      const ergonomicsWarnings = [];

      for (const el of controls) {
        if (!el.offsetParent) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;
        const width = Math.round(rect.width);
        const height = Math.round(rect.height);

        const info = {
          tag: el.tagName,
          text: (el.textContent || el.getAttribute("aria-label") || "").trim().substring(0, 24),
          width,
          height,
          id: el.id || "",
          className: (el.className || "").toString().substring(0, 30),
        };

        if (width < 24 || height < 24) {
          wcagFailures.push(info);
        } else if (width < 40 || height < 40) {
          ergonomicsWarnings.push(info);
        }
      }

      return {
        wcagFailures,
        ergonomicsWarnings,
      };
    });

    touchResults[vp.name] = vpMetrics;
  }

  // 4. Anti-patterns
  await page.setViewport({ width: 1280, height: 800 });
  const antiPatterns = await page.evaluate(() => {
    const all = Array.from(document.querySelectorAll("*"));
    let sideStripes = 0;
    let gradientTexts = 0;
    let backdropBlurs = 0;

    for (const el of all) {
      const cs = window.getComputedStyle(el);
      const blw = parseFloat(cs.borderLeftWidth) || 0;
      const brw = parseFloat(cs.borderRightWidth) || 0;
      const btw = parseFloat(cs.borderTopWidth) || 0;
      const bbw = parseFloat(cs.borderBottomWidth) || 0;

      if ((blw > 1 && btw <= 1 && bbw <= 1) || (brw > 1 && btw <= 1 && bbw <= 1)) {
        sideStripes++;
      }

      if (cs.webkitBackgroundClip === "text" || cs.backgroundClip === "text") {
        gradientTexts++;
      }

      if (cs.backdropFilter && cs.backdropFilter !== "none") {
        backdropBlurs++;
      }
    }

    return {
      sideStripes,
      gradientTexts,
      backdropBlurs,
    };
  });

  return {
    id: p.id,
    name: p.name,
    path: p.path,
    headings: {
      tree: headingTree,
      h1Count: h1s.length,
      h1Titles: h1s.map((h) => h.text),
      skips: headingSkips,
    },
    accessibleNames: missingLabels,
    touchTargets: touchResults,
    antiPatterns,
  };
}

async function runHardenedAudit() {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    if (req.url().includes("matomo") || req.url().includes("analytics.php")) {
      req.abort();
    } else {
      req.continue();
    }
  });

  const reports = [];
  for (const p of PAGES) {
    console.log(`[AUDIT] Auditing ${p.name}...`);
    const r = await auditPage(page, p);
    reports.push(r);
  }

  await browser.close();

  import("fs").then(({ writeFileSync }) => {
    writeFileSync("c:/Users/akara/Documents/Projects/finance/frontend/scripts/hardened-audit-report.json", JSON.stringify(reports, null, 2), "utf8");
    console.log("Audit report written to hardened-audit-report.json");
  });
}

runHardenedAudit().catch(console.error);
