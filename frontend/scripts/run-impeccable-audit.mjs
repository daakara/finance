import puppeteer from "puppeteer";

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

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function runAudit() {
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

  const auditReport = {
    pagesAudited: [],
    findings: {
      a11y: [],
      performance: [],
      theming: [],
      responsive: [],
      antiPatterns: [],
    },
    metrics: {},
  };

  for (const p of PAGES) {
    console.log(`Auditing: ${p.name} (${p.path})...`);
    await page.setViewport({ width: 1280, height: 800 });
    await page.goto(`${BASE_URL}${p.path}`, { waitUntil: "load", timeout: 15000 });
    await sleep(1000);

    const pageAudit = await page.evaluate((pageId, pageName) => {
      const results = {
        pageId,
        pageName,
        domNodes: document.querySelectorAll("*").length,
        headings: [],
        missingLabels: [],
        missingAlt: [],
        touchTargetsUnder40: [],
        antiPatterns: {
          sideStripeBorders: [],
          gradientText: [],
          backdropBlur: [],
        },
        responsiveOverflows: [],
        colorTokens: {
          inlineStyles: 0,
        },
      };

      // 1. Accessibility Checks
      // Headings
      const headings = Array.from(document.querySelectorAll("h1, h2, h3, h4, h5, h6"));
      results.headings = headings.map((h) => ({
        tag: h.tagName,
        text: (h.textContent || "").trim().substring(0, 40),
      }));

      // Interactive controls accessible naming
      const buttons = Array.from(document.querySelectorAll("button, a[href], input, select, textarea"));
      for (const btn of buttons) {
        const text = (btn.textContent || "").trim();
        const ariaLabel = btn.getAttribute("aria-label");
        const ariaLabelledBy = btn.getAttribute("aria-labelledby");
        const title = btn.getAttribute("title");
        const hasAccessibleName = Boolean(text || ariaLabel || ariaLabelledBy || title || btn.querySelector("svg, img"));
        if (!hasAccessibleName) {
          results.missingLabels.push({
            tag: btn.tagName,
            className: btn.className,
            outerHTML: btn.outerHTML.substring(0, 100),
          });
        }
      }

      // Images alt text
      const images = Array.from(document.querySelectorAll("img"));
      for (const img of images) {
        if (!img.hasAttribute("alt")) {
          results.missingAlt.push({
            src: img.src,
            className: img.className,
          });
        }
      }

      // Touch targets < 40px
      for (const btn of buttons) {
        if (!btn.offsetParent) continue;
        const rect = btn.getBoundingClientRect();
        if ((rect.width > 0 && rect.width < 32) || (rect.height > 0 && rect.height < 32)) {
          results.touchTargetsUnder40.push({
            tag: btn.tagName,
            text: (btn.textContent || "").trim().substring(0, 25),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            className: (btn.className || "").toString().substring(0, 40),
          });
        }
      }

      // 2. Anti-Patterns Check (Impeccable strict rules)
      const allElements = Array.from(document.querySelectorAll("*"));
      for (const el of allElements) {
        const cs = window.getComputedStyle(el);

        // BAN 1: Side-stripe borders (border-left or border-right > 1px on cards/wells/alerts)
        const blw = parseFloat(cs.borderLeftWidth) || 0;
        const brw = parseFloat(cs.borderRightWidth) || 0;
        const btw = parseFloat(cs.borderTopWidth) || 0;
        const bbw = parseFloat(cs.borderBottomWidth) || 0;

        if (blw > 1 && btw <= 1 && bbw <= 1) {
          results.antiPatterns.sideStripeBorders.push({
            tag: el.tagName,
            borderLeft: `${blw}px ${cs.borderLeftStyle} ${cs.borderLeftColor}`,
            className: (el.className || "").toString().substring(0, 60),
            text: (el.textContent || "").trim().substring(0, 40),
          });
        }
        if (brw > 1 && btw <= 1 && bbw <= 1) {
          results.antiPatterns.sideStripeBorders.push({
            tag: el.tagName,
            borderRight: `${brw}px ${cs.borderRightStyle} ${cs.borderRightColor}`,
            className: (el.className || "").toString().substring(0, 60),
          });
        }

        // BAN 2: Gradient text (background-clip: text with linear-gradient)
        if (
          cs.webkitBackgroundClip === "text" ||
          cs.backgroundClip === "text" ||
          cs.getPropertyValue("-webkit-background-clip") === "text"
        ) {
          results.antiPatterns.gradientText.push({
            tag: el.tagName,
            text: (el.textContent || "").trim().substring(0, 40),
            className: (el.className || "").toString().substring(0, 60),
            background: cs.backgroundImage.substring(0, 50),
          });
        }

        // Glassmorphism / backdrop-filter
        if (cs.backdropFilter && cs.backdropFilter !== "none") {
          results.antiPatterns.backdropBlur.push({
            tag: el.tagName,
            backdropFilter: cs.backdropFilter,
            className: (el.className || "").toString().substring(0, 60),
          });
        }

        if (el.hasAttribute("style") && (el.getAttribute("style").includes("#") || el.getAttribute("style").includes("rgb"))) {
          results.colorTokens.inlineStyles++;
        }
      }

      return results;
    }, p.id, p.name);

    // Test mobile responsive overflow at 375px
    await page.setViewport({ width: 375, height: 667 });
    await sleep(500);
    const mobileOverflow = await page.evaluate(() => {
      const scrollWidth = document.documentElement.scrollWidth;
      const clientWidth = document.documentElement.clientWidth;
      return {
        scrollWidth,
        clientWidth,
        hasHorizontalScroll: scrollWidth > clientWidth + 1,
      };
    });
    pageAudit.mobileOverflow = mobileOverflow;

    auditReport.pagesAudited.push(pageAudit);
  }

  await browser.close();
  console.log(JSON.stringify(auditReport, null, 2));
}

runAudit().catch(console.error);
