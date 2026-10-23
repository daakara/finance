/**
 * frontend/scripts/measure-viewport-bounding-boxes.mjs
 *
 * Measures bounding boxes and vertical placement at scroll position zero for:
 * 1. Shell framing / Navbar
 * 2. Collection navigation / rail
 * 3. Dossier header (symbol, company name)
 * 4. WATCH verdict badge
 * 5. Reference close tape
 * 6. Decisive reason paragraph
 *
 * Viewports evaluated:
 * - Desktop: 1280x800
 * - Mobile: 390x844 (iPhone 12/13/14)
 * - Compact Mobile: 360x740 (Standard Android)
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3338;
const OUT_DIR = path.resolve(__dirname, "../out");

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
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
      };

      res.writeHead(200, {
        "Content-Type": mimeTypes[ext] || "application/octet-stream",
        "Cache-Control": "no-cache",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.on("error", reject);
    server.listen(PORT, () => {
      console.log(`[SERVER] Serving static export on http://localhost:${PORT}`);
      resolve(server);
    });
  });
}

async function inspectViewports() {
  const server = await startServer();
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  const viewports = [
    { name: "Desktop (1280x800)", width: 1280, height: 800, isMobile: false },
    { name: "Mobile (390x844)", width: 390, height: 844, isMobile: true, hasTouch: true },
    { name: "Compact Mobile (360x740)", width: 360, height: 740, isMobile: true, hasTouch: true },
  ];

  const results = {};

  try {
    const page = await browser.newPage();

    for (const vp of viewports) {
      await page.setViewport(vp);
      const url = `http://localhost:${PORT}/research?symbol=DECK`;
      await page.goto(url, { waitUntil: "networkidle0" });

      // Detect and dismiss any onboarding modal or overlay
      const overlayStatus = await page.evaluate(() => {
        const dialog = document.querySelector('[role="dialog"]');
        const wasPresent = !!dialog;
        if (dialog) {
          const closeBtn = dialog.querySelector('button');
          if (closeBtn) closeBtn.click();
        }
        window.scrollTo(0, 0);
        return {
          wasPresent,
          scrollY: window.scrollY,
        };
      });

      await new Promise((r) => setTimeout(r, 150));

      const metrics = await page.evaluate((vpHeight) => {
        function getBox(el) {
          if (!el) return null;
          const r = el.getBoundingClientRect();
          return {
            x: Math.round(r.x),
            y: Math.round(r.y),
            top: Math.round(r.top),
            bottom: Math.round(r.bottom),
            left: Math.round(r.left),
            right: Math.round(r.right),
            width: Math.round(r.width),
            height: Math.round(r.height),
            inInitialViewport: r.top >= 0 && r.top < vpHeight,
            fullyVisible: r.top >= 0 && r.bottom <= vpHeight,
          };
        }

        // 1. Shell framing / Navbar
        // In the layout, Navbar is at the top (<header> or <nav>)
        const navbarEl = document.querySelector('header nav, nav[aria-label="Global"], nav');

        // Surface Header Banner (Research page title / qualification note)
        const surfaceBannerEl = document.querySelector('main header');

        // 2. Collection navigation / rail
        const desktopAsideEl = document.querySelector('aside');
        const mobileRailEl = document.querySelector('.lg\\:hidden.space-y-2');

        // 3. Dossier header (symbol, company name)
        // Inside DeckDecisionFirstView, it's the article header
        const dossierHeaderEl = document.querySelector('article header');
        const dossierSymbolH1El = document.querySelector('article header h1');

        // 4. WATCH verdict badge
        // Looking for badge with WATCH text
        const allSpans = Array.from(document.querySelectorAll('article header span, article header div'));
        const verdictBadgeEl = allSpans.find(s => s.innerText && s.innerText.includes('WATCH') && s.className.includes('bg-amber-950'));

        // 5. Reference close tape
        const refCloseTapeEl = allSpans.find(s => s.innerText && s.innerText.includes('USD') && s.className.includes('font-mono'));

        // 6. Decisive reason paragraph
        const decisiveReasonSection = document.querySelector('section[aria-label="WATCH Decisive Reason"]');
        const decisiveReasonParagraphEl = decisiveReasonSection ? decisiveReasonSection.querySelector('p') : null;

        return {
          scrollY: window.scrollY,
          navbar: getBox(navbarEl),
          surfaceBanner: getBox(surfaceBannerEl),
          collectionNavDesktop: getBox(desktopAsideEl),
          collectionNavMobile: getBox(mobileRailEl),
          dossierHeader: getBox(dossierHeaderEl),
          dossierSymbolH1: getBox(dossierSymbolH1El),
          verdictBadge: getBox(verdictBadgeEl),
          refCloseTape: getBox(refCloseTapeEl),
          decisiveReasonSection: getBox(decisiveReasonSection),
          decisiveReasonParagraph: getBox(decisiveReasonParagraphEl),
        };
      }, vp.height);

      results[vp.name] = {
        viewport: vp,
        overlayDismissed: overlayStatus.wasPresent,
        scrollYAtZero: metrics.scrollY === 0,
        metrics,
      };
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log(JSON.stringify(results, null, 2));
}

inspectViewports().catch(console.error);
