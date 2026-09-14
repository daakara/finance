import puppeteer from "puppeteer";

const VIEWPORTS = [
  { name: "iPhone SE / 375px", width: 375, height: 667 },
  { name: "iPhone 12/13/14 / 390px", width: 390, height: 844 },
];

const PAGES = [
  { name: "Analysis Workstation", url: "http://localhost:3000/?symbol=NVDA&mode=standard" },
  { name: "Radar Screener", url: "http://localhost:3000/radar/?mode=standard" },
  { name: "Trade Plan Setups", url: "http://localhost:3000/setups/?ticker=NVDA&mode=standard" },
  { name: "Portfolio", url: "http://localhost:3000/portfolio/?mode=standard" },
];

async function measureViewportPerformance() {
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

  const client = await page.target().createCDPSession();
  await client.send("Performance.enable");

  const results = [];

  for (const vp of VIEWPORTS) {
    await page.setViewport({ width: vp.width, height: vp.height, deviceScaleFactor: 2, isMobile: true, hasTouch: true });

    for (const pg of PAGES) {
      await page.goto(pg.url, { waitUntil: "domcontentloaded", timeout: 15000 });
      await new Promise((r) => setTimeout(r, 600));

      const metricsBefore = await client.send("Performance.getMetrics");

      const scrollMetrics = await page.evaluate(async () => {
        const start = performance.now();
        let frameCount = 0;
        let lastTimestamp = performance.now();
        const frameDeltas = [];

        function checkFrame(time) {
          frameDeltas.push(time - lastTimestamp);
          lastTimestamp = time;
          frameCount++;
        }

        // Scroll test
        for (let i = 0; i < 5; i++) {
          window.scrollBy(0, 200);
          await new Promise((r) => requestAnimationFrame((t) => {
            checkFrame(t);
            r();
          }));
          await new Promise((r) => setTimeout(r, 30));
          window.scrollBy(0, -200);
          await new Promise((r) => requestAnimationFrame((t) => {
            checkFrame(t);
            r();
          }));
          await new Promise((r) => setTimeout(r, 30));
        }

        const duration = performance.now() - start;
        const avgFrameTime = frameDeltas.reduce((a, b) => a + b, 0) / (frameDeltas.length || 1);
        const jankFrames = frameDeltas.filter((d) => d > 33.33).length;

        return {
          durationMs: Math.round(duration),
          framesSampled: frameCount,
          avgFrameTimeMs: Number(avgFrameTime.toFixed(2)),
          jankFrames,
          fps: Math.round(1000 / (avgFrameTime || 16.67)),
        };
      });

      const metricsAfter = await client.send("Performance.getMetrics");

      function getMetricDiff(name) {
        const b = metricsBefore.metrics.find((m) => m.name === name)?.value || 0;
        const a = metricsAfter.metrics.find((m) => m.name === name)?.value || 0;
        return a - b;
      }

      const layoutDuration = getMetricDiff("LayoutDuration");
      const recalcDuration = getMetricDiff("RecalcStyleDuration");

      results.push({
        viewport: vp.name,
        width: vp.width,
        page: pg.name,
        scrollMetrics,
        layoutDurationSeconds: Number(layoutDuration.toFixed(4)),
        recalcDurationSeconds: Number(recalcDuration.toFixed(4)),
      });
    }
  }

  await browser.close();
  console.log(JSON.stringify(results, null, 2));
}

measureViewportPerformance().catch(console.error);
