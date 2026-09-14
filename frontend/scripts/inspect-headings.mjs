import puppeteer from "puppeteer";

async function run() {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    if (req.url().includes("matomo") || req.url().includes("analytics.php")) req.abort();
    else req.continue();
  });

  const urls = [
    { name: "Analysis", url: "http://localhost:3000/?symbol=NVDA&mode=standard" },
    { name: "Radar", url: "http://localhost:3000/radar/?mode=standard" },
    { name: "Trade Plan", url: "http://localhost:3000/setups/?ticker=NVDA&mode=standard" },
    { name: "Portfolio", url: "http://localhost:3000/portfolio/?mode=standard" },
    { name: "Guide", url: "http://localhost:3000/guide/?mode=standard" },
    { name: "Journal", url: "http://localhost:3000/journal/?mode=standard" },
    { name: "Performance", url: "http://localhost:3000/performance/?mode=standard" },
  ];

  for (const u of urls) {
    await page.goto(u.url, { waitUntil: "load" });
    const outline = await page.evaluate(() => {
      const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6, [role="heading"]'));
      return headings.map((h) => ({
        tag: h.tagName,
        ariaLevel: h.getAttribute("aria-level"),
        text: (h.textContent || "").trim().replace(/\s+/g, " ").substring(0, 60),
        isFooter: Boolean(h.closest("footer")),
      }));
    });
    console.log(`=== ${u.name} Headings ===`);
    console.log(JSON.stringify(outline, null, 2));
  }
  await browser.close();
}

run().catch(console.error);
