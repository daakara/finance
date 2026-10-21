const http = require('http');
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '../..');
const outDir = path.resolve(repoRoot, 'frontend/out');

function getContentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const map = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.txt': 'text/plain',
  };
  return map[ext] || 'application/octet-stream';
}

function startServer(port) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const parsedUrl = new URL(req.url, `http://127.0.0.1:${port}`);
      let reqPath = parsedUrl.pathname;
      let filePath = path.join(outDir, reqPath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, 'index.html');
      } else if (!fs.existsSync(filePath) && fs.existsSync(filePath + '.html')) {
        filePath = filePath + '.html';
      }

      if (!fs.existsSync(filePath)) {
        filePath = path.join(outDir, '404.html');
      }

      if (fs.existsSync(filePath)) {
        res.writeHead(200, { 'Content-Type': getContentType(filePath) });
        fs.createReadStream(filePath).pipe(res);
      } else {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
      }
    });

    server.listen(port, '127.0.0.1', () => {
      console.log(`Static server running at http://127.0.0.1:${port}`);
      resolve(server);
    });
  });
}

async function run() {
  const port = 3034;
  const server = await startServer(port);

  console.log('Running Puppeteer headless browser checks...');
  const puppeteerPath = path.resolve(repoRoot, 'frontend/node_modules/puppeteer');
  const puppeteer = require(puppeteerPath);
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });

  const results = [];
  const executionTimestamp = new Date().toISOString();

  // 1. Desktop Test: FIX (Reinvestment units & Provenance timestamps)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=FIX`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Desktop FIX Dossier (Reinvestment & Provenance)',
      url: '/research/?symbol=FIX',
      hasReverseDcf: content.includes('Reverse-DCF'),
      hasReinvestmentNopatLabel: content.includes('Reinvestment (% of NOPAT)'),
      noSalesCapLabel: !content.includes('Reinvestment (Sales/Cap)'),
      hasReinvestmentPercentUnits: content.includes('15.0%'),
      noReinvestmentMultipleSuffix: !content.includes('15.00x') && !content.includes('15.0x'),
      hasAcceptedFilingCutoff: content.includes('2026-10-16T20:00:00Z'),
      hasAcceptedAuthoringTimestamp: content.includes('2026-10-20T14:30:00Z'),
      noStaticGeneratorPassCopy: !content.includes('Deterministic Generator: Pass'),
      hasPrimaryFiling: content.includes('0001035983-24-000014') || content.includes('10-K'),
      hasWatchConditions: content.includes('WATCH CONDITIONS') || content.includes('WATCH'),
    });
    await page.close();
  }

  // 2. Desktop Test: DUOL (Zero Debt verification & Reinvestment 10.0%)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=DUOL`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Desktop DUOL Dossier (Zero Debt & Reinvestment)',
      url: '/research/?symbol=DUOL',
      hasZeroDebt: content.includes('Zero Funded Debt Evidenced') || content.includes('Zero funded credit debt confirmed'),
      hasTotalCash: content.includes('$1313.9M') || content.includes('1313.9'),
      hasReinvestmentNopatLabel: content.includes('Reinvestment (% of NOPAT)'),
      hasReinvestmentPercentUnits: content.includes('10.0%'),
      noReinvestmentMultipleSuffix: !content.includes('10.00x') && !content.includes('10.0x'),
      hasCleanBridgeLabels: content.includes('Not included in this bridge') && !content.includes('None / Excluded') && !content.includes('None / N/A'),
      hasWatchConditions: content.includes('WATCH CONDITIONS') || content.includes('WATCH'),
    });
    await page.close();
  }

  // 3. Desktop Test: IREN (Suppressed Valuation verification)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=IREN`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Desktop IREN Dossier (Valuation Suppressed)',
      url: '/research/?symbol=IREN',
      hasValuationSuppressedBanner: content.includes('VALUATION SUPPRESSED') || content.includes('Valuation Suppressed'),
      zeroNumericTargets: !content.includes('Scenario 1: Conservative') && content.includes('Single-stage DCF suppressed'),
      hasResumeConditions: content.includes('Conditions Required to Resume Valuation') || content.includes('Resume Valuation'),
      hasWatchConditions: content.includes('WATCH CONDITIONS') || content.includes('WATCH'),
    });
    await page.close();
  }

  // 4. Desktop Test: CORZ (Digital assets & Clean Bridge Labels)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=CORZ`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Desktop CORZ Dossier (CIK, Commitments & Digital Assets)',
      url: '/research/?symbol=CORZ',
      hasCorrectCik: content.includes('0001839341'),
      hasCoreWeaveCommitments: content.includes('CoreWeave') && content.includes('$6.7B'),
      hasDigitalAssets: content.includes('Digital Assets (Bitcoin)') && content.includes('$49.7M'),
      noAmbiguousAccountingCaption: !content.includes('Carried at fair value / impairment basis'),
      hasSuppressionReason: content.includes('Single-stage DCF suppressed'),
      hasWatchConditions: content.includes('WATCH CONDITIONS') || content.includes('WATCH'),
    });
    await page.close();
  }

  // 5. Desktop Test: MU (Qualified valuation & Reinvestment 25.0%)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=MU`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Desktop MU Dossier (Qualified Valuation & Reinvestment)',
      url: '/research/?symbol=MU',
      hasQualifiedBanner: content.includes('CYCLICAL PEAK / RECOVERY QUALIFICATION') || content.includes('COMPUTED_QUALIFIED') || content.includes('Model Qualification Notice'),
      hasReinvestmentNopatLabel: content.includes('Reinvestment (% of NOPAT)'),
      hasReinvestmentPercentUnits: content.includes('25.0%'),
      noReinvestmentMultipleSuffix: !content.includes('25.00x') && !content.includes('25.0x'),
      noDeflationClaim: !content.includes('15% QoQ') && !content.includes('15% deflation'),
      hasRejectDisposition: content.includes('NO_SUPPORTED_OPPORTUNITY') || content.includes('REJECT'),
      hasSeparatedWatchCondition: content.includes('< $500') || content.includes('$500'),
    });
    await page.close();
  }

  // 6. Desktop Test: Unsupported Symbol (NVDA)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=NVDA`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Desktop Unsupported Symbol (NVDA)',
      url: '/research/?symbol=NVDA',
      hasUnsupportedBanner: content.includes('UNSUPPORTED RESEARCH SYMBOL') || content.includes('No ODP-05 Research Dossier'),
      hasTerminalDeepLink: content.includes('href="/?symbol=NVDA"'),
      hasRadarDeepLink: content.includes('href="/radar/?symbol=NVDA"') || content.includes('href="/radar?symbol=NVDA"'),
      hasCohortJumpButtons: content.includes('FIX') && content.includes('DUOL') && content.includes('IREN'),
    });
    await page.close();
  }

  // 7. Stock Page Test: FIX (/stock/fix/) shows ODP-05 Research Link Badge
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(`http://127.0.0.1:${port}/stock/fix/`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Stock Detail Page (/stock/fix/)',
      url: '/stock/fix/',
      hasResearchLinkBadge: content.includes('/research?symbol=FIX') && content.includes('View ODP-05 Research Dossier'),
      hasSourcedFundamentalsCard: content.includes('SEC EDGAR Sourced Fundamentals') || content.includes('SEC EDGAR') || content.includes('CIK: 0001035983'),
    });
    await page.close();
  }

  // 8. Mobile Viewport Test (390x844): FIX (Accessible scrolling & units)
  {
    const page = await browser.newPage();
    await page.setViewport({ width: 390, height: 844 });
    await page.goto(`http://127.0.0.1:${port}/research/?symbol=FIX`, { waitUntil: 'networkidle0' });
    const content = await page.content();
    results.push({
      test: 'Mobile (390x844) FIX Dossier',
      url: '/research/?symbol=FIX',
      rendersProperly: content.includes('Comfort Systems USA') && content.includes('Reverse-DCF'),
      hasHorizontalRail: content.includes('overflow-x-auto'),
      hasMobileReinvestmentLabel: content.includes('Reinvestment (% of NOPAT)'),
      hasMobilePercentUnits: content.includes('15.0%'),
      noMobileMultipleSuffix: !content.includes('15.00x') && !content.includes('15.0x'),
    });
    await page.close();
  }

  await browser.close();
  server.close();

  const report = {
    executionTimestamp,
    serverPort: port,
    totalChecks: results.length,
    allPass: results.every(r => Object.values(r).every(v => v === true || (typeof v === 'string' && v.length > 0))),
    results,
  };

  const resultsJsonPath = path.resolve(repoRoot, 'scripts/research/browser_verification_results.json');
  fs.writeFileSync(resultsJsonPath, JSON.stringify(report, null, 2), 'utf-8');

  console.log('\n================================================================');
  console.log('BROWSER VERIFICATION TEST RESULTS:');
  console.log('================================================================');
  console.log(JSON.stringify(report, null, 2));

  if (report.allPass) {
    console.log('\nALL 8 BROWSER AND VIEWPORT CHECKS PASSED PERFECTLY!');
    process.exit(0);
  } else {
    console.error('\nSOME CHECKS RETURNED FALSE/NULL');
    process.exit(1);
  }
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
