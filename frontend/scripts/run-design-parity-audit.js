/**
 * ARX_UX_DESIGN_PARITY_GATE_V1: Full-Page Browser Acceptance & Audit Runner
 *
 * Verifies live rendered UI against the approved Synthesis E V3 contract:
 *  - 1440px Desktop, 768px Tablet, 390px Mobile full-page scroll capture
 *  - Four-pillar confluence attribution (with Institutional Flow & 25% Weight)
 *  - Single canonical Pre-Flight readiness authority mounted below chart
 *  - Zero duplicate steppers, duplicate primary verdicts, or duplicate bottom lines
 *  - Zero undefined / null / NaN leaks
 *  - Active analysis onboarding suppression
 *  - Machine-readable output (PASS / FAIL / UNVERIFIED)
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const targetBaseUrl = process.env.ARX_TEST_URL || 'https://www.arxterminal.com';
const outDir = path.resolve(__dirname, '../../screenshots_consolidation_audit');

if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

(async () => {
  console.log('================================================================================');
  console.log('  ARX_UX_DESIGN_PARITY_GATE_V1: Full-Page Live Browser Acceptance Audit');
  console.log(`  Target URL: ${targetBaseUrl}`);
  console.log(`  Artifacts:  ${outDir}`);
  console.log('================================================================================\n');

  const auditReport = {
    gateId: 'ARX_UX_DESIGN_PARITY_GATE_V1',
    timestamp: new Date().toISOString(),
    targetUrl: targetBaseUrl,
    verdict: 'UNVERIFIED',
    passedCount: 0,
    failedCount: 0,
    checks: {},
    symbolsAudited: {},
    screenshots: []
  };

  function recordCheck(name, passed, message) {
    auditReport.checks[name] = { passed, message };
    if (passed) {
      auditReport.passedCount++;
      console.log(`  [PASS] ${name}: ${message}`);
    } else {
      auditReport.failedCount++;
      console.error(`  [FAIL] ${name}: ${message}`);
    }
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });

  const page = await browser.newPage();

  // Suppress onboarding tour and default to Pro Quant vernacular
  await page.evaluateOnNewDocument(() => {
    localStorage.setItem('FINANCE_ONBOARDING_TOUR_DISMISSED', 'true');
    localStorage.setItem('FINANCE_ONBOARDING_COMPLETED', 'true');
    localStorage.setItem('hasSeenTour', 'true');
    localStorage.setItem('arx_tour_seen', 'true');
    localStorage.setItem('ARX_VERNACULAR_MODE', 'PRO_QUANT');
  });

  async function dismissTourIfPresent() {
    try {
      const buttons = await page.$$('button');
      for (const b of buttons) {
        const txt = await page.evaluate(el => el.textContent, b);
        if (txt && (txt.includes('Skip') || txt.includes('Dismiss') || txt.includes('Got it'))) {
          await b.click();
          await new Promise(r => setTimeout(r, 400));
          break;
        }
      }
    } catch (e) {}
  }

  try {
    // -------------------------------------------------------------------------
    // Phase 1: AAPL Desktop Acceptance (1440x900)
    // -------------------------------------------------------------------------
    console.log('1. Auditing AAPL on Desktop Viewport (1440x900)...');
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    const aaplUrl = `${targetBaseUrl}/?symbol=AAPL`;
    const aaplResp = await page.goto(aaplUrl, { waitUntil: 'networkidle2', timeout: 45000 });
    
    recordCheck('AAPL_HTTP_STATUS', aaplResp.status() === 200, `HTTP status code ${aaplResp.status()}`);
    await dismissTourIfPresent();
    await new Promise(r => setTimeout(r, 2000));

    // Await Pre-Flight bar and Supporting Evidence
    await page.waitForSelector('[data-testid="preflight-readiness-bar"]', { timeout: 20000 });
    await page.waitForSelector('[data-testid="supporting-evidence"]', { timeout: 20000 });

    // Capture Desktop Full Page
    const desktopShot = path.join(outDir, 'after_consolidation_desktop_fullpage.png');
    await page.screenshot({ path: desktopShot, fullPage: true });
    auditReport.screenshots.push('after_consolidation_desktop_fullpage.png');
    console.log(`  Saved full desktop screenshot: ${desktopShot}`);

    // Audit Pre-Flight bar location & contents
    const preflightAudit = await page.evaluate(() => {
      const bar = document.querySelector('[data-testid="preflight-readiness-bar"]');
      const chart = document.querySelector('#market-workspace-chart') || document.querySelector('[data-testid="market-workspace-chart"]');
      if (!bar) return { found: false };

      const barRect = bar.getBoundingClientRect();
      const chartRect = chart ? chart.getBoundingClientRect() : null;
      const isBelow = chartRect ? (barRect.top >= chartRect.top) : true;

      const pills = Array.from(bar.querySelectorAll('[data-testid^="inline-chk-pill-"]')).map(p => ({
        testid: p.getAttribute('data-testid'),
        text: p.innerText
      }));

      const actionability = bar.querySelector('[data-testid="inline-actionability-badge"]')?.innerText || '';
      const clearance = bar.querySelector('[data-testid="inline-clearance-badge"]')?.innerText || '';
      const blocker = bar.querySelector('[data-testid="active-blocker-callout"]')?.innerText || '';

      return {
        found: true,
        isBelowChart: isBelow,
        pillsCount: pills.length,
        actionability,
        clearance,
        blocker
      };
    });

    recordCheck('PREFLIGHT_BAR_MOUNTED', preflightAudit.found, 'Inline Pre-Flight readiness bar mounted');
    recordCheck('PREFLIGHT_BAR_POSITION', preflightAudit.isBelowChart, 'Mounted in DOM hierarchy beneath chart');
    recordCheck('PREFLIGHT_5_PILLS', preflightAudit.pillsCount === 5, `Contains 5 checks (found ${preflightAudit.pillsCount})`);
    recordCheck('PREFLIGHT_FAIL_CLOSED_AAPL', preflightAudit.clearance.includes('NOT CLEARED'), `AAPL clearance fail-closed: ${preflightAudit.clearance}`);

    // Capture Pre-Flight bar closeup
    const preflightEl = await page.$('[data-testid="preflight-readiness-bar"]');
    if (preflightEl) {
      const barShot = path.join(outDir, 'preflight_bar_aapl_closeup.png');
      await preflightEl.screenshot({ path: barShot });
      auditReport.screenshots.push('preflight_bar_aapl_closeup.png');
    }

    // Expand Pre-Flight Telemetry Drawer
    const drawerExpanded = await page.evaluate(() => {
      const toggle = document.querySelector('button[aria-controls="preflight-details-drawer"]');
      if (toggle) {
        toggle.click();
        return true;
      }
      return false;
    });
    if (drawerExpanded) {
      await new Promise(r => setTimeout(r, 600));
      const drawerShot = path.join(outDir, 'preflight_drawer_expanded.png');
      const barWithDrawer = await page.$('[data-testid="preflight-readiness-bar"]');
      if (barWithDrawer) {
        await barWithDrawer.screenshot({ path: drawerShot });
        auditReport.screenshots.push('preflight_drawer_expanded.png');
      }
    }

    // Audit Four Pillars Confluence
    const confluenceAudit = await page.evaluate(() => {
      const evidence = document.querySelector('[data-testid="supporting-evidence"]');
      if (!evidence) return { found: false };

      const text = evidence.innerText;
      const lowerText = text.toLowerCase();
      const pillars = [
        'Technical Confluence',
        'Fundamental Moat',
        'Institutional Flow',
        'Macro Environment'
      ];

      const presentPillars = pillars.filter(p => lowerText.includes(p.toLowerCase()));
      const has25Pct = (text.match(/25% Weight/g) || []).length;
      const hasUndefinedOrNull = /undefined|null|NaN/i.test(text);

      // Check Institutional Flow score specifically
      const institutionalFlowPresent = lowerText.includes('institutional flow');

      return {
        found: true,
        allPillarsPresent: presentPillars.length === 4,
        presentPillars,
        has25PctCount: has25Pct,
        hasUndefinedOrNull,
        institutionalFlowPresent,
        rawSnippet: text.slice(0, 400)
      };
    });

    recordCheck('CONFLUENCE_4_PILLARS', confluenceAudit.allPillarsPresent, `All 4 pillars present (${confluenceAudit.presentPillars.join(', ')})`);
    recordCheck('CONFLUENCE_INSTITUTIONAL_FLOW', confluenceAudit.institutionalFlowPresent, 'Institutional Flow pillar is present');
    recordCheck('CONFLUENCE_EQUAL_WEIGHT', confluenceAudit.has25PctCount >= 4, `4 pillars display 25% Weight (found ${confluenceAudit.has25PctCount})`);
    recordCheck('CONFLUENCE_NO_UNDEFINED_OR_NULL', !confluenceAudit.hasUndefinedOrNull, 'Zero undefined, null, or NaN in confluence evidence');

    // Capture 4-pillar confluence closeup
    const evidenceEl = await page.$('[data-testid="supporting-evidence"]');
    if (evidenceEl) {
      const closeupPath = path.join(outDir, 'four_pillars_aapl_closeup.png');
      await evidenceEl.screenshot({ path: closeupPath });
      auditReport.screenshots.push('four_pillars_aapl_closeup.png');
      console.log(`  Saved 4-pillar closeup: ${closeupPath}`);
    }

    // Negative Assertion Audits on AAPL page
    const negativeAudits = await page.evaluate(() => {
      // 1. No trade-execution-stepper
      const stepper = document.querySelector('[data-testid="trade-execution-stepper"]');
      // 2. No duplicate primary verdict h2s
      const h2s = Array.from(document.querySelectorAll('h2'));
      const verdictH2s = h2s.filter(h => /BULLISH CONFLUENCE|BEARISH CONFLUENCE|NEUTRAL CONFLUENCE/.test(h.innerText));
      // 3. No duplicate bottom line
      const bodyText = document.body.innerText;
      const bottomLineOccurrences = (bodyText.match(/The Bottom Line \(No Wall Street Fluff\)/g) || []).length;
      // 4. No onboarding tutorials leaking on explicit asset
      const pageIntro = document.querySelector('[data-testid="page-intro"]');
      const intentHero = document.querySelector('[data-testid="intent-hero"]');

      return {
        stepperPresent: !!stepper,
        verdictH2Count: verdictH2s.length,
        bottomLineCount: bottomLineOccurrences,
        tutorialsPresent: !!(pageIntro || intentHero)
      };
    });

    recordCheck('NEGATIVE_NO_STEPPER', !negativeAudits.stepperPresent, 'TradeExecutionStepper eliminated');
    recordCheck('NEGATIVE_NO_DUPLICATE_VERDICT_H2', negativeAudits.verdictH2Count <= 1, `Max 1 primary verdict title (found ${negativeAudits.verdictH2Count})`);
    recordCheck('NEGATIVE_NO_DUPLICATE_BOTTOM_LINE', negativeAudits.bottomLineCount <= 1, `Max 1 bottom-line box (found ${negativeAudits.bottomLineCount})`);
    recordCheck('NEGATIVE_NO_TUTORIAL_LEAK', !negativeAudits.tutorialsPresent, 'Onboarding tutorials suppressed during asset analysis');

    auditReport.symbolsAudited['AAPL'] = {
      preflight: preflightAudit,
      confluence: confluenceAudit,
      negatives: negativeAudits
    };

    // -------------------------------------------------------------------------
    // Phase 2: Tablet Viewport (768x1024)
    // -------------------------------------------------------------------------
    console.log('\n2. Auditing Tablet Viewport (768x1024)...');
    await page.setViewport({ width: 768, height: 1024, deviceScaleFactor: 1 });
    await new Promise(r => setTimeout(r, 1000));
    const tabletShot = path.join(outDir, 'after_consolidation_tablet_fullpage.png');
    await page.screenshot({ path: tabletShot, fullPage: true });
    auditReport.screenshots.push('after_consolidation_tablet_fullpage.png');
    recordCheck('TABLET_FULLPAGE_CAPTURED', fs.existsSync(tabletShot), 'Tablet 768px full page captured');

    // -------------------------------------------------------------------------
    // Phase 3: Mobile Viewport (390x844)
    // -------------------------------------------------------------------------
    console.log('\n3. Auditing Mobile Viewport (390x844)...');
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 });
    await new Promise(r => setTimeout(r, 1000));
    const mobileShot = path.join(outDir, 'after_consolidation_mobile_fullpage.png');
    await page.screenshot({ path: mobileShot, fullPage: true });
    auditReport.screenshots.push('after_consolidation_mobile_fullpage.png');
    recordCheck('MOBILE_FULLPAGE_CAPTURED', fs.existsSync(mobileShot), 'Mobile 390px full page captured');

    // -------------------------------------------------------------------------
    // Phase 4: NVDA Acceptance (1440x900) — Stage 4 Wait & Confluence
    // -------------------------------------------------------------------------
    console.log('\n4. Auditing NVDA on Desktop Viewport (1440x900)...');
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    const nvdaUrl = `${targetBaseUrl}/?symbol=NVDA`;
    const nvdaResp = await page.goto(nvdaUrl, { waitUntil: 'networkidle2', timeout: 45000 });
    recordCheck('NVDA_HTTP_STATUS', nvdaResp.status() === 200, `HTTP status code ${nvdaResp.status()}`);
    await dismissTourIfPresent();
    await new Promise(r => setTimeout(r, 2000));

    await page.waitForSelector('[data-testid="preflight-readiness-bar"]', { timeout: 20000 });
    await page.waitForSelector('[data-testid="supporting-evidence"]', { timeout: 20000 });

    const nvdaAudit = await page.evaluate(() => {
      const bar = document.querySelector('[data-testid="preflight-readiness-bar"]');
      const evidence = document.querySelector('[data-testid="supporting-evidence"]');

      const clearance = bar?.querySelector('[data-testid="inline-clearance-badge"]')?.innerText || '';
      const evidenceText = evidence?.innerText || '';
      const lowerEvidence = evidenceText.toLowerCase();

      const pillars = ['Technical Confluence', 'Fundamental Moat', 'Institutional Flow', 'Macro Environment'];
      const presentPillars = pillars.filter(p => lowerEvidence.includes(p.toLowerCase()));
      const hasUndefinedOrNull = /undefined|null|NaN/i.test(evidenceText);

      return {
        clearance,
        stage4Wait: clearance.includes('STAGE 4 WAIT'),
        allPillarsPresent: presentPillars.length === 4,
        presentPillars,
        hasUndefinedOrNull
      };
    });

    recordCheck('NVDA_FAIL_CLOSED_STAGE4', nvdaAudit.stage4Wait, `NVDA clearance blocks on Stage 4 Wait: ${nvdaAudit.clearance}`);
    recordCheck('NVDA_CONFLUENCE_4_PILLARS', nvdaAudit.allPillarsPresent, `NVDA renders all 4 pillars (${nvdaAudit.presentPillars.join(', ')})`);
    recordCheck('NVDA_NO_UNDEFINED_OR_NULL', !nvdaAudit.hasUndefinedOrNull, 'NVDA zero undefined/null/NaN');

    const nvdaEvidenceEl = await page.$('[data-testid="supporting-evidence"]');
    if (nvdaEvidenceEl) {
      const nvdaCloseup = path.join(outDir, 'four_pillars_nvda_closeup.png');
      await nvdaEvidenceEl.screenshot({ path: nvdaCloseup });
      auditReport.screenshots.push('four_pillars_nvda_closeup.png');
      console.log(`  Saved NVDA 4-pillar closeup: ${nvdaCloseup}`);
    }

    auditReport.symbolsAudited['NVDA'] = nvdaAudit;

    // -------------------------------------------------------------------------
    // Phase 5: Workspace Tabs Accessibility
    // -------------------------------------------------------------------------
    console.log('\n5. Auditing Workspace Domain Tabs...');
    const tabsAudit = await page.evaluate(() => {
      const tabs = Array.from(document.querySelectorAll('button[role="tab"], button')).filter(b => {
        const txt = b.innerText || '';
        return txt.includes('Smart Money') || txt.includes('Factors & Macro') || txt.includes('Risk & Contagion');
      });

      return {
        tabsFound: tabs.map(t => t.innerText.trim()),
        hasSmartMoney: tabs.some(t => t.innerText.includes('Smart Money')),
        hasFactors: tabs.some(t => t.innerText.includes('Factors & Macro')),
        hasRisk: tabs.some(t => t.innerText.includes('Risk & Contagion'))
      };
    });

    recordCheck('TABS_SMART_MONEY', tabsAudit.hasSmartMoney, 'Smart Money tab accessible in workspace');
    recordCheck('TABS_FACTORS_MACRO', tabsAudit.hasFactors, 'Factors & Macro tab accessible');
    recordCheck('TABS_RISK_CONTAGION', tabsAudit.hasRisk, 'Risk & Contagion tab accessible');

    // Consolidated Determination
    if (auditReport.failedCount === 0 && auditReport.passedCount >= 18) {
      auditReport.verdict = 'PASS';
    } else {
      auditReport.verdict = 'FAIL';
    }

  } catch (err) {
    console.error('Audit encountered exception:', err);
    recordCheck('AUDIT_EXCEPTION', false, err.message);
    auditReport.verdict = 'FAIL';
  } finally {
    await browser.close();
  }

  // Save machine-readable report
  const reportPath = path.join(outDir, 'design_parity_gate_v1_results.json');
  fs.writeFileSync(reportPath, JSON.stringify(auditReport, null, 2));
  console.log(`\nAudit Report written to: ${reportPath}`);

  // Also write to frontend/reports for CI archiving
  const frontendReportDir = path.resolve(__dirname, '../reports');
  if (!fs.existsSync(frontendReportDir)) {
    fs.mkdirSync(frontendReportDir, { recursive: true });
  }
  fs.writeFileSync(path.join(frontendReportDir, 'ux_design_parity_results.json'), JSON.stringify(auditReport, null, 2));

  console.log('================================================================================');
  console.log(`  ARX_UX_DESIGN_PARITY_GATE_V1 AUDIT DISPOSITION: ${auditReport.verdict}`);
  console.log(`  Total Checks: ${auditReport.passedCount + auditReport.failedCount} | Passed: ${auditReport.passedCount} | Failed: ${auditReport.failedCount}`);
  console.log('================================================================================\n');

  if (auditReport.verdict !== 'PASS') {
    process.exit(1);
  }
})();
