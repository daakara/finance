import puppeteer from 'puppeteer';
import {
  calculateContrastRatio,
  parseRgb,
  getRequiredContrastRatio,
  checkReflowMetrics,
  checkExactFocusRestoration,
  auditVisibleElementsContrast,
  auditNonTextContrast,
} from './a4-verification-core.mjs';

async function runNegativeControls() {
  console.log('================================================================================');
  console.log('       A4 ACCESSIBILITY HARNESS NEGATIVE CONTROLS VALIDATION SUITE              ');
  console.log('================================================================================');
  console.log('Objective: Verify that the shared a4-verification-core module functions fail-close');
  console.log('and correctly detect intentional accessibility violations (no false passes).\n');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      const u = req.url();
      if (u.includes('matomo') || u.includes('analytics') || u.includes('fpldna.com')) {
        req.abort();
      } else {
        req.continue();
      }
    });
    await page.goto('http://localhost:3000/?symbol=NVDA&mode=standard', { waitUntil: 'domcontentloaded', timeout: 15000 });
    await new Promise((r) => setTimeout(r, 1500));

    let detectedViolations = 0;
    const totalControls = 5;

    // -------------------------------------------------------------------------
    // Control 1: Forced Horizontal Overflow via checkReflowMetrics
    // -------------------------------------------------------------------------
    console.log('Running Control 1: Injecting forced horizontal overflow (5000px wide element)...');
    await page.evaluate(() => {
      const div = document.createElement('div');
      div.id = 'negative-control-overflow';
      div.style.width = '5000px';
      div.style.height = '10px';
      div.style.backgroundColor = 'red';
      document.body.appendChild(div);
    });

    const reflowMetrics = await checkReflowMetrics(page);
    await page.evaluate(() => {
      const el = document.getElementById('negative-control-overflow');
      if (el) el.remove();
    });

    if (reflowMetrics.hasHorizontalScroll) {
      console.log(`✔ CONTROL 1 PASS: checkReflowMetrics correctly detected forced horizontal scroll violation (scrollWidth=${reflowMetrics.scrollWidth} > clientWidth=${reflowMetrics.clientWidth}).`);
      detectedViolations++;
    } else {
      console.error('✖ CONTROL 1 FAIL: checkReflowMetrics failed to detect 5000px horizontal overflow!');
    }

    // -------------------------------------------------------------------------
    // Control 2: Clipped Text via checkReflowMetrics
    // -------------------------------------------------------------------------
    console.log('\nRunning Control 2: Injecting clipped text container (overflow: hidden, no ellipsis)...');
    await page.evaluate(() => {
      const p = document.createElement('p');
      p.id = 'negative-control-clipped';
      p.style.width = '40px';
      p.style.whiteSpace = 'nowrap';
      p.style.overflow = 'hidden';
      p.style.textOverflow = 'clip';
      p.innerText = 'This is a very long string that will definitely clip without ellipsis';
      document.body.appendChild(p);
    });

    const clipMetrics = await checkReflowMetrics(page);
    await page.evaluate(() => {
      const el = document.getElementById('negative-control-clipped');
      if (el) el.remove();
    });

    if (clipMetrics.clippedTextCount > 0) {
      console.log(`✔ CONTROL 2 PASS: checkReflowMetrics correctly detected clipped text violation (clipped count: ${clipMetrics.clippedTextCount}).`);
      detectedViolations++;
    } else {
      console.error('✖ CONTROL 2 FAIL: checkReflowMetrics failed to detect clipped text element!');
    }

    // -------------------------------------------------------------------------
    // Control 3: Broken Focus Restoration via checkExactFocusRestoration
    // -------------------------------------------------------------------------
    console.log('\nRunning Control 3: Simulating broken focus restoration on modal dismiss...');
    const triggerMarker = 'data-a4-test-origin="negative-focus-test"';
    await page.evaluate(() => {
      const btn = document.getElementById('privacy-settings-btn');
      if (btn) {
        btn.setAttribute('data-a4-test-origin', 'negative-focus-test');
        btn.focus();
        btn.click();
      }
    });

    await page.waitForSelector('[role="dialog"][aria-label="Privacy and Data Telemetry Settings"]', { visible: true });

    // Intentionally divert focus away from the origin trigger (e.g. to body or another button)
    await page.evaluate(() => {
      // Blur and focus body to simulate dropped focus
      if (document.activeElement) document.activeElement.blur();
      document.body.focus();
    });

    await page.keyboard.press('Escape');
    await page.waitForSelector('[role="dialog"][aria-label="Privacy and Data Telemetry Settings"]', { hidden: true });

    // Now call shared checkExactFocusRestoration: MUST return matched: false because focus was diverted
    const focusCheck = await checkExactFocusRestoration(page, triggerMarker);

    // Clean up marker attribute
    await page.evaluate(() => {
      const btn = document.getElementById('privacy-settings-btn');
      if (btn) btn.removeAttribute('data-a4-test-origin');
    });

    if (!focusCheck.matched) {
      console.log(`✔ CONTROL 3 PASS: checkExactFocusRestoration correctly rejected diverted focus (active=${focusCheck.activeTag}#${focusCheck.activeId}, expected trigger=${focusCheck.triggerTag}#${focusCheck.triggerId}).`);
      detectedViolations++;
    } else {
      console.error('✖ CONTROL 3 FAIL: checkExactFocusRestoration incorrectly passed broken focus restoration!');
    }

    // -------------------------------------------------------------------------
    // Control 4: Insufficient Contrast via auditVisibleElementsContrast
    // -------------------------------------------------------------------------
    console.log('\nRunning Control 4: Evaluating low-contrast normal text (#777777 on #777778) via auditVisibleElementsContrast...');
    await page.evaluate(() => {
      const p = document.createElement('p');
      p.id = 'negative-control-contrast';
      p.style.color = '#777777';
      p.style.backgroundColor = '#777778';
      p.style.fontSize = '14px';
      p.style.fontWeight = '400';
      p.innerText = 'Low contrast text for negative control verification';
      document.body.appendChild(p);
    });

    const contrastAudit = await auditVisibleElementsContrast(page, 'Obsidian (Dark)');
    await page.evaluate(() => {
      const el = document.getElementById('negative-control-contrast');
      if (el) el.remove();
    });

    const lowContrastViolated = contrastAudit.violations.some((v) => v.id === 'negative-control-contrast');

    if (lowContrastViolated) {
      const violation = contrastAudit.violations.find((v) => v.id === 'negative-control-contrast');
      console.log(`✔ CONTROL 4 PASS: auditVisibleElementsContrast correctly flagged low-contrast element (ratio=${violation.actualRatio}:1, required >= ${violation.requiredRatio}:1).`);
      detectedViolations++;
    } else {
      console.error('✖ CONTROL 4 FAIL: auditVisibleElementsContrast failed to detect low contrast element!');
    }

    // -------------------------------------------------------------------------
    // Control 5: Insufficient Non-Text Contrast via auditNonTextContrast
    // -------------------------------------------------------------------------
    console.log('\nRunning Control 5: Evaluating low-contrast non-text boundary (#111622 on #090d14) via auditNonTextContrast...');
    await page.evaluate(() => {
      const input = document.createElement('input');
      input.id = 'negative-control-nontext';
      input.type = 'text';
      input.value = 'Negative non-text';
      input.style.setProperty('border-color', '#111622', 'important');
      input.style.setProperty('border-width', '2px', 'important');
      input.style.setProperty('border-style', 'solid', 'important');
      input.style.setProperty('outline', '2px solid #111622', 'important');
      input.style.setProperty('outline-color', '#111622', 'important');
      input.style.setProperty('outline-width', '2px', 'important');
      input.style.setProperty('outline-style', 'solid', 'important');
      input.style.setProperty('box-shadow', 'none', 'important');
      input.style.setProperty('background-color', '#090d14', 'important');
      document.body.appendChild(input);
    });

    const nonTextAudit = await auditNonTextContrast(page, 'Obsidian (Dark)');
    await page.evaluate(() => {
      const el = document.getElementById('negative-control-nontext');
      if (el) el.remove();
    });

    const lowNonTextViolated = nonTextAudit.violations.some((v) => v.id === 'negative-control-nontext');

    if (lowNonTextViolated) {
      const violation = nonTextAudit.violations.find((v) => v.id === 'negative-control-nontext');
      console.log(`✔ CONTROL 5 PASS: auditNonTextContrast correctly flagged low-contrast non-text indicator (type=${violation.componentType}, ratio=${violation.actualRatio}:1, required >= ${violation.requiredRatio}:1).`);
      detectedViolations++;
    } else {
      console.error('✖ CONTROL 5 FAIL: auditNonTextContrast failed to detect low non-text contrast element!');
    }

    console.log('\n================================================================================');
    console.log(`NEGATIVE CONTROLS SUMMARY: ${detectedViolations} / ${totalControls} violations correctly rejected.`);
    console.log('================================================================================');

    if (detectedViolations === totalControls) {
      console.log('VERDICT: NEGATIVE CONTROLS SUITE PASSED (Shared core harness has proven fail-close capability).\n');
      process.exit(0);
    } else {
      console.error('VERDICT: NEGATIVE CONTROLS SUITE FAILED.\n');
      process.exit(1);
    }
  } finally {
    await browser.close();
  }
}

runNegativeControls().catch((err) => {
  console.error('FATAL ERROR in negative controls suite:', err);
  process.exit(1);
});
