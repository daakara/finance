import puppeteer from 'puppeteer';

async function main() {
  const b = await puppeteer.launch({ headless: true });
  const p = await b.newPage();
  await p.setRequestInterception(true);
  p.on("request", (req) => {
    const url = req.url();
    if (url.includes("/analytics/setups/")) {
      req.respond({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ticker: "NVDA",
          setupName: "Minervini VCP Breakout",
          entryPivot: 125.50,
          stopLoss: 118.20,
          target1: 145.00,
          target2: 160.00,
          confluenceScore: 88,
          isActionable: true,
          reasonSuppressed: null,
          executionStatus: "READY_TO_BUY",
        }),
      });
    } else if (url.includes("/analytics/setups")) {
      req.respond({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          setups: [
            {
              ticker: "NVDA",
              setupName: "Minervini VCP Breakout",
              entryPivot: 125.50,
              stopLoss: 118.20,
              target1: 145.00,
              target2: 160.00,
              confluenceScore: 88,
              isActionable: true,
              reasonSuppressed: null,
              executionStatus: "READY_TO_BUY",
            },
          ],
        }),
      });
    } else if (url.includes("/risk-telemetry")) {
      req.respond({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          maxRiskPerTradePct: 1.0,
          maxOpenPositions: 5,
          accountSize: 25000,
        }),
      });
    } else {
      req.continue();
    }
  });

  await p.goto('http://localhost:3000/setups/?ticker=NVDA', { waitUntil: 'load' });
  
  // Wait for setups card to render and select it if not selected
  await p.waitForSelector('button', { timeout: 10000 });
  await p.evaluate(() => {
    const card = Array.from(document.querySelectorAll('button')).find(b => (b.innerText || '').includes('/100'));
    if (card) card.click();
  });
  await new Promise(r => setTimeout(r, 500));

  // Wait for loading to finish and button to appear
  await p.waitForFunction(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    return btns.some(b => b.innerText.toLowerCase().includes('broker fill'));
  }, { timeout: 10000 });

  const result = await p.evaluate(async () => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const trigger = buttons.find(b => b.innerText.toLowerCase().includes('broker fill'));
    if (!trigger) return { triggerFound: false };

    trigger.focus();
    trigger.click();
    await new Promise(r => setTimeout(r, 200));

    const dialog = document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!dialog) return { triggerFound: true, dialogFound: false };

    const activeEl = document.activeElement;
    const initialInside = dialog.contains(activeEl);
    const title = dialog.querySelector('h2, h3')?.innerText;
    const role = dialog.getAttribute('role') || 'dialog';
    const ariaModal = dialog.getAttribute('aria-modal') || 'true';

    return {
      triggerFound: true,
      dialogFound: true,
      role,
      ariaModal,
      title,
      initialInside,
      activeTag: activeEl?.tagName,
      activeId: activeEl?.id,
    };
  });

  console.log('Open modal result:', result);

  // Test Escape
  await p.keyboard.press('Escape');
  await new Promise(r => setTimeout(r, 200));

  const closeCheck = await p.evaluate(() => {
    const dialog = document.querySelector('[role="dialog"], [aria-modal="true"]');
    const activeEl = document.activeElement;
    const trigger = Array.from(document.querySelectorAll('button')).find(b => b.innerText.toLowerCase().includes('broker fill'));
    return {
      dialogStillVisible: !!dialog,
      focusRestoredToTrigger: activeEl === trigger,
    };
  });

  console.log('Close with Escape result:', closeCheck);
  await b.close();
}

main();
