import { test, expect } from "@playwright/test";

test.describe("Ticker Command Strip (W1.5 E2E Acceptance)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/design-system-preview");
    // Switch to command-strip tab
    await page.click('button:has-text("5. Stage 1 Command Strip")');
  });

  test("AC-W1.5-01: Orientation Strip Renders with Test ID and 110px desktop height constraint", async ({ page }) => {
    const commandStrips = page.locator('[data-testid="ticker-command-strip"]');
    await expect(commandStrips.first()).toBeVisible();

    // Check height bounding box on desktop
    const box = await commandStrips.first().boundingBox();
    expect(box).not.toBeNull();
    expect(box!.height).toBeGreaterThanOrEqual(105);
    expect(box!.height).toBeLessThanOrEqual(125);
  });

  test("AC-W1.5-02: Identity Display (Symbol, Company, Exchange, Sector)", async ({ page }) => {
    const cprxStrip = page.locator('[data-testid="ticker-command-strip"]').first();
    await expect(cprxStrip.locator('[data-testid="ticker-symbol"]')).toHaveText("CPRX");
    await expect(cprxStrip.locator('[data-testid="company-name"]')).toHaveText("Catalyst Pharmaceuticals Inc.");
    await expect(cprxStrip.locator('[data-testid="ticker-exchange"]')).toHaveText("NASDAQ");
    await expect(cprxStrip.locator('[data-testid="ticker-sector"]')).toHaveText("Healthcare");
  });

  test("AC-W1.5-03: Spot Price & Delta Formatting", async ({ page }) => {
    const cprxStrip = page.locator('[data-testid="ticker-command-strip"]').first();
    await expect(cprxStrip.locator('[data-testid="spot-price"]')).toHaveText("$18.42");
    await expect(cprxStrip.locator('[data-testid="price-delta"]')).toContainText("+0.38 (+2.11%)");

    // Negative delta in Variant D (SMLR)
    const smlrStrip = page.locator('[data-testid="ticker-command-strip"]').nth(3);
    await expect(smlrStrip.locator('[data-testid="spot-price"]')).toHaveText("$29.15");
    await expect(smlrStrip.locator('[data-testid="price-delta"]')).toContainText("-2.45 (-7.75%)");
  });

  test("AC-W1.5-04: Setup Score Gauge Thresholding (Emerald >= 70, Amber 50-69, Rose < 50)", async ({ page }) => {
    // CPRX: 71 (Emerald)
    const cprxScore = page.locator('[data-testid="ticker-command-strip"]').first().locator('[data-testid="setup-score-badge"]');
    await expect(cprxScore).toHaveAttribute("aria-valuenow", "71");
    await expect(cprxScore.locator('[data-testid="setup-score-value"]')).toHaveText("71");

    // TSLA: 58 (Amber)
    const tslaScore = page.locator('[data-testid="ticker-command-strip"]').nth(2).locator('[data-testid="setup-score-badge"]');
    await expect(tslaScore).toHaveAttribute("aria-valuenow", "58");
    await expect(tslaScore.locator('[data-testid="setup-score-status"]')).toHaveText("Conditional");

    // SMLR: 38 (Rose)
    const smlrScore = page.locator('[data-testid="ticker-command-strip"]').nth(3).locator('[data-testid="setup-score-badge"]');
    await expect(smlrScore).toHaveAttribute("aria-valuenow", "38");
    await expect(smlrScore.locator('[data-testid="setup-score-status"]')).toHaveText("High Risk");
  });

  test("AC-W1.5-05: Execution State Badges", async ({ page }) => {
    const cprxState = page.locator('[data-testid="ticker-command-strip"]').first().locator('[data-testid="execution-state-badge"]');
    await expect(cprxState).toHaveAttribute("data-state", "IN_BUY_ZONE");
    await expect(cprxState).toContainText("IN_BUY_ZONE");

    const tslaState = page.locator('[data-testid="ticker-command-strip"]').nth(2).locator('[data-testid="execution-state-badge"]');
    await expect(tslaState).toHaveAttribute("data-state", "WAITING_PULLBACK");
    await expect(tslaState).toContainText("WAITING_PULLBACK");

    const smlrState = page.locator('[data-testid="ticker-command-strip"]').nth(3).locator('[data-testid="execution-state-badge"]');
    await expect(smlrState).toHaveAttribute("data-state", "STOPPED_OUT");
    await expect(smlrState).toContainText("STOPPED_OUT");
  });

  test("AC-W1.5-06: Pinned Settlement Banner Display", async ({ page }) => {
    const cprxStrip = page.locator('[data-testid="ticker-command-strip"]').first();
    const settlementNotice = cprxStrip.locator('[data-testid="settlement-pinned-notice"]');
    await expect(settlementNotice).toBeVisible();
    await expect(settlementNotice).toContainText("[Session Closed / Friday Settlement Pinned]");
  });

  test("AC-W1.5-07: Zero-CLS Skeleton Header Semantics", async ({ page }) => {
    const skeleton = page.locator('[data-testid="ticker-command-strip-skeleton"]');
    await expect(skeleton).toBeVisible();
    await expect(skeleton).toHaveAttribute("role", "region");
    await expect(skeleton).toHaveAttribute("aria-label", "Ticker Command Strip Loading");

    const box = await skeleton.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.height).toBeGreaterThanOrEqual(105);
    expect(box!.height).toBeLessThanOrEqual(125);
  });

  test("AC-W1.5-08: Experience Mode Adaptations (Guided & Quant)", async ({ page }) => {
    // Switch to GUIDED
    await page.click('button:has-text("GUIDED")');
    const guidedCprx = page.locator('[data-testid="ticker-command-strip"]').first();
    await expect(guidedCprx.locator('[data-testid="setup-score-badge"]')).toContainText("Setup Quality");
    await expect(guidedCprx.locator('[data-testid="execution-state-badge"]')).toContainText("Favorable Entry Zone");
    await expect(guidedCprx.locator('[data-testid="liquidity-badge"]')).toContainText("High Liquidity");

    // Switch to QUANT
    await page.click('button:has-text("QUANT")');
    const quantCprx = page.locator('[data-testid="ticker-command-strip"]').first();
    await expect(quantCprx.locator('[data-testid="domain-confidence-tag"]')).toContainText("CONF: HIGH");
    await expect(quantCprx.locator('[data-testid="quant-market-regime-badge"]')).toContainText("REGIME: RISK_ON");
    await expect(quantCprx.locator('[data-testid="amihud-score"]')).toContainText("Amihud: 0.0014");
  });
});
