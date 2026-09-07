import { test, expect } from "@playwright/test";

test.describe("Experience Mode Machine (W1.4 E2E Acceptance)", () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage before each test
    await page.addInitScript(() => {
      window.localStorage.clear();
    });
  });

  test("AC-W1.4-01: Default STANDARD mode loads when no mode is specified", async ({ page }) => {
    await page.goto("/");

    // Verify Standard tab is active
    const standardTab = page.locator('[data-testid="mode-STANDARD"]');
    await expect(standardTab).toBeVisible();
    await expect(standardTab).toHaveAttribute("aria-selected", "true");

    // Verify URL synced to mode=standard
    await expect(page).toHaveURL(/mode=standard/);
  });

  test("AC-W1.4-02: Guided mode deep link activation", async ({ page }) => {
    await page.goto("/?mode=guided");

    const guidedTab = page.locator('[data-testid="mode-GUIDED"]');
    await expect(guidedTab).toBeVisible();
    await expect(guidedTab).toHaveAttribute("aria-selected", "true");

    const standardTab = page.locator('[data-testid="mode-STANDARD"]');
    await expect(standardTab).toHaveAttribute("aria-selected", "false");
  });

  test("AC-W1.4-03: Quant mode deep link activation", async ({ page }) => {
    await page.goto("/?mode=quant");

    const quantTab = page.locator('[data-testid="mode-QUANT"]');
    await expect(quantTab).toBeVisible();
    await expect(quantTab).toHaveAttribute("aria-selected", "true");

    const standardTab = page.locator('[data-testid="mode-STANDARD"]');
    await expect(standardTab).toHaveAttribute("aria-selected", "false");
  });

  test("AC-W1.4-04: Mode changes update URL without full page reload", async ({ page }) => {
    await page.goto("/");

    // Click Guided
    await page.click('[data-testid="mode-GUIDED"]');
    await expect(page).toHaveURL(/mode=guided/);

    const guidedTab = page.locator('[data-testid="mode-GUIDED"]');
    await expect(guidedTab).toHaveAttribute("aria-selected", "true");

    // Click Quant
    await page.click('[data-testid="mode-QUANT"]');
    await expect(page).toHaveURL(/mode=quant/);

    const quantTab = page.locator('[data-testid="mode-QUANT"]');
    await expect(quantTab).toHaveAttribute("aria-selected", "true");
  });

  test("AC-W1.4-05: Selected mode persists after page refresh", async ({ page }) => {
    await page.goto("/");

    // Switch to Quant
    await page.click('[data-testid="mode-QUANT"]');
    await expect(page).toHaveURL(/mode=quant/);

    // Refresh page
    await page.reload();

    // Verify Quant remains active
    const quantTab = page.locator('[data-testid="mode-QUANT"]');
    await expect(quantTab).toBeVisible();
    await expect(quantTab).toHaveAttribute("aria-selected", "true");
    await expect(page).toHaveURL(/mode=quant/);
  });

  test("AC-W1.4-06: Falls back to STANDARD when URL contains invalid mode", async ({ page }) => {
    await page.goto("/?mode=invalid_foobar");

    // Verify Standard tab is active
    const standardTab = page.locator('[data-testid="mode-STANDARD"]');
    await expect(standardTab).toBeVisible();
    await expect(standardTab).toHaveAttribute("aria-selected", "true");

    // Verify URL was rewritten to mode=standard
    await expect(page).toHaveURL(/mode=standard/);
  });

  test("AC-W1.4-07: Supports keyboard arrow navigation and tab order", async ({ page }) => {
    await page.goto("/?mode=guided");

    const guidedTab = page.locator('[data-testid="mode-GUIDED"]');
    await guidedTab.focus();

    // ArrowRight to Standard
    await page.keyboard.press("ArrowRight");
    const standardTab = page.locator('[data-testid="mode-STANDARD"]');
    await expect(standardTab).toHaveAttribute("aria-selected", "true");

    // ArrowRight to Quant
    await page.keyboard.press("ArrowRight");
    const quantTab = page.locator('[data-testid="mode-QUANT"]');
    await expect(quantTab).toHaveAttribute("aria-selected", "true");

    // ArrowRight wraps to Guided
    await page.keyboard.press("ArrowRight");
    await expect(guidedTab).toHaveAttribute("aria-selected", "true");
  });

  test("AC-W1.4-08: Screen reader semantics (role=tablist, role=tab, aria-selected)", async ({ page }) => {
    await page.goto("/");

    const tablist = page.getByRole("tablist", { name: "Experience Mode" });
    await expect(tablist).toBeVisible();

    const tabs = tablist.getByRole("tab");
    await expect(tabs).toHaveCount(3);
  });
});
