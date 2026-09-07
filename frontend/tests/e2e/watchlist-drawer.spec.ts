import { test, expect } from "@playwright/test";

test.describe("Slide-Over Watchlist Drawer (W1.6 E2E Acceptance)", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.clear();
    });
    await page.goto("/design-system-preview");
    // Navigate to drawer showcase or workstation
    await page.click('button:has-text("5. Stage 1 Command Strip")');
  });

  test("AC-W1.6-01: Watchlist Drawer is collapsed by default", async ({ page }) => {
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    // Offscreen / closed class
    await expect(drawer).toHaveClass(/-translate-x-full/);

    const trigger = page.locator('[data-testid="watchlist-drawer-trigger"]').first();
    await expect(trigger).toHaveAttribute("aria-expanded", "false");
  });

  test("AC-W1.6-02: Pressing '[' hotkey toggles open state", async ({ page }) => {
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await expect(drawer).toHaveClass(/-translate-x-full/);

    // Press '['
    await page.keyboard.press("[");
    await expect(drawer).toHaveClass(/translate-x-0/);

    // Press '[' again to close
    await page.keyboard.press("[");
    await expect(drawer).toHaveClass(/-translate-x-full/);
  });

  test("AC-W1.6-03: Pressing 'Control+b' toggles drawer", async ({ page }) => {
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await page.keyboard.press("Control+b");
    await expect(drawer).toHaveClass(/translate-x-0/);

    await page.keyboard.press("Control+b");
    await expect(drawer).toHaveClass(/-translate-x-full/);
  });

  test("AC-W1.6-04: Pressing 'Escape' closes open drawer", async ({ page }) => {
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await page.keyboard.press("[");
    await expect(drawer).toHaveClass(/translate-x-0/);

    await page.keyboard.press("Escape");
    await expect(drawer).toHaveClass(/-translate-x-full/);
  });

  test("AC-W1.6-05: Clicking backdrop closes drawer", async ({ page }) => {
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await page.keyboard.press("[");
    await expect(drawer).toHaveClass(/translate-x-0/);

    const backdrop = page.locator('[data-testid="watchlist-drawer-backdrop"]');
    await backdrop.click({ position: { x: 500, y: 300 } });
    await expect(drawer).toHaveClass(/-translate-x-full/);
  });

  test("AC-W1.6-06: State persistence survives page reload", async ({ page }) => {
    // Open drawer
    await page.keyboard.press("[");
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await expect(drawer).toHaveClass(/translate-x-0/);

    // Reload page
    await page.reload();

    // Drawer should still be open per saved preference
    const reloadedDrawer = page.locator('[data-testid="watchlist-drawer"]');
    await expect(reloadedDrawer).toHaveClass(/translate-x-0/);
  });

  test("AC-W1.6-07: Accessibility semantics & focus trap", async ({ page }) => {
    const drawer = page.locator('[data-testid="watchlist-drawer"]');
    await expect(drawer).toHaveAttribute("role", "dialog");
    await expect(drawer).toHaveAttribute("aria-modal", "true");
    await expect(drawer).toHaveAttribute("aria-label", "Watchlist Drawer");

    // Open and verify search input auto-focus
    await page.keyboard.press("[");
    const searchInput = page.locator('[data-testid="watchlist-search-input"]');
    await expect(searchInput).toBeFocused();
  });
});
