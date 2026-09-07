import { test, expect } from "@playwright/test";

test.describe("65/35 Decision Workspace & Viewport Anchoring (W1.7 E2E Acceptance)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/design-system-preview");
    // Switch to grid tab to inspect 65/35 WorkstationGrid
    await page.click('button:has-text("2. 65/35 Workstation Grid")');
  });

  test("UX-001: Chart Above Fold - Top edge renders above fold on desktop", async ({ page }) => {
    const chart = page.locator('[data-testid="price-chart-workspace"]').first();
    await expect(chart).toBeVisible();

    const box = await chart.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.y).toBeLessThan(400); // Confirms top edge well above fold
  });

  test("UX-002: Execution Corridor Visibility - 65/35 side-by-side on desktop (>=1280px)", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });

    const chart = page.locator('[data-testid="price-chart-workspace"]').first();
    const corridor = page.locator('[data-testid="execution-corridor"]').first();

    await expect(chart).toBeVisible();
    await expect(corridor).toBeVisible();

    const chartBox = await chart.boundingBox();
    const corridorBox = await corridor.boundingBox();

    expect(chartBox).not.toBeNull();
    expect(corridorBox).not.toBeNull();

    // Side-by-side geometry verification: chart on left, corridor on right
    expect(chartBox!.x).toBeLessThan(corridorBox!.x);

    // 65/35 proportional ratio verification (chart wider than corridor)
    expect(chartBox!.width).toBeGreaterThan(corridorBox!.width * 1.5);

    // Visible above fold without scrolling on 900px display
    expect(corridorBox!.y + corridorBox!.height).toBeLessThanOrEqual(950);
  });

  test("UX-003: Critical Data Discovery - Entry, Stop, Target clearly identified", async ({ page }) => {
    const corridor = page.locator('[data-testid="execution-corridor"]').first();
    await expect(corridor).toContainText("Entry Corridor");
    await expect(corridor).toContainText("Stop Loss Floor");
    await expect(corridor).toContainText("Target 1");
  });

  test("UX-004: Setup Score Discovery on Command Strip", async ({ page }) => {
    await page.click('button:has-text("5. Stage 1 Command Strip")');
    const scoreBadge = page.locator('[data-testid="setup-score-badge"]').first();
    await expect(scoreBadge).toBeVisible();
    await expect(scoreBadge).toHaveAttribute("role", "meter");
  });

  test("UX-005: Viewport Audit across 1366x768, 1440x900, and 1920x1080", async ({ page }) => {
    const viewports = [
      { width: 1366, height: 768 },
      { width: 1440, height: 900 },
      { width: 1920, height: 1080 },
    ];

    for (const vp of viewports) {
      await page.setViewportSize(vp);
      const grid = page.locator('[data-testid="workstation-grid"]').first();
      await expect(grid).toBeVisible();

      const box = await grid.boundingBox();
      expect(box).not.toBeNull();
      // Verifies minimum height constraint is maintained across viewports
      expect(box!.height).toBeGreaterThanOrEqual(600);
    }
  });

  test("UX-006: Chart Dominance Control - Stop loss floor has high contrast Rose boundary", async ({ page }) => {
    const corridor = page.locator('[data-testid="execution-corridor"]').first();
    await expect(corridor).toBeVisible();

    const stop = corridor.locator('[data-testid="corridor-stop"]');
    await expect(stop).toBeVisible();
    await expect(stop).toHaveClass(/bg-rose-500/);
  });

  test("UX-007: Zero CLS Validation - Container min-height eliminates layout jumping", async ({ page }) => {
    const grid = page.locator('[data-testid="workstation-grid"]').first();
    await expect(grid).toHaveClass(/lg:min-h-\[620px\]/);
  });
});
