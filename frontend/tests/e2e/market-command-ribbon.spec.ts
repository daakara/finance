import { test, expect } from "@playwright/test";

test.describe("Market Command Ribbon E2E Specification", () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage before each test
    await page.addInitScript(() => {
      window.localStorage.clear();
    });
  });

  test("should render persistent 36px ribbon directly beneath the 56px navbar", async ({ page }) => {
    await page.goto("/");

    // 1. Verify 56px Navbar presence
    const navbar = page.locator('[data-testid="navbar"]');
    await expect(navbar).toBeVisible();
    const navBox = await navbar.boundingBox();
    expect(navBox).not.toBeNull();
    expect(navBox!.height).toBeCloseTo(56, 1);

    // 2. Verify 36px Market Command Ribbon presence
    const ribbon = page.locator('[data-testid="market-command-ribbon"]');
    await expect(ribbon).toBeVisible();
    const ribbonBox = await ribbon.boundingBox();
    expect(ribbonBox).not.toBeNull();
    expect(ribbonBox!.height).toBeCloseTo(36, 1);

    // 3. Verify Ribbon is positioned immediately below Navbar (top = navBox.y + navBox.height)
    expect(ribbonBox!.y).toBeGreaterThanOrEqual(navBox!.y + navBox!.height - 1);
    expect(ribbonBox!.y).toBeLessThanOrEqual(navBox!.y + navBox!.height + 2);
  });

  test("should render all macro benchmarks (SPY, QQQ, VIX, 10Y) above the fold without scrolling", async ({ page }) => {
    await page.goto("/");

    const ribbon = page.locator('[data-testid="market-command-ribbon"]');
    await expect(ribbon).toBeVisible();

    // Verify benchmark elements
    const spy = ribbon.locator('[aria-label="S&P 500"]');
    const qqq = ribbon.locator('[aria-label="NASDAQ 100"]');
    const vix = ribbon.locator('[aria-label="CBOE Volatility Index"]');
    const tenYear = ribbon.locator('[aria-label="10-Year Treasury Yield"]');

    await expect(spy).toBeVisible();
    await expect(spy).toContainText("SPY");

    await expect(qqq).toBeVisible();
    await expect(qqq).toContainText("QQQ");

    await expect(vix).toBeVisible();
    await expect(vix).toContainText("VIX");

    await expect(tenYear).toBeVisible();
    await expect(tenYear).toContainText("10Y");

    // Viewport check: elements must be within top 120px (above the fold)
    const ribbonBox = await ribbon.boundingBox();
    expect(ribbonBox!.y + ribbonBox!.height).toBeLessThan(120);
  });

  test("should display consolidated 5 semantic navigation categories in desktop navbar", async ({ page }) => {
    await page.goto("/");

    const navLinks = page.locator('[data-testid="desktop-nav-links"]');
    await expect(navLinks).toBeVisible();

    // Verify 5 semantic categories
    await expect(navLinks.getByRole("link", { name: "Terminal" })).toBeVisible();
    await expect(navLinks.getByRole("link", { name: "Intelligence" })).toBeVisible();
    await expect(navLinks.getByRole("link", { name: "Portfolio" })).toBeVisible();
    await expect(navLinks.getByRole("link", { name: "Research" })).toBeVisible();
    await expect(navLinks.getByRole("link", { name: "Docs" })).toBeVisible();
  });

  test("should display Market Regime badge with correct styling", async ({ page }) => {
    await page.goto("/");

    const regimeBadge = page.locator('[data-testid="market-regime-badge"]');
    await expect(regimeBadge).toBeVisible();

    const text = await regimeBadge.innerText();
    expect(["RISK ON", "NEUTRAL", "DEFENSIVE"]).toContain(text.trim());

    if (text.includes("RISK ON")) {
      await expect(regimeBadge).toHaveClass(/text-emerald-400/);
    } else if (text.includes("DEFENSIVE")) {
      await expect(regimeBadge).toHaveClass(/text-rose-400/);
    } else {
      await expect(regimeBadge).toHaveClass(/text-amber-400/);
    }
  });

  test("should handle 503 fallback gracefully by rendering [Cached Market Snapshot]", async ({ page }) => {
    // Intercept /api/macro/ribbon and return 503
    await page.route("**/macro/ribbon", (route) => {
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error: "Service Temporarily Unavailable" }),
      });
    });

    await page.goto("/");

    const cachedBadge = page.locator('[data-testid="cached-snapshot-badge"]');
    await expect(cachedBadge).toBeVisible();
    await expect(cachedBadge).toContainText("Cached Market Snapshot");
  });

  test("should support accessible keyboard navigation and tab order", async ({ page }) => {
    await page.goto("/");

    // Tab through the navigation elements
    await page.keyboard.press("Tab"); // ARX Logo
    const homeLink = page.locator('a[aria-label="ARX Terminal Home"]');
    await expect(homeLink).toBeFocused();

    await page.keyboard.press("Tab"); // Terminal link
    const terminalLink = page.locator('[data-testid="desktop-nav-links"] a:has-text("Terminal")');
    await expect(terminalLink).toBeFocused();
  });
});
