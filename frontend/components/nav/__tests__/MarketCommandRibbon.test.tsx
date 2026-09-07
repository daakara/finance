import React from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import MarketCommandRibbon, {
  MarketCommandRibbonSkeleton,
  DEFAULT_MACRO_SNAPSHOT,
  MacroRibbonPayload,
} from "../MarketCommandRibbon";

describe("MarketCommandRibbon Component", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders MarketCommandRibbonSkeleton with exact 36px height and zero-CLS attributes", () => {
    render(<MarketCommandRibbonSkeleton />);
    const skeleton = screen.getByTestId("market-command-ribbon-skeleton");
    expect(skeleton).toBeDefined();
    expect(skeleton.className).toContain("h-9");
    expect(skeleton.className).toContain("min-h-[36px]");
    expect(skeleton.getAttribute("role")).toBe("region");
    expect(skeleton.getAttribute("aria-label")).toBe("Market Command Ribbon Loading");
  });

  it("renders macro symbols and indicators (SPY, QQQ, VIX, 10Y Yield)", () => {
    const mockData: MacroRibbonPayload = {
      spy: { price: 545.2, changePct: 0.85, change: 4.6 },
      qqq: { price: 472.1, changePct: 1.15, change: 5.37 },
      vix: { value: 14.8, changePct: -3.5, tier: "NORMAL" },
      tenYearYield: { value: 4.18, dailyChangeBp: -3.2 },
      regime: "RISK_ON",
      marketSession: "OPEN",
      settlementPinned: false,
      dataSource: "LIVE_FEED",
    };

    render(<MarketCommandRibbon initialData={mockData} />);

    const ribbon = screen.getByTestId("market-command-ribbon");
    expect(ribbon).toBeDefined();
    expect(ribbon.className).toContain("h-9");
    expect(ribbon.className).toContain("min-h-[36px]");

    // Verify SPY
    const spyContainer = screen.getByLabelText("S&P 500");
    expect(spyContainer.textContent).toContain("SPY");
    expect(spyContainer.textContent).toContain("$545.20");
    expect(spyContainer.textContent).toContain("+0.85%");

    // Verify QQQ
    const qqqContainer = screen.getByLabelText("NASDAQ 100");
    expect(qqqContainer.textContent).toContain("QQQ");
    expect(qqqContainer.textContent).toContain("$472.10");
    expect(qqqContainer.textContent).toContain("+1.15%");

    // Verify VIX
    const vixContainer = screen.getByLabelText("CBOE Volatility Index");
    expect(vixContainer.textContent).toContain("VIX");
    expect(vixContainer.textContent).toContain("14.80");
    expect(vixContainer.textContent).toContain("-3.50%");

    // Verify 10Y Yield
    const treasuryContainer = screen.getByLabelText("10-Year Treasury Yield");
    expect(treasuryContainer.textContent).toContain("10Y");
    expect(treasuryContainer.textContent).toContain("4.18%");
    expect(treasuryContainer.textContent).toContain("-3.2bp");
  });

  it("renders RISK_ON regime with Emerald color tokens and pulsing indicator", () => {
    const riskOnData: MacroRibbonPayload = {
      ...DEFAULT_MACRO_SNAPSHOT,
      regime: "RISK_ON",
    };

    render(<MarketCommandRibbon initialData={riskOnData} />);

    const regimeBadge = screen.getByTestId("market-regime-badge");
    expect(regimeBadge.textContent).toContain("RISK ON");
    expect(regimeBadge.className).toContain("text-emerald-400");
    expect(regimeBadge.className).toContain("bg-emerald-500/10");
    expect(regimeBadge.className).toContain("border-emerald-500/30");
  });

  it("renders DEFENSIVE regime with Rose color tokens", () => {
    const defensiveData: MacroRibbonPayload = {
      ...DEFAULT_MACRO_SNAPSHOT,
      regime: "DEFENSIVE",
    };

    render(<MarketCommandRibbon initialData={defensiveData} />);

    const regimeBadge = screen.getByTestId("market-regime-badge");
    expect(regimeBadge.textContent).toContain("DEFENSIVE");
    expect(regimeBadge.className).toContain("text-rose-400");
    expect(regimeBadge.className).toContain("bg-rose-500/10");
    expect(regimeBadge.className).toContain("border-rose-500/30");
  });

  it("renders NEUTRAL regime with Amber color tokens", () => {
    const neutralData: MacroRibbonPayload = {
      ...DEFAULT_MACRO_SNAPSHOT,
      regime: "NEUTRAL",
    };

    render(<MarketCommandRibbon initialData={neutralData} />);

    const regimeBadge = screen.getByTestId("market-regime-badge");
    expect(regimeBadge.textContent).toContain("NEUTRAL");
    expect(regimeBadge.className).toContain("text-amber-400");
    expect(regimeBadge.className).toContain("bg-amber-500/10");
    expect(regimeBadge.className).toContain("border-amber-500/30");
  });

  it("renders settlement pinned badge when market session is closed or pinned", () => {
    const pinnedData: MacroRibbonPayload = {
      ...DEFAULT_MACRO_SNAPSHOT,
      settlementPinned: true,
      marketSession: "CLOSED",
    };

    render(<MarketCommandRibbon initialData={pinnedData} />);

    const pinnedBadge = screen.getByTestId("settlement-pinned-badge");
    expect(pinnedBadge).toBeDefined();
    expect(pinnedBadge.textContent).toContain("Settlement Pinned");
  });

  it("handles 503 fallback and network errors by rendering Cached Market Snapshot indicator", async () => {
    // Mock fetch to simulate 503 Service Unavailable
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: "Upstream Provider Unavailable" }),
    });

    render(<MarketCommandRibbon />);

    await waitFor(() => {
      const cachedBadge = screen.getByTestId("cached-snapshot-badge");
      expect(cachedBadge).toBeDefined();
      expect(cachedBadge.textContent).toContain("[Cached Market Snapshot]");
    });
  });

  it("contains compliant accessible ARIA landmarks and labels", () => {
    render(<MarketCommandRibbon initialData={DEFAULT_MACRO_SNAPSHOT} />);

    expect(screen.getByRole("region", { name: "Market Command Ribbon" })).toBeDefined();
    expect(screen.getByLabelText("Market regime")).toBeDefined();
    expect(screen.getByLabelText("S&P 500")).toBeDefined();
    expect(screen.getByLabelText("NASDAQ 100")).toBeDefined();
    expect(screen.getByLabelText("CBOE Volatility Index")).toBeDefined();
  });
});
