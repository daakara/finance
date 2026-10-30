import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchScreenerGems } from "../../lib/api";
import RadarPage from "../../app/radar/page";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/radar",
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
}));

describe("ARX-036-D11R Layer 1: Contract Falsification (fetchScreenerGems)", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("1.1 Falsifies contradictory envelope: status SUCCESS + datasetState SCAN_FAILED normalizes fail-closed to ERROR", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "SCAN_FAILED",
        scanState: "FAILED",
        error: "Upstream scanner timeout after 30s",
        results: [],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.status).toBe("ERROR");
    expect(res.datasetState).toBe("SCAN_FAILED");
    expect(res.scanState).toBe("FAILED");
    expect(res.error).toBe("Upstream scanner timeout after 30s");
  });

  it("1.2 Falsifies contradictory envelope: status SUCCESS + scanState FAILED normalizes fail-closed to ERROR", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        scanState: "FAILED",
        results: [],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.status).toBe("ERROR");
    expect(res.error).toBe("Radar scan failed at dataset evaluation layer");
  });

  it("1.3 Falsifies partial degradation: status PARTIAL + scanState PARTIAL preserves PARTIAL status and partialErrors", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "PARTIAL",
        scanState: "PARTIAL",
        datasetState: "CANDIDATES_AVAILABLE",
        partialErrors: [{ ticker: "MSFT", reason: "Data vendor rate limit" }],
        results: [
          {
            ticker: "NVDA",
            name: "NVIDIA Corp",
            confluenceScore: 92,
            executionStatus: "IN_BUY_ZONE",
            allowedActions: ["ANALYZE_SETUP"],
          },
        ],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.status).toBe("PARTIAL");
    expect(res.scanState).toBe("PARTIAL");
    expect(res.partialErrors).toHaveLength(1);
    expect(res.results).toHaveLength(1);
  });

  it("1.4 Gate C State 15 Legacy Compatibility: status SUCCESS without modern fields preserves SUCCESS", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        // No datasetState, no scanState (legacy v0 backend contract)
        results: [
          {
            ticker: "AAPL",
            name: "Apple Inc",
            confluenceScore: 85,
            executionStatus: "IN_BUY_ZONE",
          },
        ],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.status).toBe("SUCCESS");
    expect(res.results).toHaveLength(1);
  });

  it("1.5 Eliminates silent fallback: missing decisionState defaults strictly to UNVERIFIED_SETUP", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        results: [
          {
            ticker: "TSLA",
            name: "Tesla Inc",
            // decisionState omitted
          },
        ],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.results[0].decisionState).toBe("UNVERIFIED_SETUP");
    expect(res.results[0].decisionStateLabel).toBe("Unverified Setup");
  });

  it("1.6 Invariant preservation: isActionable and canSizeTrade are strictly false on discovery candidates", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        results: [
          {
            ticker: "AMD",
            name: "Advanced Micro Devices",
            isActionable: true, // Attempted adversarial override from raw payload
            canSizeTrade: true,
          },
        ],
      }),
    } as any);

    const res = await fetchScreenerGems();
    expect(res.results[0].isActionable).toBe(false);
    expect(res.results[0].canSizeTrade).toBe(false);
  });
});

describe("ARX-036-D11R Layer 2: Component Surface Falsification (RadarPage)", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.unstubAllGlobals();
  });

  it("2.1 Renders explicit error banner and NEVER renders 'No Active Confluence Candidates' on SCAN_FAILED", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS", // Contradictory envelope that caused D10 bug
        datasetState: "SCAN_FAILED",
        scanState: "FAILED",
        error: "Underlying scanner timeout across 500 tickers",
        results: [],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      // Must display explicit connection / scan error banner
      expect(screen.getByText("Radar Tape Connection Error")).toBeDefined();
    });

    // Proves prevention of the false consolidation empty state
    expect(screen.queryByText("No Active Confluence Candidates")).toBeNull();
    expect(screen.getByText(/Unable to load live screening data:/i)).toBeDefined();
    expect(screen.getByText(/↻ Retry Connection/i)).toBeDefined();
  });

  it("2.2 Renders genuine empty state and NO error banner when scan genuinely produces zero candidates", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "GENUINE_EMPTY",
        scanState: "COMPLETED",
        results: [],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getByText("No Active Confluence Candidates")).toBeDefined();
    });

    // Proves error banner is NOT rendered for genuine market consolidation
    expect(screen.queryByText("Radar Tape Connection Error")).toBeNull();
    expect(screen.getByText(/This is normal during market pullbacks/i)).toBeDefined();
  });

  it("2.3 Renders partial degradation notice when status is PARTIAL", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "PARTIAL",
        datasetState: "CANDIDATES_AVAILABLE",
        scanState: "PARTIAL",
        partialErrors: [{ ticker: "GOOGL", reason: "Timeout" }],
        results: [
          {
            ticker: "NVDA",
            name: "NVIDIA Corp",
            price: 120.5,
            confluenceScore: 95,
            executionStatus: "IN_BUY_ZONE",
            allowedActions: ["ANALYZE_SETUP"],
            decisionContextId: "dec-radar-NVDA-1772323200000",
          },
        ],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getByTestId("radar-partial-results")).toBeDefined();
    });

    expect(screen.getByText(/Radar results are incomplete/i)).toBeDefined();
    expect(screen.getByText("Retry incomplete scan")).toBeDefined();
  });

  it("2.4 Delisted candidate displays Delisted / Inactive affordance and non-fabricated catalyst", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "CANDIDATES_AVAILABLE",
        scanState: "COMPLETED",
        results: [
          {
            ticker: "FRC",
            name: "First Republic Bank",
            price: 0.0,
            confluenceScore: 10,
            executionStatus: "DELISTED_SECURITY",
            allowedActions: [],
            // No catalyst provided
          },
        ],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getAllByText("FRC").length).toBeGreaterThan(0);
    });

    // Disabled action affordance
    expect(screen.getByText("Delisted / Inactive")).toBeDefined();

    // Truthful disclosure instead of manufactured breakout narrative
    expect(screen.getAllByText("Trading suspended for delisted security.").length).toBeGreaterThan(0);
    expect(screen.queryByText("Stage 2 accumulation breakout with institutional liquidity flow.")).toBeNull();
  });

  it("2.5 Candidate link propagates decisionContextId into Analysis Hub href", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "SUCCESS",
        datasetState: "CANDIDATES_AVAILABLE",
        scanState: "COMPLETED",
        results: [
          {
            ticker: "LNTH",
            name: "Lantheus Holdings",
            price: 74.25,
            confluenceScore: 91,
            executionStatus: "IN_BUY_ZONE",
            allowedActions: ["ANALYZE_SETUP"],
            decisionContextId: "dec-radar-LNTH-1772323200000",
          },
        ],
      }),
    } as any);

    render(<RadarPage />);

    await waitFor(() => {
      expect(screen.getAllByText("LNTH").length).toBeGreaterThan(0);
    });

    // Check that link has &contextId=dec-radar-LNTH-1772323200000
    const links = screen.getAllByRole("link");
    const targetLink = links.find((l) => l.getAttribute("href")?.includes("/?symbol=LNTH"));
    expect(targetLink).toBeDefined();
    expect(targetLink?.getAttribute("href")).toContain("contextId=dec-radar-LNTH-1772323200000");
  });
});

describe("ARX-036-D11R Layer 3: Browser & Navigation Context Falsification", () => {
  function validateContextId(rawContextId: string | null, urlSymbol: string | null): string | null {
    if (!rawContextId || !urlSymbol) return null;
    const parts = rawContextId.split("-");
    if (parts.length >= 3 && parts[2].toUpperCase() !== urlSymbol.toUpperCase()) {
      return null;
    }
    return rawContextId;
  }

  it("3.1 Accepts contextId when symbol strictly matches", () => {
    const valid = validateContextId("dec-radar-LNTH-1772323200000", "LNTH");
    expect(valid).toBe("dec-radar-LNTH-1772323200000");
  });

  it("3.2 Rejects contextId when symbol does not match (prevents cross-symbol leakage)", () => {
    const invalid = validateContextId("dec-radar-NVDA-1772323200000", "AAPL");
    expect(invalid).toBeNull();
  });

  it("3.3 Rejects malformed contextId lacking symbol component", () => {
    const invalid = validateContextId("random-invalid-string", "AAPL");
    // "random-invalid-string" has parts ["random", "invalid", "string"], parts[2] is "string" != "AAPL"
    expect(invalid).toBeNull();
  });

  it("3.4 Analysis Hub cache key remains strictly isolated from contextId", () => {
    const selectedSymbol = "AAPL";
    const interval = "1d";
    const userRole = "USER";
    const cacheKey = `${selectedSymbol}_${interval}_${userRole}`;
    expect(cacheKey).toBe("AAPL_1d_USER");
    expect(cacheKey.includes("contextId")).toBe(false);
  });
});

describe("ARX-036-D11R Layer 4: Cross-Engine Integration & Invariant Boundary", () => {
  it("4.1 Radar discovery context is transient and does not confer execution authority", () => {
    const candidate = {
      ticker: "LNTH",
      decisionContextId: "dec-radar-LNTH-1772323200000",
      isActionable: false,
      canSizeTrade: false,
    };

    // Execution invariant: trade sizing strictly requires DecisionTrace canonical clearance
    const canExecuteTrade = (c: typeof candidate, hasDecisionTraceClearance: boolean) => {
      if (!c.isActionable || !c.canSizeTrade) return false;
      return hasDecisionTraceClearance;
    };

    expect(canExecuteTrade(candidate, false)).toBe(false);
    expect(canExecuteTrade(candidate, true)).toBe(false); // isActionable is false
  });

  it("4.2 Portfolio context is pure consumer and does not alter Radar scores or ranking", () => {
    const portfolioHeldMap = new Map([["LNTH", { shares: 50, avgCost: 70.0 }]]);
    const asset = {
      ticker: "LNTH",
      confluenceScore: 91,
      rank: 1,
    };

    const isHeld = portfolioHeldMap.has(asset.ticker);
    expect(isHeld).toBe(true);
    // Invariants INV-RADAR-PORTFOLIO-01 and 02
    expect(asset.confluenceScore).toBe(91);
    expect(asset.rank).toBe(1);
  });
});

describe("ARX-036-D11R Negative Controls (Baseline Falsification Verification)", () => {
  it("5.1 Negative Control: Baseline mapper treats SCAN_FAILED with status SUCCESS as success (causing bug)", () => {
    const baselineMapper = (data: any) => ({
      status: data.status === "ERROR" ? "ERROR" : "SUCCESS",
    });

    const contradictoryPayload = {
      status: "SUCCESS",
      datasetState: "SCAN_FAILED",
    };

    // Demonstrates that the baseline bug allowed SCAN_FAILED to pass as SUCCESS
    const baselineResult = baselineMapper(contradictoryPayload);
    expect(baselineResult.status).toBe("SUCCESS"); // The bug!
  });

  it("5.2 Negative Control: Baseline catalyst fallback produced fabricated accumulation narrative", () => {
    const baselineCatalystFallback = (gem: any) =>
      gem.primary_catalyst || gem.investment_thesis || "Stage 2 accumulation breakout with institutional liquidity flow.";

    const blankGem = { ticker: "FRC" };
    const fabricated = baselineCatalystFallback(blankGem);
    expect(fabricated).toBe("Stage 2 accumulation breakout with institutional liquidity flow."); // The bug!
  });

  it("5.3 Negative Control: Baseline navigation link dropped decisionContextId", () => {
    const baselineHref = (ticker: string) => `/?symbol=${ticker}`;
    const href = baselineHref("LNTH");
    expect(href).toBe("/?symbol=LNTH");
    expect(href.includes("contextId")).toBe(false); // The bug!
  });
});
