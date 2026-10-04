import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { usePortfolioContext } from "../usePortfolioContext";
import * as portfolioModule from "../../lib/portfolio";

describe("usePortfolioContext Hook (INV-RADAR-PORTFOLIO-01..07)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("initializes in loading state and resolves to verified holdings", async () => {
    vi.spyOn(portfolioModule, "fetchAuthoritativePortfolio").mockResolvedValue({
      isVerified: true,
      positions: [
        {
          symbol: "NVDA",
          shares: 10,
          entryPrice: 120.0,
          currentPrice: 130.0,
          unrealizedPnL: 100.0,
          unrealizedPnLPct: 8.33,
        },
        {
          symbol: "BRK.B",
          shares: 5,
          entryPrice: 450.0,
          currentPrice: 460.0,
          unrealizedPnL: 50.0,
          unrealizedPnLPct: 2.22,
        },
      ],
    });

    const { result } = renderHook(() => usePortfolioContext());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isVerified).toBe(true);
    expect(result.current.isDegraded).toBe(false);
    expect(result.current.holdingsCount).toBe(2);

    // HELD tests (case-insensitive and normalized)
    expect(result.current.getOwnershipState("NVDA")).toBe("HELD");
    expect(result.current.getOwnershipState("nvda")).toBe("HELD");
    expect(result.current.getOwnershipState("BRK-B")).toBe("HELD");
    expect(result.current.getOwnershipState("brk.b")).toBe("HELD");

    // NOT_HELD test (only valid when portfolio is verified)
    expect(result.current.getOwnershipState("AAPL")).toBe("NOT_HELD");

    // Ambiguous / invalid symbol tests fail closed to UNKNOWN
    expect(result.current.getOwnershipState("")).toBe("UNKNOWN");
    expect(result.current.getOwnershipState(null)).toBe("UNKNOWN");
    expect(result.current.getOwnershipState(undefined)).toBe("UNKNOWN");
    expect(result.current.getOwnershipState("INVALID$$$")).toBe("UNKNOWN");

    // getHolding retrieval
    const nvdaHolding = result.current.getHolding("NVDA");
    expect(nvdaHolding?.shares).toBe(10);
    expect(nvdaHolding?.entryPrice).toBe(120.0);
  });

  it("fails closed to UNKNOWN when server portfolio is unavailable or unverified (INV-RADAR-PORTFOLIO-05 & 06)", async () => {
    vi.spyOn(portfolioModule, "fetchAuthoritativePortfolio").mockResolvedValue({
      isVerified: false,
      positions: [],
      error: "Network 503 Service Unavailable",
    });

    const { result } = renderHook(() => usePortfolioContext());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isVerified).toBe(false);
    expect(result.current.isDegraded).toBe(true);
    expect(result.current.error).toContain("503");
    expect(result.current.holdingsCount).toBe(0);

    // CRITICAL INVARIANT: Symbols are NEVER declared NOT_HELD when portfolio is unverified
    expect(result.current.getOwnershipState("NVDA")).toBe("UNKNOWN");
    expect(result.current.getOwnershipState("AAPL")).toBe("UNKNOWN");
    expect(result.current.getHolding("NVDA")).toBeUndefined();
  });

  it("handles exception thrown during fetch gracefully (fail-closed)", async () => {
    vi.spyOn(portfolioModule, "fetchAuthoritativePortfolio").mockRejectedValue(
      new Error("Failed to fetch")
    );

    const { result } = renderHook(() => usePortfolioContext());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isVerified).toBe(false);
    expect(result.current.isDegraded).toBe(true);
    expect(result.current.error).toBe("Failed to fetch");
    expect(result.current.getOwnershipState("NVDA")).toBe("UNKNOWN");
  });

  it("refreshes holdings when finance:portfolio-updated window event fires", async () => {
    let callCount = 0;
    vi.spyOn(portfolioModule, "fetchAuthoritativePortfolio").mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        return { isVerified: true, positions: [] };
      }
      return {
        isVerified: true,
        positions: [
          {
            symbol: "TSLA",
            shares: 20,
            entryPrice: 200.0,
            currentPrice: 220.0,
            unrealizedPnL: 400.0,
            unrealizedPnLPct: 10.0,
          },
        ],
      };
    });

    const { result } = renderHook(() => usePortfolioContext());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.getOwnershipState("TSLA")).toBe("NOT_HELD");

    // Dispatch update event
    act(() => {
      window.dispatchEvent(new Event("finance:portfolio-updated"));
    });

    await waitFor(() => {
      expect(result.current.getOwnershipState("TSLA")).toBe("HELD");
    });

    expect(result.current.holdingsCount).toBe(1);
    expect(result.current.getHolding("TSLA")?.shares).toBe(20);
  });
});
