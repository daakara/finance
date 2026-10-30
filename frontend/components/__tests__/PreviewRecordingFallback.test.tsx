// @vitest-environment-options {"url":"https://rc2.finance-xp8.pages.dev"}
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";
import PositionSizerModal from "../PositionSizerModal";
import DayTraderPositionSizer from "../DayTraderPositionSizer";
import UxReviewNotice from "../UxReviewNotice";
import { isUxReviewOnly } from "../../lib/uxReviewBoundary";
import { savePortfolioPositions, persistHoldingToApi, addPortfolioPosition, migrateLocalHoldingsToApi, removeHoldingFromApi, exitManualHoldingViaApi } from "../../lib/portfolio";

const input = { symbol: "AAPL", name: "Apple", shares: 50, entryPrice: 100, currentPrice: 100, addedAt: "2026-10-29", assetType: "Stock" as const };
beforeEach(() => { localStorage.clear(); vi.stubEnv("NEXT_PUBLIC_UX_PREVIEW_DEPLOYMENT", "false"); });
afterEach(() => { vi.unstubAllEnvs(); vi.restoreAllMocks(); });
describe("Preview recording fallback", () => {
  it("shows the global research notice without a review flag", () => {
    vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", undefined);
    render(<UxReviewNotice />);
    expect(screen.getByRole("note").textContent).toContain("Paper recording is locked");
  });
  it.each([undefined, "", "false", "invalid", "true"])("locks preview when review flag is %s", flag => {
    vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", flag);
    expect(isUxReviewOnly()).toBe(true);
    render(<PositionSizerModal isOpen onClose={() => {}} {...input} stopLoss={95} takeProfit1={115} canSizeTrade isActionable />);
    const button = screen.getByRole("button", { name: /paper recording locked/ }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    const before = { ...localStorage };
    fireEvent.click(button);
    const key = Object.keys(button).find(k => k.startsWith("__reactProps$"))!;
    act(() => (button as any)[key].onClick());
    expect({ ...localStorage }).toEqual(before);
  });
  it("blocks alternate local/API recording before identity or network writes", async () => {
    vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", undefined);
    const fetcher = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network forbidden"));
    savePortfolioPositions([input]);
    expect((await persistHoldingToApi(input)).success).toBe(false);
    expect((await addPortfolioPosition(input)).success).toBe(false);
    expect((await migrateLocalHoldingsToApi([input])).success).toBe(false);
    expect((await removeHoldingFromApi("AAPL")).success).toBe(false);
    expect((await exitManualHoldingViaApi(1, {} as any)).success).toBe(false);
    expect(fetcher).not.toHaveBeenCalled();
    expect(localStorage.length).toBe(0);
  });
  it("blocks the intraday calculator direct handler as well as its button", () => {
    vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", undefined);
    render(<DayTraderPositionSizer symbol="AAPL" data={{ currentPrice: 100, technicals: { atr_14: 2 }, analytics: {} } as any} />);
    const button = screen.getByRole("button", { name: /paper recording locked/ }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    const key = Object.keys(button).find(k => k.startsWith("__reactProps$"))!;
    act(() => (button as any)[key].onClick());
    expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
  });
});
