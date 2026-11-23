import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAssetDiscovery } from "../useAssetDiscovery";
import { discoverAssets } from "../../lib/assetDiscovery";
vi.mock("../../lib/assetDiscovery", () => ({ discoverAssets: vi.fn() }));
const mock = vi.mocked(discoverAssets);
const match = { symbol: "ANT", name: "Anthropic PBC", exchange: "NASDAQ", type: "EQUITY", source: "YAHOO_FINANCE_SEARCH" as const, identityVerified: false as const };
beforeEach(() => { vi.useFakeTimers(); mock.mockReset(); });
afterEach(() => vi.useRealTimers());
describe("bounded live asset search", () => {
  it("debounces, caches suggestions and does not search while closed", async () => {
    mock.mockResolvedValue([match]);
    const { result, rerender } = renderHook(({ q, open }) => useAssetDiscovery(q, open), { initialProps: { q: "Anthropic", open: false } });
    await act(async () => { vi.advanceTimersByTime(400); });
    expect(mock).not.toHaveBeenCalled();
    rerender({ q: "Anthropic", open: true });
    await act(async () => { vi.advanceTimersByTime(299); });
    expect(mock).not.toHaveBeenCalled();
    await act(async () => { vi.advanceTimersByTime(1); });
    expect(result.current.results).toEqual([match]);
    rerender({ q: "", open: true }); rerender({ q: "Anthropic", open: true });
    expect(result.current.status).toBe("AVAILABLE");
    expect(mock).toHaveBeenCalledTimes(1);
  });
  it("aborts outdated requests and ignores responses arriving out of order", async () => {
    let resolveOld: (v: typeof match[]) => void = () => {};
    mock.mockImplementationOnce(() => new Promise(r => { resolveOld = r; })).mockResolvedValueOnce([{ ...match, symbol: "OTHER" }]);
    const { result, rerender } = renderHook(({ q }) => useAssetDiscovery(q, true), { initialProps: { q: "Anthropic" } });
    await act(async () => { vi.advanceTimersByTime(300); });
    const signal = mock.mock.calls[0][1];
    rerender({ q: "Other Company" });
    expect(signal?.aborted).toBe(true);
    await act(async () => { vi.advanceTimersByTime(300); });
    await act(async () => { resolveOld([match]); });
    expect(result.current.results[0].symbol).toBe("OTHER");
  });
  it("provider failure clears stale matches without disabling direct lookup", async () => {
    mock.mockRejectedValue(new Error("unavailable"));
    const { result } = renderHook(() => useAssetDiscovery("New Listing", true));
    await act(async () => { vi.advanceTimersByTime(300); });
    expect(result.current.status).toBe("UNAVAILABLE");
    expect(result.current.results).toEqual([]);
  });
});
