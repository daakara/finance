import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import UniversalOmniSearch from "../UniversalOmniSearch";
import CommandPaletteModal from "../CommandPaletteModal";
// jsdom has no scrolling/layout implementation.
Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
vi.mock("../../hooks/useAssetDiscovery", () => ({ useAssetDiscovery: (query: string) => ({
  status: "AVAILABLE", results: query.toLowerCase().includes("anthropic") ? [{ symbol: "ANT", name: "Anthropic PBC", exchange: "NASDAQ", type: "EQUITY", identityVerified: false }] : [],
}) }));
vi.mock("../../lib/api", () => ({ SpotPriceRegistry: { get: () => null },
  prefetchAssetAnalytics: vi.fn(), fetchTacticalSetups: vi.fn().mockResolvedValue([]) }));
vi.mock("../../lib/matomo", () => ({ trackSearchResolve: vi.fn() }));
describe("company-name discovery in both search menus", () => {
  it("shows provider company matches absent from presets, without a price", () => {
    render(<UniversalOmniSearch />);
    act(() => { window.dispatchEvent(new CustomEvent("arx:open-omni-search")); });
    fireEvent.change(screen.getByLabelText("Type any ticker symbol or asset name"), { target: { value: "Anthropic" } });
    expect(screen.getByText("Anthropic PBC")).toBeDefined();
    expect(screen.getByText("ANT")).toBeDefined();
    expect(screen.getByText(/analysis availability checked when opened/i)).toBeDefined();
  });
  it("command palette selects the discovered symbol instead of company text", async () => {
    const select = vi.fn(); const close = vi.fn();
    await act(async () => { render(<CommandPaletteModal isOpen onClose={close} onSelectSymbol={select} />); });
    fireEvent.change(screen.getByLabelText("Command search query"), { target: { value: "Anthropic" } });
    fireEvent.click(screen.getByText("ANT — Anthropic PBC"));
    expect(select).toHaveBeenCalledWith("ANT");
    expect(close).toHaveBeenCalledTimes(1);
  });
});
