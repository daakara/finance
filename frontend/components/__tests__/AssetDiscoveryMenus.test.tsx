import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import UniversalOmniSearch from "../UniversalOmniSearch";
import DiscoveryLandingView from "../discovery/DiscoveryLandingView";
import CommandPaletteModal from "../CommandPaletteModal";

// jsdom has no scrolling/layout implementation.
Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });

let mockDiscoveryState = {
  status: "AVAILABLE" as "IDLE" | "LOADING" | "AVAILABLE" | "UNAVAILABLE",
  results: [
    { symbol: "ANT", name: "Anthropic PBC", exchange: "NASDAQ", type: "EQUITY", source: "ALPACA_ASSET_DIRECTORY" as const, identityVerified: false as const }
  ],
};

vi.mock("../../hooks/useAssetDiscovery", () => ({
  useAssetDiscovery: () => mockDiscoveryState,
}));

vi.mock("../../lib/api", () => ({
  SpotPriceRegistry: { get: () => null },
  prefetchAssetAnalytics: vi.fn(),
  fetchTacticalSetups: vi.fn().mockResolvedValue([])
}));

vi.mock("../../lib/matomo", () => ({ trackSearchResolve: vi.fn() }));

describe("company-name discovery across search menus", () => {
  beforeEach(() => {
    mockDiscoveryState = {
      status: "AVAILABLE",
      results: [
        { symbol: "ANT", name: "Anthropic PBC", exchange: "NASDAQ", type: "EQUITY", source: "ALPACA_ASSET_DIRECTORY", identityVerified: false }
      ],
    };
  });

  it("shows provider company matches absent from presets, without a price", () => {
    render(<UniversalOmniSearch />);
    act(() => { window.dispatchEvent(new CustomEvent("arx:open-omni-search")); });
    fireEvent.change(screen.getByLabelText("Type any ticker symbol or asset name"), { target: { value: "Anthropic" } });
    expect(screen.getByText("Anthropic PBC")).toBeDefined();
    expect(screen.getByText("ANT")).toBeDefined();
    expect(screen.getByText(/analysis availability checked when opened/i)).toBeDefined();
  });

  it("hides arbitrary custom symbol launch when matching provider results exist in OmniSearch", () => {
    render(<UniversalOmniSearch />);
    act(() => { window.dispatchEvent(new CustomEvent("arx:open-omni-search")); });
    fireEvent.change(screen.getByLabelText("Type any ticker symbol or asset name"), { target: { value: "Anthropic" } });
    expect(screen.queryByText(/Run Live Analysis for Custom Asset/i)).toBeNull();
  });

  it("disables OmniSearch submit and blocks execution while discovery is pending", () => {
    mockDiscoveryState = { status: "LOADING", results: [] };
    render(<UniversalOmniSearch />);
    act(() => { window.dispatchEvent(new CustomEvent("arx:open-omni-search")); });
    const input = screen.getByLabelText("Type any ticker symbol or asset name");
    fireEvent.change(input, { target: { value: "Anthropic" } });

    const submitBtn = screen.getByLabelText(/Submit search for ANTHROPIC/i);
    expect((submitBtn as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByText(/Run Live Analysis for Custom Asset/i)).toBeNull();
  });

  it("preserves direct ticker entry in OmniSearch when 0 provider matches exist", () => {
    mockDiscoveryState = { status: "AVAILABLE", results: [] };
    render(<UniversalOmniSearch />);
    act(() => { window.dispatchEvent(new CustomEvent("arx:open-omni-search")); });
    fireEvent.change(screen.getByLabelText("Type any ticker symbol or asset name"), { target: { value: "BTU" } });

    expect(screen.getByText(/Run Live Analysis for Custom Asset:/i)).toBeDefined();
    expect(screen.getByText("BTU")).toBeDefined();
  });

  it("command palette selects the discovered symbol instead of company text", async () => {
    const select = vi.fn(); const close = vi.fn();
    await act(async () => { render(<CommandPaletteModal isOpen onClose={close} onSelectSymbol={select} />); });
    fireEvent.change(screen.getByLabelText("Command search query"), { target: { value: "Anthropic" } });
    fireEvent.click(screen.getByText("ANT — Anthropic PBC"));
    expect(select).toHaveBeenCalledWith("ANT");
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("command palette blocks submission while discovery is loading", async () => {
    mockDiscoveryState = { status: "LOADING", results: [] };
    const select = vi.fn(); const close = vi.fn();
    await act(async () => { render(<CommandPaletteModal isOpen onClose={close} onSelectSymbol={select} />); });
    const input = screen.getByLabelText("Command search query");
    fireEvent.change(input, { target: { value: "Anthropic" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(select).not.toHaveBeenCalled();
    expect(close).not.toHaveBeenCalled();
  });

  it("homepage search resolves company name to ticker with Enter/click parity", () => {
    const onSelect = vi.fn();
    render(<DiscoveryLandingView onSelectSymbol={onSelect} macroData={null} />);
    const input = screen.getByTestId("landing-search-input");
    fireEvent.change(input, { target: { value: "Anthropic" } });

    // Live suggestion card appears
    expect(screen.getByText("Anthropic PBC")).toBeDefined();
    expect(screen.getByText("ANT")).toBeDefined();

    // Enter submission resolves to ANT
    fireEvent.submit(input.closest("form")!);
    expect(onSelect).toHaveBeenCalledWith("ANT");
  });

  it("homepage search blocks submission while discovery is loading", () => {
    mockDiscoveryState = { status: "LOADING", results: [] };
    const onSelect = vi.fn();
    render(<DiscoveryLandingView onSelectSymbol={onSelect} macroData={null} />);
    const input = screen.getByTestId("landing-search-input");
    fireEvent.change(input, { target: { value: "Anthropic" } });

    const analyzeBtn = screen.getByRole("button", { name: /Submit search for ANTHROPIC/i });
    expect((analyzeBtn as HTMLButtonElement).disabled).toBe(true);

    fireEvent.submit(input.closest("form")!);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("homepage search requires explicit selection for ambiguous matches", () => {
    mockDiscoveryState = {
      status: "AVAILABLE",
      results: [
        { symbol: "AAA", name: "American Airlines Group", exchange: "NASDAQ", type: "EQUITY", source: "ALPACA_ASSET_DIRECTORY", identityVerified: false },
        { symbol: "AAL", name: "American Assets Trust", exchange: "NYSE", type: "EQUITY", source: "ALPACA_ASSET_DIRECTORY", identityVerified: false },
      ],
    };
    const onSelect = vi.fn();
    render(<DiscoveryLandingView onSelectSymbol={onSelect} macroData={null} />);
    const input = screen.getByTestId("landing-search-input");
    fireEvent.change(input, { target: { value: "American" } });

    // Enter submission does NOT guess among ambiguous matches
    fireEvent.submit(input.closest("form")!);
    expect(onSelect).not.toHaveBeenCalled();

    // Explicit click on one result selects that exact symbol
    fireEvent.click(screen.getByText("American Airlines Group"));
    expect(onSelect).toHaveBeenCalledWith("AAA");
  });

  it("homepage search preserves direct ticker entry when 0 matches exist", () => {
    mockDiscoveryState = { status: "AVAILABLE", results: [] };
    const onSelect = vi.fn();
    render(<DiscoveryLandingView onSelectSymbol={onSelect} macroData={null} />);
    const input = screen.getByTestId("landing-search-input");
    fireEvent.change(input, { target: { value: "BTU" } });

    const analyzeBtn = screen.getByRole("button", { name: /Submit search for BTU/i });
    expect((analyzeBtn as HTMLButtonElement).disabled).toBe(false);

    fireEvent.submit(input.closest("form")!);
    expect(onSelect).toHaveBeenCalledWith("BTU");
  });

  it("homepage search handles provider unavailability gracefully", () => {
    mockDiscoveryState = { status: "UNAVAILABLE", results: [] };
    const onSelect = vi.fn();
    render(<DiscoveryLandingView onSelectSymbol={onSelect} macroData={null} />);
    const input = screen.getByTestId("landing-search-input");
    fireEvent.change(input, { target: { value: "AAPL" } });

    expect(screen.getByText(/Live search unavailable. Saved matches and direct ticker lookup remain available./i)).toBeDefined();

    // Direct entry still works
    fireEvent.submit(input.closest("form")!);
    expect(onSelect).toHaveBeenCalledWith("AAPL");
  });
});
