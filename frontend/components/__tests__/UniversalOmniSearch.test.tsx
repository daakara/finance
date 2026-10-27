import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import UniversalOmniSearch from "../UniversalOmniSearch";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

vi.mock("../../lib/matomo", () => ({
  trackSearchResolve: vi.fn(),
}));

describe("UniversalOmniSearch Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("is closed by default and does not render search input", () => {
    render(<UniversalOmniSearch />);

    expect(screen.queryByRole("dialog", { name: "Universal Asset Search" })).toBeNull();
    expect(document.getElementById("omni-search-input")).toBeNull();
  });

  it("opens modal when arx:open-omni-search window event is dispatched", async () => {
    render(<UniversalOmniSearch />);

    act(() => {
      window.dispatchEvent(new CustomEvent("arx:open-omni-search"));
    });

    const dialog = await screen.findByRole("dialog", { name: "Universal Asset Search" });
    expect(dialog).toBeDefined();

    const input = document.getElementById("omni-search-input") as HTMLInputElement;
    expect(input).toBeDefined();
    expect(input.placeholder).toContain("Type ANY ticker");
  });

  it("dismisses search dialog when Escape key is pressed", async () => {
    render(<UniversalOmniSearch />);

    act(() => {
      window.dispatchEvent(new CustomEvent("arx:open-omni-search"));
    });

    expect(await screen.findByRole("dialog", { name: "Universal Asset Search" })).toBeDefined();

    act(() => {
      fireEvent.keyDown(window, { key: "Escape" });
    });

    expect(screen.queryByRole("dialog", { name: "Universal Asset Search" })).toBeNull();
  });

  it("dismisses search dialog when ESC close button is clicked", async () => {
    render(<UniversalOmniSearch />);

    act(() => {
      window.dispatchEvent(new CustomEvent("arx:open-omni-search"));
    });

    const closeBtn = await screen.findByRole("button", { name: /Close search dialog/i });
    fireEvent.click(closeBtn);

    expect(screen.queryByRole("dialog", { name: "Universal Asset Search" })).toBeNull();
  });
});
