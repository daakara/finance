import React from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import WatchlistDrawer from "../WatchlistDrawer";
import WatchlistDrawerTrigger from "../WatchlistDrawerTrigger";
import { useUIStore } from "../../../state/ui-store";

describe("WatchlistDrawer & UI Store (Milestone W1.6)", () => {
  beforeEach(() => {
    localStorage.clear();
    useUIStore.setState({ watchlistOpen: false, openedAtMs: null });
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe("UIStore State Management & Persistence", () => {
    it("defaults to closed state (false)", () => {
      expect(useUIStore.getState().watchlistOpen).toBe(false);
    });

    it("openWatchlist sets state to true and updates localStorage", () => {
      useUIStore.getState().openWatchlist("mouse");
      expect(useUIStore.getState().watchlistOpen).toBe(true);
      expect(localStorage.getItem("arx-watchlist-open")).toBe("true");
    });

    it("closeWatchlist sets state to false and updates localStorage", () => {
      useUIStore.getState().openWatchlist("mouse");
      useUIStore.getState().closeWatchlist();
      expect(useUIStore.getState().watchlistOpen).toBe(false);
      expect(localStorage.getItem("arx-watchlist-open")).toBe("false");
    });

    it("toggleWatchlist inverts state", () => {
      useUIStore.getState().toggleWatchlist("keyboard");
      expect(useUIStore.getState().watchlistOpen).toBe(true);
      useUIStore.getState().toggleWatchlist("keyboard");
      expect(useUIStore.getState().watchlistOpen).toBe(false);
    });
  });

  describe("WatchlistDrawerTrigger", () => {
    it("renders trigger with accessibility attributes", () => {
      render(<WatchlistDrawerTrigger />);
      const trigger = screen.getByTestId("watchlist-drawer-trigger");
      expect(trigger).toBeDefined();
      expect(trigger.getAttribute("aria-controls")).toBe("watchlist-drawer");
      expect(trigger.getAttribute("aria-expanded")).toBe("false");
    });

    it("clicking trigger toggles open state in store", () => {
      render(<WatchlistDrawerTrigger />);
      const trigger = screen.getByTestId("watchlist-drawer-trigger");
      fireEvent.click(trigger);
      expect(useUIStore.getState().watchlistOpen).toBe(true);
      expect(trigger.getAttribute("aria-expanded")).toBe("true");
    });
  });

  describe("WatchlistDrawer Component Accessibility & Slide-Over", () => {
    it("renders dialog with proper accessibility attributes", () => {
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
          isOpen={true}
        />
      );

      const drawer = screen.getByTestId("watchlist-drawer");
      expect(drawer).toBeDefined();
      expect(drawer.getAttribute("role")).toBe("dialog");
      expect(drawer.getAttribute("aria-modal")).toBe("true");
      expect(drawer.getAttribute("aria-label")).toBe("Watchlist Drawer");
      expect(drawer.className).toContain("translate-x-0");
      expect(drawer.className).toContain("lg:w-80");
    });

    it("renders offscreen when closed (-translate-x-full)", () => {
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
          isOpen={false}
        />
      );

      const drawer = screen.getByTestId("watchlist-drawer");
      expect(drawer.className).toContain("-translate-x-full");
    });

    it("clicking close button calls onClose", () => {
      const handleClose = vi.fn();
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
          isOpen={true}
          onClose={handleClose}
        />
      );

      const closeBtn = screen.getByTestId("watchlist-drawer-close");
      fireEvent.click(closeBtn);
      expect(handleClose).toHaveBeenCalledTimes(1);
    });

    it("clicking backdrop overlay calls onClose", () => {
      const handleClose = vi.fn();
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
          isOpen={true}
          onClose={handleClose}
        />
      );

      const backdrop = screen.getByTestId("watchlist-drawer-backdrop");
      fireEvent.click(backdrop);
      expect(handleClose).toHaveBeenCalledTimes(1);
    });
  });

  describe("Keyboard Hotkeys Interaction", () => {
    it("pressing '[' toggles drawer open and closed", () => {
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
        />
      );

      expect(useUIStore.getState().watchlistOpen).toBe(false);

      // Press '['
      fireEvent.keyDown(window, { key: "[" });
      expect(useUIStore.getState().watchlistOpen).toBe(true);

      // Press '[' again
      fireEvent.keyDown(window, { key: "[" });
      expect(useUIStore.getState().watchlistOpen).toBe(false);
    });

    it("pressing 'Ctrl+B' toggles drawer", () => {
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
        />
      );

      // Press Ctrl+B
      fireEvent.keyDown(window, { key: "b", ctrlKey: true });
      expect(useUIStore.getState().watchlistOpen).toBe(true);

      fireEvent.keyDown(window, { key: "b", ctrlKey: true });
      expect(useUIStore.getState().watchlistOpen).toBe(false);
    });

    it("pressing 'Escape' closes an open drawer", () => {
      useUIStore.setState({ watchlistOpen: true });
      render(
        <WatchlistDrawer
          activeSymbol="AAPL"
          onSelectSymbol={vi.fn()}
        />
      );

      fireEvent.keyDown(window, { key: "Escape" });
      expect(useUIStore.getState().watchlistOpen).toBe(false);
    });
  });
});
