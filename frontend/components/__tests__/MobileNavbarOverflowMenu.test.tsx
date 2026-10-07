import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import Navbar from "../Navbar";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

// Mock matomo
vi.mock("../../lib/matomo", () => ({
  trackFirstHubNavigation: vi.fn(),
  trackOnboardingCompleted: vi.fn(),
}));

describe("Mobile Navigation Overflow Menu Interaction Architecture", () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("1. Initial state: closed with valid accessibility attributes", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    expect(triggerBtn).toBeDefined();
    expect(triggerBtn.id).toBe("utilities-menu-btn");
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(triggerBtn.getAttribute("aria-haspopup")).toBe("menu");
    expect(triggerBtn.getAttribute("aria-controls")).toBe("utilities-menu-dropdown");

    // Touch target >= 44x44
    expect(triggerBtn.className).toContain("min-h-[44px]");
    expect(triggerBtn.className).toContain("min-w-[44px]");

    // Menu panel must not be in DOM initially
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();
    expect(screen.queryByRole("menu", { name: "Terminal Utilities" })).toBeNull();
  });

  it("2. Mobile trigger: single tap opens menu and remains open after pointer cycle", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });

    // Emulate pointerdown -> pointerup -> click cycle on mobile tap
    fireEvent.pointerDown(triggerBtn);
    fireEvent.pointerUp(triggerBtn);
    fireEvent.click(triggerBtn);

    // Menu is open and remains open
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    const menuPanel = document.getElementById("utilities-menu-dropdown");
    expect(menuPanel).not.toBeNull();
    expect(menuPanel?.getAttribute("role")).toBe("menu");
    expect(menuPanel?.getAttribute("aria-label")).toBe("Terminal Utilities");

    // Menu items are available
    expect(screen.getByRole("menuitem", { name: /Keyboard Shortcuts/i })).toBeDefined();
    expect(screen.getByRole("menuitem", { name: /Guided Onboarding Tour/i })).toBeDefined();
    expect(screen.getByRole("menuitem", { name: /Privacy & Telemetry/i })).toBeDefined();
    expect(screen.getByRole("menuitem", { name: /Purge Cache & Re-sync/i })).toBeDefined();
  });

  it("3. Item interaction: tapping a menu item executes action and closes menu", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);

    const shortcutsItem = screen.getByRole("menuitem", { name: /Keyboard Shortcuts/i });
    fireEvent.click(shortcutsItem);

    // Menu closed
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();

    // Action executed: Shortcuts modal opened
    expect(screen.getByRole("dialog", { name: "Keyboard Shortcuts Guide" })).toBeDefined();
  });

  it("4. Outside interaction: tapping outside closes menu", () => {
    render(
      <div>
        <div data-testid="outside-canvas">Outside Canvas</div>
        <Navbar />
      </div>
    );

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();

    // Tap outside
    const outsideEl = screen.getByTestId("outside-canvas");
    fireEvent.pointerDown(outsideEl);

    // Menu closes
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();
  });

  it("5. Repeatability: open -> close -> open cycles work consistently", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });

    // Cycle 1: Open then toggle close
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();

    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();

    // Cycle 2: Open then outside close
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();

    fireEvent.pointerDown(document.body);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();

    // Cycle 3: Reopen again
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();
  });

  it("6. No double-toggle: a single tap cannot transition closed -> open -> closed", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });

    // A single mobile tap emits pointerdown -> pointerup -> click on the button
    fireEvent.pointerDown(triggerBtn);
    fireEvent.pointerUp(triggerBtn);
    fireEvent.click(triggerBtn);

    // Must be OPEN, not closed
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();
  });

  it("7. Keyboard accessibility: Escape key dismisses menu and returns focus to trigger", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    triggerBtn.focus();
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");

    // Press Escape
    fireEvent.keyDown(window, { key: "Escape" });

    // Menu closed
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();

    // Focus restored to trigger
    expect(document.activeElement).toBe(triggerBtn);
  });

  it("8. Focus containment: blurring outside composite closes menu", () => {
    render(
      <div>
        <button id="outside-button">Outside</button>
        <Navbar />
      </div>
    );

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");

    const rootContainer = triggerBtn.parentElement!;
    const outsideBtn = screen.getByRole("button", { name: "Outside" });

    // Blur from root to outside element
    fireEvent.blur(rootContainer, { relatedTarget: outsideBtn });

    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();
  });
});
