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

  it("9. WebKit/iOS touch tolerance: blur with relatedTarget=null does NOT dismiss menu", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");

    const rootContainer = triggerBtn.parentElement!;

    // On iOS Safari / WebKit, touch events emit blur with relatedTarget: null.
    // This must NOT prematurely close the menu. Outside dismissal is governed by handleOutsideInteraction.
    fireEvent.blur(rootContainer, { relatedTarget: null });

    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();
  });

  it("10. WebKit/iOS touch tolerance: focus within dropdown does NOT dismiss menu", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");

    const rootContainer = triggerBtn.parentElement!;
    const shortcutsItem = screen.getByRole("menuitem", { name: /Keyboard Shortcuts/i });

    // Focus movement inside the composite container
    fireEvent.blur(rootContainer, { relatedTarget: shortcutsItem });

    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("utilities-menu-dropdown")).not.toBeNull();
  });

  it("11. Portal architecture: dropdown panel mounts outside header into document.body", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("true");

    const menuPanel = document.getElementById("utilities-menu-dropdown");
    expect(menuPanel).not.toBeNull();

    const headerEl = document.querySelector("header");
    expect(headerEl).not.toBeNull();

    // The portal escapes the header ancestor completely
    expect(headerEl?.contains(menuPanel!)).toBe(false);
    expect(document.body.contains(menuPanel!)).toBe(true);
  });

  it("12. Clipping escape regression: OVERLAY_ESCAPES_HEADER_CLIP", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);

    const menuPanel = document.getElementById("utilities-menu-dropdown");
    expect(menuPanel).not.toBeNull();

    // Verify fixed position and z-index escaping header layer
    expect(menuPanel?.style.position).toBe("fixed");
    expect(menuPanel?.className).toContain("z-[9999]");
  });

  it("13. Visual viewport containment: panel remains bounded and scrollable", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);

    const menuPanel = document.getElementById("utilities-menu-dropdown");
    expect(menuPanel).not.toBeNull();

    // Bounded max-width and vertical scroll containment
    expect(menuPanel?.className).toContain("max-w-[calc(100vw-16px)]");
    expect(menuPanel?.className).toContain("overflow-y-auto");
  });

  it("14. Singularity: menu item click executes once and dismisses menu", () => {
    render(<Navbar />);

    const triggerBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(triggerBtn);

    const shortcutsBtn = screen.getByRole("menuitem", { name: /Keyboard Shortcuts/i });
    fireEvent.click(shortcutsBtn);

    // Menu dismissed
    expect(triggerBtn.getAttribute("aria-expanded")).toBe("false");
    expect(document.getElementById("utilities-menu-dropdown")).toBeNull();

    // Modal active
    expect(screen.getByRole("dialog", { name: "Keyboard Shortcuts Guide" })).toBeDefined();
  });
});

