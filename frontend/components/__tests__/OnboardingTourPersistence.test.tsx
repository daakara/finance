import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import Navbar from "../Navbar";
import OnboardingTourModal from "../OnboardingTourModal";

// Mock dependencies of Navbar
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("../../lib/matomo", () => ({
  trackEvent: vi.fn(),
  trackRoleSwitch: vi.fn(),
  trackSearchSubmit: vi.fn(),
  trackOnboardingCompleted: vi.fn(),
  trackPrivacySettingsChanged: vi.fn(),
}));

describe("Quick Tour Skip Persistence & Auto-Open Invariants", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    localStorage.clear();
    vi.useRealTimers();
  });

  it("1. Auto-shows onboarding tour on first visit when flag is absent", () => {
    vi.useFakeTimers();
    render(<Navbar userRole="LONG_TERM" />);

    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();

    act(() => {
      vi.advanceTimersByTime(1100);
    });

    expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();
  });

  it("2. Skip button persists dismissal to localStorage and closes modal", () => {
    vi.useFakeTimers();
    render(<Navbar userRole="LONG_TERM" />);

    act(() => {
      vi.advanceTimersByTime(1100);
    });

    expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();

    const skipBtn = screen.getByRole("button", { name: /^Skip$/i });
    act(() => {
      fireEvent.click(skipBtn);
    });

    // Modal closed
    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();

    // Invariant: Skip must persist dismissal
    expect(localStorage.getItem("FINANCE_ONBOARDING_COMPLETED")).toBe("true");
  });

  it("3. Close ✕ button persists dismissal to localStorage and closes modal", () => {
    vi.useFakeTimers();
    render(<Navbar userRole="LONG_TERM" />);

    act(() => {
      vi.advanceTimersByTime(1100);
    });

    expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();

    const closeBtn = screen.getByRole("button", { name: /Close tour modal/i });
    act(() => {
      fireEvent.click(closeBtn);
    });

    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();
    expect(localStorage.getItem("FINANCE_ONBOARDING_COMPLETED")).toBe("true");
  });

  it("4. Escape key persists dismissal to localStorage and closes modal", () => {
    vi.useFakeTimers();
    render(<Navbar userRole="LONG_TERM" />);

    act(() => {
      vi.advanceTimersByTime(1100);
    });

    expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();

    act(() => {
      fireEvent.keyDown(window, { key: "Escape" });
    });

    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();
    expect(localStorage.getItem("FINANCE_ONBOARDING_COMPLETED")).toBe("true");
  });

  it("5. Completing the tour ('Get Started') persists completion to localStorage", () => {
    const handleClose = vi.fn();
    render(<OnboardingTourModal isOpen={true} onClose={handleClose} />);

    // Step 1 -> 2
    fireEvent.click(screen.getByRole("button", { name: /Next →/i }));
    // Step 2 -> 3
    fireEvent.click(screen.getByRole("button", { name: /Next →/i }));
    // Step 3 -> 4
    fireEvent.click(screen.getByRole("button", { name: /Next →/i }));

    // Step 4: Get Started 🚀
    const getStartedBtn = screen.getByRole("button", { name: /Get Started 🚀/i });
    fireEvent.click(getStartedBtn);

    expect(handleClose).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem("FINANCE_ONBOARDING_COMPLETED")).toBe("true");
  });

  it("6. Reload / Remount suppression: Does not auto-show when flag is set in localStorage", () => {
    vi.useFakeTimers();
    localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");

    const { unmount } = render(<Navbar userRole="LONG_TERM" />);

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();

    // Simulate navigation/remount
    unmount();
    render(<Navbar userRole="LONG_TERM" />);

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();
  });

  it("7. Manual launch from menu opens tour even after prior Skip/Completion", () => {
    vi.useFakeTimers();
    localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");

    render(<Navbar userRole="LONG_TERM" />);

    // Initially closed
    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();

    // Trigger manual open from menu
    const utilitiesBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(utilitiesBtn);

    const tourMenuBtn = screen.getByRole("menuitem", { name: /Guided Onboarding Tour/i });
    act(() => {
      fireEvent.click(tourMenuBtn);
    });

    // Tour opens
    expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();
  });

  it("8. Pending auto-open timer is cancelled when tour is opened or closed early (zero timer race)", () => {
    vi.useFakeTimers();
    // Start with empty localStorage
    render(<Navbar userRole="LONG_TERM" />);

    // User opens menu and launches tour at 300ms (before the 1000ms timer fires)
    act(() => {
      vi.advanceTimersByTime(300);
    });

    const utilitiesBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
    fireEvent.click(utilitiesBtn);
    const tourMenuBtn = screen.getByRole("menuitem", { name: /Guided Onboarding Tour/i });
    act(() => {
      fireEvent.click(tourMenuBtn);
    });

    expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();

    // User clicks Skip at 500ms
    const skipBtn = screen.getByRole("button", { name: /^Skip$/i });
    act(() => {
      fireEvent.click(skipBtn);
    });

    // Modal is now closed
    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();

    // Advance past the original 1000ms auto-open time
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // Modal must NOT reopen
    expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();
  });

  it("9. Modal retains aria-modal='true' and z-[1200] background isolation", () => {
    render(<OnboardingTourModal isOpen={true} onClose={vi.fn()} />);

    const dialog = screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.className).toContain("z-[1200]");
    expect(dialog.className).toContain("fixed inset-0");
  });
});
