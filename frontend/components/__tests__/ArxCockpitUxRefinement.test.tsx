import React from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import ExperienceModeToggle from "../experience/ExperienceModeToggle";
import StandardTerminalView from "../terminal/StandardTerminalView";
import GuidedTerminalView from "../terminal/GuidedTerminalView";
import AdvancedTerminalView from "../terminal/AdvancedTerminalView";
import OnboardingTourModal from "../OnboardingTourModal";
import Navbar from "../Navbar";
import { useExperienceStore } from "../../state/experience-store";
import { QuantitativeInsight } from "../../types/insight";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

// Mock matomo
vi.mock("../../lib/matomo", () => ({
  trackFirstHubNavigation: vi.fn(),
  trackOnboardingCompleted: vi.fn(),
  trackAnalysisToSetup: vi.fn(),
  trackRoleSwitch: vi.fn(),
  trackSymbolSearch: vi.fn(),
  trackWorkspaceSwitch: vi.fn(),
}));

const mockInsight: QuantitativeInsight = {
  symbol: "AAPL",
  price: 185.5,
  changePct: 1.25,
  setupScore: 82,
  verdictLabel: "FAVORABLE SETUP",
  bias: "BULLISH",
  confidence: "HIGH",
  actionable: true,
  standard: {
    bottomLine: "Constructive multi-pillar momentum alignment.",
    signalsRatio: "4/5 models agree",
    setupSummary: "Minervini Stage 2 Continuation",
    confluenceBreakdown: [
      { dimension: "Momentum", score: 85 },
      { dimension: "Volume", score: 78 },
      { dimension: "Regime", score: 82 },
    ],
    keyLevels: {
      currentPrice: 185.5,
      watchZone: "$182.00 - $186.00",
      entry: 185.0,
      stopLoss: 180.0,
      target1: 195.0,
      target1Pct: 5.4,
      sma50: 182.0,
      sma200: 175.0,
    },
  },
  human: {
    assessmentDescription: "Strong accumulation volume.",
    reclaimMilestone: "Reclaimed 50D SMA with volume confirmation.",
    executionReadiness: "Actionable inside buy zone.",
    whyPills: [
      { category: "Momentum", sentiment: "positive", text: "Trend is intact" },
      { category: "Volume", sentiment: "positive", text: "Accumulation patterns" },
    ],
    watchLevels: {
      watchZone: "$182.00 - $186.00",
      keyLevel: "$182.00",
      stopLoss: "$180.00",
    },
  },
  advanced: {
    rsi: 58.4,
    relativeStrengthScore: 88,
    volatilityRegime: "NORMAL",
  },
  terminalState: {
    decisionState: "ACTIONABLE_SETUP",
    overallEligibility: "ELIGIBLE",
  },
};

describe("ARX Cockpit UX Refinement Suite", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    useExperienceStore.setState({ mode: "STANDARD" });
  });

  afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  describe("A2: Removal of Redundant Terminal View Strips", () => {
    it("StandardTerminalView does not render redundant STANDARD EXPERIENCE strip", () => {
      const onOpenWhy = vi.fn();
      render(<StandardTerminalView insight={mockInsight} onOpenWhy={onOpenWhy} />);

      expect(screen.queryByText(/STANDARD EXPERIENCE/i)).toBeNull();
      const whyBtn = screen.getByRole("button", { name: /Why Score 82\?/i });
      expect(whyBtn).toBeDefined();
      expect(whyBtn.className).toContain("min-h-[36px]");
      expect(whyBtn.className).toContain("text-xs");

      fireEvent.click(whyBtn);
      expect(onOpenWhy).toHaveBeenCalledTimes(1);
    });

    it("GuidedTerminalView does not render redundant GUIDED EXPERIENCE strip", () => {
      const onOpenWhy = vi.fn();
      render(<GuidedTerminalView insight={mockInsight} onOpenWhy={onOpenWhy} />);

      expect(screen.queryByText(/GUIDED EXPERIENCE/i)).toBeNull();
      const explainBtn = screen.getByRole("button", { name: /Explain Score/i });
      expect(explainBtn).toBeDefined();
      expect(explainBtn.className).toContain("min-h-[36px]");
      expect(explainBtn.className).toContain("text-xs");

      fireEvent.click(explainBtn);
      expect(onOpenWhy).toHaveBeenCalledTimes(1);
    });

    it("AdvancedTerminalView does not render redundant ADVANCED WORKSTATION strip", () => {
      const onOpenWhy = vi.fn();
      render(<AdvancedTerminalView insight={mockInsight} onOpenWhy={onOpenWhy} />);

      expect(screen.queryByText(/ADVANCED WORKSTATION/i)).toBeNull();
      const decompBtn = screen.getByRole("button", { name: /Decompose Score/i });
      expect(decompBtn).toBeDefined();
      expect(decompBtn.className).toContain("min-h-[36px]");
      expect(decompBtn.className).toContain("text-xs");

      fireEvent.click(decompBtn);
      expect(onOpenWhy).toHaveBeenCalledTimes(1);
    });
  });

  describe("A3 & A5: Unified Utilities Dropdown & Iconography in Navbar", () => {
    it("renders unified utilities menu button and expands grouped menu on click", () => {
      render(<Navbar userRole="LONG_TERM" />);

      const utilitiesBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
      expect(utilitiesBtn).toBeDefined();
      expect(utilitiesBtn.getAttribute("aria-expanded")).toBe("false");

      // Click to open
      fireEvent.click(utilitiesBtn);
      expect(utilitiesBtn.getAttribute("aria-expanded")).toBe("true");

      // Verify semantic sections: Help, Privacy, System
      expect(screen.getByText(/Help & Navigation/i)).toBeDefined();
      expect(screen.getByText(/Privacy & Diagnostics/i)).toBeDefined();
      expect(screen.getByText(/System & Cache/i)).toBeDefined();

      // Verify specific utility triggers inside the menu
      expect(screen.getByRole("menuitem", { name: /Keyboard Shortcuts/i })).toBeDefined();
      expect(screen.getByRole("menuitem", { name: /Guided Onboarding Tour/i })).toBeDefined();
      expect(screen.getByRole("menuitem", { name: /Privacy & Telemetry/i })).toBeDefined();
      expect(screen.getByRole("menuitem", { name: /Purge Cache & Re-sync/i })).toBeDefined();
    });

    it("requires user confirmation before performing destructive cache purge", () => {
      const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
      render(<Navbar userRole="LONG_TERM" />);

      const utilitiesBtn = screen.getByRole("button", { name: /Terminal Utilities and System Settings/i });
      fireEvent.click(utilitiesBtn);

      const purgeBtn = screen.getByRole("menuitem", { name: /Purge Cache & Re-sync/i });
      fireEvent.click(purgeBtn);

      expect(confirmSpy).toHaveBeenCalledTimes(1);
      expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining("Purge local market snapshots"));
    });
  });

  describe("A4: Accessibility & Touch Target Minimums", () => {
    it("ExperienceModeToggle buttons satisfy min-h-[36px] and text-xs", () => {
      render(<ExperienceModeToggle />);
      const buttons = screen.getAllByRole("tab");
      expect(buttons.length).toBe(3);
      buttons.forEach((btn) => {
        expect(btn.className).toContain("min-h-[36px]");
        expect(btn.className).not.toContain("text-[10px]");
      });
    });

    it("OnboardingTourModal carousel dots have minimum 36px touch targets and accessible labels", () => {
      render(<OnboardingTourModal isOpen={true} onClose={vi.fn()} />);

      const dots = screen.getAllByRole("button", { name: /Go to step/i });
      expect(dots.length).toBe(4);
      dots.forEach((dot) => {
        expect(dot.className).toContain("min-w-[36px]");
        expect(dot.className).toContain("min-h-[36px]");
        expect(dot.getAttribute("aria-label")).toBeTruthy();
      });
    });
  });

  describe("B4: Guide Persistence and First-Visit Auto-Show", () => {
    it("auto-shows onboarding tour on first visit when flag is absent", () => {
      vi.useFakeTimers();
      render(<Navbar userRole="LONG_TERM" />);

      act(() => {
        vi.advanceTimersByTime(1100);
      });

      expect(screen.getByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeDefined();
      vi.useRealTimers();
    });

    it("does not auto-show onboarding tour when already completed in localStorage", () => {
      vi.useFakeTimers();
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
      render(<Navbar userRole="LONG_TERM" />);

      act(() => {
        vi.advanceTimersByTime(1500);
      });
      expect(screen.queryByRole("dialog", { name: /ARX Terminal Quick Tour/i })).toBeNull();
      vi.useRealTimers();
    });
  });
});
