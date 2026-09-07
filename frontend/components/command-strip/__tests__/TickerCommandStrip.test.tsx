import React from "react";
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import TickerCommandStrip, {
  TickerCommandStripSkeleton,
} from "../TickerCommandStrip";
import SetupScoreBadge from "../SetupScoreBadge";
import ExecutionStateBadge from "../ExecutionStateBadge";
import LiquidityBadge from "../LiquidityBadge";
import { useExperienceStore } from "../../../state/experience-store";

describe("TickerCommandStrip & Badge Components (Milestone W1.5)", () => {
  beforeEach(() => {
    useExperienceStore.getState().setMode("STANDARD");
  });

  describe("TickerCommandStripSkeleton", () => {
    it("renders skeleton with strict 110px desktop height and zero-CLS semantics", () => {
      render(<TickerCommandStripSkeleton />);
      const skeleton = screen.getByTestId("ticker-command-strip-skeleton");
      expect(skeleton).toBeDefined();
      expect(skeleton.className).toContain("min-h-[110px]");
      expect(skeleton.className).toContain("lg:h-[110px]");
      expect(skeleton.getAttribute("role")).toBe("region");
      expect(skeleton.getAttribute("aria-label")).toBe("Ticker Command Strip Loading");
    });
  });

  describe("SetupScoreBadge Anti-Cyan Thresholding", () => {
    it("renders emerald styling for favorable scores >= 70", () => {
      render(<SetupScoreBadge score={75} />);
      const badge = screen.getByTestId("setup-score-badge");
      expect(badge.getAttribute("role")).toBe("meter");
      expect(badge.getAttribute("aria-valuenow")).toBe("75");
      expect(badge.className).toContain("border-emerald-500/30");

      const scoreVal = screen.getByTestId("setup-score-value");
      expect(scoreVal.textContent).toBe("75");
      expect(scoreVal.className).toContain("text-emerald-400");

      const status = screen.getByTestId("setup-score-status");
      expect(status.textContent).toBe("Favorable");
      expect(status.className).toContain("text-emerald-400");
    });

    it("renders amber styling for cautionary scores 50 - 69", () => {
      render(<SetupScoreBadge score={62} />);
      const badge = screen.getByTestId("setup-score-badge");
      expect(badge.getAttribute("aria-valuenow")).toBe("62");
      expect(badge.className).toContain("border-amber-500/30");

      const scoreVal = screen.getByTestId("setup-score-value");
      expect(scoreVal.textContent).toBe("62");
      expect(scoreVal.className).toContain("text-amber-400");

      const status = screen.getByTestId("setup-score-status");
      expect(status.textContent).toBe("Conditional");
    });

    it("renders rose styling for high-risk scores < 50", () => {
      render(<SetupScoreBadge score={42} />);
      const badge = screen.getByTestId("setup-score-badge");
      expect(badge.getAttribute("aria-valuenow")).toBe("42");
      expect(badge.className).toContain("border-rose-500/30");

      const scoreVal = screen.getByTestId("setup-score-value");
      expect(scoreVal.textContent).toBe("42");
      expect(scoreVal.className).toContain("text-rose-400");

      const status = screen.getByTestId("setup-score-status");
      expect(status.textContent).toBe("High Risk");
    });

    it("adapts labels for GUIDED mode", () => {
      render(<SetupScoreBadge score={85} mode="GUIDED" />);
      expect(screen.getByText("Setup Quality")).toBeDefined();
      expect(screen.getByText("Strong Setup")).toBeDefined();
    });

    it("displays domain confidence tag in QUANT mode", () => {
      render(<SetupScoreBadge score={85} mode="QUANT" domainConfidence="HIGH" />);
      const confTag = screen.getByTestId("domain-confidence-tag");
      expect(confTag.textContent).toContain("CONF: HIGH");
    });
  });

  describe("ExecutionStateBadge", () => {
    it("renders IN_BUY_ZONE with emerald styling", () => {
      render(<ExecutionStateBadge state="IN_BUY_ZONE" />);
      const badge = screen.getByTestId("execution-state-badge");
      expect(badge.getAttribute("data-state")).toBe("IN_BUY_ZONE");
      expect(badge.className).toContain("text-emerald-400");
      expect(badge.textContent).toContain("IN_BUY_ZONE");
    });

    it("renders WAITING_PULLBACK with amber styling", () => {
      render(<ExecutionStateBadge state="WAITING_PULLBACK" />);
      const badge = screen.getByTestId("execution-state-badge");
      expect(badge.getAttribute("data-state")).toBe("WAITING_PULLBACK");
      expect(badge.className).toContain("text-amber-400");
      expect(badge.textContent).toContain("WAITING_PULLBACK");
    });

    it("renders STOPPED_OUT with rose styling", () => {
      render(<ExecutionStateBadge state="STOPPED_OUT" />);
      const badge = screen.getByTestId("execution-state-badge");
      expect(badge.getAttribute("data-state")).toBe("STOPPED_OUT");
      expect(badge.className).toContain("text-rose-400");
      expect(badge.textContent).toContain("STOPPED_OUT");
    });

    it("adapts state labels in GUIDED mode", () => {
      render(<ExecutionStateBadge state="IN_BUY_ZONE" mode="GUIDED" />);
      expect(screen.getByText("Favorable Entry Zone")).toBeDefined();
    });
  });

  describe("LiquidityBadge", () => {
    it("renders HIGH liquidity ADV heuristic in standard mode", () => {
      render(<LiquidityBadge tier="HIGH" />);
      const badge = screen.getByTestId("liquidity-badge");
      expect(badge.textContent).toContain("ADV: High (<1.0% ADV)");
    });

    it("renders Amihud ratio in QUANT mode when provided", () => {
      render(<LiquidityBadge tier="HIGH" amihudScore={0.0014} mode="QUANT" />);
      const amihud = screen.getByTestId("amihud-score");
      expect(amihud.textContent).toContain("Amihud: 0.0014");
    });
  });

  describe("TickerCommandStrip Integration", () => {
    const defaultProps = {
      ticker: "CPRX",
      companyName: "Catalyst Pharmaceuticals Inc.",
      spotPrice: 18.42,
      priceChangePct: 2.11,
      priceChange: 0.38,
      setupScore: 71,
      domainConfidence: "HIGH" as const,
      executionState: "IN_BUY_ZONE" as const,
      liquidityTier: "HIGH" as const,
      marketRegime: "RISK_ON" as const,
      isSettlementPinned: true,
      marketSession: "CLOSED" as const,
      exchange: "NASDAQ",
      sector: "Healthcare",
      amihudScore: 0.0014,
    };

    it("renders complete Stage 1 orientation header with 110px desktop constraint", () => {
      render(<TickerCommandStrip {...defaultProps} />);
      const strip = screen.getByTestId("ticker-command-strip");
      expect(strip).toBeDefined();
      expect(strip.className).toContain("min-h-[110px]");
      expect(strip.className).toContain("lg:h-[110px]");

      // Identity
      expect(screen.getByTestId("ticker-symbol").textContent).toBe("CPRX");
      expect(screen.getByTestId("company-name").textContent).toBe("Catalyst Pharmaceuticals Inc.");
      expect(screen.getByTestId("ticker-exchange").textContent).toBe("NASDAQ");
      expect(screen.getByTestId("ticker-sector").textContent).toBe("Healthcare");

      // Price
      expect(screen.getByTestId("spot-price").textContent).toBe("$18.42");
      expect(screen.getByTestId("price-delta").textContent).toContain("+0.38 (+2.11%)");
      expect(screen.getByTestId("price-delta").className).toContain("text-emerald-400");

      // Pinned settlement banner
      const settlement = screen.getByTestId("settlement-pinned-notice");
      expect(settlement).toBeDefined();
      expect(settlement.textContent).toContain("[Session Closed / Friday Settlement Pinned]");

      // Score and State
      expect(screen.getByTestId("setup-score-badge")).toBeDefined();
      expect(screen.getByTestId("setup-score-value").textContent).toBe("71");
      expect(screen.getByTestId("execution-state-badge").textContent).toContain("IN_BUY_ZONE");
      expect(screen.getByTestId("liquidity-badge").textContent).toContain("ADV: High");
    });

    it("renders negative price change in rose", () => {
      render(
        <TickerCommandStrip
          {...defaultProps}
          spotPrice={16.2}
          priceChangePct={-3.45}
          priceChange={-0.58}
        />
      );
      const delta = screen.getByTestId("price-delta");
      expect(delta.className).toContain("text-rose-400");
      expect(delta.textContent).toContain("-0.58 (-3.45%)");
    });

    it("renders quant mode details when active", () => {
      render(<TickerCommandStrip {...defaultProps} mode="QUANT" />);
      const regimeBadge = screen.getByTestId("quant-market-regime-badge");
      expect(regimeBadge).toBeDefined();
      expect(regimeBadge.textContent).toContain("REGIME: RISK_ON");
    });
  });
});
