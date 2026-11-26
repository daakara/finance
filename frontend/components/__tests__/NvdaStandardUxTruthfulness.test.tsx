import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import OptimalEntryExitCard from "../OptimalEntryExitCard";
import PreFlightChecklistModal from "../PreFlightChecklistModal";
import EvidenceDomainStatusCard from "../terminal/EvidenceDomainStatusCard";
import CompositeConvictionCard from "../CompositeConvictionCard";
import { SharedAssetDeck } from "../terminal/SharedAssetDeck";
import { resolveNineEvidenceDomains } from "../../lib/evidenceDomains";
import { evaluatePreFlightChecks } from "../../lib/preflightEvaluator";
import { generateQuantitativeInsight } from "../../lib/insightGenerator";

describe("NVDA Standard Analysis UX & Truthfulness Fixes", () => {
  const baseExecutionPlan: any = {
    current_price: 149.30,
    optimal_entry_min: 162.56,
    optimal_entry_max: 173.19,
    stop_loss: 162.56,
    stop_loss_pct: 0.0,
    take_profit_1: 200.0,
    take_profit_1_pct: 23.0,
    take_profit_2: 220.0,
    take_profit_2_pct: 35.3,
    risk_reward_ratio: 3.5,
    setup_pattern: "Stage 4 Breakdown",
    entry_thesis: "Wait for base formation above $162.56",
    invalidation_condition: "Price below $162.56",
    stage_phase: "Stage 4",
    liquidity_defense: {
      execution_hazard: false,
      badge_color: "emerald",
      pro_label: "Adequate Depth",
      plain_label: "Normal Trading",
      pro_summary: "Adequate market liquidity",
    },
  };

  describe("1. Corporate Event & Catalyst Truthfulness", () => {
    it("renders elapsed earnings as Prior Earnings (Elapsed), NOT imminent <=1h", () => {
      // Long-term (plain language)
      const { unmount } = render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={baseExecutionPlan}
          currentPrice={149.30}
          userRole="LONG_TERM"
          corporateEvent={{
            event_date_et: "2026-11-17",
            days_to_earnings: -9.0,
            confirmation_status: "CONFIRMED",
            coverage_status: "CONFIRMED_UPCOMING",
            source_name: "Corporate IR",
            freshness_status: "LIVE",
          } as any}
        />
      );

      // Must NOT render "Earnings Imminent" or "<=1h"
      expect(screen.queryByText(/Earnings Imminent/i)).toBeNull();
      expect(screen.queryByText(/<=1h/i)).toBeNull();

      // Must explicitly disclose the elapsed status
      expect(screen.getByText(/Prior Earnings: 2026-11-17/i)).toBeDefined();
      expect(screen.getByText(/\(Elapsed\)/i)).toBeDefined();
      expect(screen.getByText(/\(IR Confirmed\)/i)).toBeDefined();
      unmount();

      // Day trader (pro mode)
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={baseExecutionPlan}
          currentPrice={149.30}
          userRole="DAY_TRADER"
          corporateEvent={{
            event_date_et: "2026-11-17",
            days_to_earnings: -9.0,
            confirmation_status: "CONFIRMED",
            coverage_status: "CONFIRMED_UPCOMING",
            source_name: "Corporate IR",
            freshness_status: "LIVE",
          } as any}
        />
      );

      expect(screen.getByText(/Prior Catalyst: 2026-11-17/i)).toBeDefined();
      expect(screen.getByText(/Elapsed · Runway Uncertified/i)).toBeDefined();
    });

    it("renders imminent earnings ONLY when event is upcoming and within stated window", () => {
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={baseExecutionPlan}
          currentPrice={149.30}
          userRole="DAY_TRADER"
          corporateEvent={{
            event_date_et: "2026-11-26",
            days_to_earnings: 0.5,
            confirmation_status: "CONFIRMED",
            coverage_status: "CONFIRMED_UPCOMING",
            source_name: "Corporate IR",
            freshness_status: "LIVE",
            results_release_time_et: "16:00",
          } as any}
        />
      );

      expect(screen.getByText(/Catalyst: 2026-11-26/i)).toBeDefined();
      expect(screen.getByText(/16:00 ET Release/i)).toBeDefined();
    });
  });

  describe("2. Setup Invalidated vs Forming Disclosures", () => {
    it("PreFlightChecklistModal shows NON-ACTIONABLE STATE: SETUP INVALIDATED (STOP BREACHED) in pro mode", () => {
      render(
        <PreFlightChecklistModal
          isOpen={true}
          onClose={vi.fn()}
          symbol="NVDA"
          currentPrice={149.30}
          stopLoss={162.56}
          takeProfit1={200.0}
          riskRewardRatio={3.5}
          isActionable={false}
          decisionState="SETUP_INVALIDATED"
          isDayTrader={true}
        />
      );

      // Must render SETUP INVALIDATED, NOT "A valid setup is forming"
      expect(screen.getByText(/NON-ACTIONABLE STATE: SETUP INVALIDATED \(STOP BREACHED\)/i)).toBeDefined();
      expect(screen.getByText(/Setup structure is invalidated \(reference price \$149\.30 <= stop loss \$162\.56\)/i)).toBeDefined();
      expect(screen.queryByText(/A valid setup is forming for NVDA/i)).toBeNull();
    });

    it("PreFlightChecklistModal plain language version states clearly that reference price is below floor", () => {
      render(
        <PreFlightChecklistModal
          isOpen={true}
          onClose={vi.fn()}
          symbol="NVDA"
          currentPrice={149.30}
          stopLoss={162.56}
          takeProfit1={200.0}
          riskRewardRatio={3.5}
          isActionable={false}
          decisionState="SETUP_INVALIDATED"
          isDayTrader={false}
        />
      );

      expect(screen.getByText(/TRADE NOT CLEARED: SETUP INVALIDATED/i)).toBeDefined();
      expect(screen.getByText(/Reference price \(\$149\.30\) is at or below the \$162\.56 invalidation floor/i)).toBeDefined();
    });
  });

  describe("3. Planned R:R Annotation at Zone Floor", () => {
    it("labels Planned R:R (Zone Floor) and indicates floor reference when spot <= stopLoss", () => {
      const plan: any = {
        ...baseExecutionPlan,
        setup_pattern: "Minervini VCP",
        stage_phase: "Stage 2",
      };

      const { unmount } = render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={plan}
          currentPrice={149.30}
          userRole="DAY_TRADER"
        />
      );

      expect(screen.getByText(/Planned R:R \(Zone Floor\)/i)).toBeDefined();
      expect(screen.getByText(/\(Floor\)/i)).toBeDefined();
      unmount();

      // Stage 4 post-pivot floor
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={{ ...baseExecutionPlan, setup_pattern: "Stage 4 Breakdown", stage_phase: "Stage 4" }}
          currentPrice={149.30}
          userRole="DAY_TRADER"
        />
      );

      expect(screen.getByText(/Planned R:R \(Post-Pivot Floor\)/i)).toBeDefined();
    });
  });

  describe("4. CHK-1 Stop-Loss Discipline Status", () => {
    it("evaluates CHK-1 as FAIL (not UNAVAILABLE) when stop loss is breached (safeStop >= safePrice)", () => {
      const res = evaluatePreFlightChecks({
        symbol: "NVDA",
        currentPrice: 149.30,
        stopLoss: 162.56,
        takeProfit1: 200.0,
        riskRewardRatio: 3.5,
        isActionable: false,
      });

      const chk1 = res.checks[0];
      expect(chk1.id).toBe("CHK-1");
      expect(chk1.status).toBe("FAIL");
      expect(chk1.outcomeCategory).toBe("FAIL");
      expect(chk1.displayStatus).toBe("FAIL");
      expect(chk1.passed).toBe(false);
      expect(chk1.badgeClass).toContain("text-rose-300");
      expect(chk1.detail).toContain("Inverted risk boundary: Stop loss ($162.56) is above current price ($149.30)");
    });
  });

  describe("5. Macro VIX Fallback Elimination", () => {
    it("evaluates Macroeconomic Regime as UNAVAILABLE when VIX is missing, without inventing 19.4", () => {
      const domains = resolveNineEvidenceDomains({
        symbol: "NVDA",
        vix: undefined,
        macroData: null,
      });

      const macroDomain = domains.find((d) => d.id === "macro_regime");
      expect(macroDomain).toBeDefined();
      expect(macroDomain?.status).toBe("UNAVAILABLE");
      expect(macroDomain?.statusLabel).toBe("Unavailable");
      expect(macroDomain?.explanation).not.toContain("19.4");
      expect(macroDomain?.explanation).toContain("Macro volatility");
    });

    it("evaluates Macroeconomic Regime as CAUTION when VIX >= 26.0", () => {
      const domains = resolveNineEvidenceDomains({
        symbol: "NVDA",
        vix: 27.1,
      });

      const macroDomain = domains.find((d) => d.id === "macro_regime");
      expect(macroDomain?.status).toBe("CAUTION");
      expect(macroDomain?.statusLabel).toContain("Caution");
      expect(macroDomain?.explanation).toContain("VIX 27.1 >= 26.0");
    });
  });

  describe("6. Placeholder Score & Delta Suppression", () => {
    it("generateQuantitativeInsight does not default setupScore to 40 when undefined", () => {
      const insight = generateQuantitativeInsight(
        "NVDA",
        "NVIDIA Corporation",
        149.30,
        0.0,
        undefined, // setupScore undefined
        undefined,
        "SWING",
        "NOT_OWNED",
        "USER_DECLARED",
        undefined,
        undefined,
        undefined // confluence undefined
      );

      // Score must be 0 (unverified / pending), never arbitrary 40
      expect(insight.setupScore).toBe(0);
    });

    it("SharedAssetDeck does not fabricate +$0.00 (+0.00%) when changePct is undefined", () => {
      const insight: any = {
        symbol: "NVDA",
        price: 149.30,
        changePct: undefined,
        companyName: "NVIDIA Corporation",
      };

      render(
        <SharedAssetDeck
          insight={insight}
          isStaleTape={false}
          isActionable={false}
          isWaitingPullback={false}
          isExtended={false}
        />
      );

      expect(screen.queryByTestId("card-delta")).toBeNull();
    });
  });

  describe("7. Control, Disclosure & Accessibility Fixes", () => {
    it("EvidenceDomainStatusCard header is a single button with aria-expanded and no nested button", () => {
      render(
        <EvidenceDomainStatusCard
          symbol="NVDA"
          domains={[
            {
              id: "price_tape",
              name: "Real-Time Exchange Price Tape",
              category: "Market Execution",
              authority: "Live Exchange Feed",
              applicability: "REQUIRED",
              status: "VERIFIED",
              statusLabel: "Verified",
              impact: "BLOCKING",
              observationTime: "Active Session",
              provenance: "Live Exchange Feed",
              explanation: "Current market tape synchronized.",
            },
          ]}
        />
      );

      const headerButton = screen.getByRole("button", { name: /Toggle Evidence Domain Status/i });
      expect(headerButton).toBeDefined();
      expect(headerButton.getAttribute("aria-expanded")).toBe("false");

      // Verify clicking toggles aria-expanded
      fireEvent.click(headerButton);
      expect(headerButton.getAttribute("aria-expanded")).toBe("true");

      // Verify no nested buttons exist inside the toggle button
      const buttonsInside = headerButton.querySelectorAll("button");
      expect(buttonsInside.length).toBe(0);
    });

    it("CompositeConvictionCard Inspect SEC Form 4 Feed button sets pending flag and dispatches event", () => {
      const dispatchSpy = vi.spyOn(window, "dispatchEvent");
      render(
        <CompositeConvictionCard
          symbol="NVDA"
          data={{
            confluence: {
              confluenceScore: 29,
              confluenceBadge: "Negative Confluence",
              bottomLine: "Setup invalidated",
              pillars: [
                {
                  pillar: "SMART_MONEY_FLOW",
                  score: 20,
                  status: "neutral",
                  detail: "SEC Form 4 transactions evaluated.",
                },
              ],
            },
          } as any}
        />
      );

      const btn = screen.getByTestId("link-sec-disclosure-drawer");
      expect(btn).toBeDefined();
      fireEvent.click(btn);

      expect((window as any).__arxPendingSecForm4).toBe(true);
      expect(dispatchSpy).toHaveBeenCalled();
      const calls = dispatchSpy.mock.calls;
      const customEventCall = calls.find((c) => (c[0] as CustomEvent).type === "arx:open-sec-disclosures");
      expect(customEventCall).toBeDefined();
    });
  });
});
