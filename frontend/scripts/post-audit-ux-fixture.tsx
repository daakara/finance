/* Local browser acceptance fixture. Never imported by an application route. */
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import Standard from "../components/terminal/StandardTerminalView";
import Guided from "../components/terminal/GuidedTerminalView";
import Quant from "../components/terminal/AdvancedTerminalView";
import ThemeToggle from "../components/ThemeToggle";
import PositionSizerModal from "../components/PositionSizerModal";
import PreFlightReadinessBar from "../components/PreFlightReadinessBar";
import WhyInspectModal from "../components/WhyInspectModal";
import { generateQuantitativeInsight } from "../lib/insightGenerator";

function Fixture() {
  const query = new URLSearchParams(location.search);
  const symbol = query.get("symbol") === "NVDA" ? "NVDA" : "AAPL";
  const mode = query.get("mode") || "standard";
  const state = query.get("state") || "blocked";
  const actionable = state === "actionable";
  const stale = state === "stale";
  const [sizer, setSizer] = useState(false);
  const [why, setWhy] = useState(false);
  // Explicit synthetic inputs exercise the presentation contract, not a live market model.
  const insight = generateQuantitativeInsight(symbol, `${symbol} fixture`, 100, 1, 75, 2, "SWING", "NOT_OWNED", "USER_DECLARED", [], "unavailable");
  insight.price = 100;
  insight.verdictLabel = stale ? "Historical Data — Execution Prohibited" : actionable ? "Actionable Setup Confirmed" : "Valid Setup — Awaiting Trigger";
  insight.terminalState = {
    ...insight.terminalState, currentPrice: 100, overallEligibility: "ELIGIBLE",
    decisionState: stale ? "STALE_DATA" : actionable ? "ACTIONABLE_SETUP" : "VALID_SETUP",
    isActionable: actionable, canSizeTrade: actionable,
    headlineExplanation: "Synthetic acceptance fixture. No financial advice or live execution.",
    whatWouldChangeAssessment: "Verified volume confirmation and complete preflight clearance.",
    modelProvenance: { ...insight.terminalState.modelProvenance, calculatedAt: "2026-10-28T12:00:00Z" },
  };
  insight.human.assessmentHeadline = insight.verdictLabel;
  insight.human.assessmentDescription = insight.terminalState.headlineExplanation;
  insight.standard.bottomLine = insight.terminalState.headlineExplanation;
  insight.standard.keyLevels = { ...insight.standard.keyLevels, currentPrice: 100, entryMin: 99, entryMax: 101, breakoutPivot: 100, stopLoss: 95, target1: 115, target2: 125, profitRiskRatio: 3 };
  const Mode = mode === "guided" ? Guided : mode === "quant" ? Quant : Standard;
  return <>
    <header className="p-3 flex flex-wrap items-center gap-3">
      <strong>LOCAL SYNTHETIC FIXTURE · {symbol} · {mode} · {state}</strong><ThemeToggle />
    </header>
    <main className="p-3" data-fixture-state={state}>
      <Mode insight={insight} onOpenSizer={() => setSizer(true)} onOpenWhy={() => setWhy(true)}
        chartSlot={<div className="bg-[var(--surface-2)] p-6">Chart placeholder — price 100 is synthetic fixture data.</div>}
        preflightSlot={<PreFlightReadinessBar symbol={symbol} currentPrice={100} stopLoss={95} takeProfit1={115}
          riskRewardRatio={3} optimalEntryMin={99} optimalEntryMax={101} breakoutPivot={100}
          isActionable={actionable} decisionState={insight.terminalState.decisionState}
          isStage4={false} vix={16} catalystState="PASS"
          catalystEvidence={{state:"PASS", daysToEarnings:20, confirmationStatus:"CONFIRMED", coverageStatus:"CONFIRMED_UPCOMING", freshnessStatus:"LIVE", riskTier:"NORMAL_RUNWAY"}} />}
      />
      <PositionSizerModal isOpen={sizer} onClose={() => setSizer(false)} symbol={symbol} entryPrice={100} stopLoss={95}
        takeProfit1={115} riskRewardRatio={3} canSizeTrade={actionable} isActionable={actionable} />
      <WhyInspectModal isOpen={why} onClose={() => setWhy(false)} symbol={symbol} setupScore={insight.setupScore}
        terminalState={insight.terminalState} items={insight.scoreAttribution.items}
        catalystToIncreaseScore={insight.scoreAttribution.catalystToIncreaseScore}
        whatWouldChangeAssessment={insight.whatWouldChangeAssessment} />
    </main>
  </>;
}

createRoot(document.getElementById("fixture-root")!).render(<Fixture />);
