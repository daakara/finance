import React from "react";
import {render, screen, fireEvent} from "@testing-library/react";
import {describe, it, expect} from "vitest";
import PreFlightReadinessBar from "../PreFlightReadinessBar";

describe("F.2 canonical blocker and subordinate criteria", () => {
 it("keeps the blocker beside the verdict while criteria stay collapsed below the chart", () => {
  function View() { const [target, setTarget] = React.useState<HTMLDivElement | null>(null); return <><article aria-label="Verdict"><div ref={setTarget} /></article><div>Chart</div><PreFlightReadinessBar summaryTarget={target} symbol="AAPL" currentPrice={343.83} optimalEntryMin={331.73} optimalEntryMax={344.37} stopLoss={321.78} takeProfit1={384.63} riskRewardRatio={2} isActionable={false} priceContext="Market observation: 345.02 (DELAYED)." /></>; }
  render(<View />);
  const summary=screen.getByTestId("inline-active-blocker-text");
  expect(summary.closest("article")?.getAttribute("aria-label")).toBe("Verdict");
  expect(screen.getAllByTestId("inline-active-blocker-text")).toHaveLength(1);
  expect(screen.queryByTestId("preflight-details-drawer")).toBeNull();
  const toggle=screen.getByTestId("inline-toggle-details-btn");
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  fireEvent.click(toggle);
  expect(toggle.getAttribute("aria-expanded")).toBe("true");
  expect(screen.getByRole("region",{name:"Detailed Preflight Telemetry"}).textContent).toContain("Market observation: 345.02 (DELAYED)");
  expect(screen.getByTestId("preflight-details-drawer").textContent).toContain("Readiness input price ($343.83)");
  expect(screen.getByTestId("inline-actionability-badge").textContent).toBe("AWAITING TRIGGER");
  fireEvent.click(toggle);
  expect(screen.queryByTestId("preflight-details-drawer")).toBeNull();
 });
 it("moves the one summary to the current component-owned target",()=>{
  const first=document.createElement("div"),second=document.createElement("div");document.body.append(first,second);
  const props={symbol:"AAPL",currentPrice:100,stopLoss:95,takeProfit1:115,riskRewardRatio:3,isActionable:false};
  const view=render(<PreFlightReadinessBar {...props} summaryTarget={first}/>);
  expect(first.querySelector('[data-testid="inline-active-blocker-text"]')).not.toBeNull();
  view.rerender(<PreFlightReadinessBar {...props} summaryTarget={second}/>);
  expect(first.childElementCount).toBe(0);
  expect(second.querySelector('[data-testid="inline-active-blocker-text"]')).not.toBeNull();
  view.unmount();first.remove();second.remove();
 });
});
