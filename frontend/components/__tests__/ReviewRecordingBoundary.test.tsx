import React from "react";
import {describe,it,expect,vi,afterEach} from "vitest";
import {render,screen,fireEvent} from "@testing-library/react";
import PositionSizerModal from "../PositionSizerModal";
afterEach(()=>vi.unstubAllEnvs());
describe("Owner review preview boundary",()=>{
 it("keeps paper recording locked even with synthetic actionable inputs",()=>{
  vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE","true");localStorage.clear();
  render(<PositionSizerModal isOpen onClose={()=>{}} symbol="AAPL" entryPrice={100} stopLoss={95} takeProfit1={115} isActionable canSizeTrade />);
  const save=screen.getByRole("button",{name:/Owner review.*paper recording locked/i}) as HTMLButtonElement;
  expect(save.disabled).toBe(true);fireEvent.click(save);expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
 });
});
