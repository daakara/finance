import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import PositionSizerModal from "../PositionSizerModal";

const base = { isOpen: true, onClose: () => {}, symbol: "AAPL", entryPrice: 100, stopLoss: 95, takeProfit1: 115, isActionable: true, canSizeTrade: true };
// Bypass disabled DOM semantics to attack the actual React handler.
function invokeHandler(button: HTMLElement) {
  const key = Object.keys(button).find(k => k.startsWith("__reactProps$"))!;
  act(() => (button as any)[key].onClick());
}
beforeEach(() => { localStorage.clear(); vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", "false"); vi.stubEnv("NEXT_PUBLIC_UX_PREVIEW_DEPLOYMENT", "false"); });
afterEach(() => { vi.unstubAllEnvs(); vi.restoreAllMocks(); });
describe("Position recording authority behavior", () => {
  it.each([
    ["entry below stop", { entryPrice: 94 }],
    ["entry equals stop", { entryPrice: 95 }],
    ["negative entry", { entryPrice: -1 }],
    ["nonactionable backend", { isActionable: false }],
    ["missing sizing permission", { canSizeTrade: false }],
    ["missing both permissions", { canSizeTrade: undefined, isActionable: undefined }],
  ])("rejects %s through control and direct handler", (_, overrides) => {
    render(<PositionSizerModal {...base} {...overrides} />);
    const save = screen.getByRole("button", { name: /Sizing Disabled/ }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    expect(screen.getByText("Shares", { exact: true }).parentElement!.textContent?.trim()).toBe("0 Shares");
    fireEvent.click(save); invokeHandler(save);
    expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
  });
  it("rejects a zero-share budget through the handler", () => {
    localStorage.setItem("FINANCE_USER_ACCOUNT_SIZE", "100");
    render(<PositionSizerModal {...base} entryPrice={100} stopLoss={1} />);
    const button = screen.getByRole("button", { name: /Save to My Portfolio/ }) as HTMLButtonElement;
    expect(button.disabled).toBe(true); invokeHandler(button);
    expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
  });
  it.each(["true", "invalid"])("locks review flag %s without changing calculation", flag => {
    vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", flag);
    render(<PositionSizerModal {...base} />);
    const button = screen.getByRole("button", { name: /paper recording locked/ }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(screen.getByText("Shares", { exact: true }).parentElement!.textContent).toContain("50");
    const before = { ...localStorage }; fireEvent.click(button); invokeHandler(button);
    expect({ ...localStorage }).toEqual(before);
  });
  it("renders permitted non-review calculation without implicit persistence", () => {
    render(<PositionSizerModal {...base} />);
    expect((screen.getByRole("button", { name: /Save to My Portfolio/ }) as HTMLButtonElement).disabled).toBe(false);
    expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
  });
  it("locks a preview build even on a canonical hostname with the review flag absent", () => {
    vi.stubEnv("NEXT_PUBLIC_UX_REVIEW_CANDIDATE", undefined);
    vi.stubEnv("NEXT_PUBLIC_UX_PREVIEW_DEPLOYMENT", "true");
    render(<PositionSizerModal {...base} />);
    const button = screen.getByRole("button", { name: /paper recording locked/ }) as HTMLButtonElement;
    expect(button.disabled).toBe(true); invokeHandler(button);
    expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
  });
  it("retains existing valid non-review paper behavior under both permissions", () => {
    render(<PositionSizerModal {...base} />);
    fireEvent.click(screen.getByRole("button", { name: /Save to My Portfolio/ }));
    const positions = JSON.parse(localStorage.getItem("FINANCE_USER_PORTFOLIO")!);
    expect(positions).toHaveLength(1); expect(positions[0]).toMatchObject({ symbol: "AAPL", shares: 50, entryPrice: 100 });
  });
});
