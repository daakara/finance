import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PositionSizerModal from "../PositionSizerModal";

beforeEach(() => localStorage.clear());
afterEach(() => cleanup());

const props = { isOpen: true, onClose: () => {}, symbol: "AAPL", entryPrice: 100, stopLoss: 95, takeProfit1: 115 };

describe("sizer canonical authority and keyboard boundaries", () => {
  it.each([[false, true], [true, false], [false, false]])("zero shares when canSizeTrade=%s, isActionable=%s", (canSizeTrade, isActionable) => {
    render(<PositionSizerModal {...props} canSizeTrade={canSizeTrade} isActionable={isActionable} />);
    expect(screen.getByText("Shares", { exact: true }).parentElement!.textContent?.trim()).toBe("0 Shares");
    expect((screen.getByRole("button", { name: /Sizing Disabled/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByText(/Buy .* shares of/)).toBeNull();
  });

  it("allows a positive simulation only when both canonical permissions are true", () => {
    render(<PositionSizerModal {...props} canSizeTrade isActionable />);
    expect(screen.getByText("Shares", { exact: true }).parentElement!.textContent?.trim()).not.toBe("0 Shares");
    expect((screen.getByRole("button", { name: /Save to My Portfolio/ }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("handles Escape and restores focus on unmount", () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger);
    trigger.focus();
    const onClose = vi.fn();
    const view = render(<PositionSizerModal {...props} onClose={onClose} />);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
    view.unmount();
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });
});
