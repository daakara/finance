import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTerminalTheme } from "../useTerminalTheme";
import { applyTerminalTheme, readTerminalTheme, toggleTerminalTheme } from "../../lib/terminalTheme";

function Control({ name }: { name: string }) {
  const { theme, toggleTheme } = useTerminalTheme();
  return <button onClick={toggleTheme}>{name}:{theme}</button>;
}

beforeEach(() => { localStorage.clear(); document.documentElement.setAttribute("data-theme", "dark"); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("shared theme persistence and controls", () => {
  it("synchronizes two controls and reloads the selected theme", () => {
    render(<><Control name="navbar" /><Control name="header" /></>);
    fireEvent.click(screen.getByText("navbar:dark"));
    expect(screen.getByText("header:paper")).toBeDefined();
    expect(localStorage.getItem("FINANCE_THEME")).toBe("paper");
    cleanup();
    document.documentElement.setAttribute("data-theme", "dark");
    render(<Control name="reloaded" />);
    expect(screen.getByText("reloaded:paper")).toBeDefined();
    expect(document.documentElement.getAttribute("data-theme")).toBe("paper");
  });

  it("imperative command uses the current document theme, even after another control changes it", () => {
    applyTerminalTheme("paper");
    expect(toggleTerminalTheme()).toBe("dark");
    expect(readTerminalTheme()).toBe("dark");
    expect(toggleTerminalTheme()).toBe("paper");
    expect(localStorage.getItem("FINANCE_THEME")).toBe("paper");
  });

  it.each(["ARX_THEME", "theme"])("migrates legacy %s without overriding the canonical preference", (key) => {
    localStorage.setItem(key, "paper");
    render(<Control name="legacy" />);
    expect(screen.getByText("legacy:paper")).toBeDefined();
    expect(localStorage.getItem("FINANCE_THEME")).toBe("paper");
    localStorage.setItem("FINANCE_THEME", "dark");
    expect(readTerminalTheme()).toBe("dark");
  });

  it("still changes the current page if browser storage throws", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("storage denied"); });
    expect(() => applyTerminalTheme("paper")).not.toThrow();
    expect(document.documentElement.getAttribute("data-theme")).toBe("paper");
  });
});
