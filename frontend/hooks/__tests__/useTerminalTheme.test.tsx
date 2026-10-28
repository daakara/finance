import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { renderToString } from "react-dom/server";
import { hydrateRoot } from "react-dom/client";
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

  it.each(["invalid", "light", "PAPER", "", "null"])("rejects unsupported stored preference %s", (value) => {
    for (const key of ["FINANCE_THEME", "ARX_THEME", "theme"]) localStorage.setItem(key, value);
    expect(readTerminalTheme()).toBe("dark");
    render(<Control name="invalid" />);
    expect(screen.getByText("invalid:dark")).toBeDefined();
  });

  it("uses canonical preference when both legacy authorities disagree", () => {
    localStorage.setItem("FINANCE_THEME", "dark");
    localStorage.setItem("ARX_THEME", "paper");
    localStorage.setItem("theme", "paper");
    render(<><Control name="one" /><Control name="two" /></>);
    expect(screen.getByText("one:dark")).toBeDefined();
    expect(screen.getByText("two:dark")).toBeDefined();
  });

  it("recovers from a denied storage read without inventing another authority", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("storage denied"); });
    expect(readTerminalTheme()).toBe("dark");
  });

  it("hydrates the dark server markup without mismatch, then restores Paper after mount", async () => {
    localStorage.setItem("FINANCE_THEME", "paper");
    const html = renderToString(<Control name="hydrated" />);
    expect(html).toContain("dark"); // Records the residual dark first render; does not certify flash prevention.
    const host = document.createElement("div");
    host.innerHTML = html;
    document.body.appendChild(host);
    const recoverable = vi.fn();
    const root = hydrateRoot(host, <Control name="hydrated" />, { onRecoverableError: recoverable });
    try {
      await waitFor(() => expect(host.textContent).toBe("hydrated:paper"));
      expect(document.documentElement.getAttribute("data-theme")).toBe("paper");
      expect(recoverable).not.toHaveBeenCalled();
    } finally { root.unmount(); host.remove(); }
  });
});
