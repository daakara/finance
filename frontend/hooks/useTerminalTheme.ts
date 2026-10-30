"use client";

import { useEffect, useState } from "react";
import { applyTerminalTheme, readTerminalTheme, TERMINAL_THEME_EVENT, toggleTerminalTheme } from "../lib/terminalTheme";
import type { TerminalTheme } from "../lib/terminalTheme";

export function useTerminalTheme() {
  const [theme, setTheme] = useState<TerminalTheme>("dark");
  useEffect(() => {
    const sync = () => setTheme(document.documentElement.getAttribute("data-theme") === "paper" ? "paper" : "dark");
    const syncStorage = () => applyTerminalTheme(readTerminalTheme());
    window.addEventListener(TERMINAL_THEME_EVENT, sync);
    window.addEventListener("storage", syncStorage);
    applyTerminalTheme(readTerminalTheme());
    return () => {
      window.removeEventListener(TERMINAL_THEME_EVENT, sync);
      window.removeEventListener("storage", syncStorage);
    };
  }, []);
  return { theme, toggleTheme: toggleTerminalTheme };
}
