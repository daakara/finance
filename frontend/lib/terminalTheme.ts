export type TerminalTheme = "dark" | "paper";
export const TERMINAL_THEME_KEY = "FINANCE_THEME";
export const TERMINAL_THEME_EVENT = "finance:theme-change";

export function readTerminalTheme(): TerminalTheme {
  try {
    for (const key of [TERMINAL_THEME_KEY, "ARX_THEME", "theme"]) {
      const value = localStorage.getItem(key);
      if (value === "paper" || value === "dark") return value;
    }
  } catch {}
  return "dark";
}

export function applyTerminalTheme(theme: TerminalTheme): TerminalTheme {
  // Storage may be unavailable in private browsing; the current page still works.
  try { localStorage.setItem(TERMINAL_THEME_KEY, theme); } catch {}
  document.documentElement.setAttribute("data-theme", theme);
  window.dispatchEvent(new CustomEvent(TERMINAL_THEME_EVENT, { detail: theme }));
  return theme;
}

export function toggleTerminalTheme(): TerminalTheme {
  return applyTerminalTheme(document.documentElement.getAttribute("data-theme") === "paper" ? "dark" : "paper");
}
