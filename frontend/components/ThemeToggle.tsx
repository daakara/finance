"use client";

import { trackMatomoEvent } from "../lib/matomo";
import { useTerminalTheme } from "../hooks/useTerminalTheme";

export default function ThemeToggle() {
  const { theme, toggleTheme: toggleSharedTheme } = useTerminalTheme();

  const toggleTheme = () => {
    const nextTheme = toggleSharedTheme();
    trackMatomoEvent("User Journey", "Toggle Theme", nextTheme);
  };

  return (
    <button
      onClick={toggleTheme}
      type="button"
      aria-label={`Switch to ${theme === "dark" ? "Paper Light" : "Cyber Dark"} theme`}
      className="flex items-center justify-center min-w-[44px] min-h-[44px] p-2.5 rounded-xl bg-[#111722] hover:bg-[#1b2537] border border-[#2b3a52] text-slate-300 hover:text-white transition-all active:scale-[0.96] motion-reduce:transform-none transition-transform duration-100 ease-out shadow cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
    >
      <span className="text-sm" aria-hidden="true">
        {theme === "dark" ? "🌙" : "☀️"}
      </span>
    </button>
  );
}
