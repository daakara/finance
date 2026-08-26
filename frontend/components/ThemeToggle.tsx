"use client";

import { useState, useEffect } from "react";
import { trackMatomoEvent } from "../lib/matomo";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "paper">("dark");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("FINANCE_THEME") as "dark" | "paper" | null;
      if (saved === "paper" || saved === "dark") {
        setTheme(saved);
        document.documentElement.setAttribute("data-theme", saved);
      } else {
        document.documentElement.setAttribute("data-theme", "dark");
      }
    } catch {
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "paper" : "dark";
    setTheme(nextTheme);
    try {
      localStorage.setItem("FINANCE_THEME", nextTheme);
    } catch {}
    document.documentElement.setAttribute("data-theme", nextTheme);
    trackMatomoEvent("User Journey", "Toggle Theme", nextTheme);
  };

  return (
    <button
      onClick={toggleTheme}
      type="button"
      aria-label={`Switch to ${theme === "dark" ? "Paper Light" : "Cyber Dark"} theme`}
      className="flex items-center justify-center min-w-[34px] min-h-[34px] p-1.5 rounded-xl bg-[#111722] hover:bg-[#1b2537] border border-[#2b3a52] text-slate-300 hover:text-white transition-all active:scale-95 shadow cursor-pointer focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
    >
      <span className="text-sm" aria-hidden="true">
        {theme === "dark" ? "🌙" : "☀️"}
      </span>
    </button>
  );
}