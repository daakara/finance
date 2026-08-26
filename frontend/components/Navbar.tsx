"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import UniversalOmniSearch from "./UniversalOmniSearch";
import ThemeToggle from "./ThemeToggle";

interface NavbarProps {
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
}

export default function Navbar({ userRole = "LONG_TERM", onRoleChange }: NavbarProps) {
  const pathname = usePathname();
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">(userRole);

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setActiveRole(saved);
      if (onRoleChange) onRoleChange(saved);
    }
  }, []);

  const handleRoleToggle = (role: "DAY_TRADER" | "LONG_TERM") => {
    setActiveRole(role);
    localStorage.setItem("FINANCE_USER_ROLE", role);
    if (onRoleChange) onRoleChange(role);
  };

  return (
    <>
      <header role="banner" className="border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur sticky top-0 z-50">
        <div className="max-w-[1750px] mx-auto px-2.5 sm:px-6 h-14 sm:h-16 flex items-center justify-between gap-2 sm:gap-4">
          {/* Left: Brand Logo & Title */}
          <div className="flex items-center space-x-2 sm:space-x-4 shrink-0 min-w-0">
            <Link href="/" aria-label="Finance Terminal Home" className="flex items-center space-x-2 group shrink-0 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded-lg">
              <div aria-hidden="true" className="w-7 h-7 sm:w-9 sm:h-9 rounded-lg bg-gradient-to-tr from-cyan-600 via-indigo-600 to-purple-600 flex items-center justify-center font-mono font-bold text-white shadow-lg shadow-cyan-950/50 group-hover:scale-105 transition-transform text-xs sm:text-sm">
                FT
              </div>
              <div className="min-w-0">
                <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base hidden xl:block leading-none">
                  FINANCE TERMINAL
                </span>
                <span className="font-bold tracking-tight text-white font-mono text-xs xl:hidden block leading-none">
                  TERMINAL
                </span>
                <span className="text-[9px] text-cyan-400 font-mono tracking-wider uppercase hidden xl:block">
                  Quantitative Intel
                </span>
              </div>
            </Link>

            {/* Desktop Navigation Links */}
            <nav aria-label="Main Navigation" className="hidden lg:flex items-center space-x-1 font-mono text-xs">
              <Link
                href="/"
                aria-current={pathname === "/" ? "page" : undefined}
                className={`px-3 py-1.5 rounded-lg transition-colors focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Terminal
              </Link>
              <Link
                href="/screener"
                aria-current={pathname === "/screener" ? "page" : undefined}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/screener" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <svg aria-hidden="true" className="w-3.5 h-3.5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
                <span>Hidden Gems</span>
              </Link>
              <Link
                href="/compare"
                aria-current={pathname === "/compare" ? "page" : undefined}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/compare" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>⚔️ Compare</span>
              </Link>
                            <Link
                href="/smart-money"
                aria-current={pathname === "/smart-money" ? "page" : undefined}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/smart-money" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>🏛️ Smart Money</span>
              </Link>
                            <Link
                href="/portfolio"
                aria-current={pathname === "/portfolio" ? "page" : undefined}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/portfolio" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>💼 Portfolio</span>
              </Link>
              <Link
                href="/guide"
                aria-current={pathname === "/guide" ? "page" : undefined}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/guide" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>📖 Guide</span>
              </Link>
            </nav>
          </div>

          {/* Center: Global Omni-Search Bar (Desktop & Mobile) */}
          <div className="flex-1 max-w-xl mx-auto flex items-center justify-center px-1 sm:px-2">
            <UniversalOmniSearch />
          </div>

          {/* Right: Theme Toggle & Trading Horizon Mode Switcher */}
          <div className="flex items-center space-x-1 sm:space-x-2 shrink-0">
            {/* 🌓 Theme Toggle */}
            <ThemeToggle />

            {/* Trading Horizon Switcher */}
            <div role="toolbar" aria-label="Trading Horizon Mode Switcher" className="bg-[#090d14] p-0.5 sm:p-1 rounded-xl border border-[#243044] flex items-center shadow-inner">
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                role="button"
                aria-pressed={activeRole === "DAY_TRADER"}
                aria-label="Switch to Day Trader mode"
                className={`flex items-center space-x-1 sm:space-x-1.5 px-2 sm:px-3 py-1 sm:py-1.5 min-h-[32px] rounded-lg text-xs font-mono font-bold transition-colors active:scale-[0.96] transition-transform duration-100 focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none ${
                  activeRole === "DAY_TRADER"
                    ? "bg-amber-500 text-slate-950 shadow-md shadow-amber-950/50 font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <span aria-hidden="true" className="text-xs">⚡</span>
                <span className="font-mono tracking-tight text-[10px] sm:text-xs">Day Trade</span>
              </button>

              <button
                onClick={() => handleRoleToggle("LONG_TERM")}
                role="button"
                aria-pressed={activeRole === "LONG_TERM"}
                aria-label="Switch to Long-Term Investor mode"
                className={`flex items-center space-x-1 sm:space-x-1.5 px-2 sm:px-3 py-1 sm:py-1.5 min-h-[32px] rounded-lg text-xs font-mono font-bold transition-colors active:scale-[0.96] transition-transform duration-100 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  activeRole === "LONG_TERM"
                    ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950/50 font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <span aria-hidden="true" className="text-xs">🏛️</span>
                <span className="font-mono tracking-tight text-[10px] sm:text-xs">Long Term</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Floating Bottom Navigation Dock for Mobile Devices */}
      <nav
        role="navigation"
        aria-label="Mobile Navigation Dock"
        className="lg:hidden fixed bottom-3 left-3 right-3 bg-[#0c1017]/95 backdrop-blur-md border border-[#243044] rounded-2xl p-1.5 shadow-2xl flex items-center justify-around z-50 font-mono text-xs"
      >
        <Link
          href="/"
          aria-current={pathname === "/" ? "page" : undefined}
          className={`flex flex-col items-center justify-center p-2 rounded-xl transition-colors min-w-[54px] min-h-[48px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <svg aria-hidden="true" className="w-5 h-5 mb-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect width="7" height="9" x="3" y="3" rx="1" />
            <rect width="7" height="5" x="14" y="3" rx="1" />
            <rect width="7" height="9" x="14" y="12" rx="1" />
            <rect width="7" height="5" x="3" y="16" rx="1" />
          </svg>
          <span className="text-[10px] tracking-tight">Terminal</span>
        </Link>

        <Link
          href="/screener"
          aria-current={pathname === "/screener" ? "page" : undefined}
          className={`flex flex-col items-center justify-center p-2 rounded-xl transition-colors min-w-[54px] min-h-[48px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/screener" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <svg aria-hidden="true" className="w-5 h-5 mb-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
          <span className="text-[10px] tracking-tight">Gems</span>
        </Link>

        <Link
          href="/compare"
          aria-current={pathname === "/compare" ? "page" : undefined}
          className={`flex flex-col items-center justify-center p-2 rounded-xl transition-colors min-w-[54px] min-h-[48px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/compare" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-base mb-0.5 leading-none">⚔️</span>
          <span className="text-[10px] tracking-tight">Compare</span>
        </Link>

        <Link
          href="/smart-money"
          aria-current={pathname === "/smart-money" ? "page" : undefined}
          className={`flex flex-col items-center justify-center p-2 rounded-xl transition-colors min-w-[54px] min-h-[48px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/smart-money" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-base mb-0.5 leading-none">🏛️</span>
          <span className="text-[10px] tracking-tight">Insiders</span>
        </Link>
      </nav>
    </>
  );
}