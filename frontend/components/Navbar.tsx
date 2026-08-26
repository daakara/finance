"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";

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
    <header role="banner" className="border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur sticky top-0 z-50">
      <div className="max-w-[1750px] mx-auto px-2.5 sm:px-6 h-14 sm:h-16 flex items-center justify-between gap-2">
        {/* Left: Brand Logo & Title */}
        <div className="flex items-center space-x-2 sm:space-x-6 min-w-0">
          <Link href="/" aria-label="Finance Terminal Home" className="flex items-center space-x-2 group shrink-0 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded-lg">
            <div aria-hidden="true" className="w-7 h-7 sm:w-9 sm:h-9 rounded-lg bg-gradient-to-tr from-cyan-600 via-indigo-600 to-purple-600 flex items-center justify-center font-mono font-bold text-white shadow-lg shadow-cyan-950/50 group-hover:scale-105 transition-transform text-xs sm:text-sm">
              FT
            </div>
            <div className="min-w-0">
              {/* Desktop Title */}
              <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base hidden sm:block leading-none">
                FINANCE TERMINAL
              </span>
              {/* Mobile Title */}
              <span className="font-bold tracking-tight text-white font-mono text-xs sm:hidden block leading-none">
                TERMINAL
              </span>
              <span className="text-[9px] text-cyan-400 font-mono tracking-wider uppercase hidden sm:block">
                Quantitative Intel
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav aria-label="Main Navigation" className="hidden md:flex items-center space-x-1 font-mono text-xs">
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
          </nav>
        </div>

        {/* Right: Dual User Journey Role Switcher */}
        <div className="flex items-center space-x-2 shrink-0">
          <div role="toolbar" aria-label="Trading Horizon Mode Switcher" className="bg-[#090d14] p-1 rounded-xl border border-[#243044] flex items-center shadow-inner">
            {/* Day Trader Toggle */}
            <button
              onClick={() => handleRoleToggle("DAY_TRADER")}
              aria-pressed={activeRole === "DAY_TRADER"}
              aria-label="Switch to Day Trader Mode"
              className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all active:scale-[0.96] transition-transform duration-100 whitespace-nowrap focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none ${
                activeRole === "DAY_TRADER"
                  ? "bg-amber-500 text-black shadow-lg shadow-amber-950/60 font-extrabold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <svg aria-hidden="true" className="w-3.5 h-3.5 shrink-0" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              <span>Day Trade</span>
            </button>

            {/* Long-Term Toggle */}
            <button
              onClick={() => handleRoleToggle("LONG_TERM")}
              aria-pressed={activeRole === "LONG_TERM"}
              aria-label="Switch to Long-Term Investor Mode"
              className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all active:scale-[0.96] transition-transform duration-100 whitespace-nowrap focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                activeRole === "LONG_TERM"
                  ? "bg-cyan-600 text-white shadow-lg shadow-cyan-950/60 font-extrabold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <svg aria-hidden="true" className="w-3.5 h-3.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="4" y="4" width="16" height="16" rx="2" />
                <line x1="9" y1="9" x2="9" y2="15" />
                <line x1="15" y1="9" x2="15" y2="15" />
              </svg>
              <span>Long Term</span>
            </button>
          </div>

          {/* Status Indicator */}
          <div aria-label="System status: live API connected" className="hidden lg:flex items-center space-x-1.5 bg-[#111722] border border-[#243044] px-2.5 py-1.5 rounded-lg font-mono text-[11px]">
            <span aria-hidden="true" className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse motion-reduce:animate-none"></span>
            <span className="text-slate-300">Live API</span>
          </div>
        </div>
      </div>
    </header>
  );
}

