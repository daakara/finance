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
    <header className="border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur sticky top-0 z-50">
      <div className="max-w-[1750px] mx-auto px-3 sm:px-6 h-16 flex items-center justify-between gap-2 sm:gap-4">
        {/* Left: Brand Logo & Title */}
        <div className="flex items-center space-x-3 sm:space-x-6">
          <Link href="/" className="flex items-center space-x-2.5 group shrink-0">
            <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-lg bg-gradient-to-tr from-cyan-600 via-indigo-600 to-purple-600 flex items-center justify-center font-mono font-bold text-white shadow-lg shadow-cyan-950/50 group-hover:scale-105 transition-transform text-xs sm:text-sm">
              FT
            </div>
            <div>
              <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base block leading-none">
                FINANCE TERMINAL
              </span>
              <span className="text-[9px] sm:text-[10px] text-cyan-400 font-mono tracking-wider sm:tracking-widest uppercase">
                Quantitative Intel
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1 font-mono text-xs">
            <Link
              href="/"
              className={`px-3 py-1.5 rounded-lg transition-colors ${
                pathname === "/" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Terminal
            </Link>
            <Link
              href="/screener"
              className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                pathname === "/screener" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <svg className="w-3.5 h-3.5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
              </svg>
              <span>Hidden Gems</span>
            </Link>
          </nav>
        </div>

        {/* Right: Dual User Journey Role Switcher */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="bg-[#090d14] p-0.5 sm:p-1 rounded-xl border border-[#243044] flex items-center shadow-inner">
            {/* Day Trader Toggle */}
            <button
              onClick={() => handleRoleToggle("DAY_TRADER")}
              className={`flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all whitespace-nowrap ${
                activeRole === "DAY_TRADER"
                  ? "bg-amber-500 text-black shadow-lg shadow-amber-950/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <svg className="w-3 h-3 sm:w-3.5 sm:h-3.5" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              <span>Day Trader</span>
            </button>

            {/* Long-Term Toggle */}
            <button
              onClick={() => handleRoleToggle("LONG_TERM")}
              className={`flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all whitespace-nowrap ${
                activeRole === "LONG_TERM"
                  ? "bg-cyan-600 text-white shadow-lg shadow-cyan-950/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <svg className="w-3 h-3 sm:w-3.5 sm:h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="4" y="4" width="16" height="16" rx="2" />
                <line x1="9" y1="9" x2="9" y2="15" />
                <line x1="15" y1="9" x2="15" y2="15" />
              </svg>
              <span>Long-Term</span>
            </button>
          </div>

          {/* Status Indicator */}
          <div className="hidden lg:flex items-center space-x-1.5 bg-[#111722] border border-[#243044] px-2.5 py-1.5 rounded-lg font-mono text-[11px]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-300">Live API</span>
          </div>
        </div>
      </div>
    </header>
  );
}

