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
      <div className="max-w-[1750px] mx-auto px-4 md:px-6 h-16 flex items-center justify-between gap-4">
        {/* Left: Brand Logo & Title */}
        <div className="flex items-center space-x-6">
          <Link href="/" className="flex items-center space-x-2.5 group">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-600 via-indigo-600 to-purple-600 flex items-center justify-center font-mono font-bold text-white shadow-lg shadow-cyan-950/50 group-hover:scale-105 transition-transform">
              FT
            </div>
            <div>
              <span className="font-bold tracking-tight text-white font-mono text-base block leading-none">
                FINANCE TERMINAL
              </span>
              <span className="text-[10px] text-cyan-400 font-mono tracking-widest uppercase">
                Quantitative Intelligence
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
              <span>??</span>
              <span>Hidden Gems</span>
            </Link>
          </nav>
        </div>

        {/* Center/Right: Dual User Journey Role Switcher */}
        <div className="flex items-center space-x-3">
          <div className="bg-[#090d14] p-1 rounded-xl border border-[#243044] flex items-center shadow-inner">
            <button
              onClick={() => handleRoleToggle("DAY_TRADER")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                activeRole === "DAY_TRADER"
                  ? "bg-amber-500 text-black shadow-lg shadow-amber-950/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <span>?</span>
              <span>Day Trader</span>
            </button>
            <button
              onClick={() => handleRoleToggle("LONG_TERM")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                activeRole === "LONG_TERM"
                  ? "bg-cyan-600 text-white shadow-lg shadow-cyan-950/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <span>???</span>
              <span>Long-Term Wealth</span>
            </button>
          </div>

          {/* Status Indicator */}
          <div className="hidden sm:flex items-center space-x-2 bg-[#111722] border border-[#243044] px-2.5 py-1.5 rounded-lg font-mono text-[11px]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-300">Live API</span>
          </div>
        </div>
      </div>
    </header>
  );
}

