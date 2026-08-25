"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="bg-[#111722] border-b border-[#243044] px-6 py-3.5 flex items-center justify-between sticky top-0 z-50 backdrop-blur-md bg-opacity-95">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center text-slate-950 font-bold font-mono text-sm shadow-md">
          S
        </div>
        <div>
          <span className="text-base font-bold text-slate-100 tracking-tight block leading-none">
            Antigravity Quantitative
          </span>
          <span className="text-[10px] text-slate-400 font-mono tracking-widest uppercase block mt-0.5">
            Market Intelligence v2.0
          </span>
        </div>
      </div>

      <div className="flex items-center space-x-1 sm:space-x-2">
        <Link
          href="/"
          className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors focus-ring ${
            pathname === "/"
              ? "bg-[#1b2434] text-cyan-400 border border-[#364866]"
              : "text-slate-300 hover:text-slate-100 hover:bg-[#162030]"
          }`}
        >
          Dashboard
        </Link>
        <Link
          href="/screener"
          className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors focus-ring ${
            pathname === "/screener"
              ? "bg-[#1b2434] text-emerald-400 border border-[#364866]"
              : "text-slate-300 hover:text-slate-100 hover:bg-[#162030]"
          }`}
        >
          Hidden Gems Screener
        </Link>
      </div>
    </nav>
  );
}

