"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import CommandPalette from "./CommandPalette";

export default function Navbar() {
  const pathname = usePathname();
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, []);

  return (
    <>
      <nav className="bg-[#111722] border-b border-[#243044] px-6 py-3 flex items-center justify-between sticky top-0 z-40 backdrop-blur-md bg-opacity-95">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center text-slate-950 font-bold font-mono text-sm shadow-md">
            S
          </div>
          <div>
            <span className="text-base font-bold text-slate-100 tracking-tight block leading-none">
              Antigravity Quantitative
            </span>
            <span className="text-[10px] text-slate-400 font-mono tracking-widest uppercase block mt-0.5">
              Market Intelligence Terminal
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Quick Ctrl+K Search Trigger Button */}
          <button
            onClick={() => setIsCommandPaletteOpen(true)}
            className="hidden sm:flex items-center space-x-2 bg-[#090d14] border border-[#243044] hover:border-cyan-500/60 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 transition-colors focus-ring"
          >
            <span>Search Command Palette...</span>
            <kbd className="bg-[#1b2434] border border-[#364866] text-[10px] text-cyan-400 px-1.5 py-0.5 rounded font-mono">
              Ctrl+K
            </kbd>
          </button>

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
        </div>
      </nav>

      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onSelectSymbol={(sym) => {
          window.location.href = `/?symbol=${encodeURIComponent(sym)}`;
        }}
      />
    </>
  );
}

