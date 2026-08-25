"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import CommandPalette from "./CommandPalette";

export default function Navbar() {
  const pathname = usePathname();
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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
      <nav className="bg-[#111722] border-b border-[#243044] px-4 sm:px-6 py-2.5 sm:py-3 sticky top-0 z-40 backdrop-blur-md bg-opacity-95">
        <div className="flex items-center justify-between">
          {/* Logo & Brand Title (Responsive) */}
          <div className="flex items-center space-x-2.5 sm:space-x-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center text-slate-950 font-bold font-mono text-sm shadow-md shrink-0">
              S
            </div>
            <div>
              <span className="text-sm sm:text-base font-bold text-slate-100 tracking-tight block leading-none">
                <span className="sm:hidden">AG Quantitative</span>
                <span className="hidden sm:inline">Antigravity Quantitative</span>
              </span>
              <span className="text-[9px] sm:text-[10px] text-slate-400 font-mono tracking-widest uppercase block mt-0.5">
                Market Terminal
              </span>
            </div>
          </div>

          {/* Desktop & Tablet Navigation Controls */}
          <div className="flex items-center space-x-2">
            {/* Desktop Ctrl+K Search Trigger */}
            <button
              onClick={() => setIsCommandPaletteOpen(true)}
              className="hidden sm:flex items-center space-x-2 bg-[#090d14] border border-[#243044] hover:border-cyan-500/60 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 transition-colors focus-ring"
            >
              <span>Search Palette...</span>
              <kbd className="bg-[#1b2434] border border-[#364866] text-[10px] text-cyan-400 px-1.5 py-0.5 rounded font-mono">
                Ctrl+K
              </kbd>
            </button>

            {/* Mobile Touch Search Trigger Icon (Visible < 640px) */}
            <button
              onClick={() => setIsCommandPaletteOpen(true)}
              className="sm:hidden p-2 rounded-lg bg-[#090d14] border border-[#243044] text-cyan-400 focus-ring min-w-[40px] min-h-[40px] flex items-center justify-center"
              aria-label="Open Search Command Palette"
            >
              ??
            </button>

            {/* Desktop Route Links (= 768px) */}
            <div className="hidden md:flex items-center space-x-2">
              <Link
                href="/"
                className={`px-3.5 py-2 sm:py-1.5 rounded-md text-sm font-medium transition-colors focus-ring ${
                  pathname === "/"
                    ? "bg-[#1b2434] text-cyan-400 border border-[#364866]"
                    : "text-slate-300 hover:text-slate-100 hover:bg-[#162030]"
                }`}
              >
                Dashboard
              </Link>
              <Link
                href="/screener"
                className={`px-3.5 py-2 sm:py-1.5 rounded-md text-sm font-medium transition-colors focus-ring ${
                  pathname === "/screener"
                    ? "bg-[#1b2434] text-emerald-400 border border-[#364866]"
                    : "text-slate-300 hover:text-slate-100 hover:bg-[#162030]"
                }`}
              >
                Hidden Gems Screener
              </Link>
            </div>

            {/* Mobile Menu Hamburger Toggle (< 768px) */}
            <button
              onClick={() => setIsMobileMenuOpen((prev) => !prev)}
              className="md:hidden p-2 rounded-lg bg-[#090d14] border border-[#243044] text-slate-300 focus-ring min-w-[40px] min-h-[40px] flex items-center justify-center font-mono text-sm"
              aria-label="Toggle Navigation Menu"
            >
              {isMobileMenuOpen ? "?" : "?"}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu Drawer (< 768px) */}
        {isMobileMenuOpen && (
          <div className="md:hidden pt-3 pb-2 space-y-2 border-t border-[#243044] mt-3 animate-in fade-in slide-in-from-top-2 duration-150">
            <Link
              href="/"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`block w-full px-4 py-2.5 rounded-lg text-sm font-medium font-mono transition-colors focus-ring ${
                pathname === "/"
                  ? "bg-[#1b2434] text-cyan-400 border border-[#364866]"
                  : "text-slate-300 hover:bg-[#162030]"
              }`}
            >
              ? Dashboard Terminal
            </Link>
            <Link
              href="/screener"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`block w-full px-4 py-2.5 rounded-lg text-sm font-medium font-mono transition-colors focus-ring ${
                pathname === "/screener"
                  ? "bg-[#1b2434] text-emerald-400 border border-[#364866]"
                  : "text-slate-300 hover:bg-[#162030]"
              }`}
            >
              ?? Hidden Gems Screener
            </Link>
          </div>
        )}
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

