"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Search, Moon, Sun, MoreVertical, Compass, Activity, ShieldCheck, Briefcase } from "lucide-react";

interface SynthesisHeaderProps {
  activeSymbol?: string | null;
  onSelectSymbol?: (symbol: string) => void;
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
}

const POPULAR_TICKERS = [
  { symbol: "NVDA", name: "NVIDIA Corp.", exchange: "NASDAQ", role: "AI Semiconductor" },
  { symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", role: "Consumer Hardware" },
  { symbol: "MSFT", name: "Microsoft Corp.", exchange: "NASDAQ", role: "Enterprise Software" },
  { symbol: "AMD", name: "Advanced Micro Devices", exchange: "NASDAQ", role: "Semiconductors" },
  { symbol: "SPY", name: "SPDR S&P 500 ETF", exchange: "NYSE", role: "Broad Market ETF" },
  { symbol: "QQQ", name: "Invesco QQQ Trust", exchange: "NASDAQ", role: "Tech Index ETF" },
  { symbol: "SMH", name: "VanEck Semiconductor", exchange: "NASDAQ", role: "Chip Sector ETF" },
];

export default function SynthesisHeader({
  activeSymbol = "AAPL",
  onSelectSymbol,
  userRole = "LONG_TERM",
  onRoleChange,
}: SynthesisHeaderProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [searchQuery, setSearchQuery] = useState("");
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "paper">("dark");
  const searchRef = useRef<HTMLDivElement>(null);

  // Initialize theme from document or localStorage
  useEffect(() => {
    if (typeof document !== "undefined") {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "paper" ? "paper" : "dark");
    }
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "paper" : "dark";
    setTheme(next);
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("ARX_THEME", next);
    }
  };

  // Keyboard shortcut Cmd/Ctrl + K or / to focus search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsSearchOpen(true);
        const input = document.getElementById("omni-search-input");
        if (input) input.focus();
      } else if (e.key === "/" && (e.target as HTMLElement)?.tagName !== "INPUT") {
        e.preventDefault();
        setIsSearchOpen(true);
        const input = document.getElementById("omni-search-input");
        if (input) input.focus();
      } else if (e.key === "Escape") {
        setIsSearchOpen(false);
        setIsMenuOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Handle outside click to close dropdowns
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setIsSearchOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = searchQuery.trim().toUpperCase();
    if (clean) {
      if (onSelectSymbol) {
        onSelectSymbol(clean);
      } else {
        router.push(`/?symbol=${clean}`);
      }
      setIsSearchOpen(false);
      setSearchQuery("");
    }
  };

  const handlePickTicker = (sym: string) => {
    if (onSelectSymbol) {
      onSelectSymbol(sym);
    } else {
      router.push(`/?symbol=${sym}`);
    }
    setIsSearchOpen(false);
    setSearchQuery("");
  };

  const symbolParam = activeSymbol ? `?symbol=${activeSymbol}` : "";

  return (
    <header className="sticky top-0 z-40 w-full bg-[var(--surface-1)] border-b border-[var(--border-subtle)] backdrop-blur-md transition-colors duration-200">
      <div className="max-w-[1750px] mx-auto px-3 sm:px-6 h-14 sm:h-16 flex items-center justify-between gap-3">
        {/* Brand & 4-Hub Nav Group */}
        <div className="flex items-center gap-4 lg:gap-8 min-w-0">
          <Link href="/" className="flex items-center gap-2 shrink-0 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 rounded-lg">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center shadow-md">
              <span className="font-mono font-black text-slate-950 text-base">A</span>
            </div>
            <div className="hidden sm:flex flex-col">
              <span className="font-black tracking-tight text-white text-sm sm:text-base leading-none group-hover:text-cyan-400 transition-colors">
                ARX TERMINAL
              </span>
              <span className="text-[10px] font-mono text-cyan-400/80 uppercase tracking-widest mt-0.5">
                Synthesis E V3
              </span>
            </div>
          </Link>

          {/* 4-Hub Navigation (Desktop) */}
          <nav className="hidden md:flex items-center gap-1 font-mono text-xs" aria-label="Main Navigation">
            <Link
              href="/radar"
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg font-bold transition-colors min-h-[44px] ${
                pathname === "/radar"
                  ? "bg-[var(--surface-active)] text-cyan-400 border border-[var(--border-strong)]"
                  : "text-[var(--text-2)] hover:text-white hover:bg-[var(--surface-hover)]"
              }`}
            >
              <Compass className="w-4 h-4" />
              <span>Radar</span>
            </Link>

            <Link
              href={`/${symbolParam}`}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg font-bold transition-colors min-h-[44px] ${
                pathname === "/" || pathname === ""
                  ? "bg-[var(--surface-active)] text-cyan-400 border border-[var(--border-strong)]"
                  : "text-[var(--text-2)] hover:text-white hover:bg-[var(--surface-hover)]"
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Analysis</span>
            </Link>

            <button
              type="button"
              onClick={() => {
                const el = document.getElementById("optimal-entry-exit-card");
                if (el) el.scrollIntoView({ behavior: "smooth" });
              }}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg font-bold text-[var(--text-2)] hover:text-white hover:bg-[var(--surface-hover)] transition-colors min-h-[44px] cursor-pointer"
            >
              <ShieldCheck className="w-4 h-4" />
              <span>Trade Plan</span>
            </button>

            <Link
              href="/portfolio"
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg font-bold transition-colors min-h-[44px] ${
                pathname === "/portfolio"
                  ? "bg-[var(--surface-active)] text-cyan-400 border border-[var(--border-strong)]"
                  : "text-[var(--text-2)] hover:text-white hover:bg-[var(--surface-hover)]"
              }`}
            >
              <Briefcase className="w-4 h-4" />
              <span>Portfolio</span>
            </Link>
          </nav>
        </div>

        {/* Omni Search & Quick Ticker Selector */}
        <div ref={searchRef} className="relative flex-1 max-w-md mx-2 sm:mx-4">
          <form onSubmit={handleSearchSubmit} className="relative flex items-center">
            <Search className="absolute left-3 w-4 h-4 text-[var(--text-muted)] pointer-events-none" />
            <input
              id="omni-search-input"
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setIsSearchOpen(true);
              }}
              onFocus={() => setIsSearchOpen(true)}
              placeholder="Search ticker, company, theme... (Cmd+K)"
              className="w-full bg-[var(--surface-2)] border border-[var(--border-subtle)] focus:border-cyan-500 rounded-xl pl-9 pr-14 py-2 text-xs sm:text-sm text-[var(--text-1)] placeholder-[var(--text-muted)] font-mono transition-all outline-none min-h-[40px] focus:ring-1 focus:ring-cyan-500"
              autoComplete="off"
            />
            <div className="absolute right-2.5 hidden sm:flex items-center gap-1 text-[10px] font-mono font-bold text-[var(--text-muted)] bg-[var(--surface-1)] px-1.5 py-0.5 rounded border border-[var(--border-subtle)] pointer-events-none">
              <span>⌘K</span>
            </div>
          </form>

          {/* Instant Ticker Dropdown */}
          {isSearchOpen && (
            <div
              id="search-dropdown"
              className="absolute left-0 right-0 top-full mt-1.5 bg-[var(--surface-elevated)] border border-[var(--border-strong)] rounded-xl shadow-2xl overflow-hidden z-50 animate-fade-in divide-y divide-[var(--border-subtle)]"
            >
              <div className="px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider text-[var(--text-muted)] bg-[var(--surface-2)] flex items-center justify-between">
                <span>Verified Assets ({POPULAR_TICKERS.length})</span>
                <span>Press Enter to Jump</span>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {POPULAR_TICKERS.filter(
                  (t) =>
                    !searchQuery ||
                    t.symbol.includes(searchQuery.toUpperCase()) ||
                    t.name.toUpperCase().includes(searchQuery.toUpperCase())
                ).map((t) => (
                  <button
                    key={t.symbol}
                    type="button"
                    onClick={() => handlePickTicker(t.symbol)}
                    className="w-full px-3 py-2.5 flex items-center justify-between text-left hover:bg-[var(--surface-hover)] transition-colors min-h-[44px] cursor-pointer"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-black text-sm text-cyan-400">{t.symbol}</span>
                      <span className="text-xs text-[var(--text-1)] font-sans">{t.name}</span>
                    </div>
                    <div className="flex items-center gap-2 font-mono text-[11px]">
                      <span className="text-[var(--text-muted)]">{t.role}</span>
                      <span className="px-1.5 py-0.5 rounded bg-[var(--surface-2)] text-[10px] text-[var(--text-2)] border border-[var(--border-subtle)]">
                        {t.exchange}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Command Utilities Group */}
        <div className="flex items-center gap-2 shrink-0">
          {/* User Role Selector */}
          {onRoleChange && (
            <div className="hidden lg:flex items-center bg-[var(--surface-2)] p-0.5 rounded-lg border border-[var(--border-subtle)] font-mono text-xs">
              <button
                type="button"
                onClick={() => onRoleChange("LONG_TERM")}
                className={`px-2.5 py-1.5 rounded-md font-bold transition-all min-h-[32px] cursor-pointer ${
                  userRole === "LONG_TERM"
                    ? "bg-cyan-600 text-slate-950 font-black shadow-sm"
                    : "text-[var(--text-2)] hover:text-white"
                }`}
              >
                Swing / Investor
              </button>
              <button
                type="button"
                onClick={() => onRoleChange("DAY_TRADER")}
                className={`px-2.5 py-1.5 rounded-md font-bold transition-all min-h-[32px] cursor-pointer ${
                  userRole === "DAY_TRADER"
                    ? "bg-cyan-600 text-slate-950 font-black shadow-sm"
                    : "text-[var(--text-2)] hover:text-white"
                }`}
              >
                Day Trader
              </button>
            </div>
          )}

          {/* Theme Toggle */}
          <button
            id="theme-btn"
            type="button"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to Editorial Paper light theme" : "Switch to Obsidian Dark theme"}
            className="p-2.5 rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-2)] hover:text-white border border-[var(--border-subtle)] transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center cursor-pointer"
          >
            {theme === "dark" ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-cyan-500" />}
          </button>

          {/* Utilities Overflow Button */}
          <div className="relative">
            <button
              id="header-utilities-btn"
              type="button"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              aria-label="Toggle utilities and keyboard shortcuts menu"
              aria-expanded={isMenuOpen}
              className="p-2.5 rounded-xl bg-[var(--surface-2)] hover:bg-[var(--surface-hover)] text-[var(--text-2)] hover:text-white border border-[var(--border-subtle)] transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center cursor-pointer"
            >
              <MoreVertical className="w-4 h-4" />
            </button>

            {isMenuOpen && (
              <div
                id="header-overflow-menu"
                className="absolute right-0 top-full mt-1.5 w-56 bg-[var(--surface-elevated)] border border-[var(--border-strong)] rounded-xl shadow-2xl p-1.5 z-50 animate-fade-in font-mono text-xs divide-y divide-[var(--border-subtle)]"
              >
                <div className="py-1">
                  <div className="px-3 py-1.5 text-[10px] uppercase text-[var(--text-muted)] font-bold">
                    ARX Utilities
                  </div>
                  <Link
                    href="/smart-money"
                    onClick={() => setIsMenuOpen(false)}
                    className="flex items-center justify-between px-3 py-2 rounded-lg text-[var(--text-1)] hover:bg-[var(--surface-hover)] transition-colors min-h-[36px]"
                  >
                    <span>Smart Money Scanner</span>
                    <span className="text-[10px] text-cyan-400 font-bold">C006</span>
                  </Link>
                  <Link
                    href="/prototype/"
                    onClick={() => setIsMenuOpen(false)}
                    className="flex items-center justify-between px-3 py-2 rounded-lg text-[var(--text-1)] hover:bg-[var(--surface-hover)] transition-colors min-h-[36px]"
                  >
                    <span>Approved V3 Prototype</span>
                    <span className="text-[10px] text-emerald-400 font-bold">SHA-256</span>
                  </Link>
                </div>
                <div className="py-1">
                  <button
                    type="button"
                    onClick={() => {
                      setIsMenuOpen(false);
                      const el = document.getElementById("guide-banner");
                      if (el) el.scrollIntoView({ behavior: "smooth" });
                    }}
                    className="w-full text-left px-3 py-2 rounded-lg text-[var(--text-1)] hover:bg-[var(--surface-hover)] transition-colors min-h-[36px] cursor-pointer"
                  >
                    Show Workflow Guide
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
