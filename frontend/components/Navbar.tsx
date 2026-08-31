"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import UniversalOmniSearch from "./UniversalOmniSearch";
import ThemeToggle from "./ThemeToggle";
import OnboardingTourModal from "./OnboardingTourModal";
import PrivacySettingsModal from "./PrivacySettingsModal";
import CommandPaletteModal from "./CommandPaletteModal";

interface NavbarProps {
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
}

export default function Navbar({ userRole = "LONG_TERM", onRoleChange }: NavbarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">(userRole);
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");
  const [isOnboardingOpen, setIsOnboardingOpen] = useState<boolean>(false);
  const [isPrivacyOpen, setIsPrivacyOpen] = useState<boolean>(false);
  const [isPurging, setIsPurging] = useState<boolean>(false);
  const [purgeToast, setPurgeToast] = useState<boolean>(false);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState<boolean>(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState<boolean>(false);

  useEffect(() => {
    const handleGlobalKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      } else if (e.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) {
        e.preventDefault();
        setIsCommandPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", handleGlobalKey);
    return () => window.removeEventListener("keydown", handleGlobalKey);
  }, []);

  const handlePurgeCache = () => {
    setIsPurging(true);
    try {
      localStorage.removeItem("FINANCE_MARKET_SNAPSHOTS_V1");
      sessionStorage.clear();
      window.dispatchEvent(new CustomEvent("finance:cache-purge"));
      setPurgeToast(true);
      setTimeout(() => setPurgeToast(false), 3000);
    } catch (err) {
      console.warn("Failed to purge client cache:", err);
    } finally {
      setTimeout(() => setIsPurging(false), 600);
    }
  };

  const handleRoleToggle = (role: "DAY_TRADER" | "LONG_TERM") => {
    setActiveRole(role);
    try { localStorage.setItem("FINANCE_USER_ROLE", role); } catch {}
    if (onRoleChange) onRoleChange(role);
    window.dispatchEvent(new CustomEvent("finance:role-change", { detail: role }));
  };

  const handleVernacularToggle = (mode: "PLAIN_ENGLISH" | "PRO_QUANT") => {
    setVernacularMode(mode);
    try { localStorage.setItem("ARX_VERNACULAR_MODE", mode); } catch {}
    window.dispatchEvent(new CustomEvent("finance:vernacular-change", { detail: mode }));
  };

  useEffect(() => {
    try {
      const savedV = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (savedV === "PLAIN_ENGLISH" || savedV === "PRO_QUANT") {
        setVernacularMode(savedV);
      }
    } catch {}
  }, []);

  // Pro-Trader Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }

      if (e.key === "d" || e.key === "D") {
        const nextRole = activeRole === "DAY_TRADER" ? "LONG_TERM" : "DAY_TRADER";
        handleRoleToggle(nextRole);
      } else if (e.key === "v" || e.key === "V") {
        const nextV = vernacularMode === "PLAIN_ENGLISH" ? "PRO_QUANT" : "PLAIN_ENGLISH";
        handleVernacularToggle(nextV);
      } else if (e.key === "s" || e.key === "S") {
        if (pathname !== "/screener") router.push("/screener");
      } else if (e.key === "p" || e.key === "P") {
        if (pathname !== "/portfolio") router.push("/portfolio");
      } else if (e.key === "t" || e.key === "T") {
        if (pathname !== "/") router.push("/");
      } else if (e.key === "?") {
        setIsShortcutsOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeRole, vernacularMode, pathname, router]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("FINANCE_USER_ROLE") as "DAY_TRADER" | "LONG_TERM" | null;
      if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
        setActiveRole(saved);
      } else if (userRole) {
        setActiveRole(userRole);
      }
    } catch {
      if (userRole) setActiveRole(userRole);
    }
  }, []);

  useEffect(() => {
    if (userRole) setActiveRole(userRole);
  }, [userRole]);

  useEffect(() => {
    const handleRoleEvent = (e: Event) => {
      const custom = e as CustomEvent<"DAY_TRADER" | "LONG_TERM">;
      if (custom.detail === "DAY_TRADER" || custom.detail === "LONG_TERM") {
        setActiveRole(custom.detail);
      }
    };
    const handleVernacularEvent = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail === "PLAIN_ENGLISH" || custom.detail === "PRO_QUANT") {
        setVernacularMode(custom.detail);
      }
    };
    const handleOnboardingEvent = () => {
      setIsOnboardingOpen(true);
    };

    window.addEventListener("finance:role-change", handleRoleEvent);
    window.addEventListener("finance:vernacular-change", handleVernacularEvent);
    window.addEventListener("open-onboarding", handleOnboardingEvent);
    return () => {
      window.removeEventListener("finance:role-change", handleRoleEvent);
      window.removeEventListener("finance:vernacular-change", handleVernacularEvent);
      window.removeEventListener("open-onboarding", handleOnboardingEvent);
    };
  }, []);

  const handleOpenOnboarding = () => {
    setIsOnboardingOpen(true);
  };

  return (
    <>
      <header role="banner" className="border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur sticky top-0 z-50">
        <div className="max-w-[1750px] mx-auto px-2 sm:px-4 lg:px-4 xl:px-6 h-14 sm:h-16 flex items-center justify-between gap-1.5 sm:gap-2 xl:gap-4">
          {/* Left: Brand Logo & Title */}
          <div className="flex items-center space-x-1.5 sm:space-x-3 shrink-0 min-w-0">
            <Link href="/" aria-label="ARX Terminal Home" className="flex items-center space-x-2 group shrink-0 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded-lg">
              <div aria-hidden="true" className="w-7 h-7 sm:w-9 sm:h-9 rounded-lg bg-cyan-600 flex items-center justify-center font-mono font-bold text-white shadow-sm group-hover:scale-105 transition-transform text-xs sm:text-sm">
                ARX
              </div>
              <div className="min-w-0">
                <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base hidden xl:block leading-none">
                  ARX TERMINAL
                </span>
                <span className="font-bold tracking-tight text-white font-mono text-xs xl:hidden block leading-none">
                  ARX
                </span>
                <span className="text-[9px] text-cyan-400 font-mono tracking-wider uppercase hidden xl:block">
                  No-BS Market Intel
                </span>
              </div>
            </Link>

            {/* Desktop Navigation Links (Always visible on all desktop and laptop resolutions >= 1024px) */}
            <nav aria-label="Main Navigation" className="hidden lg:flex items-center space-x-0.5 xl:space-x-1 font-mono text-xs shrink-0">
              <Link
                href="/"
                aria-current={pathname === "/" ? "page" : undefined}
                className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Terminal
              </Link>
              <Link
                href="/screener"
                aria-current={pathname === "/screener" ? "page" : undefined}
                className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/screener" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <svg aria-hidden="true" className="w-3.5 h-3.5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
                <span>Screener</span>
              </Link>
              <Link
                href="/compare"
                aria-current={pathname === "/compare" ? "page" : undefined}
                className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/compare" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>⚔️ Compare</span>
              </Link>
              <Link
                href="/smart-money"
                aria-current={pathname === "/smart-money" ? "page" : undefined}
                className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/smart-money" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>🏛️ Insiders</span>
              </Link>
              <Link
                href="/portfolio"
                aria-current={pathname === "/portfolio" ? "page" : undefined}
                className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/portfolio" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>💼 Portfolio</span>
              </Link>
              <Link
                href="/guide"
                aria-current={pathname === "/guide" ? "page" : undefined}
                className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  pathname === "/guide" ? "bg-[#1b2434] text-cyan-400 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>📖 Guide</span>
              </Link>
              <button
                type="button"
                onClick={handleOpenOnboarding}
                aria-label="Open Terminal Setup & Onboarding Tour"
                className="hidden 2xl:flex px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-cyan-300 hover:bg-[#162030] transition-colors items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs"
              >
                <span>✨ Tour</span>
              </button>
              <button
                type="button"
                onClick={() => setIsPrivacyOpen(true)}
                aria-label="Open Privacy & Analytics Settings"
                className="hidden 2xl:flex px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-emerald-300 hover:bg-[#162030] transition-colors items-center gap-1 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:outline-none cursor-pointer text-xs"
                title="GDPR Privacy & Data Telemetry Settings"
              >
                <span>🛡️ Privacy</span>
              </button>
            </nav>
          </div>

          {/* Center: Global Omni-Search Bar & Cmd+K Quick Button */}
          <div className="flex-1 min-w-[70px] max-w-[120px] sm:max-w-[170px] md:max-w-[200px] lg:max-w-[220px] xl:max-w-xs 2xl:max-w-md mx-1 sm:mx-1.5 flex items-center justify-center gap-1.5">
            <UniversalOmniSearch />
            <button
              type="button"
              onClick={() => setIsCommandPaletteOpen(true)}
              aria-label="Open Command Palette (Cmd+K)"
              className="hidden sm:flex items-center gap-1 px-2 py-1.5 rounded-lg border border-[#243044] bg-[#070a10] text-[11px] font-mono text-slate-400 hover:text-cyan-300 hover:border-cyan-500/50 transition-colors shadow-inner shrink-0"
              title="Open Command Palette (Cmd+K or /)"
            >
              <span>⚡</span>
              <kbd className="text-[10px] text-cyan-400 font-bold">⌘K</kbd>
            </button>
          </div>

          {/* Right: Theme Toggle & Trading Horizon Mode Switcher */}
          <div className="flex items-center space-x-1 sm:space-x-1.5 shrink-0">
            {/* Tour Button (Mobile / Tablet visible) */}
            <button
              type="button"
              onClick={handleOpenOnboarding}
              aria-label="Open Terminal Setup & Onboarding Tour"
              className="2xl:hidden p-1.5 rounded-xl border border-[#243044] bg-[#090d14] text-slate-300 hover:text-cyan-300 hover:bg-[#162030] transition-colors flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs min-h-[32px] min-w-[32px]"
              title="Terminal Tour & Guide"
            >
              <span>✨</span>
            </button>

            {/* Purge Cache & Refresh Live Feeds Button */}
            <button
              type="button"
              onClick={handlePurgeCache}
              aria-label="Purge Local Cache & Re-sync Live Feeds"
              title="Purge Local Cache & Force Live Quote Refresh"
              className={`p-1.5 rounded-xl border border-[#243044] bg-[#090d14] text-slate-300 hover:text-cyan-300 hover:bg-[#162030] transition-all flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs min-h-[32px] min-w-[32px] active:scale-90 ${
                isPurging ? "animate-spin text-cyan-400 border-cyan-500" : ""
              }`}
            >
              <svg aria-hidden="true" className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                <path d="M21 3v5h-5" />
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                <path d="M3 21v-5h5" />
              </svg>
            </button>

            {/* Vernacular Language Mode Switcher (Plain English vs Pro Quant) */}
            <div role="toolbar" aria-label="Language Vernacular Mode Switcher" className="flex bg-[#090d14] p-0.5 rounded-xl border border-[#243044] items-center shadow-inner shrink-0">
              <button
                type="button"
                onClick={() => handleVernacularToggle("PLAIN_ENGLISH")}
                aria-pressed={vernacularMode === "PLAIN_ENGLISH"}
                aria-label="Switch to Plain English explanation mode"
                title="Plain English Mode: Clear, punchy, no-BS financial explanations"
                className={`flex items-center space-x-1 px-2 py-1 min-h-[30px] sm:min-h-[32px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:outline-none cursor-pointer ${
                  vernacularMode === "PLAIN_ENGLISH"
                    ? "bg-emerald-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <span aria-hidden="true" className="text-xs">💬</span>
                <span className="font-mono tracking-tight text-[10px] sm:text-xs">
                  <span className="hidden xl:inline">Plain English</span>
                  <span className="xl:hidden">Plain</span>
                </span>
              </button>

              <button
                type="button"
                onClick={() => handleVernacularToggle("PRO_QUANT")}
                aria-pressed={vernacularMode === "PRO_QUANT"}
                aria-label="Switch to Pro Quant mathematical mode"
                title="Pro Quant Mode: Rigorous mathematical models, VaR metrics, and factor loadings"
                className={`flex items-center space-x-1 px-2 py-1 min-h-[30px] sm:min-h-[32px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:outline-none cursor-pointer ${
                  vernacularMode === "PRO_QUANT"
                    ? "bg-purple-600 text-white shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <span aria-hidden="true" className="text-xs">🤓</span>
                <span className="font-mono tracking-tight text-[10px] sm:text-xs">
                  <span className="hidden xl:inline">Pro Quant</span>
                  <span className="xl:hidden">Quant</span>
                </span>
              </button>
            </div>

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Trading Horizon Switcher (Always 100% visible and unclipped across all viewports) */}
            <div role="toolbar" aria-label="Trading Horizon Mode Switcher" className="hidden sm:flex bg-[#090d14] p-0.5 rounded-xl border border-[#243044] items-center shadow-inner shrink-0">
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                role="button"
                aria-pressed={activeRole === "DAY_TRADER"}
                aria-label="Switch to Day Trader mode"
                title="Day Trader Mode (Intraday Momentum & Quick Scalps)"
                className={`flex items-center space-x-1 px-2 2xl:px-2.5 py-1 min-h-[30px] sm:min-h-[32px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none cursor-pointer ${
                  activeRole === "DAY_TRADER"
                    ? "bg-amber-500 text-slate-950 shadow-md shadow-amber-950/50 font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <span aria-hidden="true" className="text-xs">⚡</span>
                <span className="font-mono tracking-tight text-[10px] sm:text-xs">
                  <span className="hidden 2xl:inline">Day Trade</span>
                  <span className="2xl:hidden">Day</span>
                </span>
              </button>

              <button
                onClick={() => handleRoleToggle("LONG_TERM")}
                role="button"
                aria-pressed={activeRole === "LONG_TERM"}
                aria-label="Switch to Long-Term Investor mode"
                title="Long-Term Mode (Value Compounding & Secular Growth)"
                className={`flex items-center space-x-1 px-2 2xl:px-2.5 py-1 min-h-[30px] sm:min-h-[32px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer ${
                  activeRole === "LONG_TERM"
                    ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950/50 font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <span aria-hidden="true" className="text-xs">🏛️</span>
                <span className="font-mono tracking-tight text-[10px] sm:text-xs">
                  <span className="hidden 2xl:inline">Long Term</span>
                  <span className="2xl:hidden">Long</span>
                </span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Cache Purge Notification Toast */}
      {purgeToast && (
        <div
          role="status"
          aria-live="polite"
          className="fixed top-16 right-4 z-[1000] bg-cyan-950/95 border border-cyan-500 text-cyan-200 px-3.5 py-2 rounded-xl text-xs font-mono shadow-2xl flex items-center gap-2 animate-fadeIn"
        >
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
          <span>⚡ Local cache purged — Live quotes re-synced!</span>
        </div>
      )}

      {/* Floating Bottom Navigation Dock for Mobile Devices */}
      <nav
        role="navigation"
        aria-label="Mobile Navigation Dock"
        className="lg:hidden fixed bottom-0 left-0 right-0 w-full z-[999] bg-[#0c1017]/95 backdrop-blur-xl border-t border-[#243044] px-1.5 py-1.5 pb-[max(0.6rem,env(safe-area-inset-bottom))] shadow-2xl flex items-center justify-around font-mono text-[10px] transform-gpu"
        style={{ position: 'fixed', bottom: 0, left: 0, right: 0, width: '100%', zIndex: 999 }}
      >
        <Link
          href="/"
          aria-current={pathname === "/" ? "page" : undefined}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[46px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <svg aria-hidden="true" className="w-4 h-4 mb-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect width="7" height="9" x="3" y="3" rx="1" />
            <rect width="7" height="5" x="14" y="3" rx="1" />
            <rect width="7" height="9" x="14" y="12" rx="1" />
            <rect width="7" height="5" x="3" y="16" rx="1" />
          </svg>
          <span className="text-[9px] tracking-tight">Terminal</span>
        </Link>

        <Link
          href="/screener"
          aria-current={pathname === "/screener" ? "page" : undefined}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[46px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/screener" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <svg aria-hidden="true" className="w-4 h-4 mb-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
          <span className="text-[9px] tracking-tight">Screener</span>
        </Link>

        <Link
          href="/compare"
          aria-current={pathname === "/compare" ? "page" : undefined}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[46px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/compare" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">⚔️</span>
          <span className="text-[9px] tracking-tight">Compare</span>
        </Link>

        <Link
          href="/smart-money"
          aria-current={pathname === "/smart-money" ? "page" : undefined}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[46px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/smart-money" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">🏛️</span>
          <span className="text-[9px] tracking-tight">Insiders</span>
        </Link>

        <Link
          href="/portfolio"
          aria-current={pathname === "/portfolio" ? "page" : undefined}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[46px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/portfolio" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">💼</span>
          <span className="text-[9px] tracking-tight">Portfolio</span>
        </Link>

        <Link
          href="/guide"
          aria-current={pathname === "/guide" ? "page" : undefined}
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[44px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/guide" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">📖</span>
          <span className="text-[9px] tracking-tight">Guide</span>
        </Link>

        {/* Quick Horizon Toggle on Mobile Dock */}
        <button
          type="button"
          onClick={() => handleRoleToggle(activeRole === "DAY_TRADER" ? "LONG_TERM" : "DAY_TRADER")}
          aria-label={`Toggle Trading Horizon: currently ${activeRole === "DAY_TRADER" ? "Day Trader" : "Long-Term Investor"}`}
          className={`flex flex-col items-center justify-center py-1 px-1.5 rounded-xl transition-all min-w-[46px] min-h-[44px] border ${
            activeRole === "DAY_TRADER"
              ? "bg-amber-950/40 border-amber-500/50 text-amber-400 font-bold"
              : "bg-cyan-950/40 border-cyan-500/50 text-cyan-400 font-bold"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">
            {activeRole === "DAY_TRADER" ? "⚡" : "🏛️"}
          </span>
          <span className="text-[8.5px] tracking-tight">
            {activeRole === "DAY_TRADER" ? "Day" : "Long"}
          </span>
        </button>
      </nav>

      {/* Onboarding Tour Modal */}
      <OnboardingTourModal
        isOpen={isOnboardingOpen}
        onClose={() => setIsOnboardingOpen(false)}
      />

      {/* GDPR Privacy & Analytics Settings Modal */}
      <PrivacySettingsModal
        isOpen={isPrivacyOpen}
        onClose={() => setIsPrivacyOpen(false)}
      />

      {/* Pro-Trader Keyboard Shortcuts Modal */}
      {isShortcutsOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Keyboard Shortcuts Guide"
          className="fixed inset-0 z-[1000] flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fadeIn"
          onClick={() => setIsShortcutsOpen(false)}
        >
          <div
            className="bg-[#0f1520] border border-[#223149] rounded-2xl p-5 sm:p-6 max-w-md w-full shadow-2xl space-y-4 font-sans"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-[#1b2537] pb-3">
              <div className="flex items-center space-x-2">
                <span className="text-xl">⌨️</span>
                <h3 className="text-base font-black text-white">Pro-Trader Shortcuts</h3>
              </div>
              <button
                type="button"
                onClick={() => setIsShortcutsOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-[#1b2537] text-sm"
              >
                ✕
              </button>
            </div>

            <div className="space-y-2.5 text-xs">
              {[
                { key: "/", desc: "Open Universal Omni-Search & Ticker Scanner" },
                { key: "D", desc: "Toggle Day Trader ⚡ / Long Term 🏛️ Mode" },
                { key: "T", desc: "Navigate to Main Terminal Workspace" },
                { key: "S", desc: "Navigate to Screener & Pattern Radar" },
                { key: "P", desc: "Navigate to Private Portfolio Sizer" },
                { key: "?", desc: "Open / Close Shortcuts Cheatsheet" },
                { key: "Esc", desc: "Dismiss Open Modals & Dialogs" },
              ].map((s) => (
                <div key={s.key} className="flex items-center justify-between p-2 rounded-lg bg-[#090d14] border border-[#1a2333]">
                  <span className="text-slate-300 font-medium">{s.desc}</span>
                  <kbd className="px-2 py-0.5 rounded bg-[#1c2738] border border-[#2a3a52] text-cyan-400 font-mono font-bold text-xs shadow-inner">
                    {s.key}
                  </kbd>
                </div>
              ))}
            </div>

            <div className="text-right pt-2">
              <button
                type="button"
                onClick={() => setIsShortcutsOpen(false)}
                className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold transition-all shadow"
              >
                Got It (Esc)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ⚡ Global Cmd+K Omnisearch & Action Palette Modal */}
      <CommandPaletteModal
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
      />
    </>
  );
}