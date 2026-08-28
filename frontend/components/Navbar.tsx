"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import UniversalOmniSearch from "./UniversalOmniSearch";
import ThemeToggle from "./ThemeToggle";
import OnboardingTourModal from "./OnboardingTourModal";

interface NavbarProps {
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
}

export default function Navbar({ userRole = "LONG_TERM", onRoleChange }: NavbarProps) {
  const pathname = usePathname();
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">(userRole);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState<boolean>(false);
  const [isPurging, setIsPurging] = useState<boolean>(false);
  const [purgeToast, setPurgeToast] = useState<boolean>(false);

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

  useEffect(() => {
    setActiveRole(userRole);
  }, [userRole]);

  useEffect(() => {
    const handleRoleEvent = (e: Event) => {
      const custom = e as CustomEvent<"DAY_TRADER" | "LONG_TERM">;
      if (custom.detail === "DAY_TRADER" || custom.detail === "LONG_TERM") {
        setActiveRole(custom.detail);
        if (onRoleChange) onRoleChange(custom.detail);
      }
    };
    const handleOnboardingEvent = () => {
      setIsOnboardingOpen(true);
    };

    window.addEventListener("finance:role-change", handleRoleEvent);
    window.addEventListener("open-onboarding", handleOnboardingEvent);
    return () => {
      window.removeEventListener("finance:role-change", handleRoleEvent);
      window.removeEventListener("open-onboarding", handleOnboardingEvent);
    };
  }, [onRoleChange]);

  const handleOpenOnboarding = () => {
    setIsOnboardingOpen(true);
  };

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
              <div aria-hidden="true" className="w-7 h-7 sm:w-9 sm:h-9 rounded-lg bg-cyan-600 flex items-center justify-center font-mono font-bold text-white shadow-sm group-hover:scale-105 transition-transform text-xs sm:text-sm">
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
                <span>Screener</span>
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
              <button
                type="button"
                onClick={handleOpenOnboarding}
                aria-label="Open Terminal Setup & Onboarding Tour"
                className="px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-cyan-300 hover:bg-[#162030] transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs"
              >
                <span>✨ Tour</span>
              </button>
            </nav>
          </div>

          {/* Center: Global Omni-Search Bar (Desktop & Mobile) */}
          <div className="flex-1 max-w-xl mx-auto flex items-center justify-center px-1 sm:px-2">
            <UniversalOmniSearch />
          </div>

          {/* Right: Theme Toggle & Trading Horizon Mode Switcher */}
          <div className="flex items-center space-x-1 sm:space-x-2 shrink-0">
            {/* Tour Button (Mobile visible) */}
            <button
              type="button"
              onClick={handleOpenOnboarding}
              aria-label="Open Terminal Setup & Onboarding Tour"
              className="lg:hidden p-1.5 rounded-xl border border-[#243044] bg-[#090d14] text-slate-300 hover:text-cyan-300 hover:bg-[#162030] transition-colors flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs min-h-[32px] min-w-[32px]"
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

            {/* Theme Toggle */}
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
          className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-xl transition-colors min-w-[46px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
            pathname === "/guide" ? "bg-[#1b2434] text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">📖</span>
          <span className="text-[9px] tracking-tight">Guide</span>
        </Link>
      </nav>

      {/* Onboarding Tour Modal */}
      <OnboardingTourModal
        isOpen={isOnboardingOpen}
        onClose={() => setIsOnboardingOpen(false)}
      />
    </>
  );
}