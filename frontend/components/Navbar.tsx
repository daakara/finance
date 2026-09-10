"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useEffect, useCallback } from "react";
import UniversalOmniSearch from "./UniversalOmniSearch";
import ThemeToggle from "./ThemeToggle";
import OnboardingTourModal from "./OnboardingTourModal";
import PrivacySettingsModal from "./PrivacySettingsModal";
import CommandPaletteModal from "./CommandPaletteModal";
import RealTimeAlertEngine from "./RealTimeAlertEngine";
import ArxLogo from "./ArxLogo";
import MarketCommandRibbon from "./nav/MarketCommandRibbon";
import ExperienceModeToggle from "./experience/ExperienceModeToggle";
import WatchlistDrawerTrigger from "./drawers/WatchlistDrawerTrigger";
import { CANONICAL_HUBS, buildHubHref, isHubActive, extractActiveSymbol } from "../lib/canonicalNav";

interface NavbarProps {
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
  hideMobileDock?: boolean;
  activeSymbol?: string | null;
}

export default function Navbar({
  userRole = "LONG_TERM",
  onRoleChange,
  hideMobileDock = false,
  activeSymbol,
}: NavbarProps) {
  const pathname = usePathname();
  const router = useRouter();

  const [urlSymbolState, setUrlSymbolState] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const sym = extractActiveSymbol(window.location.search);
      setUrlSymbolState(sym);
    }
  }, [pathname]);

  const effectiveSymbol = activeSymbol !== undefined ? activeSymbol : urlSymbolState;
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

  const handleRoleToggle = useCallback((role: "DAY_TRADER" | "LONG_TERM") => {
    setActiveRole(role);
    try { localStorage.setItem("FINANCE_USER_ROLE", role); } catch {}
    if (onRoleChange) onRoleChange(role);
    window.dispatchEvent(new CustomEvent("finance:role-change", { detail: role }));
  }, [onRoleChange]);

  const handleVernacularToggle = useCallback((mode: "PLAIN_ENGLISH" | "PRO_QUANT") => {
    setVernacularMode(mode);
    try { localStorage.setItem("ARX_VERNACULAR_MODE", mode); } catch {}
    window.dispatchEvent(new CustomEvent("finance:vernacular-change", { detail: mode }));
  }, []);

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
        const radarHref = buildHubHref("radar", effectiveSymbol);
        if (pathname !== "/radar") router.push(radarHref);
      } else if (e.key === "p" || e.key === "P") {
        const portfolioHref = buildHubHref("portfolio", effectiveSymbol);
        if (pathname !== "/portfolio") router.push(portfolioHref);
      } else if (e.key === "t" || e.key === "T") {
        const analysisHref = buildHubHref("analysis", effectiveSymbol);
        if (pathname !== "/") router.push(analysisHref);
      } else if (e.key === "?") {
        setIsShortcutsOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeRole, vernacularMode, pathname, router, handleRoleToggle, handleVernacularToggle]);

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
      <div className="sticky top-0 z-50">
        <header
          role="banner"
          data-testid="navbar"
          className="border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur h-14 flex items-center"
        >
          <div className="max-w-[1750px] mx-auto px-2 sm:px-4 lg:px-4 xl:px-6 w-full h-14 flex items-center justify-between gap-1.5 sm:gap-2 xl:gap-4">
            {/* Left: Brand Logo & Title */}
            <div className="flex items-center space-x-1.5 sm:space-x-3 shrink-0 min-w-0">
              <Link
                href="/"
                aria-label="ARX Terminal Home"
                className="flex items-center space-x-2 group shrink-0 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded-lg"
              >
                <ArxLogo size="sm" variant="badge" />
                <div className="min-w-0 hidden sm:block">
                  <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base block leading-none">
                    ARX TERMINAL
                  </span>
                  <span className="text-[9px] text-cyan-400 font-mono tracking-wider uppercase hidden xl:block mt-0.5">
                    No-BS Market Intel
                  </span>
                </div>
              </Link>

              {/* Desktop Navigation Links (Canonical 6 Hubs) */}
              <nav
                aria-label="Main Navigation"
                data-testid="desktop-nav-links"
                className="hidden lg:flex items-center space-x-0.5 xl:space-x-1 font-mono text-xs shrink-0"
              >
                {CANONICAL_HUBS.map((hub) => {
                  const href = buildHubHref(hub, effectiveSymbol);
                  const active = isHubActive(hub.href, pathname);
                  return (
                    <Link
                      key={hub.id}
                      href={href}
                      aria-current={active ? "page" : undefined}
                      className={`px-2 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                        active
                          ? hub.id === "performance"
                            ? "bg-emerald-950/80 text-emerald-300 font-bold border border-emerald-700/60"
                            : "bg-[#1b2434] text-cyan-400 font-semibold"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <span>{hub.label}</span>
                    </Link>
                  );
                })}
              </nav>
            </div>

          {/* Center: Global Omni-Search Bar */}
          <div className="flex-1 min-w-0 md:min-w-[140px] max-w-xs xl:max-w-sm 2xl:max-w-md mx-1.5 sm:mx-2 flex items-center justify-center">
            <UniversalOmniSearch />
          </div>

          {/* Right: Theme Toggle & Trading Horizon Mode Switcher */}
          <div className="flex items-center space-x-1 sm:space-x-1.5 shrink-0">
            {/* Purge Cache & Refresh Live Feeds Button */}
            <button
              type="button"
              onClick={handlePurgeCache}
              aria-label="Purge Local Cache & Re-sync Live Feeds"
              title="Purge Local Cache & Force Live Quote Refresh"
              className={`p-2.5 rounded-xl border border-[#243044] bg-[#090d14] text-slate-300 hover:text-cyan-300 hover:bg-[#162030] transition-all flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs min-h-[44px] min-w-[44px] active:scale-90 motion-reduce:transform-none ${
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

            {/* Trading Horizon Switcher (Always 100% visible and unclipped across all viewports) */}
            <div role="toolbar" aria-label="Trading Horizon Mode Switcher" className="hidden sm:flex bg-[#090d14] p-0.5 rounded-xl border border-[#243044] items-center shadow-inner shrink-0">
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                role="button"
                aria-pressed={activeRole === "DAY_TRADER"}
                aria-label="Switch to Day Trader mode"
                title="Day Trader Mode (Intraday Momentum & Quick Scalps)"
                className={`flex items-center space-x-1 px-3 2xl:px-3.5 py-2 sm:py-1.5 min-h-[44px] sm:min-h-[38px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] motion-reduce:transform-none transition-transform duration-100 ease-out focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none cursor-pointer ${
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
                className={`flex items-center space-x-1 px-3 2xl:px-3.5 py-2 sm:py-1.5 min-h-[44px] sm:min-h-[38px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] motion-reduce:transform-none transition-transform duration-100 ease-out focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer ${
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

      {/* Persistent 36px Market Command Ribbon directly beneath Navbar */}
      <MarketCommandRibbon />
    </div>

    {/* Cache Purge Notification Toast */}
    {purgeToast && (
      <div
        role="status"
        aria-live="polite"
        className="fixed top-24 right-4 z-[1000] bg-cyan-950/95 border border-cyan-500 text-cyan-200 px-3.5 py-2 rounded-xl text-xs font-mono shadow-2xl flex items-center gap-2 animate-fadeIn"
      >
        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
        <span>⚡ Local cache purged — Live quotes re-synced!</span>
      </div>
    )}

    {/* Floating Bottom Navigation Dock for Mobile Devices (Canonical 6 Hubs) */}
    {!hideMobileDock && (
      <nav
        role="navigation"
        aria-label="Mobile Navigation Dock"
        data-testid="mobile-nav-dock"
        className="lg:hidden fixed bottom-0 left-0 right-0 w-full z-[999] bg-[#0c1017]/95 backdrop-blur-xl border-t border-[#243044] px-1 py-1 pb-[max(0.6rem,env(safe-area-inset-bottom))] shadow-2xl flex items-center justify-around font-mono text-[10px] transform-gpu"
        style={{ position: 'fixed', bottom: 0, left: 0, right: 0, width: '100%', zIndex: 999 }}
      >
        {CANONICAL_HUBS.map((hub) => {
          const href = buildHubHref(hub, effectiveSymbol);
          const active = isHubActive(hub.href, pathname);
          return (
            <Link
              key={hub.id}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`flex flex-col items-center justify-center py-1 px-1.5 rounded-xl transition-colors min-w-[44px] sm:min-w-[48px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                active
                  ? hub.id === "performance"
                    ? "bg-emerald-950/80 text-emerald-300 font-bold border border-emerald-700/60"
                    : "bg-[#1b2434] text-cyan-400 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <span aria-hidden="true" className="text-sm mb-0.5 leading-none">{hub.icon}</span>
              <span className="text-[8.5px] sm:text-[9px] tracking-tight">{hub.label}</span>
            </Link>
          );
        })}

        {/* Quick Horizon Toggle on Mobile Dock */}
        <button
          type="button"
          onClick={() => handleRoleToggle(activeRole === "DAY_TRADER" ? "LONG_TERM" : "DAY_TRADER")}
          aria-label={`Toggle Trading Horizon: currently ${activeRole === "DAY_TRADER" ? "Day Trader" : "Long-Term Investor"}`}
          className={`flex flex-col items-center justify-center py-1 px-1.5 rounded-xl transition-all active:scale-[0.96] motion-reduce:transform-none min-w-[44px] sm:min-w-[48px] min-h-[44px] border ${
            activeRole === "DAY_TRADER"
              ? "bg-amber-950/40 border-amber-500/50 text-amber-400 font-bold"
              : "bg-cyan-950/40 border-cyan-500/50 text-cyan-400 font-bold"
          }`}
        >
          <span aria-hidden="true" className="text-sm mb-0.5 leading-none">
            {activeRole === "DAY_TRADER" ? "⚡" : "🏛️"}
          </span>
          <span className="text-[8px] sm:text-[8.5px] tracking-tight">
            {activeRole === "DAY_TRADER" ? "Day" : "Long"}
          </span>
        </button>
      </nav>
    )}

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

      {/* 🔔 Real-Time Price Level Alert Notification Engine */}
      <RealTimeAlertEngine />
    </>
  );
}