"use client";

import { trackFirstHubNavigation } from "../lib/matomo";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useEffect, useCallback, useRef } from "react";
// @ts-ignore
import { createPortal } from "react-dom";
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
import { HelpCircle, Compass, Shield, RefreshCw, Zap, Landmark, Radar, Microscope, Briefcase } from "lucide-react";

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
  const [isUtilitiesMenuOpen, setIsUtilitiesMenuOpen] = useState<boolean>(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState<boolean>(false);
  const shortcutsTriggerRef = useRef<HTMLElement | null>(null);
  const shortcutsCloseBtnRef = useRef<HTMLButtonElement | null>(null);
  const utilitiesMenuRef = useRef<HTMLDivElement | null>(null);
  const utilitiesMenuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const firstVisitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [mounted, setMounted] = useState<boolean>(false);
  const [menuPosition, setMenuPosition] = useState<{ top: number; right: number; maxHeight: number }>({
    top: 56,
    right: 16,
    maxHeight: 480,
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  const updateMenuPosition = useCallback(() => {
    if (!utilitiesMenuTriggerRef.current) return;
    const rect = utilitiesMenuTriggerRef.current.getBoundingClientRect();

    const vv = typeof window !== "undefined" ? window.visualViewport : null;
    const viewportWidth = vv ? vv.width : (typeof window !== "undefined" ? window.innerWidth : 390);
    const viewportHeight = vv ? vv.height : (typeof window !== "undefined" ? window.innerHeight : 844);
    const viewportOffsetTop = vv ? vv.offsetTop : 0;
    const viewportOffsetLeft = vv ? vv.offsetLeft : 0;

    // Anchor top to bottom of trigger plus margin
    const top = Math.max(viewportOffsetTop, rect.bottom + 6);

    // Anchor right edge inside viewport
    const rightOffsetFromViewportEdge = (viewportOffsetLeft + viewportWidth) - rect.right;
    const right = Math.max(8, rightOffsetFromViewportEdge);

    // Calculate max height within visual viewport
    const availableHeight = (viewportOffsetTop + viewportHeight) - top - 16;
    const maxHeight = Math.max(200, Math.min(availableHeight, 520));

    setMenuPosition({ top, right, maxHeight });
  }, []);

  useEffect(() => {
    if (!isUtilitiesMenuOpen) return;
    updateMenuPosition();

    const handleReposition = () => {
      updateMenuPosition();
    };

    window.addEventListener("resize", handleReposition, { passive: true });
    window.addEventListener("scroll", handleReposition, { passive: true });
    if (window.visualViewport) {
      window.visualViewport.addEventListener("resize", handleReposition);
      window.visualViewport.addEventListener("scroll", handleReposition);
    }

    const handleOutsideInteraction = (e: MouseEvent | TouchEvent | PointerEvent) => {
      const dropdownEl = document.getElementById("utilities-menu-dropdown");
      const targetNode = e.target as Node;
      const isInsideTrigger = utilitiesMenuRef.current?.contains(targetNode);
      const isInsideDropdown = dropdownEl?.contains(targetNode);
      if (!isInsideTrigger && !isInsideDropdown) {
        setIsUtilitiesMenuOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsUtilitiesMenuOpen(false);
        utilitiesMenuTriggerRef.current?.focus();
      }
    };

    if (typeof window !== "undefined" && "PointerEvent" in window) {
      document.addEventListener("pointerdown", handleOutsideInteraction);
    } else {
      document.addEventListener("mousedown", handleOutsideInteraction);
      document.addEventListener("touchstart", handleOutsideInteraction);
    }
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("resize", handleReposition);
      window.removeEventListener("scroll", handleReposition);
      if (window.visualViewport) {
        window.visualViewport.removeEventListener("resize", handleReposition);
        window.visualViewport.removeEventListener("scroll", handleReposition);
      }
      if (typeof window !== "undefined" && "PointerEvent" in window) {
        document.removeEventListener("pointerdown", handleOutsideInteraction);
      } else {
        document.removeEventListener("mousedown", handleOutsideInteraction);
        document.removeEventListener("touchstart", handleOutsideInteraction);
      }
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isUtilitiesMenuOpen, updateMenuPosition]);

  const handleUtilitiesBlur = (e: React.FocusEvent<HTMLDivElement>) => {
    const dropdownEl = document.getElementById("utilities-menu-dropdown");
    const relatedTargetNode = e.relatedTarget as Node | null;
    if (
      relatedTargetNode &&
      utilitiesMenuRef.current &&
      !utilitiesMenuRef.current.contains(relatedTargetNode) &&
      (!dropdownEl || !dropdownEl.contains(relatedTargetNode))
    ) {
      setIsUtilitiesMenuOpen(false);
    }
  };

  useEffect(() => {
    if (isShortcutsOpen) {
      shortcutsTriggerRef.current = (document.activeElement as HTMLElement) || null;
      setTimeout(() => shortcutsCloseBtnRef.current?.focus(), 50);
    } else {
      setTimeout(() => shortcutsTriggerRef.current?.focus(), 20);
    }
  }, [isShortcutsOpen]);

  useEffect(() => {
    const handleGlobalKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleGlobalKey);
    return () => window.removeEventListener("keydown", handleGlobalKey);
  }, []);

  const handlePurgeCache = () => {
    if (typeof window !== "undefined") {
      const confirmed = window.confirm("Purge local market snapshots and re-sync live quotes? This will clear locally cached data.");
      if (!confirmed) return;
    }
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

      if ((e.key === "d" || e.key === "D") && e.altKey) {
        e.preventDefault();
        const nextRole = activeRole === "DAY_TRADER" ? "LONG_TERM" : "DAY_TRADER";
        handleRoleToggle(nextRole);
      } else if ((e.key === "v" || e.key === "V") && e.altKey) {
        e.preventDefault();
        const nextV = vernacularMode === "PLAIN_ENGLISH" ? "PRO_QUANT" : "PLAIN_ENGLISH";
        handleVernacularToggle(nextV);
      } else if ((e.key === "s" || e.key === "S") && e.altKey) {
        e.preventDefault();
        const radarHref = buildHubHref("radar", effectiveSymbol);
        if (pathname !== "/radar") router.push(radarHref);
      } else if ((e.key === "p" || e.key === "P") && e.altKey) {
        e.preventDefault();
        const portfolioHref = buildHubHref("portfolio", effectiveSymbol);
        if (pathname !== "/portfolio") router.push(portfolioHref);
      } else if ((e.key === "t" || e.key === "T") && e.altKey) {
        e.preventDefault();
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
      if (firstVisitTimerRef.current) {
        clearTimeout(firstVisitTimerRef.current);
        firstVisitTimerRef.current = null;
      }
      setIsOnboardingOpen(true);
    };
    const handleShortcutsEvent = () => {
      setIsShortcutsOpen(true);
    };
    const handlePrivacyEvent = () => {
      setIsPrivacyOpen(true);
    };

    window.addEventListener("finance:role-change", handleRoleEvent);
    window.addEventListener("finance:vernacular-change", handleVernacularEvent);
    window.addEventListener("open-onboarding", handleOnboardingEvent);
    window.addEventListener("open-shortcuts", handleShortcutsEvent);
    window.addEventListener("open-privacy", handlePrivacyEvent);

    // B4: Auto-show onboarding tour on first visit
    try {
      if (!localStorage.getItem("FINANCE_ONBOARDING_COMPLETED")) {
        firstVisitTimerRef.current = setTimeout(() => {
          setIsOnboardingOpen(true);
          firstVisitTimerRef.current = null;
        }, 1000);
      }
    } catch {}

    return () => {
      window.removeEventListener("finance:role-change", handleRoleEvent);
      window.removeEventListener("finance:vernacular-change", handleVernacularEvent);
      window.removeEventListener("open-onboarding", handleOnboardingEvent);
      window.removeEventListener("open-shortcuts", handleShortcutsEvent);
      window.removeEventListener("open-privacy", handlePrivacyEvent);
      if (firstVisitTimerRef.current) {
        clearTimeout(firstVisitTimerRef.current);
        firstVisitTimerRef.current = null;
      }
    };
  }, []);

  const handleOpenOnboarding = () => {
    if (firstVisitTimerRef.current) {
      clearTimeout(firstVisitTimerRef.current);
      firstVisitTimerRef.current = null;
    }
    setIsOnboardingOpen(true);
  };

  const handleCloseOnboarding = () => {
    if (firstVisitTimerRef.current) {
      clearTimeout(firstVisitTimerRef.current);
      firstVisitTimerRef.current = null;
    }
    try {
      localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
    } catch {}
    setIsOnboardingOpen(false);
  };

  return (
    <>
      <div className="sticky top-0 z-50 pt-[env(safe-area-inset-top,0px)] bg-[#0c1017]">
        <header
          role="banner"
          data-testid="navbar"
          className="border-b border-[#243044] bg-[#0c1017]/95 backdrop-blur h-12 flex items-center overflow-x-clip max-w-full"
        >
          <div className="max-w-[1750px] mx-auto px-2 sm:px-3 xl:px-6 w-full h-14 flex items-center justify-between gap-1 xl:gap-4 min-w-0">
            {/* Left: Brand Logo & Title */}
            <div className="flex items-center space-x-1 sm:space-x-2.5 shrink-0 min-w-0">
              <Link
                href="/"
                aria-label="ARX Terminal Home"
                className="flex items-center space-x-2 group shrink-0 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none rounded-lg"
              >
                <ArxLogo size="sm" variant="badge" />
                <div className="min-w-0 hidden sm:block">
                  <span className="font-bold tracking-tight text-white font-mono text-xs sm:text-sm xl:text-base block leading-none">
                    ARX TERMINAL
                  </span>
                  <span className="text-[9px] text-cyan-400 font-mono tracking-wider uppercase hidden 2xl:block mt-0.5">
                    No-BS Market Intel
                  </span>
                </div>
              </Link>

              {/* — Canonical 4 Hubs (Journal + Performance deferred post-R1) — */}
              <nav
                aria-label="Main Navigation"
                data-testid="desktop-nav-links"
                className="hidden lg:flex items-center space-x-0.5 xl:space-x-1 font-mono text-[11px] xl:text-xs shrink min-w-0"
              >
                {CANONICAL_HUBS.map((hub) => {
                  const href = buildHubHref(hub, effectiveSymbol);
                  const active = isHubActive(hub.href, pathname);
                  return (
                    <Link
                      key={hub.id}
                      href={href}
                      onClick={() => trackFirstHubNavigation(hub.id)}
                      aria-current={active ? "page" : undefined}
                      className={`px-1.5 xl:px-2.5 2xl:px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                        active
                          ? "bg-[#1b2434] text-cyan-400 font-semibold"
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
          <div className="flex-1 min-w-0 max-w-[140px] xl:max-w-sm 2xl:max-w-md mx-1 xl:mx-2 flex items-center justify-center">
            <UniversalOmniSearch />
          </div>

          {/* Right: Theme Toggle & Trading Horizon Mode Switcher */}
          <div className="flex items-center space-x-1 shrink-0">
            {/* Theme Toggle (Always visible) */}
            <ThemeToggle />

            {/* Experience Mode Selector (Guided / Standard / Quant) - Desktop/Tablet */}
            <div className="hidden lg:flex items-center">
              <ExperienceModeToggle />
            </div>

            {/* Unified Utilities Menu (A3: Grouped by HELP, SETTINGS, SYSTEM) */}
            <div className="relative" ref={utilitiesMenuRef} onBlur={handleUtilitiesBlur}>
              <button
                ref={utilitiesMenuTriggerRef}
                id="utilities-menu-btn"
                type="button"
                onClick={() => setIsUtilitiesMenuOpen((prev) => !prev)}
                aria-expanded={isUtilitiesMenuOpen}
                aria-haspopup="menu"
                aria-controls="utilities-menu-dropdown"
                aria-label="Terminal Utilities and System Settings"
                title="Utilities & Settings"
                className={`p-2.5 rounded-xl border border-[#243044] bg-[#090d14] text-slate-300 hover:text-cyan-300 hover:bg-[#162030] active:bg-[#162030] touch-manipulation transition-colors flex items-center justify-center focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer text-xs min-h-[44px] min-w-[44px] ${
                  isUtilitiesMenuOpen ? "border-cyan-500 text-cyan-300 bg-[#162030]" : ""
                }`}
              >
                <span aria-hidden="true" className="font-mono text-sm leading-none font-bold pointer-events-none">⋯</span>
              </button>

              {/* Grouped Utilities Dropdown Menu (Portaled to document.body to escape WebKit clipping) */}
              {mounted && isUtilitiesMenuOpen && typeof document !== "undefined" && createPortal(
                <div
                  id="utilities-menu-dropdown"
                  role="menu"
                  aria-label="Terminal Utilities"
                  style={{
                    position: "fixed",
                    top: `${menuPosition.top}px`,
                    right: `${menuPosition.right}px`,
                    maxHeight: `${menuPosition.maxHeight}px`,
                  }}
                  className="z-[9999] w-64 max-w-[calc(100vw-16px)] rounded-xl border border-[#243044] bg-[#0c1017] p-2.5 shadow-2xl space-y-2 text-xs font-mono overflow-y-auto animate-fadeIn"
                >
                  {/* On smaller viewports: Terminal Experience depth switcher */}
                  <div className="lg:hidden pb-2 border-b border-slate-800 space-y-1">
                    <div className="px-1 text-xs text-slate-500 uppercase tracking-wider font-bold">
                      Terminal Experience
                    </div>
                    <div className="pt-0.5">
                      <ExperienceModeToggle />
                    </div>
                  </div>

                  {/* Section 1: HELP */}
                  <div className="space-y-1">
                    <div className="px-1 text-xs text-slate-500 uppercase tracking-wider font-bold">
                      Help & Navigation
                    </div>
                    <button
                      id="shortcuts-help-btn"
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setIsUtilitiesMenuOpen(false);
                        setIsShortcutsOpen(true);
                      }}
                      className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-slate-300 hover:bg-[#162030] hover:text-cyan-400 text-left cursor-pointer min-h-[36px] transition-colors"
                    >
                      <HelpCircle className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span>Keyboard Shortcuts</span>
                      <kbd className="ml-auto font-mono text-xs px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">?</kbd>
                    </button>
                    <button
                      id="onboarding-tour-btn"
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setIsUtilitiesMenuOpen(false);
                        handleOpenOnboarding();
                      }}
                      className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-slate-300 hover:bg-[#162030] hover:text-cyan-400 text-left cursor-pointer min-h-[36px] transition-colors"
                    >
                      <Compass className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span>Guided Onboarding Tour</span>
                    </button>
                  </div>

                  {/* Section 2: SETTINGS */}
                  <div className="border-t border-slate-800 pt-2 space-y-1">
                    <div className="px-1 text-xs text-slate-500 uppercase tracking-wider font-bold">
                      Privacy & Diagnostics
                    </div>
                    <button
                      id="privacy-settings-btn"
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setIsUtilitiesMenuOpen(false);
                        setIsPrivacyOpen(true);
                      }}
                      className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-slate-300 hover:bg-[#162030] hover:text-cyan-400 text-left cursor-pointer min-h-[36px] transition-colors"
                    >
                      <Shield className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>Privacy & Telemetry</span>
                    </button>
                  </div>

                  {/* Section 3: SYSTEM */}
                  <div className="border-t border-slate-800 pt-2 space-y-1">
                    <div className="px-1 text-xs text-slate-500 uppercase tracking-wider font-bold">
                      System & Cache
                    </div>
                    <button
                      id="purge-cache-btn"
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setIsUtilitiesMenuOpen(false);
                        handlePurgeCache();
                      }}
                      className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-slate-300 hover:bg-rose-950/40 hover:text-rose-300 text-left cursor-pointer min-h-[36px] transition-colors"
                    >
                      <RefreshCw className={`w-4 h-4 text-amber-400 shrink-0 ${isPurging ? "animate-spin text-cyan-400" : ""}`} />
                      <div className="flex flex-col">
                        <span>Purge Cache & Re-sync</span>
                        <span className="text-xs text-slate-500">Requires confirmation · Re-syncs feeds</span>
                      </div>
                    </button>
                  </div>
                </div>,
                document.body
              )}
            </div>

            {/* Trading Horizon Switcher (Always 100% visible and unclipped across all viewports) */}
            <div role="toolbar" aria-label="Trading Horizon Mode Switcher" className="hidden sm:flex bg-[#090d14] p-0.5 rounded-xl border border-[#243044] items-center shadow-inner shrink-0">
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                role="button"
                aria-pressed={activeRole === "DAY_TRADER"}
                aria-label="Switch to Day Trader mode"
                title="Day Trader Mode (Intraday Momentum & Quick Scalps)"
                className={`flex items-center space-x-1.5 px-2.5 xl:px-3 2xl:px-3.5 py-2 sm:py-1.5 min-h-[44px] sm:min-h-[38px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] motion-reduce:transform-none transition-transform duration-100 ease-out focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:outline-none cursor-pointer ${
                  activeRole === "DAY_TRADER"
                    ? "bg-amber-500 text-slate-950 shadow-md shadow-amber-950/50 font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <Zap className="w-3.5 h-3.5" aria-hidden="true" />
                <span className="font-mono tracking-tight text-xs">
                  <span className="hidden 2xl:inline">Day Trade</span>
                  <span className="hidden xl:inline 2xl:hidden">Day</span>
                </span>
              </button>

              <button
                onClick={() => handleRoleToggle("LONG_TERM")}
                role="button"
                aria-pressed={activeRole === "LONG_TERM"}
                aria-label="Switch to Long-Term Investor mode"
                title="Long-Term Mode (Value Compounding & Secular Growth)"
                className={`flex items-center space-x-1.5 px-2.5 xl:px-3 2xl:px-3.5 py-2 sm:py-1.5 min-h-[44px] sm:min-h-[38px] rounded-lg text-xs font-mono font-bold transition-all active:scale-[0.96] motion-reduce:transform-none transition-transform duration-100 ease-out focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer ${
                  activeRole === "LONG_TERM"
                    ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950/50 font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                <Landmark className="w-3.5 h-3.5" aria-hidden="true" />
                <span className="font-mono tracking-tight text-xs">
                  <span className="hidden 2xl:inline">Long Term</span>
                  <span className="hidden xl:inline 2xl:hidden">Long</span>
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
        <RefreshCw className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" /><span>Local cache purged — Live quotes re-synced!</span>
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
          const HubIcon =
            hub.id === "radar" ? Radar :
            hub.id === "analysis" ? Microscope :
            hub.id === "setups" ? Zap :
            Briefcase;

          return (
            <Link
              key={hub.id}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`flex flex-col items-center justify-center py-1 px-1.5 rounded-xl transition-colors min-w-[44px] sm:min-w-[48px] min-h-[44px] focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                active
                  ? "bg-[#1b2434] text-cyan-400 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <HubIcon className="w-4 h-4 mb-0.5 leading-none shrink-0" aria-hidden="true" />
              <span className="text-[10px] sm:text-xs tracking-tight">{hub.label}</span>
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
          {activeRole === "DAY_TRADER" ? (
            <Zap className="w-4 h-4 mb-0.5 text-amber-400 shrink-0" aria-hidden="true" />
          ) : (
            <Landmark className="w-4 h-4 mb-0.5 text-cyan-400 shrink-0" aria-hidden="true" />
          )}
          <span className="text-[10px] sm:text-xs tracking-tight">
            {activeRole === "DAY_TRADER" ? "Day" : "Long"}
          </span>
        </button>
      </nav>
    )}

      {/* Onboarding Tour Modal */}
      <OnboardingTourModal
        isOpen={isOnboardingOpen}
        onClose={handleCloseOnboarding}
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
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              e.preventDefault();
              setIsShortcutsOpen(false);
            } else if (e.key === "Tab") {
              const dialog = document.querySelector('[role="dialog"][aria-label="Keyboard Shortcuts Guide"]');
              if (!dialog) return;
              const focusables = Array.from(
                dialog.querySelectorAll<HTMLElement>('button, input, select, textarea, a[href], [tabindex="0"]')
              ).filter((el) => !el.hasAttribute("disabled") && el.tabIndex !== -1);
              if (!focusables.length) return;
              const first = focusables[0];
              const last = focusables[focusables.length - 1];
              if (e.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
                e.preventDefault();
                last.focus();
              } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault();
                first.focus();
              }
            }
          }}
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
                ref={shortcutsCloseBtnRef}
                id="shortcuts-close-x-btn"
                type="button"
                aria-label="Close keyboard shortcuts modal"
                onClick={() => setIsShortcutsOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-[#1b2537] text-sm focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
              >
                ✕
              </button>
            </div>

            <div className="space-y-2.5 text-xs">
              {[
                { key: "/", desc: "Open Universal Omni-Search & Ticker Scanner" },
                { key: "Alt+D", desc: "Toggle Day Trader ⚡ / Long Term 🏛️ Mode" },
                { key: "Alt+T", desc: "Navigate to Main Terminal Workspace" },
                { key: "Alt+S", desc: "Navigate to Screener & Pattern Radar" },
                { key: "Alt+P", desc: "Navigate to Private Portfolio Sizer" },
                { key: "Alt+V", desc: "Toggle Plain English / Pro-Quant Vernacular" },
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
                id="shortcuts-got-it-btn"
                type="button"
                onClick={() => setIsShortcutsOpen(false)}
                className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold transition-all shadow focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none"
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