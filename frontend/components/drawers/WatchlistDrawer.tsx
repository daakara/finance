"use client";

import React, { useEffect, useRef } from "react";
import { useUIStore } from "../../state/ui-store";
import WatchlistDrawerContent from "./WatchlistDrawerContent";
import WatchlistDrawerHotkeys from "./WatchlistDrawerHotkeys";

export interface WatchlistDrawerProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
  liveCurrentPrice?: number;
  livePriceChangePct?: number;
  isOpen?: boolean; // Optional prop override
  onClose?: () => void;
  className?: string;
}

export default function WatchlistDrawer({
  activeSymbol,
  onSelectSymbol,
  liveCurrentPrice,
  livePriceChangePct,
  isOpen: propIsOpen,
  onClose: propOnClose,
  className = "",
}: WatchlistDrawerProps) {
  const storeIsOpen = useUIStore((state) => state.watchlistOpen);
  const storeClose = useUIStore((state) => state.closeWatchlist);

  const isOpen = propIsOpen !== undefined ? propIsOpen : storeIsOpen;
  const handleClose = propOnClose || storeClose;

  const drawerRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  // Focus trap and accessibility management
  useEffect(() => {
    if (isOpen) {
      previousActiveElement.current = document.activeElement as HTMLElement;

      // Focus the drawer or search input inside on open
      const searchInput = drawerRef.current?.querySelector<HTMLInputElement>(
        '[data-testid="watchlist-search-input"]'
      );
      if (searchInput) {
        // Small delay to allow CSS transition
        setTimeout(() => searchInput.focus(), 50);
      } else {
        drawerRef.current?.focus();
      }

      // Trap focus inside drawer
      const handleTabKey = (e: KeyboardEvent) => {
        if (e.key !== "Tab" || !drawerRef.current) return;

        const focusable = drawerRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;

        const firstElement = focusable[0];
        const lastElement = focusable[focusable.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      };

      window.addEventListener("keydown", handleTabKey);
      return () => {
        window.removeEventListener("keydown", handleTabKey);
        // Restore focus on close
        if (previousActiveElement.current && document.contains(previousActiveElement.current)) {
          previousActiveElement.current.focus();
        }
      };
    }
  }, [isOpen]);

  return (
    <>
      {/* Global Hotkeys listener ([ and Ctrl+B to toggle, Esc to close) */}
      <WatchlistDrawerHotkeys />

      {/* Backdrop Overlay (Zero chart remounting: overlay sits in fixed layer) */}
      <div
        data-testid="watchlist-drawer-backdrop"
        onClick={handleClose}
        aria-hidden="true"
        className={`fixed inset-0 bg-black/60 backdrop-blur-sm z-40 transition-opacity duration-200 ${
          isOpen
            ? "opacity-100 pointer-events-auto"
            : "opacity-0 pointer-events-none"
        }`}
      />

      {/* Slide-Over Drawer Sheet */}
      <aside
        ref={drawerRef}
        id="watchlist-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Watchlist Drawer"
        data-testid="watchlist-drawer"
        tabIndex={-1}
        className={`fixed inset-y-0 left-0 z-50 flex flex-col bg-[#0a0e17] border-r border-[#1e293b] shadow-2xl transition-transform duration-200 ease-out ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } w-full max-w-sm sm:max-w-md md:w-[280px] lg:w-80 ${className}`}
      >
        {/* Drawer Header */}
        <div className="h-14 px-4 border-b border-[#1e293b] bg-[#0c1017] flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <svg
              className="w-4 h-4 text-emerald-400"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
            </svg>
            <h2 className="text-sm font-bold font-mono text-slate-100 uppercase tracking-wide">
              Watchlist
            </h2>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              [
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={handleClose}
              aria-label="Close Watchlist Drawer"
              data-testid="watchlist-drawer-close"
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
            >
              <svg
                className="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Drawer Content */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <WatchlistDrawerContent
            activeSymbol={activeSymbol}
            onSelectSymbol={onSelectSymbol}
            liveCurrentPrice={liveCurrentPrice}
            livePriceChangePct={livePriceChangePct}
            onClose={handleClose}
          />
        </div>
      </aside>
    </>
  );
}
