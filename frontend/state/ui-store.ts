import { create } from "zustand";
import { trackTelemetryEvent } from "../telemetry/tracker";

const STORAGE_KEY = "arx-watchlist-open";

export interface UIStore {
  watchlistOpen: boolean;
  openedAtMs: number | null;
  openWatchlist: (source?: "keyboard" | "mouse" | "touch") => void;
  closeWatchlist: () => void;
  toggleWatchlist: (source?: "keyboard" | "mouse" | "touch") => void;
}

// Initial state reading from localStorage when available on client
function getInitialWatchlistOpen(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "true";
  } catch {
    return false;
  }
}

export const useUIStore = create<UIStore>((set, get) => ({
  watchlistOpen: getInitialWatchlistOpen(),
  openedAtMs: null,

  openWatchlist: (source = "mouse") => {
    const now = typeof window !== "undefined" && window.performance ? window.performance.now() : Date.now();
    try {
      localStorage.setItem(STORAGE_KEY, "true");
    } catch {}

    trackTelemetryEvent("NAVIGATION", "watchlist_drawer_opened", { source });

    set({
      watchlistOpen: true,
      openedAtMs: now,
    });
  },

  closeWatchlist: () => {
    const { openedAtMs } = get();
    const now = typeof window !== "undefined" && window.performance ? window.performance.now() : Date.now();
    const durationMs = openedAtMs ? Math.round(now - openedAtMs) : 0;

    try {
      localStorage.setItem(STORAGE_KEY, "false");
    } catch {}

    trackTelemetryEvent("NAVIGATION", "watchlist_drawer_closed", { durationMs });

    set({
      watchlistOpen: false,
      openedAtMs: null,
    });
  },

  toggleWatchlist: (source = "keyboard") => {
    const { watchlistOpen, openWatchlist, closeWatchlist } = get();
    if (watchlistOpen) {
      closeWatchlist();
    } else {
      openWatchlist(source);
    }
  },
}));
