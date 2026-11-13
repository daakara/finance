"use client";

import { useState, useEffect, useCallback } from "react";

export type VernacularMode = "PLAIN_ENGLISH" | "PRO_QUANT";

export const VERNACULAR_STORAGE_KEY = "ARX_VERNACULAR_MODE";
export const VERNACULAR_EVENT_NAME = "finance:vernacular-change";

/**
 * Global synchronized hook for managing plain-English vs professional-quant terminology.
 * Defaults to "PLAIN_ENGLISH" for all new visitors.
 * Persists user preference in localStorage and synchronizes across all components in the window.
 */
export function useVernacularMode(): {
  mode: VernacularMode;
  isPlain: boolean;
  isQuant: boolean;
  toggleMode: () => void;
  setMode: (mode: VernacularMode) => void;
} {
  const [mode, setModeState] = useState<VernacularMode>("PLAIN_ENGLISH");

  useEffect(() => {
    // Read initial preference on client mount
    try {
      const stored = localStorage.getItem(VERNACULAR_STORAGE_KEY);
      if (stored === "PLAIN_ENGLISH" || stored === "PRO_QUANT") {
        setModeState(stored);
      }
    } catch {
      // Graceful fallback to default PLAIN_ENGLISH
    }

    // Synchronize across custom events in the same window
    const handleCustomEvent = (e: Event) => {
      const custom = e as CustomEvent<VernacularMode>;
      if (custom.detail === "PLAIN_ENGLISH" || custom.detail === "PRO_QUANT") {
        setModeState(custom.detail);
      }
    };

    // Synchronize across browser tabs
    const handleStorage = (e: StorageEvent) => {
      if (e.key === VERNACULAR_STORAGE_KEY && (e.newValue === "PLAIN_ENGLISH" || e.newValue === "PRO_QUANT")) {
        setModeState(e.newValue);
      }
    };

    window.addEventListener(VERNACULAR_EVENT_NAME, handleCustomEvent);
    window.addEventListener("storage", handleStorage);

    return () => {
      window.removeEventListener(VERNACULAR_EVENT_NAME, handleCustomEvent);
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  const setMode = useCallback((newMode: VernacularMode) => {
    setModeState(newMode);
    try {
      localStorage.setItem(VERNACULAR_STORAGE_KEY, newMode);
    } catch {}
    window.dispatchEvent(new CustomEvent(VERNACULAR_EVENT_NAME, { detail: newMode }));
  }, []);

  const toggleMode = useCallback(() => {
    const nextMode: VernacularMode = mode === "PLAIN_ENGLISH" ? "PRO_QUANT" : "PLAIN_ENGLISH";
    setMode(nextMode);
  }, [mode, setMode]);

  return {
    mode,
    isPlain: mode === "PLAIN_ENGLISH",
    isQuant: mode === "PRO_QUANT",
    toggleMode,
    setMode,
  };
}
