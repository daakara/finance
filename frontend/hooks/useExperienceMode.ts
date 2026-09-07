"use client";

import { useEffect, useState, useCallback } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  ExperienceMode,
  DEFAULT_MODE,
  useExperienceStore,
} from "../state/experience-store";

export const STORAGE_KEY = "arx-experience-mode";
export const LEGACY_STORAGE_KEY = "FINANCE_USER_EXPERIENCE_MODE";

export const VALID_MODES = ["guided", "standard", "quant"] as const;
export type ValidModeString = (typeof VALID_MODES)[number];

export function isValidMode(value: string | null): value is ValidModeString {
  if (!value) return false;
  const lower = value.toLowerCase().trim();
  return lower === "guided" || lower === "standard" || lower === "quant";
}

export function normalizeMode(value: string | null): ValidModeString | null {
  if (!value) return null;
  const lower = value.toLowerCase().trim();
  if (lower === "guided") return "guided";
  if (lower === "standard") return "standard";
  if (lower === "quant" || lower === "advanced") return "quant";
  return null;
}

export function useExperienceMode() {
  const router = useRouter();
  const pathname = usePathname();

  const mode = useExperienceStore((s) => s.mode);
  const setMode = useExperienceStore((s) => s.setMode);

  const [isHydrated, setIsHydrated] = useState<boolean>(false);

  const updateUrl = useCallback(
    (nextMode: ValidModeString) => {
      if (typeof window === "undefined") return;
      const currentUrl = new URL(window.location.href);
      if (currentUrl.searchParams.get("mode") !== nextMode) {
        currentUrl.searchParams.set("mode", nextMode);
        router.replace(`${pathname}?${currentUrl.searchParams.toString()}`, { scroll: false });
      }
    },
    [pathname, router]
  );

  const changeMode = useCallback(
    (next: ExperienceMode) => {
      const fromMode = mode;
      const normalizedStr = next.toLowerCase() === "advanced" ? "quant" : (next.toLowerCase() as ValidModeString);
      const targetMode: ExperienceMode = normalizedStr.toUpperCase() as ExperienceMode;

      setMode(targetMode);

      try {
        localStorage.setItem(STORAGE_KEY, normalizedStr);
        localStorage.setItem(LEGACY_STORAGE_KEY, targetMode);
      } catch (e) {
        console.warn("Failed to persist experience mode to localStorage:", e);
      }

      updateUrl(normalizedStr);

      // Telemetry readiness hooks (W1.8)
      try {
        let ticker: string | null = null;
        if (typeof window !== "undefined") {
          const params = new URLSearchParams(window.location.search);
          ticker = params.get("ticker") || params.get("symbol") || null;
        }
        window.dispatchEvent(
          new CustomEvent("arx:telemetry:experience_mode_changed", {
            detail: {
              from_mode: fromMode,
              to_mode: targetMode,
              ticker,
              timestamp: Date.now(),
            },
          })
        );
        window.dispatchEvent(new Event("finance:experience-mode-changed"));
      } catch {}
    },
    [mode, setMode, updateUrl]
  );

  useEffect(() => {
    setIsHydrated(true);
    if (typeof window === "undefined") return;

    const params = new URLSearchParams(window.location.search);
    const rawUrlMode = params.get("mode");
    const normalizedUrlMode = normalizeMode(rawUrlMode);

    // 1. Priority: Valid URL parameter
    if (normalizedUrlMode) {
      const targetMode = normalizedUrlMode.toUpperCase() as ExperienceMode;
      setMode(targetMode);
      try {
        localStorage.setItem(STORAGE_KEY, normalizedUrlMode);
        localStorage.setItem(LEGACY_STORAGE_KEY, targetMode);
      } catch {}
      return;
    }

    // Edge case: URL contained an invalid mode query parameter (e.g. ?mode=foobar)
    if (rawUrlMode && !normalizedUrlMode) {
      setMode(DEFAULT_MODE);
      try {
        localStorage.setItem(STORAGE_KEY, "standard");
        localStorage.setItem(LEGACY_STORAGE_KEY, "STANDARD");
      } catch {}
      updateUrl("standard");
      return;
    }

    // 2. Priority: localStorage persistence
    try {
      const savedRaw = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY);
      const normalizedSaved = normalizeMode(savedRaw);

      if (normalizedSaved) {
        const targetMode = normalizedSaved.toUpperCase() as ExperienceMode;
        setMode(targetMode);
        updateUrl(normalizedSaved);
        return;
      } else if (savedRaw) {
        // Corrupt localStorage value found -> repair and fallback to STANDARD
        localStorage.setItem(STORAGE_KEY, "standard");
        localStorage.setItem(LEGACY_STORAGE_KEY, "STANDARD");
      }
    } catch {}

    // 3. Priority: Default mode STANDARD
    setMode(DEFAULT_MODE);
    updateUrl("standard");
  }, [setMode, updateUrl]);

  return {
    mode,
    changeMode,
    setExperienceMode: changeMode,
    experienceMode: mode,
    isHydrated,
  };
}
