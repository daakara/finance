"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { ExperienceMode } from "../types/insight";

interface ExperienceModeContextType {
  experienceMode: ExperienceMode;
  setExperienceMode: (mode: ExperienceMode) => void;
}

const ExperienceModeContext = createContext<ExperienceModeContextType>({
  experienceMode: "STANDARD",
  setExperienceMode: () => {},
});

const STORAGE_KEY = "FINANCE_USER_EXPERIENCE_MODE";

export function ExperienceModeProvider({ children }: { children: React.ReactNode }) {
  const [experienceMode, setExperienceModeState] = useState<ExperienceMode>("STANDARD");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    try {
      const saved = localStorage.getItem(STORAGE_KEY) as ExperienceMode;
      if (saved === "GUIDED" || saved === "STANDARD" || saved === "ADVANCED") {
        setExperienceModeState(saved);
      }
    } catch (e) {
      console.warn("Failed to load experience mode from localStorage:", e);
    }

    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        const val = e.newValue as ExperienceMode;
        if (val === "GUIDED" || val === "STANDARD" || val === "ADVANCED") {
          setExperienceModeState(val);
        }
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  const setExperienceMode = (mode: ExperienceMode) => {
    setExperienceModeState(mode);
    try {
      localStorage.setItem(STORAGE_KEY, mode);
      window.dispatchEvent(new Event("finance:experience-mode-changed"));
    } catch (e) {
      console.warn("Failed to save experience mode to localStorage:", e);
    }
  };

  return (
    <ExperienceModeContext.Provider value={{ experienceMode, setExperienceMode }}>
      {children}
    </ExperienceModeContext.Provider>
  );
}

export function useExperienceMode() {
  const context = useContext(ExperienceModeContext);
  return context;
}
