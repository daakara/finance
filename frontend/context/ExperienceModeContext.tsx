"use client";

import React, { createContext, useContext } from "react";
import { ExperienceMode } from "../state/experience-store";
import { useExperienceMode as useHookExperienceMode } from "../hooks/useExperienceMode";

interface ExperienceModeContextType {
  experienceMode: ExperienceMode;
  setExperienceMode: (mode: ExperienceMode) => void;
  isHydrated?: boolean;
}

const ExperienceModeContext = createContext<ExperienceModeContextType>({
  experienceMode: "STANDARD",
  setExperienceMode: () => {},
  isHydrated: true,
});

export function ExperienceModeProvider({ children }: { children: React.ReactNode }) {
  // Bridge with URL-synchronized hook
  const { mode, changeMode, isHydrated } = useHookExperienceMode();

  return (
    <ExperienceModeContext.Provider
      value={{
        experienceMode: mode,
        setExperienceMode: changeMode,
        isHydrated,
      }}
    >
      {children}
    </ExperienceModeContext.Provider>
  );
}

export function useExperienceMode() {
  const context = useContext(ExperienceModeContext);
  return context;
}
