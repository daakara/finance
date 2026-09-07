import { create } from "zustand";

export type ExperienceMode = "GUIDED" | "STANDARD" | "QUANT";

export const DEFAULT_MODE: ExperienceMode = "STANDARD";

interface ExperienceState {
  mode: ExperienceMode;
  setMode: (mode: ExperienceMode) => void;
}

export const useExperienceStore = create<ExperienceState>((set) => ({
  mode: DEFAULT_MODE,
  setMode: (mode: ExperienceMode) => set({ mode }),
}));
