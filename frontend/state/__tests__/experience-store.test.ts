import { describe, it, expect, beforeEach } from "vitest";
import {
  useExperienceStore,
  DEFAULT_MODE,
  ExperienceMode,
} from "../experience-store";

describe("experience store (Zustand)", () => {
  beforeEach(() => {
    useExperienceStore.setState({ mode: DEFAULT_MODE });
  });

  it("defaults to STANDARD mode", () => {
    expect(useExperienceStore.getState().mode).toBe("STANDARD");
  });

  it("changes mode to QUANT", () => {
    useExperienceStore.getState().setMode("QUANT");
    expect(useExperienceStore.getState().mode).toBe("QUANT");
  });

  it("changes mode to GUIDED", () => {
    useExperienceStore.getState().setMode("GUIDED");
    expect(useExperienceStore.getState().mode).toBe("GUIDED");
  });

  it("reverts mode to STANDARD", () => {
    useExperienceStore.getState().setMode("QUANT");
    expect(useExperienceStore.getState().mode).toBe("QUANT");

    useExperienceStore.getState().setMode("STANDARD");
    expect(useExperienceStore.getState().mode).toBe("STANDARD");
  });
});
