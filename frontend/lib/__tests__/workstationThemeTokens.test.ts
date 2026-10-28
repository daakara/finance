import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const root = path.resolve(__dirname, "../..");
const components = [
  "AdaptiveTerminal.tsx",
  "PreFlightReadinessBar.tsx",
  "WhyInspectModal.tsx",
  "PositionSizerModal.tsx",
  "PreFlightChecklistModal.tsx",
  "AlertTriggerModal.tsx",
  "terminal/StandardTerminalView.tsx",
  "terminal/GuidedTerminalView.tsx",
  "terminal/AdvancedTerminalView.tsx",
  "terminal/SharedAssetDeck.tsx",
  "terminal/SharedExecutionLadder.tsx",
];

function luminance(hex: string) {
  const channels = hex.match(/[a-f\d]{2}/gi)!.map((channel) => {
    const value = parseInt(channel, 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
}

describe("workstation theme contrast contract", () => {
  for (const component of components) {
    it(`${component} cannot reintroduce neutral hardcoded wells`, () => {
      const source = fs.readFileSync(path.join(root, "components", component), "utf8");
      const literals = [...source.matchAll(/bg-\[#([a-f\d]{6})\]/gi)].map((match) => match[1]);
      expect(literals).toEqual([]);
      expect(source).toContain("var(--surface-");
    });
  }

  const css = fs.readFileSync(path.join(root, "styles/synthesis-e-v3-tokens.css"), "utf8");
  for (const theme of [":root", '[data-theme="paper"]']) {
    const block = css.slice(css.indexOf(theme)).split("}")[0];
    const tokens = Object.fromEntries([...block.matchAll(/(--[\w-]+):\s*(#[a-f\d]{6})/gi)]
      .map((match) => [match[1], match[2]]));
    for (const surface of ["--surface-1", "--surface-2"]) {
      for (const text of ["--text-1", "--text-2", "--text-muted"]) {
        it(`${theme}: ${text} on ${surface} meets AA normal text`, () => {
          const a = luminance(tokens[text]);
          const b = luminance(tokens[surface]);
          expect((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)).toBeGreaterThanOrEqual(4.5);
        });
      }
    }
  }
});
