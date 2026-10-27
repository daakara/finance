import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import SynthesisMobileDock from "../SynthesisMobileDock";

vi.mock("next/navigation", () => ({
  usePathname: () => "/radar",
}));

describe("SynthesisMobileDock Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("MOB-01: Asset Context & Link Normalization", () => {
    it("renders neutral destination links when no activeSymbol is provided", () => {
      render(<SynthesisMobileDock />);

      const analysisLink = screen.getByRole("link", { name: /Analysis/i });
      expect(analysisLink.getAttribute("href")).toBe("/");

      const planLink = screen.getByRole("link", { name: /Plan/i });
      expect(planLink.getAttribute("href")).toBe("/setups");
    });

    it("renders neutral destination links when activeSymbol is null or empty whitespace", () => {
      const { rerender } = render(<SynthesisMobileDock activeSymbol={null} />);

      let analysisLink = screen.getByRole("link", { name: /Analysis/i });
      expect(analysisLink.getAttribute("href")).toBe("/");

      let planLink = screen.getByRole("link", { name: /Plan/i });
      expect(planLink.getAttribute("href")).toBe("/setups");

      rerender(<SynthesisMobileDock activeSymbol="   " />);
      analysisLink = screen.getByRole("link", { name: /Analysis/i });
      expect(analysisLink.getAttribute("href")).toBe("/");
      planLink = screen.getByRole("link", { name: /Plan/i });
      expect(planLink.getAttribute("href")).toBe("/setups");
    });

    it("normalizes and preserves activeSymbol in links when provided", () => {
      render(<SynthesisMobileDock activeSymbol=" tsla " />);

      const analysisLink = screen.getByRole("link", { name: /Analysis/i });
      expect(analysisLink.getAttribute("href")).toBe("/?symbol=TSLA");

      const planLink = screen.getByRole("link", { name: /Plan/i });
      expect(planLink.getAttribute("href")).toBe("/setups?ticker=TSLA");
    });

    it("preserves URL encoding for symbols with special characters", () => {
      render(<SynthesisMobileDock activeSymbol="BRK.B" />);

      const analysisLink = screen.getByRole("link", { name: /Analysis/i });
      expect(analysisLink.getAttribute("href")).toBe("/?symbol=BRK.B");

      const planLink = screen.getByRole("link", { name: /Plan/i });
      expect(planLink.getAttribute("href")).toBe("/setups?ticker=BRK.B");
    });
  });

  describe("MOB-02: Universal Omni Search Trigger", () => {
    it("dispatches arx:open-omni-search custom event on Search button click", () => {
      const dispatchSpy = vi.spyOn(window, "dispatchEvent");
      render(<SynthesisMobileDock />);

      const searchBtn = screen.getByRole("button", { name: /Search/i });
      fireEvent.click(searchBtn);

      expect(dispatchSpy).toHaveBeenCalled();
      const event = dispatchSpy.mock.calls[0][0] as CustomEvent;
      expect(event.type).toBe("arx:open-omni-search");
    });
  });
});
