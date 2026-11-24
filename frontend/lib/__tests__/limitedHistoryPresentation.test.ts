import { describe, it, expect } from "vitest";
import {
  resolveListingIdentity,
  resolveSecFilingCoverage,
  formatLimitedHistoryExplanation,
  resolveLimitedHistoryCapabilities,
  type LimitedHistoryContext,
} from "../limitedHistoryPresentation";

describe("limitedHistoryPresentation - Decoupled Listing Identity and SEC Filing Evidence", () => {
  describe("resolveListingIdentity", () => {
    it("returns 'Verified' when supported by Security Master evidence (ACTIVE + VERIFIED)", () => {
      const context: LimitedHistoryContext = {
        symbol: "ANT",
        instrument: {
          symbol: "ANT",
          listing_status: "ACTIVE",
          classification_status: "VERIFIED",
          primary_exchange: "NASDAQ",
          security_type: "COMMON_STOCK",
        },
      };
      const result = resolveListingIdentity(context);
      expect(result.status).toBe("Verified");
      expect(result.description).toContain("Security listing verified in Security Master");
      expect(result.description).toContain("NASDAQ");
    });

    it("returns 'Verified' when error message indicates a recognized exchange asset", () => {
      const context: LimitedHistoryContext = {
        symbol: "XYZ",
        error: "Asset \"XYZ\" is a recognized NASDAQ security, but has insufficient daily trading sessions.",
      };
      const result = resolveListingIdentity(context);
      expect(result.status).toBe("Verified");
    });

    it("returns 'Not verified' when instrument is unverified or missing", () => {
      const context: LimitedHistoryContext = {
        symbol: "UNKNOWN",
        instrument: {
          symbol: "UNKNOWN",
          listing_status: "UNVERIFIED",
          classification_status: "UNVERIFIED",
        },
      };
      const result = resolveListingIdentity(context);
      expect(result.status).toBe("Not verified");
      expect(result.description).toContain("unavailable or unconfirmed");
    });
  });

  describe("resolveSecFilingCoverage", () => {
    it("strictly defaults to 'Not verified' when coverage metadata is missing", () => {
      const context: LimitedHistoryContext = {
        symbol: "ANT",
        instrument: {
          listing_status: "ACTIVE",
          classification_status: "VERIFIED",
        },
        coverageCertified: null,
        coverageStatus: undefined,
      };
      const result = resolveSecFilingCoverage(context);
      expect(result.status).toBe("Not verified");
      expect(result.description).toContain("No certified Form 4 or ownership filing history");
      expect(result.description).toContain("listing does not imply filing completeness");
    });

    it("returns 'PARTIAL / UNCERTIFIED' when coverage is explicitly partial or uncertified", () => {
      const context: LimitedHistoryContext = {
        symbol: "PARTIAL_SYM",
        coverageCertified: false,
        coverageStatus: "PARTIAL",
      };
      const result = resolveSecFilingCoverage(context);
      expect(result.status).toBe("PARTIAL / UNCERTIFIED");
      expect(result.description).toContain("Partial historical filing ingestion");
    });

    it("returns 'Certified' only when authoritative coverage is certified or complete", () => {
      const context: LimitedHistoryContext = {
        symbol: "VERIFIED_SYM",
        coverageCertified: true,
        coverageStatus: "COMPLETE",
      };
      const result = resolveSecFilingCoverage(context);
      expect(result.status).toBe("Certified");
      expect(result.description).toContain("Historical SEC Form 4 coverage certified");
    });

    it("invariant: an active listing NEVER implies filing completeness or ownership verification", () => {
      const contextWithActiveListing: LimitedHistoryContext = {
        symbol: "NEW_LISTING",
        instrument: {
          listing_status: "ACTIVE",
          classification_status: "VERIFIED",
          primary_exchange: "NYSE",
        },
        coverageCertified: undefined,
        coverageStatus: undefined,
      };
      const listing = resolveListingIdentity(contextWithActiveListing);
      const filings = resolveSecFilingCoverage(contextWithActiveListing);

      expect(listing.status).toBe("Verified");
      expect(filings.status).toBe("Not verified");
    });
  });

  describe("formatLimitedHistoryExplanation", () => {
    it("clarifies that 15 daily sessions is the baseline modeling gate and does not promise automatic activation", () => {
      const explanation = formatLimitedHistoryExplanation(
        "ANT",
        "Asset ANT has only 3 daily sessions on record (observed 2026-11-19 to 2026-11-23). Minimum 15 daily sessions required"
      );
      expect(explanation).toContain("baseline modeling gate before quantitative evaluation begins");
      expect(explanation).toContain("Individual indicators require longer histories");
      expect(explanation).toContain("50 SMA, 200 SMA");
      expect(explanation).toContain("do not activate automatically at 15 sessions");
    });
  });

  describe("resolveLimitedHistoryCapabilities", () => {
    it("verified listing + unknown filings renders Listing Identity: Verified and SEC Filing Coverage: Not verified", () => {
      const context: LimitedHistoryContext = {
        symbol: "ANT",
        error: "Asset \"ANT\" is a recognized NASDAQ COMMON_STOCK, but has insufficient daily sessions.",
        instrument: {
          symbol: "ANT",
          listing_status: "ACTIVE",
          classification_status: "VERIFIED",
          primary_exchange: "NASDAQ",
        },
      };

      const caps = resolveLimitedHistoryCapabilities(context);
      const capMap = new Map(caps.map((c) => [c.id, c]));

      // 1. Listing Identity: Verified
      const listing = capMap.get("listing");
      expect(listing).toBeDefined();
      expect(listing?.name).toBe("Listing Identity");
      expect(listing?.status).toBe("Verified");

      // 2. SEC Filing Coverage: Not verified
      const filings = capMap.get("filings");
      expect(filings).toBeDefined();
      expect(filings?.name).toBe("SEC Filing Coverage");
      expect(filings?.status).toBe("Not verified");

      // 3. Multi-Factor Confluence: UNAVAILABLE
      const confluence = capMap.get("confluence");
      expect(confluence?.status).toBe("UNAVAILABLE");
      expect(confluence?.requirement).toContain("minimum modeling gate");

      // 4. Momentum & RSI: UNAVAILABLE
      const momentum = capMap.get("momentum");
      expect(momentum?.status).toBe("UNAVAILABLE");
      expect(momentum?.requirement).toContain("14-period Wilder smoothing");

      // 5. Cornish-Fisher VaR: UNAVAILABLE
      const varCap = capMap.get("var");
      expect(varCap?.status).toBe("UNAVAILABLE");

      // 6. Minervini Trend & VCP: UNAVAILABLE
      const vcp = capMap.get("vcp");
      expect(vcp?.status).toBe("UNAVAILABLE");

      // 7. Execution Clearance: INELIGIBLE
      const execution = capMap.get("execution");
      expect(execution?.status).toBe("INELIGIBLE");
    });

    it("applies generically to any arbitrary ticker without company-specific exceptions", () => {
      const genericSymbols = ["ACME", "FOOBAR", "NEWIPO", "ANT"];
      for (const sym of genericSymbols) {
        const caps = resolveLimitedHistoryCapabilities({
          symbol: sym,
          instrument: {
            symbol: sym,
            listing_status: "ACTIVE",
            classification_status: "VERIFIED",
          },
        });
        const listing = caps.find((c) => c.id === "listing");
        const filings = caps.find((c) => c.id === "filings");
        expect(listing?.status).toBe("Verified");
        expect(filings?.status).toBe("Not verified");
      }
    });
  });
});
