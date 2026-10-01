import { describe, it, expect, beforeEach } from "vitest";
import {
  classifyCandidateRecord,
  ingestCandidate,
  createAccountingLedger,
  EpochContract,
  RawCandidateRecord,
  AccountingLedger,
} from "../etfDenominatorEngine";

describe("Unified P1 Denominator Decision Engine (Section 9)", () => {
  const epoch: EpochContract = {
    epoch_id: "EPOCH_P1_PROSPECTIVE_01",
    epoch_start_utc: "2026-10-01T00:00:00Z",
    epoch_end_utc: "2026-10-31T23:59:59Z",
    authorized_releases: ["72914a58757790aec42edc0995ea43a40dd06105"],
    permitted_deployments: ["https://www.arxterminal.com", "https://finance-xp8.pages.dev"],
  };

  let ledger: AccountingLedger;

  beforeEach(() => {
    ledger = createAccountingLedger();
  });

  const baseNaturalCandidate: RawCandidateRecord = {
    event_id: "EVT_1001",
    observation_unit_id: "OBS_1001",
    session_id: "SES_9999",
    timestamp: "2026-10-01T04:00:00Z",
    symbol: "SPY",
    normalized_symbol: "SPY",
    environment: "production",
    deployment_identity: "https://www.arxterminal.com",
    release_sha: "72914a58757790aec42edc0995ea43a40dd06105",
    intent_boundary: "ETF_COCKPIT_INTENT",
    source_component: "OmniSearch",
  };

  it("R01: Positively classifies clean natural production candidate as VALID", () => {
    const res = ingestCandidate(baseNaturalCandidate, epoch, ledger);
    expect(res.classification_state).toBe("VALID");
    expect(res.traffic_class).toBe("NATURAL_PRODUCTION");
    expect(res.is_denominator_member).toBe(true);
    expect(ledger.prospective_denominator).toBe(1);
    expect(ledger.valid_observations).toBe(1);
  });

  it("R02: Structural invalidity defeats all lower classifications (precedence INVALID > ALL)", () => {
    const malformedWithCi: RawCandidateRecord = {
      ...baseNaturalCandidate,
      malformed_structure: true,
      ci_marker: true,
    };
    const res = ingestCandidate(malformedWithCi, epoch, ledger);
    expect(res.classification_state).toBe("INVALID");
    expect(res.is_denominator_member).toBe(false);
    expect(ledger.invalid_observations).toBe(1);
    expect(ledger.prospective_denominator).toBe(0);
  });

  it("R03: Missing release identity produces QUARANTINED (precedence QUARANTINED > EXCLUDED)", () => {
    const missingReleaseWithSynthetic: RawCandidateRecord = {
      ...baseNaturalCandidate,
      release_sha: "",
      synthetic_marker: true,
    };
    const res = ingestCandidate(missingReleaseWithSynthetic, epoch, ledger);
    expect(res.classification_state).toBe("QUARANTINED");
    expect(res.quarantine_reason).toBe("RELEASE_UNATTRIBUTABLE");
    expect(res.is_denominator_member).toBe(false);
    expect(ledger.quarantined_observations).toBe(1);
  });

  it("R04: Deterministic synthetic test markers produce EXCLUDED with specific code", () => {
    const e2eCandidate: RawCandidateRecord = {
      ...baseNaturalCandidate,
      event_id: "EVT_SYNTH_01",
      synthetic_marker: true,
    };
    const res = ingestCandidate(e2eCandidate, epoch, ledger);
    expect(res.classification_state).toBe("EXCLUDED");
    expect(res.traffic_class).toBe("SYNTHETIC_E2E");
    expect(res.exclusion_code).toBe("SYNTHETIC_E2E");
    expect(res.is_denominator_member).toBe(false);
  });

  it("R05: CI probe markers produce EXCLUDED / CI", () => {
    const ciCandidate: RawCandidateRecord = {
      ...baseNaturalCandidate,
      event_id: "EVT_CI_01",
      ci_marker: true,
    };
    const res = ingestCandidate(ciCandidate, epoch, ledger);
    expect(res.classification_state).toBe("EXCLUDED");
    expect(res.traffic_class).toBe("CI");
    expect(res.exclusion_code).toBe("CI");
  });

  it("R06: Unauthorized release SHA produces EXCLUDED / WRONG_RELEASE", () => {
    const wrongRelease: RawCandidateRecord = {
      ...baseNaturalCandidate,
      release_sha: "3590f7f0ba4f7e20869ea48c1b353a105c080d4d",
    };
    const res = ingestCandidate(wrongRelease, epoch, ledger);
    expect(res.classification_state).toBe("EXCLUDED");
    expect(res.traffic_class).toBe("NON_AUTHORIZED_RELEASE");
    expect(res.exclusion_code).toBe("WRONG_RELEASE");
  });

  it("R07: Timestamp outside epoch produces EXCLUDED / OUTSIDE_EPOCH", () => {
    const oldCandidate: RawCandidateRecord = {
      ...baseNaturalCandidate,
      timestamp: "2026-09-30T23:59:59Z", // 1 second before epoch start
    };
    const res = ingestCandidate(oldCandidate, epoch, ledger);
    expect(res.classification_state).toBe("EXCLUDED");
    expect(res.exclusion_code).toBe("OUTSIDE_EPOCH");
  });

  it("R08: Non-ETF intent attempt produces EXCLUDED / NON_ETF_INTENT", () => {
    const stockCandidate: RawCandidateRecord = {
      ...baseNaturalCandidate,
      symbol: "AAPL",
      normalized_symbol: "AAPL",
      intent_boundary: "STOCK_INTENT",
    };
    const res = ingestCandidate(stockCandidate, epoch, ledger);
    expect(res.classification_state).toBe("EXCLUDED");
    expect(res.exclusion_code).toBe("NON_ETF_INTENT");
  });

  it("R09: Deterministic duplicates are EXCLUDED / DUPLICATE", () => {
    // Ingest first valid observation
    ingestCandidate(baseNaturalCandidate, epoch, ledger);
    expect(ledger.prospective_denominator).toBe(1);

    // Ingest exact duplicate telemetry for the same attempt
    const dupRes = ingestCandidate(baseNaturalCandidate, epoch, ledger);
    expect(dupRes.classification_state).toBe("EXCLUDED");
    expect(dupRes.exclusion_code).toBe("DUPLICATE");
    // Denominator remains 1!
    expect(ledger.prospective_denominator).toBe(1);
    expect(ledger.excluded_observations).toBe(1);
  });

  it("R10: Multiple distinct ETF intent attempts in same session are both counted", () => {
    // Attempt 1: SPY
    const attempt1: RawCandidateRecord = {
      ...baseNaturalCandidate,
      event_id: "EVT_SPY_01",
      symbol: "SPY",
      normalized_symbol: "SPY",
    };
    const res1 = ingestCandidate(attempt1, epoch, ledger);
    expect(res1.classification_state).toBe("VALID");

    // Attempt 2: QQQ in same session
    const attempt2: RawCandidateRecord = {
      ...baseNaturalCandidate,
      event_id: "EVT_QQQ_02",
      symbol: "QQQ",
      normalized_symbol: "QQQ",
    };
    const res2 = ingestCandidate(attempt2, epoch, ledger);
    expect(res2.classification_state).toBe("VALID");

    // Both distinct ETF attempts are counted
    expect(ledger.prospective_denominator).toBe(2);
  });

  it("R11: Section 9.17 Failure-Path Invariant: Downstream failures DO NOT remove VALID records from denominator", () => {
    const failedRenderCandidate: RawCandidateRecord = {
      ...baseNaturalCandidate,
      event_id: "EVT_FAIL_RENDER_01",
      downstream_routing_success: true,
      downstream_render_success: false,
      cost_of_ownership_available: false,
      downstream_error: "Cost of Ownership feed unreachable",
    };
    const res = ingestCandidate(failedRenderCandidate, epoch, ledger);

    // Initial ETF intent remains VALID observation in prospective denominator
    expect(res.classification_state).toBe("VALID");
    expect(res.is_denominator_member).toBe(true);
    expect(res.downstream_outcome?.render_success).toBe(false);
    expect(ledger.prospective_denominator).toBe(1);
  });

  it("R12: Section 9.18 Conservation Law: Total raw events exactly equal sum of classified partitions", () => {
    // Ingest diverse mix
    ingestCandidate(baseNaturalCandidate, epoch, ledger); // VALID (1)
    ingestCandidate({ ...baseNaturalCandidate, malformed_structure: true }, epoch, ledger); // INVALID (1)
    ingestCandidate({ ...baseNaturalCandidate, release_sha: "missing" }, epoch, ledger); // QUARANTINED (1)
    ingestCandidate({ ...baseNaturalCandidate, ci_marker: true }, epoch, ledger); // EXCLUDED (1)
    ingestCandidate({ ...baseNaturalCandidate, synthetic_marker: true }, epoch, ledger); // EXCLUDED (2)

    expect(ledger.raw_events_count).toBe(5);
    const sumPartitions =
      ledger.valid_events_count +
      ledger.quarantined_events_count +
      ledger.excluded_events_count +
      ledger.invalid_events_count;

    expect(ledger.raw_events_count).toBe(sumPartitions);
    expect(ledger.prospective_denominator).toBe(1);
  });

  it("R13: Section 9.19 Required Audit Fields: All 16 fields present without undefined or fabricated placeholders", () => {
    const res = ingestCandidate(baseNaturalCandidate, epoch, ledger);
    const requiredFields = [
      "event_id",
      "observation_unit_id",
      "session_id",
      "timestamp",
      "normalized_symbol",
      "environment",
      "deployment_identity",
      "release_sha",
      "traffic_class",
      "classification_state",
      "classification_reason",
      "exclusion_code",
      "quarantine_reason",
      "deduplication_key",
      "replay_identity",
      "source_component",
    ] as const;

    for (const field of requiredFields) {
      expect(res[field]).toBeDefined();
      expect(typeof res[field]).toBe("string");
      expect(res[field].length).toBeGreaterThan(0);
    }
  });
});
