# SMART MONEY WAVE 2A — UNIVERSE BUILD & COVERAGE CONTRACT

## 1. Executive Summary

This contract defines the deterministic construction, mathematical accounting closure, immutable manifest attestation, and coverage denominator governing the Smart Money eligible universe.

---

## 2. Mathematical Accounting Closure

Every universe build must satisfy exact arithmetic closure across four distinct population tiers:

```text
SOURCE_POPULATION =
  Total raw instruments received from authoritative upstream directory.

ELIGIBLE_POPULATION =
  Instruments satisfying all Smart Money eligibility rules (Common Stock, US Major Exchange, Active, USD).

CIK_MAPPED_POPULATION =
  Eligible instruments possessing a verified, non-ambiguous 10-digit SEC CIK.

SCANNABLE_POPULATION =
  Eligible instruments that are CIK-mapped and satisfy all scanner ingestion prerequisites.
```

### Invariant Equations
```text
SOURCE_POPULATION = ELIGIBLE_POPULATION + INELIGIBLE_POPULATION + UNRESOLVED_POPULATION
ELIGIBLE_POPULATION = CIK_MAPPED_POPULATION + UNMAPPED_ELIGIBLE_POPULATION
SCANNABLE_POPULATION <= CIK_MAPPED_POPULATION <= ELIGIBLE_POPULATION <= SOURCE_POPULATION
```

---

## 3. Authoritative Coverage Metrics

Coverage cannot be calculated without an authoritative denominator:

```text
COVERAGE_DENOMINATOR =
  ELIGIBLE_POPULATION

CIK_MAPPING_COVERAGE =
  CIK_MAPPED_POPULATION / ELIGIBLE_POPULATION

SCANNABLE_COVERAGE =
  SCANNABLE_POPULATION / ELIGIBLE_POPULATION
```

If `ELIGIBLE_POPULATION` authority is not established or equals 0:
```text
SMART_MONEY_COVERAGE =
  NOT_ESTABLISHED
```

---

## 4. Immutable Build Manifest (`SmartMoneyUniverseBuildManifest`)

Every build produces an immutable manifest sealed with cryptographic digests:

```json
{
  "universe_build_id": "UB_SM_20261010_143000_a1b2c3",
  "universe_id": "SMART_MONEY_US_EQUITIES",
  "source_authority": "SEC_EDGAR_COMPANY_TICKERS_EXCHANGE",
  "source_snapshot_timestamp": "2026-10-10T00:00:00Z",
  "source_snapshot_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "eligibility_policy_version": "sm-eligibility-v1.0",
  "mapping_policy_version": "cik-mapping-v1.0",
  "normalization_version": "norm-v1.0",
  "source_population_count": 10435,
  "eligible_count": 4820,
  "ineligible_count": 5615,
  "cik_mapped_count": 4820,
  "scannable_count": 4820,
  "member_identity_hash": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "exclusion_summary": {
    "INELIGIBLE_EXCHANGE_OTC": 2542,
    "INELIGIBLE_SECURITY_TYPE_ADR_OR_FUND": 3073
  },
  "generated_at": "2026-10-10T14:30:00Z"
}
```

### Deterministic Member Identity Hash
The `member_identity_hash` is computed by:
1. Sorting all scannable member records alphabetically by their canonical identifier (`symbol`).
2. Creating a normalized list of `"{symbol}:{cik}:{exchange}"`.
3. Computing the SHA-256 hex digest over the JSON serialization of that list.
4. Independent of database iteration order, thread scheduling, or memory layout.

---

## 5. Universe Build Binding in Scanner Contract

```text
FORM4_SCANNER_REQUIRES_UNIVERSE_BUILD_MARKET_WIDE =
  YES

SNAPSHOT_WITHOUT_UNIVERSE_BUILD_IN_MARKET_WIDE_MODE =
  PROHIBITED
```

In `MARKET_WIDE` mode, `SmartMoneyForm4Scanner` cannot accept a raw symbol list. It strictly requires a validated `universe_build_id`. It loads the scannable membership from that build and embeds the `universe_build_id` and `universe_content_hash` directly into the immutable snapshot envelope.
