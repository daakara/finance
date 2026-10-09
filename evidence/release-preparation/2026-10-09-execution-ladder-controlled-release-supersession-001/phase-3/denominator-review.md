# Denominator review — P3-I06

Authoritative Read-Only Production SQLite Denominator Reconstruction:
- DENOMINATOR_AS_OF_UTC: 2026-10-09T20:36:49Z
- PHYSICAL_ROW_COUNT: 9
- HISTORICAL_01683A3_PHYSICAL_ROW_COUNT: 4
- HISTORICAL_01683A3_CANONICAL_IDENTITY_COUNT: 3
- CURRENT_5A90B91_PHYSICAL_ROW_COUNT: 5
- CURRENT_5A90B91_CANONICAL_IDENTITY_COUNT: 3
- CUMULATIVE_CANONICAL_IDENTITY_COUNT: 6

Population Definitions & Invariants:
1. HISTORICAL_DENOMINATOR_DEFINITION: canonical-identity count for the frozen historical cohort (Release commit SHA-1 `01683a39a19f3f74720f798459cec717698e2ab2`). Evaluates 4 physical rows to 3 unique canonical plans (`HISTORICAL_DENOMINATOR = 3`).
2. CUMULATIVE_DENOMINATOR_DEFINITION: canonical-identity count for the defined cumulative production population across release commit SHA-1 `01683a39a19f3f74720f798459cec717698e2ab2` and release commit SHA-1 `5a90b918b0975151b74e936b3fbfa536b575edd7` at DENOMINATOR_AS_OF_UTC (`2026-10-09T20:36:49Z`). Evaluates 9 physical rows to 6 unique canonical plans (`CUMULATIVE_DENOMINATOR = 6`).
