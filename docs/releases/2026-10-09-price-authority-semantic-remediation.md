# ARX Terminal — Production Release Notes

## Price Authority & Execution Ladder Semantic Remediation

### Release Identity

```ini
RELEASE_DATE =
  2026-10-09
UTC_DEPLOYMENT_TIMESTAMP =
  2026-10-09T18:03:52Z
AUTHORIZED_FUNCTIONAL_RELEASE_SHA =
  d97801e783620294454d1989164c907534ed4358
PRE_REMEDIATION_HEAD =
  74af302ab2c8f584a5288d24896895a6963c1f49
PREVIOUS_ORIGIN_MAIN_SHA =
  01683a39a19f3f74720f798459cec717698e2ab2
PREVIOUS_PRODUCTION_SHA =
  01683a39a19f3f74720f798459cec717698e2ab2
FRONTEND_PRODUCTION_DEPLOYMENT_ID =
  ef378562-0fee-4529-a94f-e4d95a803cc5
FRONTEND_PRODUCTION_SHA =
  d97801e783620294454d1989164c907534ed4358
BACKEND_PRODUCTION_DEPLOYMENT_ID =
  0685134c-134c-4cce-8e0a-850299e18c34
BACKEND_PRODUCTION_SHA =
  d97801e783620294454d1989164c907534ed4358
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway & Cloudflare Pages deployment
DEPLOYMENT_STATUS =
  SUCCESS
PRODUCTION_VERIFICATION =
  VERIFIED_AT_TARGET_SHA
```

---

### Release Purpose & Classification

```ini
RELEASE_PURPOSE =
  Remediate price-authority conflation between live spot quote ($383.23) and completed session reference price ($375.00), eliminate misleading static "• Live Spot" label, introduce dedicated Setup Reference row, declare explicit target percentage baselines, eliminate false unestablished purchase price copy, and align frontend inZone actionability strictly with backend eval_price authority.

RELEASE_CLASSIFICATION =
  BUG_FIX_AND_DATA_CONTRACT_ENHANCEMENT
  SEMANTIC_INTEGRITY
  PRESENTATION_DISAMBIGUATION
  ACTIONABILITY_GUARD

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  YES / INTENTIONAL (Execution Ladder separates Live Spot benchmark from Setup Reference baseline, declares explicit percentage baselines, and rejects straddle false actionability)

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

PROSPECTIVE_CAPTURE_CHANGE =
  NONE

MODEL_PARAMETER_CHANGE =
  NONE

MODEL_TUNING =
  FROZEN

EMPIRICAL_QUALITY =
  INSUFFICIENT_EVIDENCE

LEARNING_CLAIM =
  NOT_AUTHORIZED

DATABASE_SCHEMA_CHANGE =
  NONE
```

---

### Defect Summary & Root Cause

* **Observed Production Discrepancy**:
  On TSLA analysis surfaces:
  - Surface A (PriceChart Header) rendered TSLA Live Spot as **`$383.23`** with green badge `● LIVE SPOT` alongside `Analysis Ref: $375.00 (Prior Close)`.
  - Surface B (OptimalEntryExitCard / Recommended Price Ladder) rendered `Current Spot: $375.00` and `⚡ CURRENT MARKET PRICE: $375.00 • Live Spot`, with targets `Profit Goal 1: $430.03 (+14.67%)`, `Profit Goal 2: $460.68 (+22.85%)`, and execution ratchet copy referencing `purchase price ($375.00)`.
* **Root Cause**:
  1. *Contract Conflation*: Backend `OptimalExecutionEngine` output `raw_plan["current_price"] = spot` ($375.00), representing the completed daily session close, while tracking live spot separately in `live_spot_price`.
  2. *Component Decoupling*: `OptimalEntryExitCard.tsx` only received `executionPlan={data?.optimalExecution}`, omitting `liveSpotPrice` and `marketPriceState`.
  3. *Static Mislabeling*: `OptimalEntryExitCard.tsx` hardcoded static text `• Live Spot` directly adjacent to `${current_price.toFixed(2)}` ($375.00).
  4. *Undeclared Target Baselines*: Return percentages (+14.67% and +22.85%) were measured from the reference baseline ($375.00) rather than the displayed price ($383.23), but appeared without declaring their basis.
  5. *Unestablished Purchase Copy*: Tactical ratchet copy assumed unentered prospective setups represented an executed position with cost basis $375.00.
  6. *Frontend Straddle Actionability*: Frontend evaluated `inZone` against `current_price` ($375.00) rather than live spot ($383.23).

---

### Backend Contract Changes

File: `analyst_dashboard/analyzers/optimal_execution.py`
- Added explicit semantic price attributes to `raw_plan` and `_enforce_execution_invariants`:
  - `analysis_reference_price = spot` ($375.00)
  - `analysis_reference_type = "COMPLETED_SESSION_CLOSE"`
  - `target_percentage_basis = "ANALYSIS_REFERENCE_PRICE"`
  - `target_1_pct_from_reference = 14.67`
  - `target_2_pct_from_reference = 22.85`
  - `target_1_pct_from_live = 12.21` (dynamic when live spot exists)
  - `target_2_pct_from_live = 20.21` (dynamic when live spot exists)
  - `live_spot_price = live_spot_price`
  - `eval_price = eval_price`
- Enforced synchronization: `plan["is_in_buy_zone"] = plan["execution_status"] in ACTIONABLE_EXECUTION_STATUSES`.
- Preserved `current_price = spot` for full backward compatibility as legacy analysis reference price.

File: `api/routes/analytics.py`
- Propagates `analysis_reference_as_of`, `live_spot_as_of`, `live_freshness`, and `market_session` to `optimal_execution_plan`.

---

### Frontend Price Presentation Changes

File: `frontend/lib/api.ts`
- Extended `OptimalExecutionPlan` interface with additive typed attributes.

File: `frontend/app/page.tsx`
- Passes unified price props to `OptimalEntryExitCard`:
  - `liveSpotPrice`, `analysisReferencePrice`, `marketPriceState`, `liveFreshness`, `marketSession`.
  - Both `PriceChart` and `OptimalEntryExitCard` now consume the identical market-price authority.

File: `frontend/components/OptimalEntryExitCard.tsx`
- **Benchmark Row**: Displays live market spot ($383.23) with dynamic freshness badge (`● LIVE SPOT` in realtime, `DELAYED SPOT` in delayed, `PRIOR CLOSE (REF)` in closed). Static `• Live Spot` text eliminated.
- **Setup Reference Row**: Added dedicated row preserving visibility of frozen model anchor ($375.00 Prior Close).
- **Target Returns**: Explicitly labeled `+14.67% from Setup Ref` with secondary `(+12.21% from Live)`.
- **Ratchet Copy**: Replaced false "purchase price" with conditional copy ("filled entry price upon execution").

---

### Actionability Corrections & Straddle Resolution

- Actionability and in-zone evaluation strictly consume `evalPrice` (`canonicalLiveSpot ?? eval_price ?? current_price`).
- **Straddle Boundary**: When reference is inside corridor ($372.00) but live spot is above corridor ($380.00), system resolves to `EXTENDED_ABOVE_BUY_ZONE`, `is_in_buy_zone = false`, and `market_location = BETWEEN_BASE_AND_TP1`. Reference price cannot falsely trigger entry readiness.

---

### Quantitative Invariance Proof

- Optimal Entry Corridor: `[$353.27, $373.32]` (Zero delta)
- Invalidation Stop Loss: `$342.67` (Zero delta)
- Profit Goal 1: `$430.03` (Zero delta)
- Profit Goal 2: `$460.68` (Zero delta)
- R:R Floor: $\ge 1.85:1$ (Preserved)
- Technical Indicators: Daily EMA, ATR, SMA arrays remain strictly immutable under spot changes.

---

### Test Evidence & Results

1. **Dedicated Price Authority Suite**:
   - `tests/test_price_authority_reproduction.py`: 8/8 PASSED
   - `frontend/tests/priceAuthorityLadderInvariants.test.ts`: 7/7 PASSED
2. **Core Execution & Provenance Suites**:
   - `tests/test_optimal_execution.py`: 7/7 PASSED
   - `tests/test_price_provenance_truthfulness.py`: 5/5 PASSED
3. **Frontend Architecture & Regression Suite**:
   - `npm run test:arch`: 22/22 test suites PASSED
   - `npm run type-check`: 0 errors
   - `npm run build`: 145/145 pages built cleanly

---

### Governance Isolation Statement

- `PROSPECTIVE_DENOMINATOR`: 0 (authoritatively verified in `governance.db`)
- `FIRST_NATURAL_CAPTURE`: AWAITING_VERIFICATION
- `EMPIRICAL_SCANNER_QUALITY`: INSUFFICIENT_EVIDENCE
- `MODEL_TUNING`: FROZEN
- `LEARNING_CLAIM`: NOT_AUTHORIZED
- Zero synthetic prospective records injected during testing or deployment.

---

### Rollback Procedure

If emergency rollback is required:
```bash
git revert d97801e783620294454d1989164c907534ed4358 --no-edit
git push origin main
```
Automatic deployment on Railway and Cloudflare will restore previous baseline `01683a39a19f3f74720f798459cec717698e2ab2` within 3 minutes with zero database schema migrations needed.

---

### Final Production Verification Verdict

* **Production Deployed SHA**: `d97801e783620294454d1989164c907534ed4358`
* **Frontend Production URL**: `https://www.arxterminal.com` (Status: 200 OK)
* **Backend Production URL**: `https://web-production-470560.up.railway.app` (Status: 200 OK)
* **Runtime Verification**: Railway container runtime verified startup on release SHA `d97801e783620294454d1989164c907534ed4358`.
* **VERDICT**: **PASS — RELEASE DEPLOYED AND VERIFIED**
