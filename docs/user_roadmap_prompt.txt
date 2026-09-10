<USER_REQUEST>
Perform an evidence-based gap audit against the user-provided “ARX Horizon Redesign Roadmap: From Engine Collection → Unified Intelligence Operating System,” then remediate and verify the H14 foundation.

Use that roadmap as the governing product reference. Existing Horizon 14.3 reports and differently numbered plans are implementation history, not proof of roadmap completion.

The current priorities are:
1. API-backed, authentic production data.
2. One consistent shared read model.
3. Working semantic zoom and context-preserving navigation.
4. Honest verification and reporting.

Do not start H16–H18 feature expansion or another broad visual redesign during this task.

PART A — ROADMAP GAP AUDIT

Map the implementation against:
- H14.1: Unified Read Model.
- H14.2: Semantic Zoom.
- H15.1: Four Core Hubs.
- H15.2: Command Palette.
- H16.1: Narrative Intelligence.
- H16.2: Adaptive Personalization.
- H17: Trading & Investment Experience.
- H17.2: Trader Discipline Engine.
- H18: Specialist Workbenches.
- Cross-cutting Design System v2.

Create a matrix with:
- Roadmap requirement.
- Current implementation and exact file references.
- API dependencies.
- Verification evidence.
- Status: verified, partial, missing, or untested.
- Gap and user impact.
- Priority and dependencies.
- Concrete acceptance criteria.

Distinguish “code exists” from “behavior verified.” Do not treat documentation, source-text assertions, route existence, or a successful build as proof of a working feature.

Explain naming conflicts between this roadmap and existing Horizon reports. Do not infer completion from matching phase numbers.

Identify the unresolved relationship between the four core hubs and the six trading hubs. Document whether available product evidence establishes an investor-specific experience or a replacement product direction. Do not remove hubs, rename routes, or choose a new product direction without an explicit requirement. Continue H14 work that does not depend on that decision.

PART B — VERIFY AND REMEDIATE H14.1

Goal: all views consume consistent, authoritative API projections through a shared read model.

1. Trace production data end to end
Inspect frontend pages, shared stores, selectors, API clients, backend endpoints, calculation engines, database seeds, and caches.

Inventory all displayed business fields, including:
- LHI, HHI, IAI where relevant.
- Signal quality, runway, next best action, constraints, forecasts, and drift.
- Asset identity, prices, setups, entry/stop/targets, and position sizing.
- Holdings, exposure, portfolio risk, journal metrics, performance attribution, and Governor status.

For each field record:
- Authoritative source and API response field.
- Upstream provider, saved user record, or analytical computation.
- Units, currency, timestamp, and applicable asset/user context.
- Availability and freshness semantics.
- Every screen consuming it.

2. Remove fabricated production data
No hardcoded assets, business metrics, synthetic histories, sample holdings, invented explanations, or fallback prices may populate production screens.

Explicitly inspect CANONICAL_TACTICAL_SETUPS, baseline-price catalogs, simulation-store defaults, fixed headline statistics, and first-asset selection.

Do not move fixtures into an API and call them authentic. Trace API values to legitimate upstream data or documented computations.

Static UI labels, styling, mathematical constants, and documented algorithm configuration may remain. They must not substitute for missing observations. Roadmap examples are not production values or approved risk policies.

Keep test fixtures isolated from production imports and runtime fallback paths.

3. Establish a coherent read model
Inspect UnifiedCockpitState and unifiedCockpitStore.ts before deciding what to change. Reuse sound existing architecture.

- Centralize business metric computation in authoritative backend services.
- Pages and shared frontend stores must not independently recompute business metrics.
- Frontend selectors may select, filter, sort, and format projections without redefining metrics.
- Keep overview projections compact; load detailed evidence on demand.
- Define stable identifiers, schema version, snapshot/version identity, source timestamps, and availability metadata.
- Ensure projections are scoped correctly to user/account, asset, currency, and reporting period.
- Prevent mixed snapshots from producing contradictory headline values.
- Prevent slow responses for a previous asset/account from overwriting current state.
- Propagate successful mutations or invalidate affected projections consistently.

Measure the roadmap’s <12 kB read-model target. State exactly what is measured, including uncompressed serialized size and any compressed measurement. Use representative populated payloads, not an empty example. Do not remove provenance or necessary state merely to meet the size target.

4. Handle absence and failures honestly
Distinguish loading, empty, unavailable, unsupported, stale, unauthorized, and failed states.

- Never substitute another asset.
- Never turn missing values into zero or a neutral score.
- Preserve legitimate API zeros.
- Never imply an incomplete total is complete.
- Never invent a reason why an asset lacks a setup.
- Do not silently replace failed current-data requests with cached or sample values.
- Intentionally requested historical data is valid when clearly dated.
- Disable actions whose required inputs are unavailable, stale, invalid, or mismatched.
- Preserve existing user holdings, including fractional quantities, during any persistence migration.
- Persist saved business records through the API; unsaved form input remains local form state.

If an authoritative source or endpoint is absent, implement it where supported by available providers and requirements. Otherwise render an honest unavailable state and report the dependency. Do not fabricate a replacement.

PART C — VERIFY AND REMEDIATE H14.2

Goal: every in-scope intelligence artifact supports:
Overview → Explanation → Workbench.

Inventory artifacts across the four core hubs and six trading hubs, including currently available Governor views.

For each artifact record:
- Overview location and primary user question.
- Explanation drawer/page and supporting evidence.
- Workbench destination.
- Stable artifact ID and required context.
- Availability of each level.
- Actual navigation result.

Ensure:
- The overview presents the main finding and appropriate next action.
- The explanation uses the same authoritative metric and snapshot.
- The workbench exposes corresponding evidence and analysis.
- Asset, account, period, and artifact context survive drilldown, refresh, and direct linking.
- Back navigation restores relevant source context.
- Missing deeper capabilities are explicitly unavailable rather than linked to unrelated pages.
- Merely renaming tabs Standard/Guided/Quant does not count as semantic zoom.
- Explanations make no claims unsupported by API evidence.
- Keyboard, mobile, loading, empty, and error behavior remain usable.

Reuse existing detailed tools where appropriate. Do not build the entire H18 workbench system to disguise an H14 coverage gap. Record missing dependencies explicitly.

PART D — MEANINGFUL VERIFICATION

Run targeted automated integration tests and browser/network checks. Separate mocked-contract tests from actual API checks.

H14.1 checks:
- The same metric agrees across relevant pages for the same snapshot and context.
- An API value change propagates consistently.
- No business calculations are duplicated in frontend pages or shared selectors.
- A newly API-returned asset outside the former hardcoded list works.
- Missing fields, valid zero values, malformed responses, timeouts, authorization failures, and rate limits produce honest states.
- Rapid asset/account switching cannot leak or overwrite context.
- Initial load, refresh, and API failure never reveal demo values or a default asset.
- Fractional holdings survive API save/reload and update dependent projections.
- Read-model payload size is measured.

H14.2 checks:
- Exercise each inventoried Overview → Explanation → Workbench path.
- Verify destination content and context, not just URL strings.
- Test refresh, direct links, back navigation, keyboard activation, and mobile behavior.
- Verify unavailable drilldowns and actions communicate their actual limitation.

Use isolated test accounts/data for mutations. Do not place real orders or alter real holdings during verification.

The 30-second executive-view goal requires observed usability testing. If no participant testing is performed, label it unverified. Do not claim comprehension or cognitive-load improvements from code checks.

PART E — DELIVERABLES AND COMPLETION GATES

Produce:
1. Roadmap gap matrix.
2. H14 field-level provenance and data-flow map.
3. Semantic-zoom coverage matrix.
4. Prioritized remaining backlog.
5. H14 verification report.
6. Summary of code changes, root causes, and actual test results.

Update existing certification documents to match the evidence. Reconcile obsolete assertion counts and completion claims without deleting historical context.

For H14-Gate-01, explicitly report:
- Cross-page metric consistency.
- Duplicate frontend business calculations found and remaining.
- Hardcoded/fallback production data paths found and remaining.
- Read-model size and measurement method.
- API/provider dependencies still missing.
- Semantic-zoom coverage as verified artifacts / total inventoried artifacts.
- Failed, blocked, and untested cases.

Do not claim H14 complete while required data, consistency, or drilldown coverage remains missing or unverified. Honest unavailable states satisfy failure-handling requirements but do not establish that the missing capability is complete.

End with the next recommended roadmap step based on verified gaps. Keep implementation focused on H14; report later-phase work separately.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-10T01:04:54+02:00.
</ADDITIONAL_METADATA>