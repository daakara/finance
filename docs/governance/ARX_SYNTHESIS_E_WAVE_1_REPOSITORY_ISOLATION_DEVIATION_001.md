# Governance Deviation Record: ARX Synthesis E Wave 1 Repository Isolation Deviation

## 0. Gate Identity & Attestation

```ini
DEVIATION_ID = ARX_SYNTHESIS_E_WAVE_1_REPOSITORY_ISOLATION_DEVIATION_001
RECORD_DATE = 2026-10-07
CLASSIFICATION = PROCESS_GOVERNANCE_DEVIATION
TECHNICAL_IMPLEMENTATION_FAILURE = NO
GOVERNANCE_IMPACT = PROCESS_COMPLIANCE_NON_CONFORMANCE
TECHNICAL_IMPACT = NONE_DETECTED
PRODUCTION_STATE = TECHNICALLY_VERIFIED
PRODUCTION_ROLLBACK_REQUIRED = NO
```

---

## 1. Executive Summary

During the implementation of Synthesis E Wave 1 (Bounded Declutter & Redundant State Removal) on 2026-10-07, the controlling pre-flight governance directive:
```text
MUTATING_CHANGE_REQUIRES = DEDICATED_WORKTREE + DEDICATED_BRANCH
```
was omitted at the onset of mutation. Mutating code edits and test scaffolding were introduced directly within the primary working tree (`C:/Users/akara/Documents/Projects/finance`) on branch `main` prior to creating an isolated worktree and feature branch.

The technical changes themselves were verified comprehensively across unit tests, architectural boundary suites, TypeScript type checks, Next.js linting, and full headless browser smoke tests (Desktop 1440x900 and Mobile 390x844). Crucially, no commits were created on `main`, and no push to `origin/main` occurred.

This document formalizes the process deviation, records root cause forensics, documents the corrective action of transferring the verified candidate with byte-level SHA-256 parity to dedicated worktree `C:/Users/akara/Documents/Projects/finance-synthesis-e-wave1` and branch `feature/synthesis-e-wave-1`, and establishes binding fail-closed invariants for all future implementation waves.

---

## 2. Authoritative Repository Provenance & Forensic Facts

```ini
WAVE_1_STARTING_HEAD = 12e36a8e191c53876c4e5a14e56f8399f6887767
OBSERVED_WORKTREE = PRIMARY_WORKTREE (C:/Users/akara/Documents/Projects/finance)
OBSERVED_BRANCH = main

DEDICATED_WORKTREE_REQUIRED = YES
DEDICATED_WORKTREE_USED = NO
DEDICATED_BRANCH_REQUIRED = YES
DEDICATED_BRANCH_USED = NO

PRIMARY_WORKTREE_WAVE_1_COMMIT_CREATED = NO
DIRECT_PUSH_TO_MAIN = NO
```

### Forensic Narrative:
1. **Omission at First Mutation**: Wave 1 began editing `frontend/app/page.tsx`, `frontend/components/AdaptiveTerminal.tsx`, and `frontend/components/terminal/StandardTerminalView.tsx` within the primary worktree without first initializing `git worktree add ... -b feature/synthesis-e-wave-1`.
2. **Containment on Primary**: While the working tree was modified, no git commit was recorded on `main` and no remote push was initiated. The changes remained uncommitted in the local working directory.
3. **Audit Discovery**: During the Wave 1 Independent Closure Gate, isolation forensics identified `WAVE_1_REPOSITORY_ISOLATION = FAIL_PROCESS_DEVIATION` and triggered gate state `PASS_WITH_PROCESS_DEVIATION`.

---

## 3. Technical Verification & Absence of Scope Leakage

```ini
WAVE_1_TECHNICAL_VERIFICATION = PASS
WHY_SCORE_LOCATION_CHANGED = NO
WHY_SCORE_RELOCATION_REQUIRED = NO_ALREADY_IN_TARGET_LOCATION
WHY_SCORE_RELOCATION_STATUS = NOT_APPLICABLE
WHY_SCORE_HANDLER_PARITY = PASS

WAVE_1_DESKTOP_BROWSER_SMOKE = PASS
WAVE_1_MOBILE_BROWSER_SMOKE = PASS

OUT_OF_SCOPE_DIFF_LINES = 0
CROSS_WAVE_SCOPE_LEAKAGE = 0
UNEXPECTED_MODIFIED_FILES = 0
DOMAIN_LOGIC_FILES_MODIFIED = 0
TECHNICAL_IMPACT = NONE_DETECTED
PRODUCTION_ROLLBACK_REQUIRED = NO
```

### Verified Scope Invariants:
- **Declutter Execution**: Redundant header bar displaying duplicate horizon state in `AdaptiveTerminal.tsx` removed. Introductory discovery blocks (`PageIntro`, `IntentHero`, `WeeklyConfluenceSpotlight`) relocated below terminal analysis tabs in `app/page.tsx`. Compact `Demo Asset` badge added to verdict header when in demonstration mode.
- **Why Score Parity**: Why Score button was verified to already reside inside the Confluence Breakdown header of Supporting Evidence prior to Wave 1; only an explicit accessibility/testing identifier (`id="why-score-btn"`) was added. Click handler `onOpenWhy` remains 100% intact.
- **Boundary Containment**: Zero changes made to `PriceChart.tsx`, domain logic, pricing formulas, sizing engine, or routing. Zero work leaked from Waves 2–6 (no two-column grid, no ResizeObserver, no onboarding persistence, no ETF context branches).

---

## 4. Corrective Action

```ini
CORRECTIVE_ACTION = VERIFIED_WAVE_1_DIFF_TRANSFERRED_WITH_FILE_PARITY_TO_DEDICATED_WORKTREE_AND_BRANCH_PRIOR_TO_CANONICAL_COMMIT
ISOLATED_WORKTREE = C:/Users/akara/Documents/Projects/finance-synthesis-e-wave1
ISOLATED_BRANCH = feature/synthesis-e-wave-1
ISOLATED_BRANCH_BASE_SHA = 1d61c1d321bd677614fe00bf15f1bb091c4a2cb8
WAVE_1_TRANSFER_PARITY = PASS
FILE_CONTENT_MISMATCHES = 0
```

The verified files were transferred byte-for-byte to `feature/synthesis-e-wave-1` and verified to have exact SHA-256 match against primary candidates:
1. `frontend/app/page.tsx`: `6b88e7114bee36cdae5349fd33d926a1c1f804f6fd4110bb672c8b265d9b5c22`
2. `frontend/components/AdaptiveTerminal.tsx`: `bf608202fdcba97f0a23c1d26286eb230b6fcaf91c47114fe710d34dc1456fbc`
3. `frontend/components/terminal/StandardTerminalView.tsx`: `654690d0f357c781f5c3ffae03bc1f4fc500735236e0d1369b07ae2709b75145`
4. `frontend/components/__tests__/Wave1DeclutterPreservation.test.tsx`: `fbdc56971e7c8ad7d29ed929e91ed7fb67dfb5cd29d83b26070aaa41b7f568c0`

The uncommitted edits on primary `main` are strictly cleaned up only after full isolated verification and push of `feature/synthesis-e-wave-1`.

---

## 5. Binding Pre-Mutation Governance Invariants

To guarantee fail-closed enforcement across all subsequent waves (Wave 2 through Wave 6):

```text
INVARIANT-ISOLATION-01:
IF_MUTATION_REQUIRED
AND DEDICATED_WORKTREE_CONFIRMED != YES
THEN STOP_BEFORE_EDIT

INVARIANT-ISOLATION-02:
IF CURRENT_BRANCH = main
THEN MUTATION_NOT_AUTHORIZED
```

Any task instructing mutation without an existing isolated worktree must immediately initialize the worktree and switch branches before touching any file.
