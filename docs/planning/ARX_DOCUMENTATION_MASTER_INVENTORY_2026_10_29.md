# ARX TERMINAL — DOCUMENTATION MASTER INVENTORY
## Repository-Wide Documentation Audit, Provenance & Authority Classification
**Audit Date:** 29 October 2026  
**Baseline Git SHA:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Governing Directive:** `MASTER DIRECTIVE ARX-DOC-2026-10-29`  
**Authority Level:** `EVIDENCE_BOUND / CANONICAL_ACTIVE`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Summary & Classification Model

This master inventory provides a comprehensive, repository-wide accounting of all 1,478 documentation, specification, evidence, and governance artifacts across the ARX Terminal codebase. Every document is assigned to one of eight authoritative governance classifications per Section 2 of Directive `ARX-DOC-2026-10-29`:

1. `CANONICAL_ACTIVE` (38 documents): The active, authoritative sources of truth governing system architecture, production contracts, API semantics, active invariant registries, and current design guidelines.
2. `HISTORICAL_IMMUTABLE` (1,284 documents): Formal audit records, directive logs, gate execution receipts, sprint retrospectives, holdout epoch custodial evidence, and previous deployment receipts that are permanent audit artifacts and must never be mutated.
3. `ACTIVE_SUPPORTING_EVIDENCE` (86 documents): Live verification matrices, golden benchmark datasets, regression test harnesses, and candidate freeze evidence validating ongoing releases.
4. `SUPERSEDED` (42 documents): Outdated design notes, contaminated historical scoring analyses (e.g. historical `confluenceScore` 65.0 fallback contamination), and pre-PR #129 unverified localStorage sizing references.
5. `DUPLICATED` (12 documents): Legacy draft copies or mirrored sprint checklists superseded by unified backlog registers.
6. `CONTRADICTORY` (6 documents): Historical reports claiming production deployment based solely on local test passes or describing unratified research flags as live execution authorization.
7. `MISSING` (0 remaining): Key missing gap areas identified in Section 2 (traceability matrix, cross-engine reconciliation, PR status register) are formally addressed and created by this reconciliation.
8. `UNKNOWN_AUTHORITY` (10 documents): Unattributed root-level scratch notes or deprecated exploration scripts.

---

## 2. Canonical Active Documentation (`CANONICAL_ACTIVE`)

| Path | Purpose | Owner / Workstream | Scope | Last Substantive Update | Authority Level | Dependencies | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `README.md` | Primary repository portal, system overview, and architectural navigation | Product / Engineering | Full Repository | 2026-10-29 | `CANONICAL_ACTIVE` | Core docs | Update in-place to reflect Sprint 1–4 outcomes, PR #129, and engine contracts |
| `docs/ARX_MASTER_PROJECT_STATUS_2026_10_29.md` | Master integrated project state across research, UX, engines, engineering, and governance | Product Owner / Governance | Global ARX Estate | 2026-10-29 | `CANONICAL_ACTIVE` | Mainline `e10de3b`, PR #129 | Authoritative source of truth for repository state |
| `docs/SYSTEM_ARCHITECTURE_AND_IMPLEMENTATION.md` | Central technical architecture, data flow, container topologies, and fail-closed security invariants | Architecture | Full System | 2026-10-29 | `CANONICAL_ACTIVE` | Model E+, SQLite, FastAPI | Incorporate Section 5 Market-Wide Discovery ratification |
| `docs/architecture/ARX_CANONICAL_SECURITY_MASTER_DESIGN.md` | Authoritative security master specification, ticker mapping, and Model E+ security registry core | Architecture / Quant | Security Data Lineage | 2026-10-29 | `CANONICAL_ACTIVE` | Model E+, EODHD | Maintain as single source of security identity |
| `docs/architecture/ARX_ARCHITECTURE_AND_ENGINE_CONTRACT_RECONCILIATION.md` | Comprehensive 12-engine contract reconciliation, input/output data boundaries, and fallbacks | Architecture / Engines | All 12 Engines | 2026-10-29 | `CANONICAL_ACTIVE` | Engine codebases | Adopt as authoritative multi-engine specification |
| `docs/LIVE_API_ARCHITECTURE_STANDARDS.md` | API standards, envelope definitions, error formats, and response caching rules | Engineering / API | Backend / Gateway | 2026-10-28 | `CANONICAL_ACTIVE` | FastAPI, Pydantic | Cross-reference PR #130 client API standardization |
| `docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md` | Core mathematical, financial, and architectural invariants (INV-01 through INV-28) | Governance Guardian | Platform-wide | 2026-10-26 | `CANONICAL_ACTIVE` | Risk / Sizing engines | Maintain strict fail-closed enforcement |
| `docs/planning/arx_canonical_backlog.json` | Machine-readable backlog schema and task authorization state | Project Brain / Governance | Sprints 1–5 | 2026-10-29 | `CANONICAL_ACTIVE` | Schema v2.1 | Preserve `sprint5_execution_authorized = false` |
| `docs/research/ARX_MARKET_WIDE_RESEARCH_DECISION_AND_EVIDENCE_REGISTER.md` | Consolidated MW-004 research findings, Model E+ ratification, candidate freeze, and scoring reconciliation | Quant Research / Owner | Market-Wide Discovery | 2026-10-29 | `CANONICAL_ACTIVE` | Candidate `f154032`, PR #128 | Establish canonical research authority |
| `docs/ux/ARX_UX_V3_APPROVAL_IMPLEMENTATION_AND_RELEASE_STATUS.md` | Approved Synthesis E V3 UX design, mode architecture, Paper Light, and PR #131 engineering status | UX / Frontend | Synthesis E V3 UI | 2026-10-29 | `CANONICAL_ACTIVE` | PR #131, Next.js | Authoritative UX specification |
| `docs/releases/ARX_CI_RELEASE_AND_DEPLOYMENT_AUTHORITY_MATRIX.md` | Authoritative release qualification rules, GitHub rulesets, and deployment trigger permissions | Release Guardian | CI/CD / Infrastructure | 2026-10-29 | `CANONICAL_ACTIVE` | Ruleset `25187391`, Railway | Maintain strict deployment gate boundaries |
| `docs/planning/ARX_SPRINT1_4_OBLIGATION_AND_HOLD_REGISTER.md` | Deduplicated historical findings, unresolved obligations, and active platform holds | Governance / Planning | Sprints 1–4 Closure | 2026-10-29 | `CANONICAL_ACTIVE` | Sprint 1–4 Ledgers | Authoritative tracking for all open issues and holds |
| `docs/planning/ARX_PR_AND_INTEGRATION_STATUS_REGISTER.md` | Exhaustive verification of all 16 open and merged PRs, commit SHAs, and dependencies | Engineering / Release | Pull Requests | 2026-10-29 | `CANONICAL_ACTIVE` | GitHub API | Reference for all cross-PR dependencies |
| `docs/planning/ARX_DOCUMENTATION_CONFLICT_AND_CORRECTION_LOG.md` | Log of substantive contradictions across legacy docs and their authoritative resolutions | Governance / QA | Repository Documentation | 2026-10-29 | `CANONICAL_ACTIVE` | Documentation Audit | Reference for all resolved documentation ambiguities |
| `docs/planning/ARX_DOCUMENTATION_EVIDENCE_TRACEABILITY_MATRIX.csv` | Machine-readable traceability mapping from claims to source evidence and canonical docs | Governance / QA | Claims Traceability | 2026-10-29 | `CANONICAL_ACTIVE` | Full Repository | Primary evidence-to-document audit artifact |
| `docs/planning/ARX_DOCUMENTATION_QA_REPORT.md` | Independent verification results, link checks, arithmetic reconciliation, and diff enforcement | QA / Governance | Documentation Estate | 2026-10-29 | `CANONICAL_ACTIVE` | Verification Suite | Quality certification for documentation baseline |
| `docs/planning/ARX_DOCUMENTATION_OWNER_REVIEW_AND_PUBLICATION_HANDOFF.md` | Executive handoff package for Product/Research Owner formal review and publication | Governance / Owner | Executive Review | 2026-10-29 | `CANONICAL_ACTIVE` | Directive ARX-DOC-2026-10-29 | Master handoff deliverable |
| `docs/BRAND_GUIDE.md` | Brand visual identity, typographic standards, color palette, and UI token scales | Design / UX | Frontend Presentation | 2026-10-15 | `CANONICAL_ACTIVE` | Tailwind / CSS | Preserve as visual identity standard |
| `docs/scoring-semantics.md` | Mathematical semantics for confluence, technical, and fundamental scoring | Quant Research | Confluence Engine | 2026-10-20 | `CANONICAL_ACTIVE` | Score normalization | Maintain canonical mathematical definitions |

---

## 3. Historical Immutable Audit & Execution Records (`HISTORICAL_IMMUTABLE`)

Over 1,280 documents in the repository record historical milestones, sprint retrospectives, gate execution receipts, and forensic investigations. These documents are permanent audit records:

- **Sprint 1–4 Gate Receipts & Handshakes (`ARX_*_RECEIPT.json`, `ARX_*_HANDSHAKE.md`):** Capture exact commit SHAs, timestamps, CI run IDs, and deployment receipts.
- **Smart Money Governance Control Plane (`docs/domain/smart_money/governance_control_plane/*`):** 202 custodial documents recording SEC Form 4 historical ingestion experiments, held under `HOLD_SM1_001`.
- **Holdout Epoch 002 VCP Custodial Evidence (`docs/domain/vcp/holdout_epoch_002/*`):** 26 cryptographically hashed ground-truth files for out-of-sample pattern testing.
- **Candidate 006 Production Preflight Records (`docs/releases/candidate_006/*`):** 110 preflight and rehearsal records for previous release iterations.
- **Sprint 2 Package B / C Convergence Reviews (`docs/reviews/sprint2-*`):** 91 review documents capturing historical convergence, SIP ingestion, and pilot authorizations.

---

## 4. Active Supporting Evidence (`ACTIVE_SUPPORTING_EVIDENCE`)

- `docs/research/ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json`: Verified golden benchmark set for ETF denominator testing.
- `docs/research/vcp-production-universe-inputs/*`: Validated daily OHLCV reference inputs for Minervini VCP screening.
- `docs/research/multi-asset-coverage/*`: Bounded 16-asset comparison dossiers and verified source evidence.
- `docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md`: Live constraint registry enforced by CI gates.

---

## 5. Superseded & Contaminated Analyses (`SUPERSEDED`)

The documentation audit identified 42 historical analyses that contain superseded premises or falsified data that must not be cited as current truth:

1. **Scoring Fallback Contamination:** Early MW-004 research drafts that evaluated scoring based on the unpatched `confluenceScore` attribute resulting in silent 65.0 default fallbacks. Falsified by MW-004B and superseded by Model E+ candidate `f154032`.
2. **Pre-PR #129 Untrusted LocalStorage Sizing:** Historical portfolio documentation stating that risk position sizing falls back to localStorage when server telemetry is absent. Superseded by PR #129 verified equity provenance isolation.
3. **Draft PR #128 Implementation Assumptions:** Previous documentation describing PR #128 as the complete corrective candidate. Superseded by research candidate `f154032` (PR #128 remains an open unmerged reference).
4. **Direct Cloudflare Pages Live API Proxies:** Architecture drafts assuming frontend routes query external broker APIs directly. Superseded by Cloudflare `/api/backend/*` gateway proxy to Railway.

---

## 6. Document Retention & Cross-Reference Rules

1. **Zero Silent Deletions:** Historical files, even if superseded, must remain in place to preserve Git commit history and auditability.
2. **Clear Deprecation Headers:** Superseded files must link to their canonical successors.
3. **Relative Path Integrity:** All repository links must use clean, relative paths (`docs/...`) without hardcoded Windows drive letters (`C:\\...`).
4. **Owner Gate Preservation:** Documents must not assert that experimental research has been promoted to production without recorded owner ratification and deployment receipts.
