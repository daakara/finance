# ARX Terminal — UX Redesign Implementation Plan & Work Backlog (Directive 025)

**Document ID:** `UX-PLN-025-001`  
**Date:** 2026-11-14  
**Target Repository:** `daakara/finance` (`release/production` @ `6ea232838a53e482753d5705ba533ad9ce041e44`)  
**Branch:** `feat/ux-brain-redesign-025`  
**Classification:** Canonical Phased Implementation Backlog & Acceptance Matrix  

---

## 1. Executive Summary & Phased Delivery Strategy

This implementation plan defines the phased engineering and editorial tasks required to execute the ARX Terminal institutional redesign.

To protect production stability and avoid regression vectors in the trading engine, the redesign strictly follows a four-phase rollout strategy:
- **Phase 0:** Evidence Gathering, Baseline Audit, and Design Critique (Completed in Deliverables 1–5).
- **Phase 1:** Global Navigation Integration & Shell Foundation (Elevating Learn Hub in `Navbar.tsx` and mobile utilities).
- **Phase 2:** Complete Learn Hub Editorial Transformation (Rewriting `GuideContent.tsx`, upgrading `glossaryCatalog.ts`, and refining `guide/page.tsx`).
- **Phase 3:** Core Terminal Workflow Harmony & Contextual Deep-Links (Adding educational links from analysis cards without altering calculations).
- **Phase 4:** Comprehensive Verification, Accessibility Testing, and Final Quality Gate Handover.

---

## 2. Existing Workstream & Pull Request Reconciliation

| PR / Issue Reference | Status on Baseline | Scope & Interaction with Directive 025 | Governance Constraint |
| :--- | :--- | :--- | :--- |
| **PR #273** | **MERGED** | Mobile touch target sizing optimization across header and filters. | Must be preserved: all new interactive elements must maintain $\ge 44\times 44$px (`UI-0`). |
| **PR #274 (Phase A3)** | **HELD (DRAFT)** | Progressive disclosure, 3-tier view modes (Guided, Standard, Quant), and decision-first card layout. | **Strictly preserved under `PHASE_A3_UX_HOLD`.** Requires 5-participant real-human study. Do not merge or overwrite. |
| **PR #275** | **MERGED** | Tactical Setups timeout hardening and client reliability (<150ms). | Must be preserved: zero changes to setups data-fetching timeouts. |
| **PR #278** | **MERGED** | PG16 backup verification and worker queue remediation. | Preserved: zero backend modifications. |
| **PR #279** | **MERGED** | Radar recommendation integrity (sub-45 scores styled defensive rose; `READY TO BUY` demoted). | Must be preserved: copy in Learn Hub must align 100% with `NEAR SCREENING ZONE`. |
| **PR #280** | **MERGED** | Post-cycle queue idempotency and calendar skip remediation. | Preserved: Smart Money worker remains in fail-closed `STANDBY`. |

---

## 3. Work Breakdown Structure & Task Backlog

```
PHASE 1: GLOBAL NAVIGATION & SHELL FOUNDATION
├── TASK-025-01: Update `frontend/components/Navbar.tsx`
│   ├── Add "Learn" link to desktop navigation links (`/guide`)
│   ├── Highlight active state when pathname starts with `/guide` or `/glossary`
│   ├── Add "Learn & Field Manual" to Utilities Menu (`⋯`) under Help & Navigation
│   └── Verify 44x44px touch targets across all header triggers
└── TASK-025-02: Update mobile navigation accessibility
    └── Ensure mobile viewports can discover the Field Manual via utilities and omni-search

PHASE 2: LEARN HUB EDITORIAL TRANSFORMATION
├── TASK-025-03: Complete Rewrite of `frontend/components/GuideContent.tsx`
│   ├── Remove "Plain English vs Pro Quant" split toggle; replace with unified layered human text
│   ├── Implement 5-module taxonomy:
│   │   ├── Module 1: Foundations (Capital preservation, 1% rule, 4-tier universe)
│   │   ├── Module 2: Technical Analysis (Stage 2 trend, VCP contraction cycles, 4 ATR states)
│   │   ├── Module 3: Fundamental DNA (5-pillar profile, Piotroski 9-point score, ROIC)
│   │   ├── Module 4: Smart Money Forensics (SEC Form 4, STOCK Act, staleness decay)
│   │   └── Module 5: Using ARX Terminal (Confluence score, Pre-Flight gate, Cornish-Fisher VaR)
│   ├── Apply the 8-part human lesson structure to all topics
│   ├── Fix broken section anchors (`id="module-..."`)
│   └── Ensure clean responsive layout on 390px mobile screens (zero horizontal overflow)
├── TASK-025-04: Upgrade `frontend/lib/glossaryCatalog.ts`
│   ├── Rewrite all 10 glossary terms with human definitions, concrete examples, and explicit limitations
│   └── Ensure clean LaTeX KaTeX formulas
└── TASK-025-05: Update `frontend/app/guide/page.tsx`
    ├── Remove buzzword salad from title, description, and OpenGraph tags
    └── Provide clean Schema.org `TechArticle` and `FAQPage` structured data

PHASE 3: VERIFICATION & TESTING
├── TASK-025-06: Typecheck and Architectural Test Suite
│   ├── Run `npx tsc --noEmit` (100% clean required)
│   └── Run `npm run test:arch` (48/48 tests passing required)
└── TASK-025-07: Responsive Layout & Accessibility Verification
    ├── Verify 390px mobile viewport rendering
    └── Verify WCAG 2.2 AA contrast ratios and focus states

PHASE 4: DOCUMENTATION & REPORTING
└── TASK-025-08: Complete Deliverables 7 and 8 and final PR preparation
```

---

## 4. Measurable Acceptance Criteria

1. **Navigation Parity:** "Learn" is immediately visible in desktop navigation and accessible in 1 click from the Utilities Menu on mobile.
2. **Zero AI Slop:** Zero instances of banned marketing phrases ("Unlock the power of", "Piotroski BS Detector", "game-changing", "seamlessly leverage").
3. **Pedagogical Structure:** All 5 educational modules adhere to the 8-part human lesson blueprint.
4. **Architectural Purity:** All 48 existing architectural tests pass without modification.
5. **Mobile Integrity:** Zero horizontal layout overflow on 390px width viewports.
6. **Governance Preservation:** Zero modifications to backend files, trading engine formulas, or operational hold registers.
