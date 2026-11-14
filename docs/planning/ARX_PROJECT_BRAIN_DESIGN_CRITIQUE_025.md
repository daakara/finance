# ARX Terminal — Project Brain Multi-Perspective Design Critique (Directive 025)

**Document ID:** `UX-CRT-025-001`  
**Date:** 2026-11-14  
**Target Repository:** `daakara/finance` (`release/production` @ `6ea232838a53e482753d5705ba533ad9ce041e44`)  
**Methodology:** Project Brain Multi-Perspective Critical Panel & Architectural Trade-off Analysis  
**Classification:** Canonical Design Critique & Directional Recommendation  

---

## 1. Executive Summary

This Project Brain critique brings together seven specialized design and engineering perspectives to evaluate the ARX Terminal user experience, critique its educational surfaces, and determine the optimal architectural redesign path.

Rather than offering generic praise, each specialist rigorously challenges current assumptions, identifies concrete failure modes, and debates three distinct structural redesign alternatives.

The panel unanimously rejects superficial cosmetic restyling and ungrounded "AI copilot" patterns. It selects **Approach B: Integrated Decision Context & First-Class Learn Hub** as the highest-leverage, lowest-risk, and most institutionally sound architectural path.

---

## 2. Seven Specialized Independent Perspectives

### Perspective 1: UX Researcher
- **Primary Finding:** The user mental model is fractured by the disconnection between discovery, analysis, and education. Users encountering complex metrics like "Cornish-Fisher VaR" or "Legislative Alignment Index" have no immediate inline cognitive bridge to understand what these numbers mean.
- **Critical Assessment:** The current 4-hub navigation (`Radar → Analysis → Trade Plan → Portfolio`) is logical for seasoned quantitative traders, but completely abandons intermediate and discretionary investors who require conceptual grounding. Because `/guide` and `/glossary` are hidden from the primary navigation, users perceive ARX as an intimidating "black box" rather than an empowering institutional tool.
- **Unverified Assumptions Challenged:** The assumption that users will naturally press `/` to open Omnisearch or dig through footers to discover the Field Manual is thoroughly invalidated. Users either exit or trade with false confidence based on misunderstood numbers.

### Perspective 2: Product Designer
- **Primary Finding:** Information hierarchy in high-density analytical screens is flattened by visual competition. The active ticker hero card displays up to 8 competing badges and tags with similar visual weights, diluting the primary signal.
- **Critical Assessment:** The design system currently relies on excessive border dividers and high-contrast glass panels (`bg-[#0c1017]/95 border-[#243044]`). Visual grouping should instead rely on whitespace hierarchy, subtle surface elevation shifts, and disciplined semantic color tokens (emerald for valid accumulation, rose for risk/stop-loss, cyan for primary analytical focus, amber for caution).
- **Redesign Mandate:** Establish a decision-first visual hierarchy:
  1. Primary Level: Active Asset Identity, Current Price, and Invariant Actionability State.
  2. Secondary Level: Core Execution Levels (Entry Corridor, Invalidation Floor, Target).
  3. Tertiary Level: Supporting Multi-Pillar Confluence and Fundamental Evidence.

### Perspective 3: Financial Domain Specialist
- **Primary Finding:** Severe terminology conflation between *relative screening rank* and *executable buy recommendation*.
- **Critical Assessment:** In the legacy copy, screening candidates were labeled with directives like "READY TO BUY" simply because price was resting within an ATR band. While PR #279 successfully corrected this in code (demoting to `NEAR SCREENING ZONE` and applying defensive styling for scores $\le 45$), the educational guide still uses casual phrases like "Where to Buy" and "Piotroski BS Detector".
- **Quantitative Rigor Mandate:** Every financial concept must clearly state:
  - What the metric measures.
  - What the metric **does not** measure.
  - Why historical backtests (e.g. Minervini win rates) do not guarantee prospective forward returns.
  - The exact statutory limitations (e.g. 45-day STOCK Act reporting lag, SEC Form 4 2-day filing latency).

### Perspective 4: Content Designer / Human Editor
- **Primary Finding:** The Learn Hub (`/guide`) is saturated with classic AI-writing markers: promotional puffery, three-part repetitive sentences, excessive em-dashes, and artificial dichotomies.
- **Critical Assessment:** Phrases like *"The No-BS Trader Field Manual: How to Use This Terminal"* and *"Old financial websites clutter your screen with 50 complicated widgets that make your head spin..."* sound like a low-budget marketing pitch, completely destroying institutional credibility. The split toggle between "Plain English" and "Pro Quant" patronizes retail users with colloquial slang while dumping impenetrable Greek formulas on professional users.
- **Editorial Transformation Mandate:** Eliminate the binary toggle. Write a single, authoritative, layered human curriculum that explains the intuition first, provides concrete numerical examples second, and provides the mathematical formulation for advanced quantitative verification third. Follow the 8-part human lesson structure.

### Perspective 5: Accessibility Specialist (WCAG 2.2 AA)
- **Primary Finding:** Target size violations and contrast degradation in auxiliary controls.
- **Critical Assessment:** Several header buttons hover near 36–40px, failing the mandatory 44x44px touch target invariant (`UI-0`). Certain muted text classes (`text-slate-500` on `#0c1017`) drop below the 4.5:1 contrast ratio required for normal text under WCAG 2.2 AA.
- **Accessibility Mandate:**
  - Elevate muted text to minimum `#94a3b8` (Slate-400, 5.2:1 contrast against `#0c1017`).
  - Enforce `min-h-[44px] min-w-[44px]` on all interactive elements across mobile and desktop.
  - Ensure all mathematical formulas rendered with KaTeX have accessible textual fallbacks for screen readers.

### Perspective 6: Frontend Architect
- **Primary Finding:** High architectural resilience in the core test suite (48/48 tests passing), but unnecessary route sprawl and monolithic component debt in `GuideContent.tsx` (816 lines).
- **Critical Assessment:** `GuideContent.tsx` is an unmaintainable static monolithic blob with broken section IDs (`chapter-2` repeated, `chapter-7` skipped). It should be refactored into clean, modular, data-driven lesson components sharing a typed curriculum registry.
- **Implementation Safeguards:**
  - Zero modification to backend APIs or calculations.
  - Zero alteration to canonical hub routing contracts (`canonicalNav.ts`).
  - Strict preservation of all 6 operational holds (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`).

### Perspective 7: Adversarial Reviewer
- **Primary Finding:** Risk of "cosmetic theatre"—redesigning pages without solving actual user comprehension failure.
- **Critical Assessment:** If we simply change colors and reword paragraphs without integrating education into the user's active decision loop, users will continue making the exact same execution mistakes. Furthermore, any attempt to present simulated usability testing as "proven human results" violates the core integrity mandate of ARX.
- **Adversarial Gate:** All claimed usability improvements must be strictly qualified: code and content improvements are verified; user comprehension gains remain design hypotheses until the Phase A3 5-participant real-human study is formally executed.

---

## 3. Evaluation of Competing Redesign Approaches

The panel evaluated three distinct structural options:

```
┌────────────────────────────────────────────────────────────────────────┐
│ OPTION A: Standalone Academy Portal (/learn subdomain)                │
│ • Separate external documentation website or isolated sub-portal       │
│ • High editorial freedom, but 100% decoupled from active trading       │
│ • Pros: Zero risk to trading engine. Cons: Lowest discovery, siloed.   │
├────────────────────────────────────────────────────────────────────────┤
│ OPTION B: Integrated Decision Context & First-Class Learn Hub (CHOSEN) │
│ • "Learn" promoted to first-class global navigation alongside hubs     │
│ • Modular Field Manual + Quantitative Glossary with deep mutual links  │
│ • Analytical cards link directly to specific lesson anchors            │
│ • Pros: High discovery, unified mental model, low architectural risk.  │
├────────────────────────────────────────────────────────────────────────┤
│ OPTION C: Split-Screen Terminal Inspector / Docked Copilot Drawer      │
│ • Persistent slide-out documentation drawer alongside active charts   │
│ • Complex state management, high viewport friction on laptops/mobile   │
│ • Pros: Immediate contextual reference. Cons: High technical debt.    │
└────────────────────────────────────────────────────────────────────────┘
```

### Detailed Trade-Off Matrix

| Evaluation Dimension | Option A: Standalone Academy Portal | Option B: Integrated Decision Context (Selected) | Option C: Split-Screen Inspector Drawer |
| :--- | :--- | :--- | :--- |
| **User Discovery & Access** | **Poor.** Requires leaving the terminal; high bounce rate. | **Excellent.** 1-click access from top nav, mobile dock, and card tooltips. | **High.** Always present in viewport, but crowds analytical space. |
| **Cognitive Continuity** | **Low.** Forces context switching across separate tabs. | **High.** Preserves active ticker context; seamless return path. | **Moderate.** Constant visual competition with chart canvas. |
| **Mobile Feasibility (390px)** | **Moderate.** Standard responsive web, but disconnected. | **High.** Clean dedicated reading layout with zero horizontal overflow. | **Fails.** Cannot comfortably fit chart + inspector on mobile screen. |
| **Implementation Complexity** | High (requires separate deployment or docs framework). | **Low to Moderate.** Surgical refactor within existing Next.js App Router. | Very High (requires complex drawer docking and state synchronization). |
| **Risk to Trading Governance** | Low. | **Zero.** Strictly frontend presentation and editorial content. | Moderate (risk of layout shifting and event listener collisions). |
| **Compliance with Holds** | Preserves all holds. | **Preserves all holds completely.** | Risk of touch target and viewport conflicts. |

### Final Directional Recommendation

**Project Brain formally recommends Option B: Integrated Decision Context & First-Class Learn Hub.**

Option B achieves the optimal synthesis of discoverability, institutional rigor, editorial excellence, and implementation safety. It respects existing navigation contracts while elevating education to its rightful role as an institutional pillar.
