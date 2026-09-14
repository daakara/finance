# ARX Terminal: Phase A3 Formative Human Usability Validation Pack
**Focus**: Phase A3 — Page Clarity & Task-Driven Guidance (Release 1 Four-Hub Scope)  
**Study Type**: Moderated, Think-Aloud, Direct-Entry 30-Second Comprehension Evaluation  
**Target Sample**: N = 5 Real Human Participants  
**Authoritative Status**: `A3: NOT YET TESTED (REAL PARTICIPANT STUDY REQUIRED)`  
**Release 1 Journey Scope**: **Radar → Analysis → Trade Plan → Portfolio** (`Find → Understand → Plan → Manage`)  
**Deferred Surfaces Scope**: **Journal** and **Performance** are `DEFERRED TO POST-R1` and explicitly excluded from Release 1 human validation.

---

## 1. Study Overview & Integrity Rules

This package is a ready-to-run instrument for a human moderator conducting direct-entry usability sessions with 5 real first-time human participants.

### Repository & Environment Binding
- **Target Git Revision**: Bound to current commit SHA (`HEAD`)
- **Execution Target**: `http://localhost:3000` (Production static build)
- **Data State**: **POPULATED** (pre-loaded with representative active holdings for NVDA/AAPL, authentic trade log history, and active setups; participants must evaluate operational affordances, not empty states).

### Release 1 Scope Clarification
Under the deliberate product-scope revision for ARX Release 1:
- The four core trading hubs evaluated are: **Radar**, **Analysis**, **Trade Plan** (Setups), and **Portfolio**.
- The dedicated review surfaces **Journal** and **Performance** are **DEFERRED TO POST-R1**.
- Direct navigation to `/journal` or `/performance` renders an explicit Option A Deferred State Page with return CTAs to the 4 core hubs. These deferred pages are verified via automated accessibility/contrast audits and are not part of the active 4-hub Release 1 task evaluation.

### Mandatory Non-Negotiables
1. **Zero AI/Automated Surrogates**: Automated scripts, LLM role-play, synthetic personas, and headless audits are strictly forbidden as substitutes for real participant observation.
2. **Direct Entry Only**: Participants must land directly on each target URL without prior product walkthroughs, four-hub explanations, or guided tours.
3. **Strict Objective Status**: Until real participant observations are collected, all hubs remain classified as `NOT TESTED — NO HUMAN EVIDENCE`.

---

## 2. Moderator Script (Verbatim)

### Session Introduction
> *"Thank you for taking part.*
>
> *This is a usability study of an investment analysis interface.*
>
> *We are testing the product, not you.*
>
> *I will show you a page and give you 30 seconds to look at it.*
>
> *Please think aloud.*
>
> *After 30 seconds, I will ask you:*
> 1. *What do you think this page is for?*
> 2. *What would you do next?*
> 3. *What do you expect that action to do?*
>
> *I will not explain the page while you are looking at it."*

### Prohibited Moderator Behaviors
- **DO NOT** mention the core-hub architecture (`Radar → Analysis → Trade Plan → Portfolio`).
- **DO NOT** point out or name the `PageIntro` component.
- **DO NOT** hint at the intended primary call-to-action (CTA).
- **DO NOT** confirm or deny whether their interpretation is "correct" during the 30-second observation.

---

## 3. Participant Eligibility & Device Profile

### Target Sample: N = 5 First-Time Participants

### Screening Criteria
- **Inclusion**:
  - Basic familiarity with investing or trading concepts (stocks, tickers, orders, portfolios).
  - Comfortable using modern desktop and mobile web browsers.
- **Exclusion**:
  - Anyone involved in the design, engineering, or development of ARX Terminal.
  - Anyone who has reviewed the UX Roadmap (`OPTION_A_UX_ROADMAP.md`) or architecture specifications.
  - Anyone previously trained on the platform.

### Device Coverage & Demographic Matrix
Record non-sensitive identifiers, viewport width, and experience bands:

| Participant ID | Viewport Width / Device Class | Experience Band | Direct-Entry Initial Condition |
| :--- | :--- | :--- | :--- |
| **P01** | 1440 CSS px (Desktop) | Experienced Trader | Populated State |
| **P02** | 375 CSS px (Mobile Portrait / iPhone SE) | Intermediate Investor | Populated State |
| **P03** | 1280 CSS px (Laptop / Desktop) | Beginner Investor | Populated State |
| **P04** | 390 CSS px (Mobile Portrait / iPhone 14) | Intermediate Investor | Populated State |
| **P05** | 1440 CSS px (Desktop) | Experienced Trader | Populated State |

---

## 4. Test Routes & Direct-Entry URLs (Release 1 Retained Hubs)

Test the four primary Release 1 hubs via direct URL entry:

1. **Radar (`Find`)**: `http://localhost:3000/radar`
2. **Analysis (`Understand`)**: `http://localhost:3000/?symbol=NVDA` *(Note: `?ticker=NVDA` is also accepted as legacy fallback)*
3. **Trade Plan (`Plan`)**: `http://localhost:3000/setups?ticker=NVDA` *(or `/setups` for catalog browse)*
4. **Portfolio (`Manage`)**: `http://localhost:3000/portfolio`

*(Note: `/journal` and `/performance` are deferred to post-R1 and excluded from active task validation).*

---

## 5. Counterbalanced Cyclic Rotation (Complete First-Position Balance across 4 Hubs with N=5)

With exactly 4 canonical hubs and $N=5$ participants, complete first-position balance is mathematically achieved. Each of the 4 hubs appears as the first tested page for at least one participant, with P05 cycling back to Radar on desktop:

| Participant ID | Device Class | Target Experience | Route Presentation Order |
| :--- | :--- | :--- | :--- |
| **P01** | 1440px Desktop | Experienced Trader | `Radar → Analysis → Trade Plan → Portfolio` |
| **P02** | 375px Mobile | Intermediate Investor | `Analysis → Trade Plan → Portfolio → Radar` |
| **P03** | 1280px Desktop | Beginner Investor | `Trade Plan → Portfolio → Radar → Analysis` |
| **P04** | 390px Mobile | Intermediate Investor | `Portfolio → Radar → Analysis → Trade Plan` |
| **P05** | 1440px Desktop | Experienced Trader | `Radar → Analysis → Trade Plan → Portfolio` |

---

## 6. 30-Second Task Protocol: Comprehension After 30 Seconds of Unassisted Exposure

For every page presented:
1. Open the direct-entry URL in the participant's designated browser viewport.
2. Start the 30-second stopwatch immediately upon initial page render.
3. Maintain silence for 30 seconds. Do not intervene unless an unexpected browser crash occurs.
4. Record live think-aloud utterances with relative timestamps (e.g., `[00:08] "I see a chart for Nvidia"`, `[00:22] "There's a sizing calculator below"`).
5. At the 30-second mark, call time and prompt:
   - **Q1**: *"What do you think this page is for?"*
   - **Q2**: *"What would you do next?"*
   - **Q3**: *"What do you expect that action to do?"*
6. Transcribe the participant's responses near-verbatim.

---

## 7. Scoring Rubric

Each question is evaluated independently using `PASS`, `FAIL`, or `AMBIGUOUS`:

| Dimension | Evaluation Focus | PASS Criteria | FAIL Criteria |
| :--- | :--- | :--- | :--- |
| **Location** | Where am I? | Identifies the domain context (e.g., "trade plan for NVDA", "portfolio risk manager", "stock discovery/screener"). Exact page name not required. | Completely misidentifies the domain (e.g., mistaking Radar for their personal portfolio). |
| **Purpose** | What is this for? | Materially explains the core job (e.g., "finding momentum candidates", "evaluating company strength", "sizing a conditional trade plan", "tracking active holdings and stop floors"). | Fails to state the main job or repeats top heading without understanding. |
| **Next Action** | What do I do next? | Identifies a sensible primary/secondary action (e.g., "review the entry/stop levels", "click Record Broker Fill", "inspect an asset in Analysis"). | Confused, clicks randomly, or claims there is nothing to do. |
| **Expected Outcome** | What will happen? | Understands the consequence of the action (e.g., "opens a fill recording form", "takes me to deep analysis", "copies plan to clipboard"). | Believes a read-only or copy action executes an order or places a live market trade. |

---

## 8. Critical Safety Checks

### Safety Check 1: Copy Plan ≠ Trade Execution (`Trade Plan / Setups`)
When the participant evaluates the **Trade Plan** hub (`/setups?ticker=NVDA`), the moderator must ask:
> *"What do you think will happen if you press 'Copy Trade Plan'?"*

- **PASS**: Participant clearly understands it copies text/order parameters to their clipboard (e.g., *"It copies the order ticket so I can paste it into my broker or notes"*).
- **FAIL**: Participant believes it places a trade, routes an order, records an execution, or mutates their portfolio (e.g., *"It executes the trade for me"*, *"It buys the stock"*).

> [!CAUTION]
> **Zero Misconception Tolerance**: The acceptance threshold requires **0 of 5 participants** to mistake copying a trade plan for trade execution. A single execution misconception constitutes a material safety defect.

### Safety Check 2: Setup Score ≠ Guaranteed Win Rate (`Trade Plan / Setups`)
When the participant evaluates the **Trade Plan** hub (`/setups?ticker=NVDA`), the moderator must ask:
> *"What do you think the setup score (e.g. 73) means?"*

- **PASS**: Interpreted as relative setup quality, confluence strength, multi-factor evaluation rating, or systematic attractiveness score.
- **FAIL**: Interpreted as guaranteed win probability, promised percentage return, or mathematical certainty that the trade will be profitable (e.g., *"73% chance this trade wins"*).

### Safety Check 3: Portfolio ≠ Live Broker Account (`Portfolio`)
When the participant evaluates the **Portfolio** hub (`/portfolio`), the moderator must ask:
> *"Does this page automatically execute orders with your brokerage or show your live broker cash balance?"*

- **PASS**: Interpreted as an institutional risk management and position tracking system where positions and stop floors are recorded and monitored.
- **FAIL**: Believes ARX is custodial or executing trades directly through an invisible broker integration without manual record/fill confirmation.

---

## 9. Observation Recording Template

Use this table to record observations live during the sessions:

| Participant | Experience | Hub | Purpose Quote | Purpose | Next Action Quote | Next Action | Outcome Quote | Outcome | ≤30s | Copy≠Exec | Score Realism | Port≠Broker | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| P01 | Experienced | Radar | | | | | | | | N/A | N/A | N/A | |
| P01 | Experienced | Analysis | | | | | | | | N/A | N/A | N/A | |
| P01 | Experienced | Trade Plan | | | | | | | | | | N/A | |
| P01 | Experienced | Portfolio | | | | | | | | N/A | N/A | | |
| P02 | Intermediate | Analysis | | | | | | | | N/A | N/A | N/A | |
| P02 | Intermediate | Trade Plan | | | | | | | | | | N/A | |
| P02 | Intermediate | Portfolio | | | | | | | | N/A | N/A | | |
| P02 | Intermediate | Radar | | | | | | | | N/A | N/A | N/A | |
| P03 | Beginner | Trade Plan | | | | | | | | | | N/A | |
| P03 | Beginner | Portfolio | | | | | | | | N/A | N/A | | |
| P03 | Beginner | Radar | | | | | | | | N/A | N/A | N/A | |
| P03 | Beginner | Analysis | | | | | | | | N/A | N/A | N/A | |
| P04 | Intermediate | Portfolio | | | | | | | | N/A | N/A | | |
| P04 | Intermediate | Radar | | | | | | | | N/A | N/A | N/A | |
| P04 | Intermediate | Analysis | | | | | | | | N/A | N/A | N/A | |
| P04 | Intermediate | Trade Plan | | | | | | | | | | N/A | |
| P05 | Experienced | Radar | | | | | | | | N/A | N/A | N/A | |
| P05 | Experienced | Analysis | | | | | | | | N/A | N/A | N/A | |
| P05 | Experienced | Trade Plan | | | | | | | | | | N/A | |
| P05 | Experienced | Portfolio | | | | | | | | N/A | N/A | | |

---

## 10. Success Thresholds & Acceptance Rules

### Participant-Level Pass
A participant passes a hub if and only if they demonstrate understanding across **Purpose**, **Next Action**, and **Expected Outcome** within **30 seconds** without assistance.

### Overall Phase A3 Gate Closure Criteria
1. **Comprehension**: At least **4 of 5 participants (>= 80%)** achieve a full PASS across all 4 tested hubs.
2. **Safety Integrity**: Exactly **0 of 5 participants (0%)** confuse `Copy Trade Plan` with order execution.
3. **Score Realism**: Exactly **0 of 5 participants (0%)** interpret the setup score as guaranteed win probability.
4. **Portfolio Independence**: Exactly **0 of 5 participants (0%)** mistake portfolio tracking for live automated broker custody.

---

## 11. Hub Classification & Defect Standards

### Hub Statuses (Post-Study)
- `CLEAR`: No repeated comprehension friction.
- `MINOR_FRICTION`: Minor hesitation or phrasing ambiguity, but core task correctly understood.
- `MATERIAL_COMPREHENSION_FAILURE`: Repeated (>1 participant) failure on purpose, primary action, or outcome.

### Defect Remediation Trigger
- **Single Participant Defect**: Modifies UI only if it reveals a critical trade safety/execution misunderstanding.
- **Repeated Defect (2+ Participants)**: Triggers surgical copy or layout remediation (`PageIntro` copy, CTA label clarity, setup score tooltip refinement). Speculative or broad redesigns are prohibited.

---

## 12. Final Analysis Reporting Template

```text
## Participants
N: [5]
experience distribution: [X Beginner, Y Intermediate, Z Experienced]
study date: [YYYY-MM-DD]

## Overall Gate
participants meeting full comprehension threshold: [X / 5]
required: [4 / 5]
PASS/FAIL: [PASS | FAIL]

## Copy ≠ Execution
participants tested: [5]
misconceptions: [0]
PASS/FAIL: [PASS | FAIL]

## Setup Score Realism
correct interpretations: [X / 5]
material misconceptions: [0]
PASS/FAIL: [PASS | FAIL]

## Portfolio Independence
correct interpretations: [X / 5]
material misconceptions: [0]
PASS/FAIL: [PASS | FAIL]

## Hub Summary
| Hub | Observations | Pass Rate | Repeated Misconception | Status |
| :--- | :---: | :---: | :--- | :---: |
| Radar (Find) | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |
| Analysis (Understand) | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |
| Trade Plan (Plan) | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |
| Portfolio (Manage) | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |

## Deferred Hubs (Post-R1)
| Hub | Status | Reason |
| :--- | :---: | :--- |
| Journal | DEFERRED TO POST-R1 | Review/calibration surface deferred; persistence active in background |
| Performance | DEFERRED TO POST-R1 | Realized attribution surface deferred; calculation primitives preserved |

## Findings
material: [None | Specific findings]
minor: [None | Specific findings]
participant-specific: [None | Idiosyncratic observations]

## Roadmap
A3 before: A3: IMPLEMENTATION VERIFIED — FORMATIVE COMPREHENSION VALIDATION OPEN (4-Hub R1 Scope)
A3 after:  [COMPLETE / CLOSED — FORMATIVE COMPREHENSION VALIDATION PASSED | REMAINS OPEN]
```
