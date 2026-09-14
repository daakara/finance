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
- **Target Git Revision**: `42ab9b572ce3264958d5c2cdf519cba437502eb3` (Pinned before Session 1)
- **Execution Target**: Fixed Staging URL or Local LAN (`http://<YOUR-PC-LAN-IP>:3000`)
  > [!IMPORTANT]
  > **Physical Mobile Access**: Do NOT use `localhost:3000` for physical mobile devices (it resolves to the phone itself). Use either a fixed staging deployment or your host workstation's LAN IP (e.g., `http://192.168.x.x:3000`).
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
4. **Mid-Study Code Freeze**: Zero code, UI, copy, or model changes allowed during participant sessions (`MID_STUDY_UI_CHANGES_ALLOWED = false`).

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

### Target Sample: N = 5 First-Time Participants (Tightened Distribution)
The target audience is representative first-time users with basic trading literacy. The participant distribution is strictly calibrated to avoid knowledge bias:
- **1 Experienced Trader**
- **2 Intermediate Investors**
- **2 Beginner / Basic-Trading-Literate Users**

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

| Participant ID | Viewport Width / Device Class | Experience Band | Direct-Entry Initial Condition | First-Entry Hub |
| :--- | :--- | :--- | :--- | :--- |
| **P01** | 1440 CSS px (Desktop) | Experienced Trader | Populated State | **Radar** |
| **P02** | 375 CSS px (Mobile Portrait / iPhone SE) | Intermediate Investor | Populated State | **Analysis** |
| **P03** | 1280 CSS px (Laptop / Desktop) | Beginner / Basic Literate | Populated State | **Trade Plan** |
| **P04** | 390 CSS px (Mobile Portrait / iPhone 14) | Beginner / Basic Literate | Populated State | **Portfolio** |
| **P05** | 1440 CSS px (Desktop) | Intermediate Investor | Populated State | **Analysis** |

---

## 4. Test Routes & Direct-Entry URLs (Release 1 Retained Hubs)

Test the four primary Release 1 hubs via direct URL entry (using `<HOST-OR-LAN-IP>:3000` or staging):

1. **Radar (`Find`)**: `http://<HOST>:3000/radar`
2. **Analysis (`Understand`)**: `http://<HOST>:3000/?symbol=NVDA` *(Note: `?ticker=NVDA` is also accepted as legacy fallback)*
3. **Trade Plan (`Plan`)**: `http://<HOST>:3000/setups?ticker=NVDA` *(or `/setups` for catalog browse)*
4. **Portfolio (`Manage`)**: `http://<HOST>:3000/portfolio`

*(Note: `/journal` and `/performance` are deferred to post-R1 and excluded from active task validation).*

---

## 5. Counterbalanced Cyclic Rotation & First-Page Separation

### First-Page vs. Subsequent-Page Disambiguation
To prevent learning effects from skewing the evaluation, analysis must strictly separate:
- **`FIRST_PAGE_RESULT`**: The initial direct-entry page for each participant. This constitutes the cleanest, uncontaminated comprehension evidence.
- **`SUBSEQUENT_PAGE_RESULT`**: Subsequent pages evaluated in sequence, which may reflect cross-page learning.

### Balanced Starting Positions
Analysis is evaluated twice as a first-entry hub because it was materially updated in Phase UX-R2:

| Participant ID | Device Class | Target Experience | First-Entry Hub | Full Route Presentation Order |
| :--- | :--- | :--- | :--- | :--- |
| **P01** | 1440px Desktop | Experienced Trader | **Radar** | `Radar → Analysis → Trade Plan → Portfolio` |
| **P02** | 375px Mobile | Intermediate Investor | **Analysis** | `Analysis → Trade Plan → Portfolio → Radar` |
| **P03** | 1280px Desktop | Beginner / Basic Literate | **Trade Plan** | `Trade Plan → Portfolio → Radar → Analysis` |
| **P04** | 390px Mobile | Beginner / Basic Literate | **Portfolio** | `Portfolio → Radar → Analysis → Trade Plan` |
| **P05** | 1440px Desktop | Intermediate Investor | **Analysis** | `Analysis → Radar → Trade Plan → Portfolio` |

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

#### Safety Check 1: Copy Plan ≠ Trade Execution (`Trade Plan / Setups`)
When the participant evaluates the **Trade Plan** hub (`/setups?ticker=NVDA`), the moderator must ask:
> *"What do you think will happen if you press 'Copy Trade Plan'?"*

- **PASS**: Participant clearly understands it copies text/order parameters to their clipboard (e.g., *"It copies the order ticket so I can paste it into my broker or notes"*).
- **FAIL**: Participant believes it places a trade, routes an order, records an execution, or mutates their portfolio (e.g., *"It executes the trade for me"*, *"It buys the stock"*).

> [!CAUTION]
> **Hard Fail Condition**: The acceptance threshold requires **0 of 5 participants (0%)** to mistake copying a trade plan for trade execution. A single execution misconception is an automatic phase fail.

### Safety Check 2: Setup Score ≠ Guaranteed Win Rate (`Trade Plan / Setups`)
When the participant evaluates the **Trade Plan** hub (`/setups?ticker=NVDA`), the moderator must ask:
> *"What do you think the setup score (e.g. 73) means?"*

- **PASS**: Interpreted as relative setup quality, confluence strength, multi-factor evaluation rating, or systematic attractiveness score.
- **FAIL**: Interpreted as guaranteed win probability, promised percentage return, or mathematical certainty that the trade will be profitable (e.g., *"73% chance this trade wins"*).

> [!CAUTION]
> **Hard Fail Condition**: The acceptance threshold requires **0 of 5 participants (0%)** to interpret the score as guaranteed win probability. A single misconception is an automatic phase fail.

### Safety Check 3: Portfolio ≠ Live Broker Account (`Portfolio`)
When the participant evaluates the **Portfolio** hub (`/portfolio`), the moderator must ask:
> *"Does this page automatically execute orders with your brokerage or show your live broker cash balance?"*

- **PASS**: Interpreted as an institutional risk management and position tracking system where positions and stop floors are recorded and monitored.
- **FAIL**: Believes ARX is custodial or executing trades directly through an invisible broker integration without manual record/fill confirmation.

> [!CAUTION]
> **Hard Fail Condition**: The acceptance threshold requires **0 of 5 participants (0%)** to mistake portfolio tracking for live broker custody. A single misconception is an automatic phase fail.

---

## 9. Observation Recording Template

Use this table to record observations live during the sessions. Note whether each observation represents a clean direct-entry (`FIRST_PAGE`) or subsequent exposure (`SUBSEQUENT`):

| Participant | Experience | Hub | Entry Type | Purpose Quote | Purpose | Next Action Quote | Next Action | Outcome Quote | Outcome | ≤30s | Copy≠Exec | Score Realism | Port≠Broker | Notes |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| P01 | Experienced | Radar | **FIRST_PAGE** | | | | | | | | N/A | N/A | N/A | |
| P01 | Experienced | Analysis | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P01 | Experienced | Trade Plan | SUBSEQUENT | | | | | | | | | | N/A | |
| P01 | Experienced | Portfolio | SUBSEQUENT | | | | | | | | N/A | N/A | | |
| P02 | Intermediate | Analysis | **FIRST_PAGE** | | | | | | | | N/A | N/A | N/A | |
| P02 | Intermediate | Trade Plan | SUBSEQUENT | | | | | | | | | | N/A | |
| P02 | Intermediate | Portfolio | SUBSEQUENT | | | | | | | | N/A | N/A | | |
| P02 | Intermediate | Radar | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P03 | Beginner / Basic | Trade Plan | **FIRST_PAGE** | | | | | | | | | | N/A | |
| P03 | Beginner / Basic | Portfolio | SUBSEQUENT | | | | | | | | N/A | N/A | | |
| P03 | Beginner / Basic | Radar | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P03 | Beginner / Basic | Analysis | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P04 | Beginner / Basic | Portfolio | **FIRST_PAGE** | | | | | | | | N/A | N/A | | |
| P04 | Beginner / Basic | Radar | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P04 | Beginner / Basic | Analysis | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P04 | Beginner / Basic | Trade Plan | SUBSEQUENT | | | | | | | | | | N/A | |
| P05 | Intermediate | Analysis | **FIRST_PAGE** | | | | | | | | N/A | N/A | N/A | |
| P05 | Intermediate | Radar | SUBSEQUENT | | | | | | | | N/A | N/A | N/A | |
| P05 | Intermediate | Trade Plan | SUBSEQUENT | | | | | | | | | | N/A | |
| P05 | Intermediate | Portfolio | SUBSEQUENT | | | | | | | | N/A | N/A | | |

---

## 10. Success Thresholds & Acceptance Rules

### Participant-Level Pass
A participant passes a hub if and only if they demonstrate understanding across **Purpose**, **Next Action**, and **Expected Outcome** within **30 seconds** without assistance.

### Overall Phase A3 Gate Closure Criteria
1. **Core Comprehension Gate**: At least **4 of 5 participants (>= 80%)** achieve a full PASS across all tested hubs.
2. **First-Page Direct-Entry Pure Gate**: First-page direct-entry results (`FIRST_PAGE_RESULT`) must independently confirm comprehension without reliance on subsequent cross-page learning.
3. **Hard-Fail Safety Boundaries (Independent Zero-Tolerance Gating)**:
   - `COPY TRADE PLAN MISUNDERSTOOD AS ORDER EXECUTION = 0 / 5`
   - `SETUP SCORE MISUNDERSTOOD AS WIN PROBABILITY = 0 / 5`
   - `PORTFOLIO MISUNDERSTOOD AS LIVE BROKER CUSTODY = 0 / 5`
   *(Any single failure on these 3 checks triggers an automatic phase FAIL regardless of overall score).*

### Canonical Frozen Pre-Study State
```text
A3_STATUS = READY_FOR_HUMAN_EXECUTION

A3_TEST_BUILD_COMMIT = 42ab9b572ce3264958d5c2cdf519cba437502eb3
A3_TEST_ENVIRONMENT = STAGING OR LAN BUILD

PARTICIPANTS_REQUIRED = 5
MOBILE_REQUIRED >= 2
DESKTOP_REQUIRED >= 2

AI_SIMULATION_ALLOWED = false
MID_STUDY_UI_CHANGES_ALLOWED = false

CORE_COMPREHENSION_THRESHOLD >= 80%
SAFETY_MISCONCEPTION_THRESHOLD = 0% (HARD FAIL BOUNDARY)

A5 = BLOCKED_PENDING_A3
UX_R3_PLUS = BLOCKED_PENDING_A3
```

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
experience distribution: [2 Beginner/Basic, 2 Intermediate, 1 Experienced]
study date: [YYYY-MM-DD]
test build commit: 42ab9b572ce3264958d5c2cdf519cba437502eb3
environment: [LAN Build / Staging URL]

## Overall Gate
participants meeting full comprehension threshold: [X / 5]
required: [4 / 5]
PASS/FAIL: [PASS | FAIL]

## First-Page Pure Direct-Entry Gate
first-page pure passes: [X / 5]
subsequent-page passes: [Y / 15]

## Hard-Fail Safety Checks
copy trade plan misunderstood as order execution: [0 / 5] (Required: 0)
setup score misunderstood as win probability: [0 / 5] (Required: 0)
portfolio misunderstood as live broker custody: [0 / 5] (Required: 0)
SAFETY GATE: [PASS | FAIL]

## Hub Summary
| Hub | First-Page Obs | Total Obs | Pass Rate | Repeated Misconception | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| Radar (Find) | 1 | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |
| Analysis (Understand) | 2 | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |
| Trade Plan (Plan) | 1 | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |
| Portfolio (Manage) | 1 | 5 | % | None | [CLEAR | MINOR_FRICTION | MATERIAL_FAILURE] |

## Deferred Hubs (Post-R1)
| Hub | Status | Reason |
| :--- | :---: | :--- |
| Journal | DEFERRED TO POST-R1 | Review/calibration surface deferred; persistence active in background |
| Performance | DEFERRED TO POST-R1 | Realized attribution surface deferred; calculation primitives preserved |

## Findings
material: [None | Specific findings]
minor: [None | Specific findings]
participant-specific: [None | Idiosyncratic observations]

## Roadmap Gate Verdict
A3 Verdict: [PASS | PASS WITH REMEDIATION | FAIL / REMEDIATION REQUIRED]
A5 Release Gate: [UNBLOCKED | BLOCKED]
```
