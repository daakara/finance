# ARX Terminal — Observer Submission Instructions (v2.1.0)

> [!IMPORTANT]
> **PROVENANCE DISCLOSURE:**  
> These are synthetic calibration transcripts developed to evaluate observer scoring consistency. They are not recordings of actual ARX Terminal research participants.

---

## 1. Research Overview & Concept Identities

During the upcoming human research studies, observers will evaluate investor interactions across three competing prototype interfaces:
- **Concept A:** Simplified Linear Analysis (Sequential step-by-step disclosure).
- **Concept B:** Decision Cockpit (High-density telemetry with multi-card overview).
- **Concept C:** Evidence-First Workspace (Unified single-page command canvas with 5-question memo).

---

## 2. Study & Scoring Rules

1. Study the `ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md` thoroughly before scoring.
2. Complete evaluations independently without consulting other raters or accessing outside materials.
3. For each trial, evaluate both verbal think-aloud dialogue and UI telemetry events.
4. Record orthogonal states: `interpretation_state` and `action_state`.
5. Verify whether a critical decision error occurred (`critical_decision_error`: `true` or `false`).
6. Validate all rating objects against `OBSERVER_RATING_SCHEMA.json`.

---

## 3. Authoritative Trial Identifiers

Your ratings must reference the exact trial IDs present in the transcripts:
- **Transcript Alpha (`TRANS_ALPHA_BLINDED.json`):** `ALPHA-01` through `ALPHA-10`
- **Transcript Beta (`TRANS_BETA_BLINDED.json`):** `BETA-01` through `BETA-10`
- **Transcript Gamma (`TRANS_GAMMA_BLINDED.json`):** `GAMMA-01` through `GAMMA-10`

*(Do NOT invent or remap identifiers to arbitrary prefixes such as `TR-`.)*

---

## 4. Lockstep Submission & Restricted Custody

1. Save your completed submission files as:
   - `OBS_[YOUR_ID]_ALPHA.json`
   - `OBS_[YOUR_ID]_BETA.json`
   - `OBS_[YOUR_ID]_GAMMA.json`
2. Submit your files simultaneously to the Lead Investment Decision Systems Auditor via secure intake.
3. Submissions are hashed immediately upon arrival (SHA-256) and preserved in a restricted, de-identified custody lockbox. Identifiable personal data is decoupled from rater IDs and never committed to shared public version control.
4. Scoring will be triggered simultaneously once both observers have delivered complete, validated files.
