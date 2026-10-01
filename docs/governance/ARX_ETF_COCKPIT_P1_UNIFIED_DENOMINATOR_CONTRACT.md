# ARX TERMINAL — ETF COCKPIT P1 UNIFIED DENOMINATOR, ELIGIBILITY, EXCLUSION, AND NATURAL-TRAFFIC DECISION CONTRACT

**Document Version**: `1.0.0`  
**Governance Authority**: `ARX_GOVERNANCE_FRAMEWORK` (Section 9)  
**Status**: `RATIFIED_CANONICAL_CONTRACT`  
**Scope**: Prospective Observation Accounting for ARX ETF Cockpit P1 (`72914a58757790aec42edc0995ea43a40dd06105`)  

---

## 9. UNIFIED P1 DENOMINATOR, ELIGIBILITY, EXCLUSION, AND NATURAL-TRAFFIC DECISION CONTRACT

This section is the sole authority for determining whether a candidate interaction is counted in the prospective P1 denominator.

All later sections must use the states, precedence, decision order, natural-traffic procedure, and definitions in this section. No later section may redefine eligibility, exclusion, traffic classification, release attribution, replay handling, or denominator membership.

### 9.1 Canonical observation unit

The canonical observation unit is:

```text
ETF_INTENT_ATTEMPT
```

An `ETF_INTENT_ATTEMPT` is one deterministic, user-originated attempt to initiate the P1 ETF Cockpit journey for one normalized symbol within one session.

The unit is created at the earliest authoritative P1 intent boundary at which the following are available or classifiable:
- `session_id`
- normalized symbol
- intent event
- release attribution
- environment
- traffic-classification inputs
- event timestamp

The unit is not defined by successful routing, successful rendering, Cost of Ownership availability, or any later outcome.

A session may contain multiple observation units when the user makes multiple distinct ETF intent attempts. Repeated delivery of telemetry for the same attempt is not a new observation unit.

---

### 9.2 Canonical states and absolute precedence

Every candidate record must receive exactly one primary accounting state:
- `VALID`
- `QUARANTINED`
- `EXCLUDED`
- `INVALID`

The absolute state precedence is:

```text
INVALID
  >
QUARANTINED
  >
EXCLUDED
  >
VALID
```

The highest-precedence applicable state always wins.

The states mean:
- **INVALID**: The candidate is malformed or internally impossible such that it cannot be interpreted as a candidate P1 observation.
- **QUARANTINED**: The candidate is sufficiently interpretable to represent a possible P1 observation, but a required provenance, release, environment, traffic, replay, duplicate, or other classification fact is missing, ambiguous, contradictory, or not deterministically resolvable. It is withheld from the denominator and may not be silently promoted to `VALID`.
- **EXCLUDED**: The candidate is structurally valid and deterministically known to be outside the natural, production, authorized-release P1 observation population, or is a deterministic replay or duplicate under the rules in this contract.
- **VALID**: The candidate is structurally valid, positively classified as `NATURAL_PRODUCTION`, attributable to an authorized production release, not a deterministic replay or duplicate, within the authorized epoch, associated with the ETF intent boundary, and satisfies every denominator eligibility condition.

Only `VALID` records may enter the prospective denominator.

The precedence rules resolve all apparent conflicts:
- malformed + synthetic marker $\implies$ `INVALID`
- malformed + replay-like evidence $\implies$ `INVALID`
- structurally valid + missing release identity $\implies$ `QUARANTINED`
- structurally valid + contradictory release identity $\implies$ `QUARANTINED`
- structurally valid + deterministic synthetic marker $\implies$ `EXCLUDED`
- structurally valid + deterministic replay $\implies$ `EXCLUDED`
- structurally valid + ambiguous replay evidence $\implies$ `QUARANTINED`
- structurally valid + natural production + failed eligibility condition $\implies$ `EXCLUDED`
- structurally valid + natural production + all eligibility conditions satisfied $\implies$ `VALID`

A downstream success or failure never overrides the primary accounting state.

---

### 9.3 Canonical decision algorithm

Every candidate record must be processed by this algorithm in exactly this order:

1. Parse and structurally validate the candidate.
2. Validate the minimum fields and internal relationships required to interpret the candidate as an auditable record.
3. If the candidate is malformed, unparsable, truncated, or internally impossible:
   - `state = INVALID`
   - retain every safely recoverable diagnostic attribute
   - stop primary-state classification.
4. Validate required provenance and classification inputs, including:
   - `environment`
   - `deployment identity`
   - `release identity`
   - `session provenance`
   - `traffic-classification inputs`
   - `event timestamp`
   - `event identity`
   - `observation identity inputs`
5. If any required provenance, environment, release, or traffic fact is missing, contradictory, malformed, or unresolved:
   - `state = QUARANTINED`
   - `traffic_class = UNKNOWN` or unresolved class
   - retain the applicable quarantine reason
   - stop primary-state classification.
6. Evaluate deterministic exclusion evidence on the structurally valid candidate, including:
   - explicit synthetic/test/QA/CI/developer markers
   - monitoring or uptime-probe provenance
   - deterministic bot or crawler classification
   - deterministic replay evidence
   - deterministic duplicate evidence
   - local, preview, test, staging, or other non-production identity
   - valid but unauthorized release identity
   - timestamp outside the authorized epoch
   - non-ETF intent
   - any other deterministic exclusion defined by this contract
7. If deterministic exclusion evidence is present:
   - `state = EXCLUDED`
   - assign the most specific applicable traffic class and exclusion code
   - retain the evidence and stop primary-state classification.
8. Positively evaluate `NATURAL_PRODUCTION`.
9. If all `NATURAL_PRODUCTION` conditions are not satisfied:
   - `state = QUARANTINED`
   - `traffic_class = UNKNOWN`
   - retain the unresolved or failed natural-traffic condition
   - stop primary-state classification.
10. Verify the candidate's P1 eligibility conditions:
    - authoritative ETF intent boundary
    - valid normalized symbol
    - production environment
    - authorized release
    - valid event ID
    - valid timestamp
    - valid session ID
    - valid observation-unit identity
    - valid deduplication key
    - no replay
    - no duplicate of an already accepted observation
    - within the authorized epoch
    - no structural or sequence contradiction
    - ETF P1 intent rather than a stock, crypto, administrative, preview, test, or unrelated interaction
11. If replay or duplicate status is ambiguous:
    - `state = QUARANTINED`
    - retain the ambiguity and stop primary-state classification.
12. If replay or duplicate status is deterministically established:
    - `state = EXCLUDED`
    - assign `REPLAY` or `DUPLICATE`
    - retain the canonical prior observation reference where available
    - stop primary-state classification.
13. If every eligibility condition is satisfied:
    - `state = VALID`
    - `traffic_class = NATURAL_PRODUCTION`
    - create or retain the distinct `ETF_INTENT_ATTEMPT` observation unit
    - count the observation unit in the denominator.
14. Otherwise:
    - `state = EXCLUDED`
    - retain the failed eligibility condition and exclusion code.
15. Classify downstream success, failure, or neutral behavior separately as an outcome. Never use the outcome to alter the primary accounting state or remove a `VALID` observation from the denominator.

This algorithm is the canonical resolution of decision order and precedence.

The decision order determines when conditions are evaluated. The precedence determines which state wins when multiple conditions apply. A later step may not override a state already assigned by a higher-precedence step.

The algorithm must be read with the following qualification:
- A condition may be evaluated earlier for diagnostic purposes, but it may not determine the primary accounting state until all higher-precedence conditions have been resolved.
- No later step may replace `INVALID` or `QUARANTINED` with `EXCLUDED` or `VALID`.
- No later step may replace `EXCLUDED` with `VALID`.

---

### 9.4 Structural validation

Structural validation is intentionally narrow. A candidate need not be eligible or natural to be structurally valid. It must only be sufficiently parseable and internally coherent to support deterministic classification and auditable accounting.

A candidate is `INVALID` when it is:
- unparseable
- truncated such that required structure cannot be recovered
- missing mandatory structural delimiters or envelope fields
- internally impossible
- inconsistent in a way that prevents deterministic interpretation

A candidate is not `INVALID` merely because it is:
- synthetic
- non-production
- bot traffic
- a replay
- a duplicate
- outside the authorized release
- outside the epoch
- non-ETF
- ineligible

Those conditions produce `EXCLUDED` only when they are deterministically established on a structurally valid candidate.

Examples:
- valid event envelope + explicit CI marker $\implies$ `EXCLUDED / CI`
- valid event envelope + explicit synthetic marker + unsupported symbol $\implies$ `EXCLUDED / SYNTHETIC_E2E`
- truncated payload + bytes resembling a CI marker $\implies$ `INVALID`
- valid envelope + missing release identity $\implies$ `QUARANTINED / RELEASE_UNATTRIBUTABLE`

A synthetic marker is not a waiver of structural validation. It is an exclusion fact only after the candidate has passed the structural threshold required to interpret and retain that marker.

---

### 9.5 Required provenance and unresolved facts

The following are required classification inputs:
- `environment`
- `deployment identity`
- `release identity`
- `event timestamp`
- `event identity`
- `session provenance`
- `traffic-classification inputs`

A required fact is unresolved when it is:
- missing
- ambiguous
- contradictory
- malformed
- inconsistent with another authoritative identity
- not deterministically attributable

An unresolved required fact produces:
```text
state = QUARANTINED
```
unless the record is structurally impossible, in which case:
```text
state = INVALID
```

This rule applies even when another signal suggests that the candidate is probably synthetic or otherwise excludable.

For example:
- `explicit synthetic marker = true`
- `release identity = missing`

produces:
```text
state = QUARANTINED
traffic_class = UNKNOWN
quarantine_reason = RELEASE_UNATTRIBUTABLE
observed_synthetic_marker = true
```

The marker must be retained as an audit attribute, but it does not produce an `EXCLUDED` accounting record because the required release fact remains unresolved.

---

### 9.6 Release identity decision rules

Release identity is mandatory for denominator eligibility and positive natural-production classification.

The release identity must be:
- present
- well-formed
- attributable to the deployment identity
- authorized by the epoch contract

Apply these rules:
- missing release identity $\implies$ `QUARANTINED / RELEASE_UNATTRIBUTABLE`
- malformed release identity $\implies$ `QUARANTINED / RELEASE_UNATTRIBUTABLE`, unless the malformed value makes the entire record structurally impossible, in which case `INVALID`
- contradictory release identity $\implies$ `QUARANTINED / RELEASE_UNATTRIBUTABLE`
- release identity conflicts with deployment identity $\implies$ `QUARANTINED / RELEASE_UNATTRIBUTABLE`
- valid release identity outside the authorized epoch release set $\implies$ `EXCLUDED / WRONG_RELEASE`
- valid authorized release identity $\implies$ continue processing

Missing or contradictory release identity never produces `VALID`.

Missing or contradictory release identity also does not produce `EXCLUDED / WRONG_RELEASE`, because the release cannot be deterministically established as wrong. It produces `QUARANTINED / RELEASE_UNATTRIBUTABLE`.

A candidate with a valid but unauthorized release is structurally interpretable and deterministically outside the epoch, so it is:
```text
EXCLUDED / WRONG_RELEASE
```

---

### 9.7 Synthetic markers and traffic classification

Synthetic, test, QA, CI, developer, monitoring, uptime-probe, bot, crawler, and replay signals must be evaluated only after structural validation and required provenance validation.

A deterministic synthetic or automation marker on a structurally valid candidate produces:
```text
state = EXCLUDED
```
with the most specific applicable classification.

Examples:
- explicit CI marker $\implies$ `EXCLUDED / CI`
- explicit end-to-end test marker $\implies$ `EXCLUDED / SYNTHETIC_E2E`
- authenticated manual QA provenance $\implies$ `EXCLUDED / MANUAL_QA`
- deterministic monitoring provenance $\implies$ `EXCLUDED / MONITORING`
- deterministic uptime-probe provenance $\implies$ `EXCLUDED / UPTIME_PROBE`
- deterministic bot signal $\implies$ `EXCLUDED / BOT`
- deterministic crawler signal $\implies$ `EXCLUDED / CRAWLER`

If the marker is malformed, contradictory, or cannot be interpreted deterministically:
```text
state = QUARANTINED
```
unless the record is structurally impossible:
```text
state = INVALID
```

An ordinary user-agent appearance alone is insufficient to establish `NATURAL_PRODUCTION`. A user-agent heuristic may establish `BOT` or `CRAWLER` only when the classification is deterministic under the approved traffic-classification contract.

---

### 9.8 Replay and duplicate detection

Replay and duplicate detection are distinct controls.

- A **replay** is a previously observed event or observation unit being resent, reintroduced, or reproduced as a later candidate.
- A **duplicate** is repeated telemetry for an observation unit that has already been accepted as `VALID`.

The implementation must retain:
- `observation_unit_id`
- `deduplication_key`
- `event_id`
- `replay_identity` where available
- canonical prior observation reference where available

Apply these rules:
- deterministically established replay $\implies$ `EXCLUDED / REPLAY`
- deterministically established duplicate of an already accepted observation $\implies$ `EXCLUDED / DUPLICATE`
- ambiguous replay or duplicate status $\implies$ `QUARANTINED`
- distinct ETF intent attempt in the same session $\implies$ remains eligible as a separate observation unit when deterministically distinguishable

A candidate may be classified as `DUPLICATE` only when its deduplication key identifies an observation unit already accepted as `VALID`.

Repeated delivery of telemetry for a candidate that was previously `QUARANTINED` or `INVALID` must not automatically become `DUPLICATE`. It must be reprocessed under the same contract, with the original classification retained for audit.

Replay or duplicate evidence does not override structural invalidity or unresolved required provenance:
- malformed + replay-like evidence $\implies$ `INVALID`
- valid + ambiguous replay evidence $\implies$ `QUARANTINED`
- valid + deterministic replay evidence $\implies$ `EXCLUDED / REPLAY`

---

### 9.9 Natural-production classification

`NATURAL_PRODUCTION` is a positive classification. It is not equivalent to “not known to be synthetic.”

A candidate is classified as `NATURAL_PRODUCTION` only when all of the following are true:
- **N1**. `environment = production`
- **N2**. deployment identity is a permitted production deployment
- **N3**. release attribution is present, valid, and consistent with the deployment identity
- **N4**. no explicit synthetic, test, QA, CI, monitoring, developer, or replay provenance is present
- **N5**. no deterministic bot or crawler signal is present
- **N6**. session and event provenance are consistent with a real production interaction
- **N7**. the candidate is not generated by a known automated probe or replay
- **N8**. no required traffic-classification signal is contradictory

Authoritative signals include:
- explicit synthetic/test/QA marker
- CI or test-run provenance
- monitoring or uptime-probe marker
- deployment environment
- release identity
- authenticated internal developer marker
- known bot/crawler classification
- replay identifier
- session provenance

No personal identity is required. Classification must use event, session, deployment, and provenance metadata rather than unnecessary personal data.

If all positive conditions cannot be established:
```text
state = QUARANTINED
traffic_class = UNKNOWN
```

A candidate may become `VALID` only after positive natural-production classification succeeds.

---

### 9.10 Canonical natural-traffic decision procedure

After structural validation and required provenance validation, apply this procedure in the following order:

1. **Is the environment or deployment identity malformed, contradictory, or unresolved?**  
   - **YES**: `state = QUARANTINED`, `traffic_class = UNKNOWN`, `quarantine_reason = ENVIRONMENT_OR_DEPLOYMENT_UNRESOLVED`, stop (unless structural validation requires `INVALID`).
2. **Is the release identity missing, malformed, contradictory, or inconsistent with deployment identity?**  
   - **YES**: `state = QUARANTINED`, `traffic_class = UNKNOWN`, `quarantine_reason = RELEASE_UNATTRIBUTABLE`, stop.
3. **Is a deterministic synthetic, test, QA, CI, developer, monitoring, uptime-probe, bot, crawler, or replay signal present?**  
   - **YES**: `state = EXCLUDED`, assign the most specific applicable traffic class and exclusion code, stop.
4. **Is the environment local, preview, test, staging, or otherwise non-production?**  
   - **YES, and environment identity is valid/deterministic**: `state = EXCLUDED`, `traffic_class = NON_PRODUCTION`, assign specific environment exclusion code, stop.  
   - **YES, but environment identity is malformed/contradictory/unresolved**: `state = QUARANTINED`, `traffic_class = UNKNOWN`, stop (unless structural validation requires `INVALID`).
5. **Is the release identity valid but outside the authorized epoch release set?**  
   - **YES**: `state = EXCLUDED`, `traffic_class = NON_AUTHORIZED_RELEASE`, `exclusion_code = WRONG_RELEASE`, stop.
6. **Is replay or duplicate status ambiguous?**  
   - **YES**: `state = QUARANTINED`, `traffic_class = UNKNOWN` or unresolved class, stop.
7. **Are all NATURAL_PRODUCTION conditions N1–N8 satisfied?**  
   - **YES**: `traffic_class = NATURAL_PRODUCTION`, continue to epoch, boundary, and eligibility checks.
8. **Otherwise**:  
   - `state = QUARANTINED`, `traffic_class = UNKNOWN`, retain the failed natural-classification condition, stop.

The procedure must not treat absence of a synthetic marker as proof of natural traffic.

---

### 9.11 Epoch and boundary validation

A candidate may be `VALID` only when:
- the candidate is attributable to an authorized release;
- the candidate timestamp is valid;
- the candidate timestamp is within the authorized observation window;
- the candidate represents the authoritative ETF intent boundary.

Apply these rules:
- valid release, not admitted by epoch contract $\implies$ `EXCLUDED / WRONG_RELEASE`
- missing or contradictory release $\implies$ `QUARANTINED / RELEASE_UNATTRIBUTABLE`
- timestamp outside authorized epoch $\implies$ `EXCLUDED / OUTSIDE_EPOCH`
- missing or invalid timestamp $\implies$ `INVALID` when structurally impossible; otherwise `QUARANTINED` when unresolved
- non-ETF or unrelated intent $\implies$ `EXCLUDED / NON_ETF_INTENT`

Eligibility is determined at the earliest authoritative P1 intent boundary, not at the first successful downstream event.

The denominator must not be defined by:
- `ETF_COCKPIT_ARRIVAL`
- `COST_OF_OWNERSHIP_VIEW`
- successful route completion
- successful data fetch
- successful render

---

### 9.12 Denominator eligibility

A candidate may be `VALID` only if every condition below is true:

- **E1**. Represents an ETF intent attempt at the authoritative P1 intent boundary.
- **E2**. Contains a valid normalized symbol or equivalent deterministic asset identifier sufficient to identify the attempted P1 interaction.
- **E3**. Event occurred in the production environment.
- **E4**. Event is attributable to the authorized P1 release or to a release explicitly admitted by the epoch contract.
- **E5**. Event is classified as `NATURAL_PRODUCTION`.
- **E6**. Event has a valid event ID, timestamp, session ID, observation-unit identity, and canonical observation deduplication key.
- **E7**. Event is not a replay and is not a duplicate of an already accepted `ETF_INTENT_ATTEMPT`.
- **E8**. Event is not malformed, internally contradictory, or part of an impossible event sequence.
- **E9**. Event occurs within the authorized observation window.
- **E10**. Event is associated with the ETF Cockpit P1 intent boundary and not with a stock, crypto, test, administrative, preview, or unrelated route.

A candidate that fails any deterministic eligibility condition after natural-production classification is `EXCLUDED` with the applicable exclusion code.

A candidate whose eligibility cannot be determined because a required fact is missing, ambiguous, or contradictory is `QUARANTINED`.

The following are **not** eligibility conditions:
- successful asset classification
- successful ETF routing
- successful cockpit arrival
- successful component rendering
- Cost of Ownership data availability
- verified TER data
- completion of the event chain
- absence of a failure event

---

### 9.13 Canonical denominator

The prospective denominator is:

```text
P1_PROSPECTIVE_DENOMINATOR =
  count of distinct VALID ETF_INTENT_ATTEMPT observation units
  within the authorized epoch
```

The denominator is initialized only after the separate observation-readiness gate passes.

Before that gate passes:
```text
CURRENT_PROSPECTIVE_DENOMINATOR_STATE = NOT_INITIALIZED
```

The denominator includes every eligible ETF intent attempt regardless of downstream outcome.

---

### 9.14 Canonical exclusion taxonomy

Primary exclusion codes:
- `SYNTHETIC_E2E`
- `MANUAL_QA`
- `DEVELOPER`
- `CI`
- `UPTIME_PROBE`
- `MONITORING`
- `BOT`
- `CRAWLER`
- `REPLAY`
- `DUPLICATE`
- `LOCAL`
- `PREVIEW_DEPLOYMENT`
- `NON_PRODUCTION`
- `WRONG_RELEASE`
- `OUTSIDE_EPOCH`
- `NON_ETF_INTENT`
- `UNSUPPORTED_INPUT`

Reporting and accounting codes (unresolved facts produce `QUARANTINED`):
- `UNRESOLVED_TRAFFIC_CLASS`
- `RELEASE_UNATTRIBUTABLE`
- `ENVIRONMENT_OR_DEPLOYMENT_UNRESOLVED`

---

### 9.15 Canonical classification table

| Order | Condition | Primary state | Traffic class | Required action |
|---|---|---|---|---|
| **1** | Candidate is malformed, unparsable, structurally incomplete, or internally impossible | `INVALID` | `NOT_APPLICABLE` unless determinable | Stop primary-state classification; retain invalidity reason and safely recoverable diagnostics |
| **2** | Required provenance, environment, release, or traffic fact is missing, contradictory, malformed, or unresolved | `QUARANTINED` | `UNKNOWN` or unresolved class | Stop primary-state classification; retain quarantine reason |
| **3** | Structurally valid candidate has deterministic synthetic, test, QA, CI, developer, monitoring, uptime-probe, bot, crawler, or replay evidence | `EXCLUDED` | Specific applicable class | Retain provenance and exclusion code |
| **4** | Structurally valid candidate is deterministically local, preview, test, staging, or otherwise non-production | `EXCLUDED` | `NON_PRODUCTION` | Retain environment and exclusion code |
| **5** | Structurally valid candidate has a valid release outside the authorized epoch release set | `EXCLUDED` | `NON_AUTHORIZED_RELEASE` | Retain observed and authorized release identities |
| **6** | Structurally valid candidate is outside the authorized observation window | `EXCLUDED` | `OUTSIDE_EPOCH` | Retain timestamp and epoch boundary |
| **7** | Structurally valid candidate is not associated with the ETF Cockpit P1 intent boundary | `EXCLUDED` | `NON_ETF_INTENT` | Retain route, asset type, or intent reason |
| **8** | Structurally valid candidate is a deterministic duplicate of an already accepted observation unit | `EXCLUDED` | `DUPLICATE` | Retain canonical observation-unit reference |
| **9** | Structurally valid candidate is a deterministic replay of a previously observed event or observation unit | `EXCLUDED` | `REPLAY` | Retain replay identity and prior reference |
| **10** | Structurally valid candidate is positively classified as `NATURAL_PRODUCTION` and satisfies every eligibility condition | `VALID` | `NATURAL_PRODUCTION` | Create or retain distinct observation unit and count it in denominator |
| **11** | Structurally valid candidate fails a deterministic eligibility condition not covered above | `EXCLUDED` | Applicable known class | Retain failed eligibility condition and exclusion code |

---

### 9.16 Contradictory signals

Contradictory signals never resolve in favor of natural traffic:
- structural impossibility $\implies$ `INVALID`
- unresolved or contradictory required provenance $\implies$ `QUARANTINED`
- deterministically established synthetic or out-of-population provenance $\implies$ `EXCLUDED`

---

### 9.17 Failure-path handling

Downstream outcomes do not affect denominator membership.

The following are denominator observations when initial ETF intent is otherwise `VALID`:
- `ETF_INTENT` $\rightarrow$ successful classification
- `ETF_INTENT` $\rightarrow$ unresolved asset
- `ETF_INTENT` $\rightarrow$ classification conflict
- `ETF_INTENT` $\rightarrow$ successful route
- `ETF_INTENT` $\rightarrow$ route failure
- `ETF_INTENT` $\rightarrow$ runtime failure
- `ETF_INTENT` $\rightarrow$ Cost of Ownership unavailable
- `ETF_INTENT` $\rightarrow$ Cost of Ownership rendered with verified data
- `ETF_INTENT` $\rightarrow$ Cost of Ownership rendered fail-closed with unverified data

The absolute rule is:
```text
failure outcome = outcome classification
failure outcome ≠ exclusion reason
```

---

### 9.18 Accounting conservation

Required event-level conservation:
```text
RAW_EVENTS =
  VALID_EVENTS
  + QUARANTINED_EVENTS
  + EXCLUDED_EVENTS
  + INVALID_EVENTS
```

Required observation-level conservation:
```text
RAW_OBSERVATION_CANDIDATES =
  VALID_OBSERVATIONS
  + QUARANTINED_OBSERVATIONS
  + EXCLUDED_OBSERVATIONS
  + INVALID_OBSERVATIONS
```

Denominator formula:
```text
P1_PROSPECTIVE_DENOMINATOR = VALID_OBSERVATIONS
```
(after deduplication and within the authorized epoch).

---

### 9.19 Required audit fields

Every classified candidate must retain at minimum:
1. `event_id`
2. `observation_unit_id`
3. `session_id`
4. `timestamp`
5. `normalized_symbol`
6. `environment`
7. `deployment_identity`
8. `release_sha`
9. `traffic_class`
10. `classification_state`
11. `classification_reason`
12. `exclusion_code`
13. `quarantine_reason`
14. `deduplication_key`
15. `replay_identity`
16. `source_component`

Fields that do not apply must be represented explicitly as `NOT_APPLICABLE`.  
For a malformed record, unparseable fields must be represented as `UNAVAILABLE_DUE_TO_INVALID_STRUCTURE`.

---

### 9.20 Contract summary

1. **Structural Validation**: Malformed or internally impossible $\implies$ `INVALID`.
2. **Required Provenance Validation**: Missing/contradictory provenance, environment, release, or traffic fact $\implies$ `QUARANTINED`.
3. **Deterministic Exclusion**: Structurally valid candidate deterministically known to be synthetic, test, QA, CI, developer, probe, bot, crawler, replay, duplicate, non-production, wrong release, or outside epoch $\implies$ `EXCLUDED`.
4. **Positive Natural-Traffic Classification**: All positive conditions N1–N8 satisfied $\implies$ `NATURAL_PRODUCTION`. Otherwise $\implies$ `QUARANTINED`.
5. **Eligibility**: Satisfies E1–E10 $\implies$ `VALID`. Deterministic failure $\implies$ `EXCLUDED`. Unresolved fact $\implies$ `QUARANTINED`.
6. **Denominator**: Count of distinct `VALID` `ETF_INTENT_ATTEMPT` units.
7. **Outcomes**: Downstream success/failure classified separately; never removes `VALID` observation from denominator.
