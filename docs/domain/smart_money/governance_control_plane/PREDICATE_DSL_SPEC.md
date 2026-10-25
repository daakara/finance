# ARX Terminal — Smart Money
# Predicate Domain Specific Language (DSL) Specification V1

**Identifier:** `urn:arx:governance:predicate_dsl_spec:v1`  
**Status:** RATIFIED_NORMATIVE  
**DSL Version:** `ARX-PREDICATE-DSL-V1`  
**Normalization:** `ARX-GOV-NORMALIZE-V1`  
**Canonicalization:** `RFC8785-JCS`

---

## 1. Overview & Constraints

The **ARX Predicate DSL** is a restricted, declarative expression language designed for deterministic, fail-closed policy evaluation within the Smart Money Governance Control Plane.

### Strict Prohibitions:
* **No Arbitrary Code Execution:** No Python bytecode, JavaScript, shell script, or dynamic `eval`/`exec`.
* **No Unbound Runtime Reads:** Prohibits calling `datetime.now()`, inspecting environment variables, reading arbitrary filesystem paths, invoking network endpoints, or querying mutable process states.
* **No Side Effects:** Predicate evaluation is purely functional and referentially transparent.
* **No Non-Deterministic Operations:** Evaluation results are mathematical functions of the normalized AST and the canonical fact context.

---

## 2. Operator Grammar & Syntax

The DSL supports three strictly defined operator families:

### 2.1 Logical Operators
* **`all` (Conjunction / AND):** Accepts a non-empty list of expressions. Evaluates to:
  * `FALSE` if ANY child evaluates to `FALSE`.
  * `CONFLICT` if ANY child evaluates to `CONFLICT` (and no child is `FALSE`).
  * `UNKNOWN` if ANY child evaluates to `UNKNOWN` (and no child is `FALSE` or `CONFLICT`).
  * `TRUE` if ALL children evaluate to `TRUE`.
* **`any` (Disjunction / OR):** Accepts a non-empty list of expressions. Evaluates to:
  * `TRUE` if ANY child evaluates to `TRUE`.
  * `CONFLICT` if ANY child evaluates to `CONFLICT` (and no child is `TRUE`).
  * `UNKNOWN` if ANY child evaluates to `UNKNOWN` (and no child is `TRUE` or `CONFLICT`).
  * `FALSE` if ALL children evaluate to `FALSE`.
* **`not` (Negation):** Accepts a single child expression. Evaluates to:
  * `TRUE` if child is `FALSE`.
  * `FALSE` if child is `TRUE`.
  * `UNKNOWN` if child is `UNKNOWN`.
  * `CONFLICT` if child is `CONFLICT`.

### 2.2 Comparison Operators
Accepts `{left: Operand, right: Operand}`, where an `Operand` is either:
* `{fact: "fact_name"}`: Reference to a declared fact in the context.
* `{literal: value}`: A literal primitive value (`string`, `number`, `boolean`, `null`, `list`).

Supported comparisons:
* **`eq`**: Tests equality ($left == right$).
* **`neq`**: Tests inequality ($left \neq right$).
* **`in`**: Tests set containment ($left \in right$, where right is a list or set).
* **`gt`**: Greater than ($left > right$, numbers only).
* **`gte`**: Greater than or equal ($left \ge right$, numbers only).
* **`lt`**: Less than ($left < right$, numbers only).
* **`lte`**: Less than or equal ($left \le right$, numbers only).

### 2.3 State Operators
* **`is_established`**: Accepts `{fact: "fact_name"}`. Evaluates to `TRUE` if the fact is present in context with `establishment: ESTABLISHED`, else `FALSE`.
* **`is_current`**: Accepts `{fact: "fact_name"}`. Evaluates to `TRUE` if the fact is present with `validity: CURRENT`, else `FALSE`.
* **`has_capability`**: Accepts `{capability: "capability_name"}`. Evaluates to `TRUE` if the capability is authoritatively granted in the context, else `FALSE`.

---

## 3. Truth Tables for 4-Valued Partial Logic

The evaluation engine adheres to the following truth tables:

### 3.1 Logical Disjunction (`any` / OR)

| Left $\setminus$ Right | `TRUE` | `FALSE` | `UNKNOWN` | `CONFLICT` |
|:---:|:---:|:---:|:---:|:---:|
| **`TRUE`** | `TRUE` | `TRUE` | `TRUE` | `TRUE` |
| **`FALSE`** | `TRUE` | `FALSE` | `UNKNOWN` | `CONFLICT` |
| **`UNKNOWN`** | `TRUE` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` / `CONFLICT` |
| **`CONFLICT`** | `TRUE` | `CONFLICT` | `UNKNOWN` / `CONFLICT` | `CONFLICT` |

*Note on short-circuiting:* `TRUE OR UNKNOWN = TRUE`. An unknown on one branch does not block disjunction when another branch is authoritatively `TRUE`.

### 3.2 Logical Conjunction (`all` / AND)

| Left $\setminus$ Right | `TRUE` | `FALSE` | `UNKNOWN` | `CONFLICT` |
|:---:|:---:|:---:|:---:|:---:|
| **`TRUE`** | `TRUE` | `FALSE` | `UNKNOWN` | `CONFLICT` |
| **`FALSE`** | `FALSE` | `FALSE` | `FALSE` | `FALSE` |
| **`UNKNOWN`** | `UNKNOWN` | `FALSE` | `UNKNOWN` | `UNKNOWN` / `CONFLICT` |
| **`CONFLICT`** | `CONFLICT` | `FALSE` | `UNKNOWN` / `CONFLICT` | `CONFLICT` |

*Note on short-circuiting:* `FALSE AND UNKNOWN = FALSE`. A known false requirement fails conjunction immediately without blocking on unrelated unknowns.

### 3.3 Logical Negation (`not`)

| Input | `not(Input)` |
|:---:|:---:|
| `TRUE` | `FALSE` |
| `FALSE` | `TRUE` |
| `UNKNOWN` | `UNKNOWN` |
| `CONFLICT` | `CONFLICT` |

---

## 4. AST Normalization & Hashing

1. **Normalization Pipeline:**
   * Lists in commutative operators (`all`, `any`) are sorted deterministically by their JCS JSON canonical string representation.
   * Dict keys are sorted lexicographically by UTF-8 code point.
   * Unused whitespace and formatting variations are eliminated.
2. **Hash Computation:**
   $$\text{predicate\_hash} = \text{SHA-256}(\text{JCS}(\text{domain\_normalize}(\text{normalized\_ast})))$$
