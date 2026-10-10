# ARX Terminal — Smart Money Canonicalization Standard Contract

**Canon ID:** `SM_CERT_CANON_V1`  
**Applies To:** All Smart Money certification payloads, proofs, receipt envelopes, manifests, and hashes.  
**Implementation:** `analyst_dashboard/smart_money/canonicalization.py`  
**Status:** `FROZEN_LOCAL_ENGINEERING_AUTHORITY`

---

## 1. Motivation & Determinism Requirement

Cryptographic verification across distributed or multi-platform environments requires bit-for-bit determinism. Any variation in character encoding, object key ordering, whitespace separators, timestamp precision, or numeric representation causes catastrophic hash divergence.

`SM_CERT_CANON_V1` guarantees that two agents or systems processing identical logical objects always compute identical SHA-256 hashes.

---

## 2. The Nine Canonical Rules

### Rule 1: Unicode Normalization (NFC)
All string values and dictionary keys are recursively normalized using **Unicode Normalization Form C (NFC)** prior to serialization.

### Rule 2: Lexicographical Key Sorting
All object/dictionary keys are sorted strictly ascending in ASCII byte order (`sort_keys=True`).

### Rule 3: Canonical Sorting of Set-Like Arrays
Collections representing mathematical sets (such as partition CIKs, accession number lists, or receipt hash collections) must be deduplicated and sorted ascending by their canonical identifier before serialization.

### Rule 4: Preservation of Ordered Sequences
Sequences where order carries semantic meaning (such as audit trace logs or monotonic gap event history) preserve their explicit sequence indices.

### Rule 5: UTC RFC3339 Fixed Precision Timestamps
All timestamps must be ISO 8601 / RFC 3339 strings in UTC with uppercase `Z` designator. Microseconds/nanoseconds are prohibited in certification envelopes; standard precision is fixed millisecond (`YYYY-MM-DDTHH:MM:SS.000Z`) or whole second (`YYYY-MM-DDTHH:MM:SSZ`). Naive datetimes fail closed with `ValueError`.

### Rule 6: Explicit Nulls
Optional fields that are absent or unassigned must serialize explicitly as JSON `null`, rather than omitting the key or substituting arbitrary placeholders.

### Rule 7: Strict Float Prohibition
Floating-point numbers are strictly forbidden in any certification or proof payload:
$$\text{Floating-point values} \implies \text{ValueError}(\texttt{"FLOAT\_PROHIBITED\_IN\_CERTIFICATION"})$$
*Rationale:* IEEE 754 floating-point representations exhibit non-deterministic rounding and serialization across CPU architectures, compilers, and language runtimes. Financial amounts and quantities must be represented as exact integers (e.g. cents, raw share counts) or fixed-precision strings.

### Rule 8: Strict Deserialization Schema Enforcement
Deserializers must reject unexpected or unknown fields to prevent shadow data injection.

### Rule 9: Prohibition of Duplicate Keys
JSON payloads containing duplicate keys are strictly rejected.

---

## 3. Formatting & Serialization Parameters

When emitting canonical JSON, the serializer uses minimal delimiters with zero extraneous whitespace:
- Separators: `(",", ":")`
- Encoding: UTF-8 without Byte Order Mark (BOM)
- Hashing: Canonical UTF-8 bytes are passed directly into `hashlib.sha256()`.
