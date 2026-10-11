# ARX Terminal — Public Availability Channel Semantics (Epoch 001)

**Document Identifier:** `PUBLIC_AVAILABILITY_CHANNEL_SEMANTICS`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Observation Authority  
**Operating Mode:** `MULTI_CHANNEL` + `TRANSPORT_AWARE`

---

## 1. Multi-Channel SEC Endpoint Taxonomy

ARX Terminal models official public SEC dissemination channels independently. No artificial universal channel-precedence ordering is assumed ($RSS \not> Content$, $Content \not> RSS$).

| Channel Code | Endpoint Class | Observation Role | Proves |
|---|---|---|---|
| `RSS` | RSS / Atom Feeds | `METADATA_PUBLIC_OBSERVATION` | SEC metadata exposure in public syndication feed |
| `LATEST_FILINGS` | EDGAR Current Filings | `METADATA_PUBLIC_OBSERVATION` | Submission presence in SEC web search index |
| `FILING_INDEX` | Accession Directory / Index | `CONTENT_OR_STRUCTURAL_PUBLIC_OBSERVATION` | Directory existence and manifest file presence |
| `COMPLETE_SUBMISSION_TEXT` | `.txt` Complete Submission | `CONTENT_PUBLIC_OBSERVATION` | Full raw submission package availability |
| `PRIMARY_FORM_DOCUMENT` | HTML / Primary Form Doc | `CONTENT_PUBLIC_OBSERVATION` | Primary human-readable document availability |
| `OWNERSHIP_XML` | Structured XML Doc | `CONTENT_PUBLIC_OBSERVATION` | Machine-readable Form 4 XML content availability |
| `SUBMISSIONS_API` | Submissions JSON API | `STRUCTURED_METADATA_OBSERVATION` | Entity historical metadata indexing |
| `PDS` | Private Contractor Stream | `OUT_OF_SCOPE_FOR_FIRST_PUBLIC_OBSERVER` | Internal contractor feed; non-authority for public web timing |

---

## 2. Public Observation Classifications

Observations are mapped to ten exhaustive, non-collapsible classes:

1. `METADATA_PUBLIC`: Verified accession and issuer metadata observed in public syndication/index.
2. `CONTENT_PUBLIC`: Complete response retrieved proving filing content is accessible on SEC.gov.
3. `DECISION_INPUT_COMPLETE`: Content observation satisfying all required dependencies of the frozen decision contract.
4. `ENDPOINT_NOT_PRESENT`: HTTP 404 on the specific endpoint observed. Does **not** imply SEC-wide absence.
5. `TRANSPORT_INDETERMINATE`: HTTP 403, 429, 5xx, socket timeout, or DNS failure. Treated as inconclusive network state.
6. `IDENTITY_INVALID`: HTTP 200 returned, but accession, CIK, or form type does not match expected target.
7. `TRANSPORT_INCOMPLETE`: Premature socket closure, decompression failure, or `Content-Length` header mismatch.
8. `STRUCTURALLY_INVALID`: XML malformed or failing schema validation.
9. `DECISION_INPUT_INCOMPLETE`: Structurally valid document that lacks required decision-input fields (e.g., missing price/shares).
10. `CONTENT_VERSION_CONFLICT`: Competing observations of the same endpoint returning irreconcilable content states.

---

## 3. Negative Observation Semantics

1. **Local Endpoint Boundary**: A 404 or absence on RSS proves only that the specific endpoint did not present the item at that moment. It **never** proves `SEC_WIDE_NOT_PUBLIC = YES`.
2. **Network/Rate Limit Independence**: HTTP 429 (rate limited) and HTTP 403 (WAF block) are classified as `TRANSPORT_INDETERMINATE`, never as absence.
3. **Cross-Channel Positive Precedence**:
   $$\text{RSS PRESENT @ } T_1 \implies \text{METADATA\_PUBLIC PROVEN @ } T_1$$
   $$\text{Canonical URL 404 @ } T_2 \implies \text{CONTENT\_PUBLIC NOT YET PROVEN @ } T_2$$
   The negative result at $T_2$ **cannot** erase the positive metadata proof established at $T_1$.
   `CROSS_CHANNEL_NEGATIVE_OVERRIDES_POSITIVE = NO`.
