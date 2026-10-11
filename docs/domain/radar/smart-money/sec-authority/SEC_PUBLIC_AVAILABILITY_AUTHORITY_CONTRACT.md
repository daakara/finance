# ARX Terminal — SEC Public Availability Authority Contract (Epoch 001)

**Document Identifier:** `SEC_PUBLIC_AVAILABILITY_AUTHORITY_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. Separation of Public Availability from Feed Receipt

The timestamp at which an institution receives a PDS packet is not the timestamp of first public availability. SEC Fair Access policy mandates that filings appear on SEC.gov before or concurrently with PDS feed dissemination.

### Distinct Timestamp Registry:
- **`SEC_ACCEPTANCE_AT`**: When EDGAR accepted the filing into internal processing.
- **`FIRST_PROVEN_METADATA_PUBLIC_AT`**: Earliest verified contemporaneous observation of filing metadata on public endpoints (RSS, Submissions API, Latest Filings).
- **`FIRST_PROVEN_CONTENT_PUBLIC_AT`**: Earliest verified contemporaneous observation of full XML/text on canonical document endpoints.
- **`FIRST_PROVEN_DECISION_INPUT_COMPLETE_AT`**: Earliest observation exposing all fields required by the active quantitative decision contract.
- **`PDS_DELIVERED_AT`**: Authenticated subscriber DRS receipt time.
- **`EXACT_FIRST_PUBLIC_AVAILABLE_AT`**: Strictly `UNKNOWN` unless a native SEC public publication receipt is provided.

---

## 2. Public Observation Channel Classes

| Channel Class | Primary Evidence Role | Notes |
|---|---|---|
| `RSS_FEED` | Metadata Public Observation | Low-latency discovery list |
| `LATEST_FILINGS` | Metadata Public Observation | Web HTML index |
| `SUBMISSIONS_API` | Structured Metadata Observation | Daily JSON submissions index |
| `CANONICAL_FILING_ENDPOINT` | Content Public Observation | Complete XML / submission text |
| `PDS` | Delivery & Population Evidence | Push feed audit population |

---

## 3. Fail-Closed Transport Rules

- HTTP `403`, `429`, `5xx`, timeout, or network error is strictly classified as **`TRANSPORT_INDETERMINATE`**. It is **never** proof that a filing does not exist.
- Truncated downloads (`bytes_received < Content-Length`) are classified as **`TRANSPORT_INCOMPLETE`**, never source content deficiency.
- Accession mismatches in body content are classified as **`IDENTITY_INVALID`**.
- Conservative knowledge time binds strictly to **`response_completed_at`**, never `request_started_at`.
