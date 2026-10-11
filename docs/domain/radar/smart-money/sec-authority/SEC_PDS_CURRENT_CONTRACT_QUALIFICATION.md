# ARX Terminal — SEC PDS Current Contract Qualification (Epoch 001)

**Document Identifier:** `SEC_PDS_CURRENT_CONTRACT_QUALIFICATION`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  
**Status:** **`PASS / CONTRACT QUALIFIED`**  

---

## 1. Authoritative SEC Documents Qualified

This qualification evaluates the CURRENT official SEC documentation governing the EDGAR Public Dissemination Service (PDS) without reliance on remembered narrative or obsolete historical specifications:

| Doc ID | Title | Version | Update Date | SHA-256 |
|---|---|---|---|---|
| `SEC_DOC_PDS_TECH_SPEC_2025` | PDS Technical Specification | 2.8 | 2025-05-19 | `12cc2b39111b6c42e1b23f9eea5273bed046cbbf8ebf75a2289f1527720d57eb` |
| `SEC_DOC_PDS_NEW_SUBSCRIBER_2025` | PDS New Subscriber Document | 2025.1 | 2025-08-01 | `e8fda8688657a420209622c088c1f1739b24eca5fb71948e9bb5e91251962d96` |
| `SEC_DOC_PDS_SERVICE_PAGE` | SEC PDS Service Information Page | 2025-CURRENT | 2025-08-01 | `5730aa86c068e40cf7301e1c71c76ec5705a9ab3c5018b109cf52311416588e0` |
| `SEC_DOC_PUBLIC_WEBSITE_AVAILABILITY_GUIDANCE` | Dissemination & Web Fair Access Policy | 2025.1 | 2025-05-19 | `4a4777f36ebaf5009f04c72d4f78ac75402f9c1deb32b5c90320743392f63a39` |
| `SEC_DOC_EDGAR_FILER_MANUAL_PAC_GUIDANCE` | Filer Manual PAC Rules | EDGAR Release 25.1 | 2025-05-19 | `c8e9c25cecc6d4d95052d2a3a8e9bf7cb676ddb2a71ccab35d72111b117dcabc` |

Manifest Content Hash: `3c3275012b57401be5d88196de149f065cfb6de71c9ea076d27b2befd9a3266c`

---

## 2. Artifact Class Resolution

The current PDS specification defines four discrete file classes disseminated across the PDS feed during an EDGAR business day:

1. **`*.dissem`**: Live, accepted electronic public submissions. Contains the complete submission envelope, header tags, primary document, and exhibits.
2. **`*.paper`**: Paper submissions converted into electronic stub tracking records.
3. **`*.pr`**: Official documents (such as `COVER` or `CORRESP`) disseminated following staff review or approval.
4. **`*.pc`**: Post-Acceptance Corrections (PAC) disseminated when administrative metadata updates, form changes, or deletion notices are processed.

---

## 3. Daily End-of-Day (EOD) Audit Analysis

### 3.1. EOD Filename Pattern
- Format: `dissem2_MM-DD-YYYY.EOD` (22 characters) or `dissem_MM-DD-YYYY.EOD` (21 characters).
- Example: `dissem2_10-09-2026.EOD`.

### 3.2. EOD Enumeration Scope
The current technical specification explicitly states:
> *"The EOD file contains an ASCII listing of all disseminated submissions (including `*.dissem`, `*.paper`, `*.pr`, and `*.pc` files) sent from EDGAR to the PDS site between 6:00 a.m. and the end of the business day."*

Therefore:
- `CURRENT_EOD_ENUMERATES_DISSEM = YES`
- `CURRENT_EOD_ENUMERATES_PAPER = YES`
- `CURRENT_EOD_ENUMERATES_PR = YES`
- `CURRENT_EOD_ENUMERATES_PC = YES`

### 3.3. Terminal Marker Semantics
The technical specification establishes:
- Transmission window: Disseminated between 10:01 p.m. and 10:15 p.m. Eastern Time.
- Terminal Meaning: **Final transmission from EDGAR for the business day.** Its presence confirms that EDGAR has concluded dissemination for that business day.
- `EOD_IS_LAST_TRANSMISSION_FOR_BUSINESS_DAY = YES`

---

## 4. Contract Conflict Audit

No conflicting clauses exist between the Technical Specification, the New Subscriber Document, and the Service Information Page regarding EOD enumeration or terminal semantics:
- `SOURCE_CONTRACT_CONFLICT = NO`
- `PDS_PROTOCOL_QUALIFICATION = PASS`
- `PDS_EOD_CAN_SUPPORT_SOURCE_NATIVE_CLOSED_INTERVAL = YES`
