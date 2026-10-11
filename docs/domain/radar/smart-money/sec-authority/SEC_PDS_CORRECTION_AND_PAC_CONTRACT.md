# ARX Terminal — SEC PDS Post-Acceptance Correction (PAC) Contract (Epoch 001)

**Document Identifier:** `SEC_PDS_CORRECTION_AND_PAC_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. Post-Acceptance Correction (PAC) Architecture

In SEC EDGAR, filings are legal records and cannot be overwritten in place. When administrative errors occur, SEC staff executes a Post-Acceptance Correction (PAC).

1. **Dissemination Artifact**:
   - Disseminated as discrete files bearing the `*.pc` extension.
   - Priority code: `3`.
   - Contains parent `<ACCESSION-NUMBER>` and `<TIMESTAMP>` indicating correction execution time.

2. **Permissible PAC Actions**:
   - `HEADER_TAG_UPDATE`: Modifies header tags such as CIK, company name, or fiscal dates.
   - `DELETION_NOTICE`: Withdraws mistakenly submitted filings from active indexation.
   - `FORM_TYPE_ALTERATION`: Reclassifies an erroneous form type.

3. **Hard Invariant: Document Text Immutability**:
   > **`PAC_DOES_NOT_UPDATE_DOCUMENT_TEXT = YES`**
   A PAC **never** alters the text, numbers, or footnote disclosures of a submitted Form 4. Text modifications strictly require an official Form 4/A filing by the reporting owner.

4. **Multiple PACs Permitted**:
   A single accession may be subject to multiple PAC actions over its lifecycle. Each PAC is tracked with independent lineage.
