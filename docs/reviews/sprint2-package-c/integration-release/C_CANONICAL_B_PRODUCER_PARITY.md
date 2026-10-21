# ARX TERMINAL — SPRINT 2 TRACK C
# CANONICAL PACKAGE B PRODUCER PARITY & CONTRACT VERIFICATION

## 1. Authoritative Package B Release Coordinates

- **Governing PR:** GitHub PR #35 (`fix/sprint2-package-b-canonical-ingestion-release`)
- **Authoritative Producer Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint2-package-b-canonical-release`
- **Authoritative Producer HEAD Commit:** **`695cd986f10b46d315ada8afa13058e81f4b4652`**
- **Authoritative Commit Message:** `fix(package-b): canonical historical SIP ingestion adapter and guarded recovery controls (#34)`
- **Authoritative Artifact Path:** `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`

> [!NOTE]
> Stale references in earlier draft notes referencing intermediate engineering commit `5e86b5b5c22dcb28cfa350e25562eeb04eac55ee` are formally corrected to `695cd986f10b46d315ada8afa13058e81f4b4652`, which represents the authoritative release HEAD associated with GitHub PR #35.

---

## 2. Bit-for-Bit Parity Verification

Track C performed an exact cryptographic comparison between the authoritative Package B artifact and the local Package C test fixture:

| Parity Property | Authoritative Package B Artifact | Package C Test Fixture | Agreement |
| :--- | :--- | :--- | :--- |
| **Location** | `finance-sprint2-package-b-canonical-release/.../PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | `tests/fixtures/package_b_canonical_v3/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | Exact Path Match |
| **Byte Size** | **`9,972 bytes`** | **`9,972 bytes`** | **100% Bit-for-Bit** |
| **SHA-256 Digest** | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | **100% Bit-for-Bit** |
| **Manifest ID** | `MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL` | `MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL` | **Identical** |
| **Feed Contract** | `tape_feed="sip"`, `adjustment_convention="split"` | `tape_feed="sip"`, `adjustment_convention="split"` | **Identical** |
| **Window** | `2025-08-27` to `2026-10-19` (288 NYSE sessions) | `2025-08-27` to `2026-10-19` (288 NYSE sessions) | **Identical** |
| **Authorization State** | `PROPOSED_NOT_AUTHORIZED` (`is_authorized=false`) | `PROPOSED_NOT_AUTHORIZED` (`is_authorized=false`) | **Identical** |

---

## 3. Candidate Cohort & Association Parity

All 4 pilot candidates declared in the canonical manifest match the Package A adopted master records:

```json
[
  {
    "symbol": "AAPL",
    "durable_security_id": "ARX_SEC_0B4C728FE0BE",
    "listing_association_id": "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
    "operating_mic": "XNAS",
    "mandated_action": "REPLACE",
    "expected_bars": 288
  },
  {
    "symbol": "GOOG",
    "durable_security_id": "ARX_SEC_A11A24D442E2",
    "listing_association_id": "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17",
    "operating_mic": "XNAS",
    "share_class": "CLASS_C",
    "mandated_action": "REPLACE",
    "expected_bars": 288
  },
  {
    "symbol": "GOOGL",
    "durable_security_id": "ARX_SEC_B7EF9ED7A35A",
    "listing_association_id": "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17",
    "operating_mic": "XNAS",
    "share_class": "CLASS_A",
    "mandated_action": "REPLACE",
    "expected_bars": 288
  },
  {
    "symbol": "ABBV",
    "durable_security_id": "ARX_SEC_BFA51AC4C0B5",
    "listing_association_id": "ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19",
    "operating_mic": "XNYS",
    "mandated_action": "BACKFILL",
    "expected_bars": 288
  }
]
```

- **Validation:** Every `listing_association_id` conforms strictly to the canonical regex `^ASSOC_ARX_SEC_[0-9A-F]{12}_[A-Z0-9]{4}_[A-Z0-9\._]+_\d{4}-\d{2}-\d{2}$`.
- **Operating MICs:** Correctly mapped to `XNAS` (Apple, Alphabet) and `XNYS` (AbbVie).

---

## 4. Reconfirmation of Superseded Manifest Invalidation

Track C enforces zero tolerance for superseded manifests. The following predecessors are explicitly registered in `SUPERSEDED_PREDECESSOR_DIGESTS` in `analyst_dashboard/universe/isolated_publication.py` and fail closed in all consumer execution:

1. **Predecessor 1 (`65F5D9AB...`, 3,489 bytes):** Early Workstream A prototype manifest. Completely removed from the filesystem. Attempting to load or specify this digest immediately raises `PolicyViolationError`.
2. **Predecessor 2 (`8923DAD9...`, 13,833 bytes):** Early Workstream B manifest. Invalidated due to the historical boundary defect (`2025-09-15` start date leaving 12 unprovenanced sessions). Attempting to load or specify this digest immediately raises `PolicyViolationError`.
