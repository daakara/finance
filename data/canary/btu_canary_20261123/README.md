# ARX Known-Positive Form 4 Canary (BTU) — Execution & Persistence Receipt

Verdict: **PASS for retained-response discovery, XML parsing, isolated SQLite idempotency, and canonical local PostgreSQL persistence mechanics (Gates 1–4). BTU production dispatch, production authorization, and scoring integration: NOT_EXERCISED. The existing AAPL/NVDA production trial remains distinct and unchanged.**

---

## 1. Source Identity & Scoped Parity

- **Full Checkout SHA**: `f1c5e0d33cda1d73d959a43fef1c5b5044bfae56` (`git rev-parse HEAD` on branch `fix/sec-insider-flow-presentation`).
- **Worker Release Parity Anchor**: `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.
- **Scoped File Parity**: Only the 8 discovery, parser, and persistence files listed below were tested and verified against worker release `2eccc64a`. The entire repository checkout is **not** claimed to equal `2eccc64a`; only these specific files match with **0 diffs**:

| Subsystem File Path | SHA-256 Digest | Bytes | Diff vs Worker Release `2eccc64a` |
|---|---|:---:|:---:|
| `analyst_dashboard/smart_money/continuous_pipeline.py` | `0bb10186e1a9e04004f1be9bd3dbd65cb5c60b9906d648a11983c2c167baf0d2` | 105,626 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/form4_parser.py` | `68d95e24feaa72cefad219b95c421cbedb85fda06a94fc9772e8473564c07031` | 16,811 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/form4_ingestion.py` | `4993be447c56a4c7410061bcd988b1b1da6f6beb8597e9e7839e47138869f46b` | 59,806 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/worker.py` | `c32f01da6d774cb4dad55fd5ecfd2e3108cf6764a127fc2f2d509819075b18dc` | 50,210 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/reconciliation_engine.py` | `3604c70a56a3b39c01fe86b92d5c5dd4dd2d88f3df275efa9ff0c1d8d0f19771` | 22,070 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/sec_source_contract.py` | `4882938f97dfdc1fb2caa8947898dfa1160cf33335f18b130518ce7ee6b329c8` | 29,229 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/storage/postgres_schema.py` | `040499fde80ee3874a748b2256e9d399ef7bf51fc11c7f46a31c7920e518557e` | 13,414 | 0 diffs (empty) |
| `analyst_dashboard/smart_money/storage/postgres_repository.py` | `9894afe621b287cc77906d8865a196b696aa3bb2dc50ac885443e1395a8af9cd` | 131,782 | 0 diffs (empty) |

---

## 2. Source-Confirmed Target & Retained Response Custody

- **Issuer**: Peabody Energy Corp (`BTU` / CIK `0001064728`).
- **Accession**: `0001570382-26-000015`, Form 4, filed November 23, 2026.
- **Accepted Datetime**: `2026-11-23T13:01:04.000Z` (14:01:04 Berlin).
- **Reporting Owner**: Grech James C.
- **Transaction Date**: November 20, 2026 (transaction date decoupled from filing date).
- **Transaction Details**: 1 non-derivative transaction, code `P`, 4,411 shares @ $15.99 per share, total value $70,531.89 USD.
- **Official Filing Index**: `https://www.sec.gov/Archives/edgar/data/1570382/000157038226000015/0001570382-26-000015-index.htm`

### Retained Source Bytes (Unchanged Across Replays)
| Retained Filename | Bytes | SHA-256 Digest | Role in Replay |
|---|:---:|---|---|
| `peabody_0001570382-26-000015.xml` | 3,971 | `7a9ce89854a10df267715b2b0e9c4c6e11ff12553c73aa4e296abd3e4ed9e530` | Bare Form 4 XML response (resolved fallback) |
| `submissions.json` | 159,337 | `cb37c8ec4f38cc9ee9795ec165c6f0ec876f59079897240c685c45afa6ffeb2d` | Submissions catalog for CIK `0001064728` |
| `primary_response.raw` | 16,089 | `53b35e201c2e9ecffbf1a58f16d1ed8721299233ac038e43559818b115cd414f` | Rendered HTML from XSL path (triggers fallback) |
| `custody.json` | 273 | `35de37432ff524e627e375bf5d376f9ffc22f69791bf3acc9856d5e053df5c34` | XML custody metadata |
| `catalog_custody.json` | 236 | `ef8d57073b556375c7dac7fa05c2343828e729242f7db72f52422c960742bd97` | Catalog custody metadata |
| `primary_custody.json` | 315 | `1a17748db8d1be3a71a8aae24eb18520278389966aef164911974646015b475a` | HTML primary response custody metadata |

---

## 3. Package Provenance & Archive Hash Distinction

- **Original Ingestion Archive**:
  - Name: `ARX_BTU_KNOWN_POSITIVE_CANARY_2026-11-23.zip`
  - SHA-256: `fc2b14b61c8f79f1b33df212727fd78c5778f3cc2181b1d2d5b74f68064b8977`
  - Byte Size: 176,966 bytes
  - Status: Contains initial SQLite-only replay script and retained response bytes.
- **Updated Canonical PostgreSQL Package**:
  - Name: `ARX_BTU_KNOWN_POSITIVE_CANARY_POSTGRES_2026-11-23.zip`
  - Contains: All original source bytes, custody receipts, plus `postgres_replay.py`, updated `result.json`, updated `README.md`, and complete `bundle_manifest.json`.

---

## 4. Replay and Acceptance Gates

| Gate | Result | Evidence / Direct SQL Assertion |
|---|---|---|
| **Today's Form 4 independently identified** | PASS | Official SEC EDGAR index and retained submissions entry. |
| **Catalog issuer/ticker/accession match** | PASS | BTU, CIK `0001064728`, exact accession `0001570382-26-000015` and UTC acceptance. |
| **Raw-source integrity** | PASS | SHA-256 and size receipts for all retained responses verified 100%. |
| **Discovery from retained catalog** | PASS | `_fetch_filings_from_sec` finds 1 accession after explicit prospective boundary. |
| **Rendered-document fallback** | PASS | Retained HTML triggers actual method's clean XML fallback to unstyled archive XML. |
| **Parser/source-field parity** | PASS | One `P` transaction: 4,411 shares @ $15.99, tx date 2026-11-20, Grech James C. |
| **Isolated storage (SQLite primitive)** | PASS | `Form4IngestionManager.ingest_accession` writes 1 filing, 1 parsed transaction. |
| **Known-accession discovery replay** | PASS | Returns zero new accessions (`[]`) and issues only retained-catalog read. |
| **Five identical ingestion replays** | PASS | Row counts remain 1 filing, 1 transaction, 1 availability record; 0 duplicates. |
| **Changed content at same accession** | PASS | `SOURCE_INTEGRITY_CONFLICT` raised; transaction rolled back with 0 mutations. |
| **PostgreSQL canonical persistence mechanics** | PASS | Canonical Step 7 in disposable PostgreSQL 16.2: 1 raw filing, 1 parsed tx, 1 availability record; 6 reconciliation receipts after initial run + 5 replays. |
| **BTU Production Dispatch & Authorization** | NOT_EXERCISED | BTU production dispatch, production authorization and scoring integration were not exercised or authorized. |
| **AAPL/NVDA Production Trial** | SEPARATE_AND_UNCHANGED | Separate AAPL/NVDA production trial remains distinct and unchanged. |
| **Historical completeness / scoring eligibility** | NOT_ESTABLISHED | Coverage remains uncertified (`PARTIAL_UNCERTIFIED`); scoring remains held (`HELD`). |

---

## 5. Direct Local PostgreSQL Verification Results

- **PostgreSQL Version**: `PostgreSQL 16.2 on aarch64-apple-darwin23.5.0`
- **Initial Execution Census**:
  - `form4_raw_filings`: 1
  - `form4_parsed_transactions`: 1
  - `sm_source_availability_ledger`: 1
  - `sm_reconciliation_receipts`: 1
- **Post-5x Replay Census**:
  - `form4_raw_filings`: 1 (0 duplicate filings created; timestamps and hashes identical)
  - `form4_parsed_transactions`: 1 (0 duplicate transactions; record key and amounts identical)
  - `sm_source_availability_ledger`: 1 (0 duplicate availability entries)
  - `sm_reconciliation_receipts`: 6 (1 initial receipt + 5 execution audit receipts)
- **Conflicting Content Rollback**:
  - Injected conflicting XML hash (`0000000000...`) under accession `0001570382-26-000015`.
  - Result: `RuntimeError: SOURCE_INTEGRITY_CONFLICT`.
  - Atomic transaction rolled back cleanly. Table census remained strictly 1 filing, 1 transaction, 1 availability record, 6 receipts. Zero partial writes committed.

---

## 6. Reproduce Offline

To reproduce offline from the repository root:
```bash
PYTHONPATH=. python3 data/canary/btu_canary_20261123/postgres_replay.py
```
To reproduce the legacy SQLite run:
```bash
PYTHONPATH=. python3 data/canary/btu_canary_20261123/sqlite_replay.py
```

Both scripts read only retained HTTP bytes, strictly prohibit outbound HTTP, and run against isolated ephemeral databases.

---

## 7. Scope Boundaries & Remaining Gaps

1. **Local Mechanics Only**: This test proves that the discovery fallback, Form 4 parser, and canonical PostgreSQL persistence mechanics work as designed for BTU Form 4 filings without schema errors or partial writes.
2. **BTU Dispatch NOT_EXERCISED**: BTU production dispatch, production authorization, and scoring integration were not exercised or authorized.
3. **AAPL/NVDA Trial Separate**: The existing AAPL/NVDA production trial is separate and remains unchanged.
4. **Historical Coverage Uncertified**: Historical coverage across the governed trial universe remains uncertified; Smart Money scoring integration remains held under `HOLD_SEC_001` / `PHASE_3_HOLD`.
