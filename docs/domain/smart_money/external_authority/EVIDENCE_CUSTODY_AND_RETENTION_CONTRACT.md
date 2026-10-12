# ARX TERMINAL — SMART MONEY
## EXTERNAL AUTHORITY ARCHITECTURE CONTRACT
### ARTIFACT 3: EVIDENCE CUSTODY AND RETENTION CONTRACT

**Document Identity**: `urn:arx:contract:smart_money:evidence_custody:v1`  
**Protocol Version**: `1.0.0`  
**Governance Scope**: Theorem D Evidence Custody, WORM Immutability, and Retention Mechanics  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE FREEZE  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Durability Invariants

This contract specifies the physical and cryptographic requirements governing raw SEC evidence custody. It establishes the WORM (Write Once, Read Many) compliance retention model, immutable metadata custody tuples, quarantine protocols, tamper-detection verification, and disaster-recovery boundaries.

The core durability invariants are:
1. **Verbatim Byte Preservation**: The exact stream of octets received from SEC EDGAR servers is persisted unmodified. No preprocessing, compression, stripping, or re-encoding is permitted prior to hash anchoring.
2. **True Delete Resistance**: Objects are protected by storage-layer Compliance Mode Object Locks. Neither software runners, operators, nor cloud account root users can delete objects during the active retention epoch.
3. **Cryptographic Self-Attestation**: Content identity is derived strictly from SHA-256 content hashes. Any mismatch between retrieved bytes and anchored hashes constitutes an immediate security incident.
4. **Permanent Lineage**: Downstream processing errors quarantine the interpretation, but never erase or mutate the underlying raw evidence.

---

### 2. Evidence Lifecycle & State Transition Model

Evidence progresses through a deterministic, directed state machine:

```
                    EVIDENCE STATE TRANSITION GRAPH
                    
   [SEC API]
       |
       v
   [STAGED] -------- (WORM Storage Write Failure) -------> [DISCARDED_STAGING]
       |
       | (Successful WORM Write + SHA-256 Match)
       v
  [DURABLE_RAW]
       |
       +-------- (Validation Failure / Downstream Error) ----> [QUARANTINED]
       |                                                            |
       | (Full Candidate Validation Pass)                           v
       v                                                    [AUDIT_LEDGERED]
  [VALIDATED]
       |
       | (Atomic State Transaction Success)
       v
[REFERENCED_BY_ADMISSION]
       |
       | (Subsequent Non-Semantic Metadata Correction)
       v
[SUPERSEDED_METADATA]  <--- (Raw Bytes Remain Immutable)
```

#### 2.1 State Transition Matrix

```
+---------------------------------------------------------------------------------------------------------------------+
|                                            EVIDENCE STATE TRANSITIONS                                               |
+--------------------------+--------------------------+---------------+-----------------------------------------------+
| Source State             | Target State             | Validity      | Transition Enforcement Rule                   |
+--------------------------+--------------------------+---------------+-----------------------------------------------+
| STAGED                   | DURABLE_RAW              | ALLOWED       | PutObject with ObjectLock Compliance Mode     |
| STAGED                   | DISCARDED_STAGING        | ALLOWED       | Ephemeral runner buffer cleared on error      |
| DURABLE_RAW              | VALIDATED                | ALLOWED       | Candidate 004 schema & model checks pass      |
| DURABLE_RAW              | QUARANTINED              | ALLOWED       | Validation failure; records rejection reason  |
| VALIDATED                | REFERENCED_BY_ADMISSION  | ALLOWED       | Committed atomically in state transaction     |
| REFERENCED_BY_ADMISSION  | SUPERSEDED_METADATA      | ALLOWED       | Append-only metadata correction note created  |
| DURABLE_RAW              | DELETED                  | FORBIDDEN     | Blocked by S3/GCS Compliance Lock             |
| QUARANTINED              | DELETED                  | FORBIDDEN     | Blocked by S3/GCS Compliance Lock             |
| REFERENCED_BY_ADMISSION  | DELETED                  | FORBIDDEN     | Blocked by S3/GCS Compliance Lock             |
| REFERENCED_BY_ADMISSION  | DURABLE_RAW              | FORBIDDEN     | State regression prohibited                   |
| QUARANTINED              | REFERENCED_BY_ADMISSION  | FORBIDDEN     | Quarantined evidence cannot be admitted       |
+--------------------------+--------------------------+---------------+-----------------------------------------------+
```

---

### 3. Raw Evidence Immutability & Retention Semantics

#### 3.1 Retention Mode Specification

The storage architecture strictly defines and enforces retention semantics:

```yaml
RETENTION_MODE: COMPLIANCE_LOCK
RETENTION_PERIOD_DAYS: 2557  # Exactly 7 Tropical Years (7 * 365.25 days)
LEGAL_HOLD_POLICY: INDEFINITE_SUSPENSION_OF_DELETION
DELETE_AFTER_RETENTION_POLICY: PROHIBITED_IN_CORE_SCOPE
VERSION_EXPIRATION_POLICY: DISABLED
STORAGE_CLASS: STANDARD_IA_WITH_CROSS_REGION_REPLICATION
```

#### 3.2 Immutability Tiers Defined

1. **Object Versioning Alone (Insufficient)**: Prevents accidental overwrite by preserving previous versions, but permits `DeleteObjectVersion`. **Rejected as non-compliant.**
2. **Governance Mode Lock (Insufficient)**: Permits users with specific IAM permissions (`s3:BypassGovernanceRetention`) to delete objects or shorten retention periods. **Rejected as non-compliant.**
3. **Compliance Mode Lock (Mandatory)**: No user, including the root account or AWS Support, can shorten the retention period, disable the lock, or delete any object version until the retention period expires. **Mandatory for Theorem D.**

---

### 4. Canonical Custody Metadata Tuple

Every raw evidence object is associated with an immutable custody record frozen upon creation:

```json
{
  "$schema": "https://arx.terminal/schemas/v1/evidence-custody-tuple.json",
  "accession": "0001062993-26-000042",
  "endpoint": "https://data.sec.gov/submissions/CIK0001062993.json",
  "http_status": 200,
  "request_timestamp": "2026-10-12T00:15:02.104Z",
  "response_timestamp": "2026-10-12T00:15:02.482Z",
  "capture_timestamp": "2026-10-12T00:15:02.510Z",
  "clock_evidence_id": "urn:arx:timestamp:rfc3161:nist:2026-10-12-0015",
  "object_id": "sec/evidence/0001062993-26-000042/3b8d91c7a4.raw",
  "object_version": "v1.0.3847291048",
  "byte_length": 849204,
  "sha256": "3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019",
  "content_type": "application/json",
  "capture_run_id": "run-qual-20261012T001500Z-7f8e3d21",
  "protocol_version": "1.0.0"
}
```

#### 4.1 Metadata Mutation Invariants
- The raw byte file (`.raw`) is never modified or overwritten.
- If diagnostic metadata requires augmentation (e.g. associating an incident ID), an append-only metadata descriptor (`.meta.v2.json`) is stored adjacent to the record.
- Historical metadata versions remain queryable via immutable object versioning.

---

### 5. Quarantine Protocol

When a raw SEC capture succeeds at the network and storage layer, but fails downstream interpretation:

1. **Quarantine Triggers**:
   - SEC JSON schema malformation or unparseable payload.
   - Non-matching accession metadata.
   - Candidate 004 predicate failure or data integrity inconsistency.
   - Upstream rate-limiting payload masquerading as valid accession.
2. **Quarantine Execution**:
   - The raw byte object remains securely anchored in the compliance bucket.
   - An immutable quarantine record is published to the state backend:
     `QUARANTINE_RECORD: { accession, object_id, sha256, failure_code, capture_run_id }`
   - Status is set to `CAPTURED_QUARANTINED`.
3. **Safety Invariants**:
   - Quarantine **never mutates or deletes the raw evidence bytes**.
   - Quarantined records **never increment the natural denominator**.
   - Quarantined accessions **cannot be referenced by an admission record**.

---

### 6. Replay & Tamper-Detection Algorithm

The replay verification protocol verifies the forensic integrity of the entire evidence store:

```python
def verify_evidence_replay(custody_tuple: dict, storage_client) -> bool:
    """
    Forensically validates that stored raw bytes match the anchored SHA-256.
    Fails closed on any inconsistency.
    """
    object_id = custody_tuple["object_id"]
    expected_sha = custody_tuple["sha256"]
    expected_length = custody_tuple["byte_length"]

    # 1. Fetch exact raw bytes directly from WORM object storage
    raw_bytes = storage_client.get_object_bytes(object_id)

    # 2. Assert byte length equivalence
    if len(raw_bytes) != expected_length:
        raise EvidenceTamperDetectedError(
            f"Byte length mismatch: expected {expected_length}, got {len(raw_bytes)}"
        )

    # 3. Assert cryptographic hash equivalence
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise EvidenceTamperDetectedError(
            f"Cryptographic hash mismatch: expected {expected_sha}, got {actual_sha}"
        )

    return True
```

Any discrepancy raises `EvidenceTamperDetectedError` and forces the entire system into `TERMINAL_HOLD`.

---

### 7. Disaster Recovery & Availability Service Levels

The disaster recovery contract specifies recovery objectives across physical infrastructure loss:

```yaml
RPO: 0  # Recovery Point Objective: Zero committed data loss
RTO: 3600  # Recovery Time Objective: 1 Hour to full operational rehydration
MAX_ACCEPTABLE_STATE_LOSS: 0
MAX_ACCEPTABLE_EVIDENCE_LOSS: 0
REGION_FAILURE_POLICY: CROSS_REGION_REPLICATION_WITH_FAILOVER
BACKUP_VALIDATION_FREQUENCY: WEEKLY_AUTOMATED_DRILL
```

#### 7.1 Cross-Region Replication Topology & Synchronous Commit Architecture
- **Committed Transaction Durability (RPO = 0)**: To guarantee zero committed data loss (RPO = 0), all governance-critical transactions, epoch transitions, and ledger accession events require synchronous dual-region durable write acknowledgment before acknowledgment is returned to callers. An event is never committed on the basis of asynchronous replication alone.
- **Bulk Evidence & Secondary Replication**: Asynchronous cross-region replication (CRR) applies strictly to non-critical bulk evidence artifacts, secondary raw logs, and compliance-locked WORM archive bundles within $\le 900$ seconds SLA.
- **Failover Contract**: In the event of primary cloud region failure, the secondary region becomes read-authoritative. State cannot resume writing without a formal recovery quorum certificate (`RECOVERY_APPROVER`).
- **Operational Status Boundary**: The architectural guarantee of zero-event RPO via synchronous commit protocol is proven and enforced by design. The physical multi-region cloud infrastructure operational guarantee remains `NOT_ESTABLISHED` pending formal provider provisioning and qualification.
