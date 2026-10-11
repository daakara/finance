"""
Certification Authority Engine for ARX Terminal Smart Money Wave 2C Hardening.
Implements:
1. Discovery Completeness and Partition Enumeration Receipts
2. Pagination & Continuation Exhaustion Closure
3. Governed Holdback Stability and Post-Holdback Re-enumeration
4. Append-Only Gap Event Ledger
5. Expected Accession Set Authority & Terminal Accounting Reconciliation
6. Merkle-Linked Certification Receipts V2 & Receipt Chain
"""

import os
import json
import sqlite3
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set

from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash
from analyst_dashboard.smart_money.partition_set_authority import (
    PartitionSetAuthority,
    REQUIRED_PARTITION_COUNT,
    GOVERNED_UNIVERSE_BUILD_ID,
    GOVERNED_UNIVERSE_CONTENT_HASH,
)
from analyst_dashboard.smart_money.policy_bundle import PolicyBundleAuthority
from analyst_dashboard.smart_money.certification_models import (
    EnumerationStatus,
    PartitionEnumerationReceipt,
    SourceSnapshot,
    ExpectedAccession,
    GapEvent,
    ClosureProof,
    CertificationReceiptV2,
)
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    DEFAULT_SMART_MONEY_DB,
)


class CertificationAuthorityEngine:
    """
    Independent Certification Authority Engine enforcing mathematical proof of discovery completeness
    and issuing cryptographically sealed CertificationReceiptV2 envelopes.
    """

    CERTIFIER_COMPONENT_VERSION = "cert-auth-v2.2.0"
    CERTIFIER_FUNCTIONAL_SHA = "sha256:wave2c-certification-authority-hardened"

    def __init__(
        self,
        db_path: str = DEFAULT_SMART_MONEY_DB,
        partition_authority: Optional[PartitionSetAuthority] = None,
    ):
        if "shadow_evidence" in db_path:
            raise AssertionError("CANDIDATE_004_FIREWALL_BREACH: Certification engine cannot target shadow_evidence.db")
        self.db_path = db_path
        self.partition_auth = partition_authority or PartitionSetAuthority()
        self.ingestion_mgr = Form4IngestionManager(db_path=self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # 1. Source Snapshot & Partition Enumeration Receipts
    # -------------------------------------------------------------------------

    def record_source_snapshot(
        self,
        partition_id: str,
        issuer_cik: str,
        request_identity: str,
        window_start: str,
        window_end: str,
        raw_payload: str,
        pagination_metadata: Dict[str, Any],
        fetched_at: str,
        source_contract_version: str = "1.0.0",
    ) -> SourceSnapshot:
        """Stores immutable raw SEC payload for offline reconstruction."""
        cik_clean = issuer_cik.strip().zfill(10)
        payload_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        snapshot_id = f"SNAP-{cik_clean}-{hashlib.sha256(request_identity.encode('utf-8')).hexdigest()[:12]}"

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO sm_source_snapshots (
                    snapshot_id, partition_id, issuer_cik, request_identity,
                    window_start, window_end, raw_payload, payload_hash,
                    pagination_metadata, fetched_at, source_contract_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id, partition_id, cik_clean, request_identity,
                    window_start, window_end, raw_payload, payload_hash,
                    json.dumps(pagination_metadata, sort_keys=True), fetched_at, source_contract_version,
                ),
            )
            conn.commit()
            return SourceSnapshot(
                snapshot_id=snapshot_id,
                partition_id=partition_id,
                issuer_cik=cik_clean,
                request_identity=request_identity,
                window_start=window_start,
                window_end=window_end,
                raw_payload=raw_payload,
                payload_hash=payload_hash,
                pagination_metadata=pagination_metadata,
                fetched_at=fetched_at,
                source_contract_version=source_contract_version,
            )
        finally:
            conn.close()

    def record_partition_enumeration(
        self,
        issuer_cik: str,
        window_start: str,
        window_end: str,
        pages_expected: int,
        pages_consumed: int,
        pagination_exhausted: bool,
        continuation_token_remaining: bool,
        discovered_accessions: List[str],
        post_holdback_accessions: List[str],
        source_snapshot_manifest_hash: str,
        discovery_gap_count: int = 0,
        enumerated_at: Optional[str] = None,
    ) -> PartitionEnumerationReceipt:
        """
        Creates and seals a lower-level PartitionEnumerationReceipt.
        Validates pagination exhaustion and post-holdback stability.
        """
        cik_clean = issuer_cik.strip().zfill(10)
        part_id = f"PART-{cik_clean}"
        now_utc = enumerated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Canonicalize accession sets
        sorted_first = sorted(list(set(discovered_accessions)))
        sorted_holdback = sorted(list(set(post_holdback_accessions)))

        first_hash = compute_canonical_hash(sorted_first)
        holdback_hash = compute_canonical_hash(sorted_holdback)
        accession_set_stable = (first_hash == holdback_hash)

        # Determine status
        if not pagination_exhausted or continuation_token_remaining or not accession_set_stable or discovery_gap_count > 0:
            status = EnumerationStatus.FAILED.value
        else:
            status = EnumerationStatus.CLOSED.value

        receipt_id = f"ENUM-{cik_clean}-{hashlib.sha256(f'{window_start}:{window_end}:{first_hash}'.encode('utf-8')).hexdigest()[:12]}"
        req_part_hash = self.partition_auth.compute_required_partition_set_hash()

        receipt = PartitionEnumerationReceipt(
            partition_enumeration_receipt_id=receipt_id,
            partition_id=part_id,
            issuer_cik=cik_clean,
            universe_build_id=GOVERNED_UNIVERSE_BUILD_ID,
            required_partition_set_hash=req_part_hash,
            window_start=window_start,
            window_end=window_end,
            source_contract_version="1.0.0",
            discovery_policy_version="1.0.0",
            pagination_policy_version="1.0.0",
            source_snapshot_manifest_hash=source_snapshot_manifest_hash,
            pages_expected_or_discovered=pages_expected,
            pages_consumed=pages_consumed,
            pagination_exhausted=pagination_exhausted,
            continuation_token_remaining=continuation_token_remaining,
            first_enumeration_accession_set_hash=first_hash,
            post_holdback_accession_set_hash=holdback_hash,
            accession_set_stable=accession_set_stable,
            discovery_gap_count=discovery_gap_count,
            enumeration_status=status,
            enumerated_at=now_utc,
        )
        receipt.receipt_hash = receipt.compute_hash()

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO sm_partition_enumeration_receipts (
                    partition_enumeration_receipt_id, partition_id, issuer_cik, universe_build_id,
                    required_partition_set_hash, window_start, window_end, source_contract_version,
                    discovery_policy_version, pagination_policy_version, source_snapshot_manifest_hash,
                    pages_expected_or_discovered, pages_consumed, pagination_exhausted,
                    continuation_token_remaining, first_enumeration_accession_set_hash,
                    post_holdback_accession_set_hash, accession_set_stable, discovery_gap_count,
                    enumeration_status, enumerated_at, receipt_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.partition_enumeration_receipt_id, receipt.partition_id, receipt.issuer_cik,
                    receipt.universe_build_id, receipt.required_partition_set_hash, receipt.window_start,
                    receipt.window_end, receipt.source_contract_version, receipt.discovery_policy_version,
                    receipt.pagination_policy_version, receipt.source_snapshot_manifest_hash,
                    receipt.pages_expected_or_discovered, receipt.pages_consumed,
                    1 if receipt.pagination_exhausted else 0,
                    1 if receipt.continuation_token_remaining else 0,
                    receipt.first_enumeration_accession_set_hash, receipt.post_holdback_accession_set_hash,
                    1 if receipt.accession_set_stable else 0, receipt.discovery_gap_count,
                    receipt.enumeration_status, receipt.enumerated_at, receipt.receipt_hash,
                ),
            )
            conn.commit()
            return receipt
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 2. Append-Only Gap Event Ledger
    # -------------------------------------------------------------------------

    def emit_gap_event(
        self,
        gap_id: str,
        event_type: str,
        partition_id: str,
        issuer_cik: str,
        gap_type: str,
        window_start: str,
        window_end: str,
        payload: Dict[str, Any],
        occurred_at: Optional[str] = None,
    ) -> GapEvent:
        """Emits an immutable, monotonically sequenced gap event."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COALESCE(MAX(event_sequence), 0) + 1 FROM sm_gap_events")
            seq = cur.fetchone()[0]

            now_utc = occurred_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            event_id = f"GEVT-{seq:08d}-{hashlib.sha256(f'{gap_id}:{event_type}:{seq}'.encode('utf-8')).hexdigest()[:8]}"

            event = GapEvent(
                event_id=event_id,
                event_sequence=seq,
                gap_id=gap_id,
                event_type=event_type,
                partition_id=partition_id,
                issuer_cik=issuer_cik.strip().zfill(10),
                gap_type=gap_type,
                window_start=window_start,
                window_end=window_end,
                event_payload=payload,
                occurred_at=now_utc,
            )
            event.event_hash = event.compute_hash()

            cur.execute(
                """
                INSERT INTO sm_gap_events (
                    event_id, event_sequence, gap_id, event_type, partition_id, issuer_cik,
                    gap_type, window_start, window_end, event_payload, occurred_at, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id, event.event_sequence, event.gap_id, event.event_type,
                    event.partition_id, event.issuer_cik, event.gap_type, event.window_start,
                    event.window_end, canonicalize_json(event.event_payload), event.occurred_at, event.event_hash,
                ),
            )
            conn.commit()
            return event
        finally:
            conn.close()

    def get_gap_event_high_watermark(self) -> int:
        """Returns the highest monotonic gap event sequence number."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COALESCE(MAX(event_sequence), 0) FROM sm_gap_events")
            return cur.fetchone()[0]
        finally:
            conn.close()

    def compute_gap_event_prefix_hash(self, high_watermark: Optional[int] = None) -> str:
        """Computes deterministic SHA256 over all gap events up to high_watermark."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if high_watermark is not None:
                cur.execute(
                    "SELECT event_hash FROM sm_gap_events WHERE event_sequence <= ? ORDER BY event_sequence ASC",
                    (high_watermark,),
                )
            else:
                cur.execute("SELECT event_hash FROM sm_gap_events ORDER BY event_sequence ASC")
            hashes = [r[0] for r in cur.fetchall()]
            return compute_canonical_hash(hashes)
        finally:
            conn.close()

    def compute_open_gap_set_hash(self) -> Tuple[str, int]:
        """
        Reconstructs active open/retrying gap set and returns (canonical_hash, count).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT gap_id, partition_id, issuer_cik, gap_type, window_start, window_end, source_event_time
                FROM sm_source_gaps
                WHERE status IN ('OPEN', 'RETRYING')
                ORDER BY gap_id ASC
                """
            )
            rows = [dict(r) for r in cur.fetchall()]
            open_hash = compute_canonical_hash(rows)
            return open_hash, len(rows)
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 3. Expected Accession Sets & Terminal Reconciliation
    # -------------------------------------------------------------------------

    def register_expected_accessions(
        self,
        candidate_frontier: str,
        expected_items: List[ExpectedAccession],
    ) -> str:
        """Stores canonical expected accessions for candidate frontier."""
        exp_set_id = f"EXP-{compute_canonical_hash([asdict(e) for e in expected_items])[:16]}"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            for item in expected_items:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO sm_expected_accession_sets (
                        expected_set_id, candidate_frontier, accession_number, issuer_cik,
                        public_availability_at, source_snapshot_identity, source_content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exp_set_id, candidate_frontier, item.accession_number,
                        item.issuer_cik.strip().zfill(10), item.public_availability_at,
                        item.source_snapshot_identity, item.source_content_hash,
                    ),
                )
            conn.commit()
            return exp_set_id
        finally:
            conn.close()

    def reconcile_accession_sets(
        self,
        candidate_frontier: str,
    ) -> Dict[str, Any]:
        """
        Reconciles:
        1. Expected accession set
        2. Registered accession set (raw filings / source availability ledger)
        3. Terminally accounted accession set (parsed transactions / support dispositions)
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # 1. Expected
            cur.execute(
                """
                SELECT issuer_cik, public_availability_at, accession_number, source_content_hash
                FROM sm_expected_accession_sets
                WHERE candidate_frontier = ?
                ORDER BY issuer_cik ASC, public_availability_at ASC, accession_number ASC
                """,
                (candidate_frontier,),
            )
            expected_rows = [dict(r) for r in cur.fetchall()]
            expected_set = {r["accession_number"] for r in expected_rows}
            expected_hash = compute_canonical_hash(expected_rows)

            # 2. Registered (ledger with public_availability_at <= candidate_frontier for required partitions)
            req_ciks = self.partition_auth.get_required_partition_ciks()
            placeholders = ",".join(["?"] * len(req_ciks))

            cur.execute(
                f"""
                SELECT l.issuer_cik, l.public_availability_at, l.accession_number, l.source_hash AS source_content_hash
                FROM sm_source_availability_ledger l
                WHERE l.public_availability_at <= ? AND l.public_availability_at != 'UNKNOWN'
                  AND l.issuer_cik IN ({placeholders})
                ORDER BY l.issuer_cik ASC, l.public_availability_at ASC, l.accession_number ASC
                """,
                [candidate_frontier] + req_ciks,
            )
            registered_rows = [dict(r) for r in cur.fetchall()]
            registered_set = {r["accession_number"] for r in registered_rows}
            registered_hash = compute_canonical_hash(registered_rows)

            # 3. Terminally Accounted (raw filings with disposition FETCHED_AND_PARSED for required partitions)
            cur.execute(
                f"""
                SELECT DISTINCT r.issuer_cik, r.acceptance_datetime AS public_availability_at, r.accession_number, r.source_content_hash
                FROM form4_raw_filings r
                WHERE r.acceptance_datetime <= ? AND r.disposition IN ('FETCHED_AND_PARSED', 'SUPPORTED_DOCUMENT')
                  AND r.issuer_cik IN ({placeholders})
                ORDER BY r.issuer_cik ASC, r.acceptance_datetime ASC, r.accession_number ASC
                """,
                [candidate_frontier] + req_ciks,
            )
            terminal_rows = [dict(r) for r in cur.fetchall()]
            terminal_set = {r["accession_number"] for r in terminal_rows}
            terminal_hash = compute_canonical_hash(terminal_rows)

            exp_minus_reg = len(expected_set - registered_set)
            reg_minus_exp = len(registered_set - expected_set)
            exp_minus_term = len(expected_set - terminal_set)

            reconciled = (exp_minus_reg == 0 and reg_minus_exp == 0 and exp_minus_term == 0)

            return {
                "reconciled": reconciled,
                "expected_accession_set_hash": expected_hash,
                "registered_accession_set_hash": registered_hash,
                "terminal_accession_set_hash": terminal_hash,
                "expected_count": len(expected_set),
                "registered_count": len(registered_set),
                "terminal_count": len(terminal_set),
                "expected_minus_registered_count": exp_minus_reg,
                "registered_minus_expected_count": reg_minus_exp,
                "expected_minus_terminally_accounted_count": exp_minus_term,
            }
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 4. Amendment State Hash
    # -------------------------------------------------------------------------

    def compute_amendment_state_hash(self, candidate_frontier: str) -> Tuple[str, int]:
        """Computes canonical hash over all amendment relationships through candidate_frontier."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT accession_number, amended_accession_number, relationship, detected_at
                FROM sm_amendments
                WHERE detected_at <= ?
                ORDER BY accession_number ASC, amended_accession_number ASC
                """,
                (candidate_frontier,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            unresolved = sum(1 for r in rows if r["relationship"] == "UNRESOLVED_AMENDMENT")
            return compute_canonical_hash(rows), unresolved
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 5. Certification Receipt V2 Generation
    # -------------------------------------------------------------------------

    def issue_certification_receipt_v2(
        self,
        candidate_frontier: str,
        universe_content_hash: Optional[str] = None,
    ) -> CertificationReceiptV2:
        """
        Executes complete fail-closed certification proof and issues CertificationReceiptV2.
        """
        effective_universe_content_hash = universe_content_hash or GOVERNED_UNIVERSE_CONTENT_HASH
        conn = self._get_connection()
        try:
            cur = conn.cursor()

            # 1. Check Candidate 004 firewall invariant
            # Smart money cache DB must NOT touch shadow_evidence
            assert not self.db_path.endswith("shadow_evidence.db")

            # 2. Check active late-arrival integrity breaches
            cur.execute("SELECT COUNT(*) FROM sm_late_arrival_events WHERE is_integrity_breach = 1")
            if cur.fetchone()[0] > 0:
                raise RuntimeError("WATERMARK_INTEGRITY_BREACH_ACTIVE: Certification advancement frozen")

            # 3. Check required partition set coverage
            req_ciks = self.partition_auth.get_required_partition_ciks()
            placeholders = ",".join("?" for _ in req_ciks)
            req_part_hash = self.partition_auth.compute_required_partition_set_hash()

            # 4. Verify enumeration receipts for all 106 partitions
            cur.execute(
                """
                SELECT issuer_cik, receipt_hash, window_end, enumeration_status,
                       pagination_exhausted, continuation_token_remaining, accession_set_stable, discovery_gap_count
                FROM sm_partition_enumeration_receipts
                WHERE universe_build_id = ? AND window_end >= ?
                ORDER BY issuer_cik ASC
                """,
                (GOVERNED_UNIVERSE_BUILD_ID, candidate_frontier),
            )
            enum_rows = [dict(r) for r in cur.fetchall()]
            enum_by_cik = {r["issuer_cik"]: r for r in enum_rows}

            for cik in req_ciks:
                if cik not in enum_by_cik:
                    raise RuntimeError(f"MISSING_REQUIRED_PARTITION_ENUMERATION: CIK {cik} has no enumeration receipt through {candidate_frontier}")
                er = enum_by_cik[cik]
                if er["enumeration_status"] != EnumerationStatus.CLOSED.value:
                    raise RuntimeError(f"PARTITION_ENUMERATION_NOT_CLOSED: CIK {cik} status is {er['enumeration_status']}")
                if not er["pagination_exhausted"] or er["continuation_token_remaining"]:
                    raise RuntimeError(f"PAGINATION_NOT_EXHAUSTED: CIK {cik} pagination unclosed")
                if not er["accession_set_stable"]:
                    raise RuntimeError(f"HOLDBACK_STABILITY_FAILURE: CIK {cik} accession set changed after holdback")
                if er["discovery_gap_count"] > 0:
                    raise RuntimeError(f"DISCOVERY_GAP_BLOCKS_CERTIFICATION: CIK {cik} has open discovery gaps")

            enum_receipt_set_hash = compute_canonical_hash(sorted([r["receipt_hash"] for r in enum_rows]))

            # 5. Check blocking source gaps
            open_gap_hash, open_gap_count = self.compute_open_gap_set_hash()
            cur.execute(
                """
                SELECT COUNT(*) FROM sm_source_gaps
                WHERE status IN ('OPEN', 'RETRYING') AND (
                    (source_event_time IS NOT NULL AND source_event_time <= ?) OR
                    (source_event_time IS NULL AND window_start <= ?)
                )
                """,
                (candidate_frontier, candidate_frontier),
            )
            blocking_gaps = cur.fetchone()[0]
            if blocking_gaps > 0:
                raise RuntimeError(f"BLOCKING_GAPS_EXIST: {blocking_gaps} gaps block frontier {candidate_frontier}")

            # 6. Reconcile expected, registered, and terminal accession sets
            recon = self.reconcile_accession_sets(candidate_frontier)
            if not recon["reconciled"]:
                raise RuntimeError(
                    f"ACCESSION_SET_MISMATCH: expected - registered = {recon['expected_minus_registered_count']}, "
                    f"registered - expected = {recon['registered_minus_expected_count']}, "
                    f"expected - terminal = {recon['expected_minus_terminally_accounted_count']}"
                )

            # 7. Check amendment state
            amend_hash, unresolved_amends = self.compute_amendment_state_hash(candidate_frontier)
            if unresolved_amends > 0:
                raise RuntimeError(f"UNRESOLVED_AMENDMENTS_EXIST: {unresolved_amends} unresolved amendments through {candidate_frontier}")

            # 8. Hash other inputs
            cur.execute("SELECT snapshot_id, payload_hash FROM sm_source_snapshots ORDER BY snapshot_id ASC")
            snap_rows = [dict(r) for r in cur.fetchall()]
            snap_manifest_hash = compute_canonical_hash(snap_rows)

            cur.execute("SELECT partition_id, candidate_complete_through FROM sm_incremental_partitions ORDER BY partition_id ASC")
            part_frontier_rows = [dict(r) for r in cur.fetchall()]
            part_frontier_manifest_hash = compute_canonical_hash(part_frontier_rows)

            cur.execute(
                f"""
                SELECT accession_number, public_availability_at, source_hash
                FROM sm_source_availability_ledger
                WHERE public_availability_at <= ? AND issuer_cik IN ({placeholders})
                ORDER BY accession_number ASC
                """,
                [candidate_frontier] + req_ciks,
            )
            ledger_hash = compute_canonical_hash([dict(r) for r in cur.fetchall()])

            policy_bundle_hash = PolicyBundleAuthority.compute_policy_bundle_hash()
            gap_prefix_hash = self.compute_gap_event_prefix_hash()

            # 9. Build Deterministic ClosureProof
            # Retrieve previous receipt
            cur.execute("SELECT * FROM sm_certification_receipts_v2 ORDER BY receipt_sequence DESC LIMIT 1")
            prev_row = cur.fetchone()
            if prev_row:
                prev_receipt_id = prev_row["receipt_id"]
                prev_receipt_sha = prev_row["receipt_sha256"]
                prev_watermark = prev_row["certified_frontier"]
                seq = prev_row["receipt_sequence"] + 1
            else:
                prev_receipt_id = "RCPT-V2-000000-GENESIS"
                prev_receipt_sha = "0000000000000000000000000000000000000000000000000000000000000000"
                prev_watermark = "NOT_ESTABLISHED"
                seq = 1

            closure_proof = ClosureProof(
                previous_certified_frontier=prev_watermark,
                candidate_frontier=candidate_frontier,
                universe_build_id=GOVERNED_UNIVERSE_BUILD_ID,
                universe_content_hash=effective_universe_content_hash,
                required_partition_set_hash=req_part_hash,
                partition_coverage_manifest_hash=part_frontier_manifest_hash,
                source_snapshot_manifest_hash=snap_manifest_hash,
                enumeration_receipt_set_hash=enum_receipt_set_hash,
                expected_accession_set_hash=recon["expected_accession_set_hash"],
                registered_accession_set_hash=recon["registered_accession_set_hash"],
                terminal_accession_set_hash=recon["terminal_accession_set_hash"],
                gap_event_prefix_hash=gap_prefix_hash,
                open_gap_set_hash=open_gap_hash,
                amendment_state_hash=amend_hash,
                source_ledger_hash=ledger_hash,
                policy_bundle_hash=policy_bundle_hash,
            )
            closure_proof_sha256 = closure_proof.compute_hash()

            # 10. Construct CertificationReceiptV2 Envelope
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            receipt_id = f"RCPT-V2-{seq:06d}-{closure_proof_sha256[:12]}"

            receipt = CertificationReceiptV2(
                schema_version="2.2.0",
                receipt_id=receipt_id,
                receipt_sequence=seq,
                previous_receipt_id=prev_receipt_id,
                previous_receipt_sha256=prev_receipt_sha,
                previous_certified_frontier=prev_watermark,
                candidate_frontier=candidate_frontier,
                certified_frontier=candidate_frontier,
                closure_proof_sha256=closure_proof_sha256,
                universe_build_id=GOVERNED_UNIVERSE_BUILD_ID,
                universe_content_hash=effective_universe_content_hash,
                required_partition_count=REQUIRED_PARTITION_COUNT,
                required_partition_set_hash=req_part_hash,
                certified_at=now_utc,
                certifier_component_version=self.CERTIFIER_COMPONENT_VERSION,
                certifier_functional_sha=self.CERTIFIER_FUNCTIONAL_SHA,
            )
            receipt.receipt_sha256 = receipt.compute_hash()

            # 11. Transactional Commit
            cur.execute(
                """
                INSERT INTO sm_certification_receipts_v2 (
                    receipt_id, schema_version, receipt_sequence, previous_receipt_id, previous_receipt_sha256,
                    previous_certified_frontier, candidate_frontier, certified_frontier, closure_proof_sha256,
                    universe_build_id, universe_content_hash, required_partition_count, required_partition_set_hash,
                    certified_at, certifier_component_version, certifier_functional_sha, receipt_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id, receipt.schema_version, receipt.receipt_sequence,
                    receipt.previous_receipt_id, receipt.previous_receipt_sha256,
                    receipt.previous_certified_frontier, receipt.candidate_frontier,
                    receipt.certified_frontier, receipt.closure_proof_sha256,
                    receipt.universe_build_id, receipt.universe_content_hash,
                    receipt.required_partition_count, receipt.required_partition_set_hash,
                    receipt.certified_at, receipt.certifier_component_version,
                    receipt.certifier_functional_sha, receipt.receipt_sha256,
                ),
            )

            # Update certified_complete_through across all 106 partitions
            cur.execute(
                "UPDATE sm_incremental_partitions SET certified_complete_through = ? WHERE universe_build_id = ?",
                (candidate_frontier, GOVERNED_UNIVERSE_BUILD_ID),
            )
            conn.commit()
            return receipt
        finally:
            conn.close()
