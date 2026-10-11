"""
Audit Dispute Governance and SLA State Machine for ARX Terminal Smart Money.
Adheres strictly to Wave 2C Hardening requirements:
- 4 orthogonal state machines: DisputeState, SlaState, EscalationLevel, CertificationDisputeState
- SLA Certification Firewall: Timer/SLA events NEVER move the certified frontier, certify receipts,
  or invalidate receipts.
- Dispute Resolution: Disputed receipts are NEVER reused; post-resolution certification requires
  a new independent proof and a new receipt.
"""

import os
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash
from analyst_dashboard.smart_money.certification_models import (
    DisputeState,
    SlaState,
    EscalationLevel,
    CertificationDisputeState,
    AuditDispute,
)
from analyst_dashboard.smart_money.form4_ingestion import DEFAULT_SMART_MONEY_DB


class AuditDisputeEngine:
    """
    Manages the lifecycle of auditor disagreements and SLA escalations.
    Guarantees strict isolation between process timers and cryptographic completeness proofs.
    """

    SLA_POLICY_VERSION = "sla-v1.0"
    SLA_POLICY_CONTENT = (
        "TRIAGE_SLA_HOURS = 4\n"
        "ROOT_CAUSE_SLA_HOURS = 24\n"
        "RECONCILIATION_SLA_HOURS = 48\n"
        "GOVERNANCE_ESCALATION_SLA_HOURS = 120\n"
        "FIREWALL: Timer and SLA events have ZERO certification effect."
    )
    SLA_POLICY_SHA256 = hashlib.sha256(SLA_POLICY_CONTENT.encode("utf-8")).hexdigest()

    def __init__(self, db_path: str = DEFAULT_SMART_MONEY_DB):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def open_dispute(
        self,
        receipt_id: str,
        candidate_frontier: str,
        first_divergent_step: str,
        minimal_counterexample: Dict[str, Any],
        detected_at: Optional[str] = None,
    ) -> AuditDispute:
        """
        Registers an auditor disagreement and immediately blocks the disputed candidate.
        The last undisputed frontier remains active.
        """
        now = datetime.now(timezone.utc)
        base_dt = datetime.fromisoformat(detected_at.replace("Z", "+00:00")) if detected_at else now
        now_iso = detected_at or now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Compute deadlines relative to detected_at
        triage_dl = (base_dt + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rc_dl = (base_dt + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        recon_dl = (base_dt + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        gov_dl = (base_dt + timedelta(hours=120)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        dispute_id = f"DISP-{hashlib.sha256(f'{receipt_id}:{candidate_frontier}:{now_iso}'.encode('utf-8')).hexdigest()[:12]}"

        dispute = AuditDispute(
            dispute_id=dispute_id,
            receipt_id=receipt_id,
            candidate_frontier=candidate_frontier,
            dispute_state=DisputeState.DETECTED.value,
            sla_state=SlaState.WITHIN_TRIAGE_SLA.value,
            escalation_level=EscalationLevel.L0_ENGINEERING.value,
            certification_state=CertificationDisputeState.CANDIDATE_BLOCKED_BY_DISPUTE.value,
            first_divergent_step=first_divergent_step,
            minimal_counterexample=minimal_counterexample,
            detected_at=now_iso,
            triage_deadline=triage_dl,
            root_cause_deadline=rc_dl,
            reconciliation_deadline=recon_dl,
            governance_deadline=gov_dl,
            sla_policy_version=self.SLA_POLICY_VERSION,
            sla_policy_sha256=self.SLA_POLICY_SHA256,
        )

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO sm_audit_disputes (
                    dispute_id, receipt_id, candidate_frontier, dispute_state, sla_state,
                    escalation_level, certification_state, first_divergent_step, minimal_counterexample,
                    detected_at, triage_deadline, root_cause_deadline, reconciliation_deadline,
                    governance_deadline, sla_policy_version, sla_policy_sha256, resolved_at, resolution_artifact_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dispute.dispute_id, dispute.receipt_id, dispute.candidate_frontier,
                    dispute.dispute_state, dispute.sla_state, dispute.escalation_level,
                    dispute.certification_state, dispute.first_divergent_step,
                    canonicalize_json(dispute.minimal_counterexample), dispute.detected_at,
                    dispute.triage_deadline, dispute.root_cause_deadline,
                    dispute.reconciliation_deadline, dispute.governance_deadline,
                    dispute.sla_policy_version, dispute.sla_policy_sha256, None, None,
                ),
            )
            # Record initial history
            cur.execute(
                """
                INSERT INTO sm_audit_dispute_history (
                    history_id, dispute_id, sequence, from_dispute_state, to_dispute_state,
                    from_sla_state, to_sla_state, from_escalation_level, to_escalation_level,
                    transition_trigger, transition_payload, transitioned_at
                ) VALUES (?, ?, 1, 'NONE', ?, 'NONE', ?, 'NONE', ?, 'DISPUTE_DETECTED', ?, ?)
                """,
                (
                    f"DHIST-{dispute_id}-001", dispute_id,
                    dispute.dispute_state, dispute.sla_state, dispute.escalation_level,
                    canonicalize_json({"first_divergent_step": first_divergent_step}),
                    now_iso,
                ),
            )
            conn.commit()
            return dispute
        finally:
            conn.close()

    def process_timer_tick(self, dispute_id: str, current_time_iso: str) -> Dict[str, Any]:
        """
        Evaluates SLA timer deadlines and updates SLA/Escalation state if overdue.
        ENFORCES SLA CERTIFICATION FIREWALL:
        - NEVER moves certified frontier
        - NEVER issues a certification receipt
        - NEVER validates or invalidates cryptographic receipts
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM sm_audit_disputes WHERE dispute_id = ?", (dispute_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"DISPUTE_NOT_FOUND: {dispute_id}")

            current_dt = datetime.fromisoformat(current_time_iso.replace("Z", "+00:00"))
            triage_dt = datetime.fromisoformat(row["triage_deadline"].replace("Z", "+00:00"))
            rc_dt = datetime.fromisoformat(row["root_cause_deadline"].replace("Z", "+00:00"))
            recon_dt = datetime.fromisoformat(row["reconciliation_deadline"].replace("Z", "+00:00"))
            gov_dt = datetime.fromisoformat(row["governance_deadline"].replace("Z", "+00:00"))

            old_sla = row["sla_state"]
            old_esc = row["escalation_level"]
            old_disp = row["dispute_state"]

            new_sla = old_sla
            new_esc = old_esc

            if current_dt > gov_dt:
                new_sla = SlaState.MAX_SLA_EXCEEDED.value
                new_esc = EscalationLevel.L3_GOVERNANCE_HOLD.value
            elif current_dt > recon_dt:
                new_sla = SlaState.GOVERNANCE_ESCALATED.value
                new_esc = EscalationLevel.L2_GOVERNANCE_CLOSURE.value
            elif current_dt > rc_dt:
                new_sla = SlaState.ROOT_CAUSE_OVERDUE.value
                new_esc = EscalationLevel.L1_TECHNICAL_ADJUDICATION.value
            elif current_dt > triage_dt and old_disp == DisputeState.DETECTED.value:
                new_sla = SlaState.TRIAGE_OVERDUE.value
                new_esc = EscalationLevel.L1_TECHNICAL_ADJUDICATION.value

            if new_sla != old_sla or new_esc != old_esc:
                cur.execute(
                    """
                    UPDATE sm_audit_disputes
                    SET sla_state = ?, escalation_level = ?
                    WHERE dispute_id = ?
                    """,
                    (new_sla, new_esc, dispute_id),
                )
                cur.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM sm_audit_dispute_history WHERE dispute_id = ?", (dispute_id,))
                seq = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO sm_audit_dispute_history (
                        history_id, dispute_id, sequence, from_dispute_state, to_dispute_state,
                        from_sla_state, to_sla_state, from_escalation_level, to_escalation_level,
                        transition_trigger, transition_payload, transitioned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'TIMER_TICK_EVALUATION', '{}', ?)
                    """,
                    (
                        f"DHIST-{dispute_id}-{seq:03d}", dispute_id, seq,
                        old_disp, old_disp, old_sla, new_sla, old_esc, new_esc,
                        current_time_iso,
                    ),
                )
                conn.commit()

            return {
                "dispute_id": dispute_id,
                "dispute_state": old_disp,
                "sla_state": new_sla,
                "escalation_level": new_esc,
                "certification_effect": "NONE",
            }
        finally:
            conn.close()

    def resolve_dispute(
        self,
        dispute_id: str,
        resolution_artifact_hash: str,
        resolved_at: Optional[str] = None,
    ) -> bool:
        """
        Marks dispute as RESOLVED.
        Transitions certification state to POST_RESOLUTION_CERTIFICATION_PENDING.
        The disputed receipt is NEVER reused; a new independent proof and receipt is required.
        """
        now_iso = resolved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE sm_audit_disputes
                SET dispute_state = ?, certification_state = ?, resolved_at = ?, resolution_artifact_hash = ?
                WHERE dispute_id = ?
                """,
                (
                    DisputeState.RESOLVED.value,
                    CertificationDisputeState.POST_RESOLUTION_CERTIFICATION_PENDING.value,
                    now_iso,
                    resolution_artifact_hash,
                    dispute_id,
                ),
            )
            cur.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM sm_audit_dispute_history WHERE dispute_id = ?", (dispute_id,))
            seq = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO sm_audit_dispute_history (
                    history_id, dispute_id, sequence, from_dispute_state, to_dispute_state,
                    from_sla_state, to_sla_state, from_escalation_level, to_escalation_level,
                    transition_trigger, transition_payload, transitioned_at
                ) VALUES (?, ?, ?, 'UNDER_RECONCILIATION', ?, 'ACTIVE', 'RESOLVED', 'L0', 'L0', 'EVIDENTIARY_RESOLUTION', ?, ?)
                """,
                (
                    f"DHIST-{dispute_id}-{seq:03d}", dispute_id, seq,
                    DisputeState.RESOLVED.value,
                    canonicalize_json({"resolution_hash": resolution_artifact_hash}),
                    now_iso,
                ),
            )
            conn.commit()
            return True
        finally:
            conn.close()
