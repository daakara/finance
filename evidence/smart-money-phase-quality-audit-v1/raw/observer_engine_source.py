"""
SEC Public Availability Observer Engine — Epoch 001.
Orchestrates prospective multi-channel SEC observation, append-only SQLite persistence,
transport integrity evaluation, content version tracking, and deterministic knowledge-time derivation.
"""

import hashlib
import json
import sqlite3
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.smart_money.canonicalization import compute_canonical_hash
from analyst_dashboard.smart_money.public_availability.models import (
    EndpointChannel,
    ChannelObservationRole,
    CHANNEL_ROLE_MAP,
    PublicObservationClass,
    TransportIntegrity,
    ClockStatus,
    DecisionInputCompleteness,
    DecisionInputReasonCode,
    ContentChangeClass,
    ChannelOrder,
    CANONICAL_DECISION_INPUT_CONTRACT_ID,
    CANONICAL_DECISION_INPUT_CONTRACT_SHA256,
    ClockAuthorityRecord,
    PublicObservationRecord,
    PublicContentVersionRecord,
    PublicKnowledgeDerivationRecord,
)
from analyst_dashboard.smart_money.public_availability.clock_authority import ClockAuthority
from analyst_dashboard.smart_money.public_availability.fair_access import (
    get_arx_access_policy_record,
    ARX_POLICY_ID,
)
from analyst_dashboard.smart_money.public_availability.decision_validator import (
    parse_and_validate_form4_decision_inputs,
)


from contextlib import contextmanager


class PublicObserverEngine:
    """
    Append-only public observation authority engine.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()

            # 1. Source Policies
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sec_public_observer_source_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_version TEXT NOT NULL,
                    official_sec_source_url TEXT NOT NULL,
                    official_sec_policy_sha256 TEXT NOT NULL,
                    official_sec_fetched_at TEXT NOT NULL,
                    max_requests_per_second REAL NOT NULL,
                    user_agent_format TEXT NOT NULL,
                    policy_content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # 2. Observer Runs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sec_public_observer_runs (
                    run_id TEXT PRIMARY KEY,
                    policy_id TEXT NOT NULL,
                    policy_hash TEXT NOT NULL,
                    implementation_git_sha TEXT NOT NULL,
                    clock_source TEXT NOT NULL,
                    clock_sync_method TEXT NOT NULL,
                    clock_offset_ms REAL NOT NULL,
                    clock_uncertainty_ms REAL NOT NULL,
                    clock_status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    FOREIGN KEY (policy_id) REFERENCES sec_public_observer_source_policies(policy_id)
                )
            """)

            # 3. Observations
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sec_public_observations (
                    observation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    endpoint_class TEXT NOT NULL,
                    endpoint_url TEXT NOT NULL,
                    request_sequence INTEGER NOT NULL,
                    request_started_at TEXT NOT NULL,
                    response_completed_at TEXT NOT NULL,
                    clock_uncertainty_ms REAL NOT NULL,
                    conservative_observation_time TEXT NOT NULL,
                    earliest_possible_time TEXT NOT NULL,
                    latest_possible_time TEXT NOT NULL,
                    http_status INTEGER NOT NULL,
                    transport_integrity TEXT NOT NULL,
                    bytes_received INTEGER NOT NULL,
                    raw_response_sha256 TEXT NOT NULL,
                    raw_response_retained INTEGER NOT NULL,
                    identity_valid INTEGER NOT NULL,
                    structural_valid INTEGER NOT NULL,
                    decision_input_completeness TEXT NOT NULL,
                    decision_input_reason_code TEXT,
                    observation_class TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES sec_public_observer_runs(run_id)
                )
            """)

            # 4. Content Versions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sec_public_content_versions (
                    version_id TEXT PRIMARY KEY,
                    observation_id TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    endpoint_class TEXT NOT NULL,
                    raw_content_sha256 TEXT NOT NULL,
                    normalized_semantic_hash TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    content_change_class TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (observation_id) REFERENCES sec_public_observations(observation_id)
                )
            """)

            # 5. Knowledge Derivations
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sec_public_knowledge_derivations (
                    derivation_id TEXT PRIMARY KEY,
                    accession_number TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    decision_input_contract_id TEXT NOT NULL,
                    decision_input_contract_sha256 TEXT NOT NULL,
                    first_proven_metadata_public_at TEXT,
                    metadata_observation_id TEXT,
                    first_proven_content_public_at TEXT,
                    content_observation_id TEXT,
                    first_proven_decision_input_complete_at TEXT,
                    decision_input_observation_id TEXT,
                    conservative_knowledge_time TEXT,
                    exact_first_public_available_at TEXT NOT NULL,
                    channel_order TEXT NOT NULL,
                    derived_from_observation_ids_json TEXT NOT NULL,
                    derivation_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Indices
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_acc ON sec_public_observations(accession_number)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_run ON sec_public_observations(run_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cv_acc ON sec_public_content_versions(accession_number)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kd_acc ON sec_public_knowledge_derivations(accession_number)")
            conn.commit()

    def register_policy(self) -> None:
        policy = get_arx_access_policy_record()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO sec_public_observer_source_policies (
                    policy_id, policy_version, official_sec_source_url, official_sec_policy_sha256,
                    official_sec_fetched_at, max_requests_per_second, user_agent_format,
                    policy_content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                policy.policy_id, policy.policy_version, policy.official_sec_source_url,
                policy.official_sec_policy_sha256, policy.official_sec_fetched_at,
                policy.max_requests_per_second, policy.user_agent_format,
                policy.policy_content_hash, now_iso
            ))
            conn.commit()

    def start_observer_run(
        self,
        run_id: str,
        implementation_git_sha: str,
        clock_record: ClockAuthorityRecord,
    ) -> None:
        self.register_policy()
        policy = get_arx_access_policy_record()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO sec_public_observer_runs (
                    run_id, policy_id, policy_hash, implementation_git_sha,
                    clock_source, clock_sync_method, clock_offset_ms, clock_uncertainty_ms,
                    clock_status, started_at, completed_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'RUNNING')
            """, (
                run_id, policy.policy_id, policy.policy_content_hash, implementation_git_sha,
                clock_record.clock_source, clock_record.clock_sync_method, clock_record.clock_offset_ms,
                clock_record.clock_uncertainty_ms, clock_record.clock_status.value,
                clock_record.clock_observed_at
            ))
            conn.commit()

    def complete_observer_run(self, run_id: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE sec_public_observer_runs
                SET completed_at = ?, status = 'COMPLETED'
                WHERE run_id = ?
            """, (now_iso, run_id))
            conn.commit()

    @staticmethod
    def normalize_xml_semantic(xml_str: str) -> str:
        """Strips whitespace and comments to create canonical semantic hash."""
        s = re.sub(r"<!--.*?-->", "", xml_str, flags=re.DOTALL)
        s = re.sub(r">\s+<", "><", s.strip())
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def record_observation(
        self,
        run_id: str,
        accession_number: str,
        issuer_cik: str,
        form_type: str,
        endpoint_class: EndpointChannel,
        endpoint_url: str,
        request_sequence: int,
        request_started_at: str,
        response_completed_at: str,
        clock_uncertainty_ms: float,
        http_status: int,
        raw_body: bytes,
        content_length_header: Optional[int] = None,
        observed_accession: Optional[str] = None,
        observed_cik: Optional[str] = None,
        observed_form: Optional[str] = None,
    ) -> PublicObservationRecord:
        """
        Processes a raw observation response and persists it to the append-only ledger.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        bytes_received = len(raw_body)
        raw_sha256 = hashlib.sha256(raw_body).hexdigest()

        # Hard Invariant: REQUEST_START_USED_AS_KNOWLEDGE_TIME = NO
        earliest_time, conservative_time, latest_time = ClockAuthority.derive_observation_intervals(
            response_completed_at, clock_uncertainty_ms
        )

        # 1. Transport integrity evaluation
        if content_length_header is not None and content_length_header != bytes_received:
            transport_integrity = TransportIntegrity.TRUNCATED
        else:
            transport_integrity = TransportIntegrity.COMPLETE

        # 2. Classification logic
        decision_completeness = DecisionInputCompleteness.UNKNOWN
        decision_reason: Optional[str] = None
        identity_valid = False
        structural_valid = False

        if http_status in (403, 429, 500, 502, 503, 504, -1):
            # Indeterminate network/server failure -> TRANSPORT_INDETERMINATE
            obs_class = PublicObservationClass.TRANSPORT_INDETERMINATE
            transport_integrity = TransportIntegrity.UNKNOWN
        elif transport_integrity == TransportIntegrity.TRUNCATED:
            obs_class = PublicObservationClass.TRANSPORT_INCOMPLETE
        elif http_status == 404:
            # 404 on endpoint is ENDPOINT_NOT_PRESENT, NEVER SEC_WIDE_NOT_PUBLIC
            obs_class = PublicObservationClass.ENDPOINT_NOT_PRESENT
        elif http_status == 200:
            # Validate identity where target is specified
            expected_acc_clean = accession_number.strip().replace("-", "")
            obs_acc_clean = (observed_accession or accession_number).strip().replace("-", "")
            expected_cik_clean = issuer_cik.lstrip("0")
            obs_cik_clean = (observed_cik or issuer_cik).lstrip("0")
            expected_form_clean = form_type.strip().upper()
            obs_form_clean = (observed_form or form_type).strip().upper()

            identity_valid = (
                (expected_acc_clean == obs_acc_clean)
                and (expected_cik_clean == obs_cik_clean)
                and (expected_form_clean == obs_form_clean)
            )

            if not identity_valid:
                obs_class = PublicObservationClass.IDENTITY_INVALID
                decision_completeness = DecisionInputCompleteness.INCOMPLETE
                decision_reason = DecisionInputReasonCode.SOURCE_IDENTITY_CONFLICT.value
            else:
                # Identity matches. Evaluate channel and content
                role = CHANNEL_ROLE_MAP.get(endpoint_class, ChannelObservationRole.CONTENT_PUBLIC_OBSERVATION)
                if role == ChannelObservationRole.METADATA_PUBLIC_OBSERVATION:
                    structural_valid = True
                    obs_class = PublicObservationClass.METADATA_PUBLIC
                elif endpoint_class in (EndpointChannel.OWNERSHIP_XML, EndpointChannel.COMPLETE_SUBMISSION_TEXT, EndpointChannel.PRIMARY_FORM_DOCUMENT):
                    # Form 4 XML / content evaluation
                    xml_text = raw_body.decode("utf-8", errors="replace")
                    # Check basic XML structure
                    try:
                        import xml.etree.ElementTree as ET
                        ET.fromstring(xml_text)
                        structural_valid = True
                    except Exception:
                        structural_valid = False

                    if not structural_valid:
                        obs_class = PublicObservationClass.STRUCTURALLY_INVALID
                        decision_completeness = DecisionInputCompleteness.INCOMPLETE
                        decision_reason = DecisionInputReasonCode.REQUIRED_DOCUMENT_MISSING.value
                    else:
                        completeness, reason_code, _ = parse_and_validate_form4_decision_inputs(xml_text)
                        decision_completeness = completeness
                        decision_reason = reason_code.value if reason_code else None

                        if completeness == DecisionInputCompleteness.COMPLETE:
                            obs_class = PublicObservationClass.DECISION_INPUT_COMPLETE
                        else:
                            obs_class = PublicObservationClass.DECISION_INPUT_INCOMPLETE
                else:
                    # Generic content observation (e.g. index page)
                    structural_valid = True
                    obs_class = PublicObservationClass.CONTENT_PUBLIC
        else:
            obs_class = PublicObservationClass.TRANSPORT_INDETERMINATE

        obs_id = f"obs_{hashlib.sha256(f'{run_id}_{accession_number}_{endpoint_class.value}_{request_sequence}_{raw_sha256}'.encode('utf-8')).hexdigest()[:16]}"

        record = PublicObservationRecord(
            observation_id=obs_id,
            run_id=run_id,
            accession_number=accession_number,
            issuer_cik=issuer_cik,
            form_type=form_type,
            endpoint_class=endpoint_class,
            endpoint_url=endpoint_url,
            request_sequence=request_sequence,
            request_started_at=request_started_at,
            response_completed_at=response_completed_at,
            clock_uncertainty_ms=clock_uncertainty_ms,
            conservative_observation_time=conservative_time,
            earliest_possible_time=earliest_time,
            latest_possible_time=latest_time,
            http_status=http_status,
            transport_integrity=transport_integrity,
            bytes_received=bytes_received,
            raw_response_sha256=raw_sha256,
            raw_response_retained=True,
            identity_valid=identity_valid,
            structural_valid=structural_valid,
            decision_input_completeness=decision_completeness,
            decision_input_reason_code=decision_reason,
            observation_class=obs_class,
            created_at=now_iso,
        )

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO sec_public_observations (
                    observation_id, run_id, accession_number, issuer_cik, form_type,
                    endpoint_class, endpoint_url, request_sequence, request_started_at,
                    response_completed_at, clock_uncertainty_ms, conservative_observation_time,
                    earliest_possible_time, latest_possible_time, http_status, transport_integrity,
                    bytes_received, raw_response_sha256, raw_response_retained, identity_valid,
                    structural_valid, decision_input_completeness, decision_input_reason_code,
                    observation_class, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.observation_id, record.run_id, record.accession_number, record.issuer_cik,
                record.form_type, record.endpoint_class.value, record.endpoint_url,
                record.request_sequence, record.request_started_at, record.response_completed_at,
                record.clock_uncertainty_ms, record.conservative_observation_time,
                record.earliest_possible_time, record.latest_possible_time, record.http_status,
                record.transport_integrity.value, record.bytes_received, record.raw_response_sha256,
                1 if record.raw_response_retained else 0, 1 if record.identity_valid else 0,
                1 if record.structural_valid else 0, record.decision_input_completeness.value,
                record.decision_input_reason_code, record.observation_class.value, record.created_at
            ))

            # Handle Content Versioning if content endpoint returned 200
            if http_status == 200 and identity_valid and structural_valid:
                norm_hash = self.normalize_xml_semantic(raw_body.decode("utf-8", errors="replace"))
                cur.execute("""
                    SELECT raw_content_sha256, normalized_semantic_hash
                    FROM sec_public_content_versions
                    WHERE accession_number = ? AND endpoint_class = ?
                    ORDER BY rowid DESC LIMIT 1
                """, (accession_number, endpoint_class.value))
                prev = cur.fetchone()

                if not prev:
                    change_class = ContentChangeClass.INITIAL
                elif prev["raw_content_sha256"] == raw_sha256:
                    change_class = ContentChangeClass.INITIAL  # Idempotent re-observation
                elif prev["normalized_semantic_hash"] == norm_hash:
                    change_class = ContentChangeClass.REPRESENTATION_CHANGE
                else:
                    # Semantic hash differs
                    if record.decision_input_completeness == DecisionInputCompleteness.INCOMPLETE:
                        change_class = ContentChangeClass.CONTENT_AVAILABILITY_REGRESSION
                    else:
                        change_class = ContentChangeClass.SEMANTIC_CONTENT_CHANGE

                version_id = f"ver_{hashlib.sha256(f'{accession_number}_{endpoint_class.value}_{raw_sha256}'.encode('utf-8')).hexdigest()[:16]}"
                cur.execute("""
                    INSERT OR IGNORE INTO sec_public_content_versions (
                        version_id, observation_id, accession_number, endpoint_class,
                        raw_content_sha256, normalized_semantic_hash, observed_at,
                        content_change_class, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version_id, obs_id, accession_number, endpoint_class.value,
                    raw_sha256, norm_hash, response_completed_at, change_class.value, now_iso
                ))

            conn.commit()

        return record

    def derive_knowledge_for_accession(
        self,
        accession_number: str,
    ) -> Optional[PublicKnowledgeDerivationRecord]:
        """
        Deterministically derives knowledge-time records for an accession from all
        retained observations. Positive observations cannot be overridden by negative ones.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM sec_public_observations
                WHERE accession_number = ?
                ORDER BY response_completed_at ASC
            """, (accession_number,))
            rows = [dict(r) for r in cur.fetchall()]

            if not rows:
                return None

            issuer_cik = rows[0]["issuer_cik"]
            form_type = rows[0]["form_type"]

            # Filter valid observations by class
            valid_meta_obs = [
                r for r in rows
                if r["observation_class"] == PublicObservationClass.METADATA_PUBLIC.value
                and r["identity_valid"] == 1
            ]

            valid_content_obs = [
                r for r in rows
                if r["observation_class"] in (
                    PublicObservationClass.CONTENT_PUBLIC.value,
                    PublicObservationClass.DECISION_INPUT_COMPLETE.value,
                    PublicObservationClass.DECISION_INPUT_INCOMPLETE.value,
                )
                and r["identity_valid"] == 1
                and r["structural_valid"] == 1
            ]

            valid_decision_obs = [
                r for r in rows
                if r["observation_class"] == PublicObservationClass.DECISION_INPUT_COMPLETE.value
                and r["identity_valid"] == 1
                and r["structural_valid"] == 1
            ]

            # 1. Earliest metadata public conservative time
            if valid_meta_obs:
                min_meta = min(valid_meta_obs, key=lambda x: x["conservative_observation_time"])
                first_meta_time = min_meta["conservative_observation_time"]
                meta_obs_id = min_meta["observation_id"]
                meta_interval = (min_meta["earliest_possible_time"], min_meta["latest_possible_time"])
            else:
                first_meta_time = None
                meta_obs_id = None
                meta_interval = None

            # 2. Earliest content public conservative time
            if valid_content_obs:
                min_content = min(valid_content_obs, key=lambda x: x["conservative_observation_time"])
                first_content_time = min_content["conservative_observation_time"]
                content_obs_id = min_content["observation_id"]
                content_interval = (min_content["earliest_possible_time"], min_content["latest_possible_time"])
            else:
                first_content_time = None
                content_obs_id = None
                content_interval = None

            # 3. Earliest decision input complete conservative time
            if valid_decision_obs:
                min_decision = min(valid_decision_obs, key=lambda x: x["conservative_observation_time"])
                first_decision_time = min_decision["conservative_observation_time"]
                decision_obs_id = min_decision["observation_id"]
            else:
                first_decision_time = None
                decision_obs_id = None

            # Conservative knowledge time: requires decision input complete for Form 4
            conservative_knowledge = first_decision_time

            # 4. Channel Order evaluation (Clock Authority interval comparison)
            if meta_interval and content_interval:
                order_cmp = ClockAuthority.compare_clock_intervals(meta_interval, content_interval)
                if order_cmp == "A_PRECEDES_B":
                    channel_order = ChannelOrder.METADATA_PRECEDED_CONTENT
                elif order_cmp == "B_PRECEDES_A":
                    channel_order = ChannelOrder.CONTENT_PRECEDED_METADATA
                else:
                    channel_order = ChannelOrder.INDETERMINATE
            elif meta_interval:
                channel_order = ChannelOrder.METADATA_ONLY
            elif content_interval:
                channel_order = ChannelOrder.CONTENT_ONLY
            else:
                channel_order = ChannelOrder.NONE

            # Content version count and regression check
            cur.execute("""
                SELECT COUNT(*), content_change_class FROM sec_public_content_versions
                WHERE accession_number = ?
                GROUP BY content_change_class
            """, (accession_number,))
            version_rows = cur.fetchall()
            version_count = sum(r[0] for r in version_rows)
            has_semantic_change = any(r[1] == ContentChangeClass.SEMANTIC_CONTENT_CHANGE.value for r in version_rows)
            has_regression = any(r[1] == ContentChangeClass.CONTENT_AVAILABILITY_REGRESSION.value for r in version_rows)

            all_obs_ids = sorted([r["observation_id"] for r in rows])
            now_iso = datetime.now(timezone.utc).isoformat()

            derivation_payload = {
                "accession_number": accession_number,
                "issuer_cik": issuer_cik,
                "form_type": form_type,
                "contract_id": CANONICAL_DECISION_INPUT_CONTRACT_ID,
                "contract_sha256": CANONICAL_DECISION_INPUT_CONTRACT_SHA256,
                "first_proven_metadata_public_at": first_meta_time,
                "first_proven_content_public_at": first_content_time,
                "first_proven_decision_input_complete_at": first_decision_time,
                "conservative_knowledge_time": conservative_knowledge,
                "exact_first_public_available_at": "NOT_ESTABLISHED",
                "channel_order": channel_order.value,
                "observation_ids": all_obs_ids,
            }
            derivation_hash = compute_canonical_hash(derivation_payload)
            derivation_id = f"kd_{hashlib.sha256(f'{accession_number}_{derivation_hash}'.encode('utf-8')).hexdigest()[:16]}"

            derivation_record = PublicKnowledgeDerivationRecord(
                derivation_id=derivation_id,
                accession_number=accession_number,
                issuer_cik=issuer_cik,
                form_type=form_type,
                decision_input_contract_id=CANONICAL_DECISION_INPUT_CONTRACT_ID,
                decision_input_contract_sha256=CANONICAL_DECISION_INPUT_CONTRACT_SHA256,
                first_proven_metadata_public_at=first_meta_time,
                metadata_observation_id=meta_obs_id,
                first_proven_content_public_at=first_content_time,
                content_observation_id=content_obs_id,
                first_proven_decision_input_complete_at=first_decision_time,
                decision_input_observation_id=decision_obs_id,
                conservative_knowledge_time=conservative_knowledge,
                exact_first_public_available_at="NOT_ESTABLISHED",
                channel_order=channel_order,
                derived_from_observation_ids=all_obs_ids,
                derivation_hash=derivation_hash,
                created_at=now_iso,
                content_version_count=max(version_count, 1),
                has_semantic_content_change=has_semantic_change,
                has_content_regression=has_regression,
            )

            cur.execute("""
                INSERT OR REPLACE INTO sec_public_knowledge_derivations (
                    derivation_id, accession_number, issuer_cik, form_type,
                    decision_input_contract_id, decision_input_contract_sha256,
                    first_proven_metadata_public_at, metadata_observation_id,
                    first_proven_content_public_at, content_observation_id,
                    first_proven_decision_input_complete_at, decision_input_observation_id,
                    conservative_knowledge_time, exact_first_public_available_at,
                    channel_order, derived_from_observation_ids_json, derivation_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                derivation_record.derivation_id, derivation_record.accession_number,
                derivation_record.issuer_cik, derivation_record.form_type,
                derivation_record.decision_input_contract_id, derivation_record.decision_input_contract_sha256,
                derivation_record.first_proven_metadata_public_at, derivation_record.metadata_observation_id,
                derivation_record.first_proven_content_public_at, derivation_record.content_observation_id,
                derivation_record.first_proven_decision_input_complete_at, derivation_record.decision_input_observation_id,
                derivation_record.conservative_knowledge_time, derivation_record.exact_first_public_available_at,
                derivation_record.channel_order.value, json.dumps(derivation_record.derived_from_observation_ids),
                derivation_record.derivation_hash, derivation_record.created_at
            ))
            conn.commit()

            return derivation_record
