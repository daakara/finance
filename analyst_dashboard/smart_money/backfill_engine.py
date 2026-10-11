"""
Deterministic, Idempotent, and Resumable Historical SEC Form 4 Backfill Engine.
Enforces Knowledge-Time Semantics, Source Availability Ledger, and Survivorship Bias Safeguards.
Adheres strictly to ARX Terminal Smart Money Wave 2B Governance.
"""

import os
import json
import time
import sqlite3
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyFiling,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    DEFAULT_SMART_MONEY_DB,
)
from analyst_dashboard.smart_money.universe_builder import SmartMoneyUniverseBuilder
from analyst_dashboard.smart_money.backfill_models import (
    BackfillMode,
    BackfillStatus,
    ExecutionScopeType,
    FilingDisposition,
    AmendmentRelationship,
    SmartMoneyBackfillSpec,
    SmartMoneyBackfillManifest,
    SourceAvailabilityRecord,
    BackfillPlan,
    BackfillExecution,
    SealedExecutionManifest,
)

logger = logging.getLogger(__name__)


class SmartMoneyBackfillEngine:
    """
    Executes governed historical SEC Form 4 ingestion across a universe build.
    Provides crash-resilience, checkpoint-based resumption, strict idempotency,
    and point-in-time as-of retrieval.
    Enforces Backfill Identity V2: distinct execution scopes generate distinct execution IDs.
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        db_path: str = DEFAULT_SMART_MONEY_DB,
        ingestion_mgr: Optional[Form4IngestionManager] = None,
        universe_builder: Optional[SmartMoneyUniverseBuilder] = None,
    ):
        self.db_path = db_path
        self.ingestion_mgr = ingestion_mgr or Form4IngestionManager(db_path=self.db_path)
        self.universe_builder = universe_builder or SmartMoneyUniverseBuilder()

    def create_backfill_plan(
        self,
        universe_build_id: str,
        backfill_mode: str,
        requested_date_start: str,
        requested_date_end: str,
        source_contract_version: str = "1.0.0",
        normalization_version: str = "1.0.0",
        parser_version: str = "1.0.0",
        eligibility_policy_version: str = "sm-eligibility-v2.0",
        cik_mapping_policy_version: str = "cik-mapping-v1.0",
        request_policy_version: str = "1.0.0",
    ) -> SmartMoneyBackfillSpec:
        """
        Creates and validates an immutable backfill specification.
        Enforces Mode A vs Mode B gating.
        """
        if backfill_mode == BackfillMode.PIT_HISTORICAL_UNIVERSE_BACKFILL.value:
            raise ValueError(
                "PIT_HISTORICAL_UNIVERSE_BACKFILL_NOT_AUTHORIZED: "
                "Point-in-time historical universe authority is BLOCKED_PENDING_HISTORICAL_UNIVERSE_AUTHORITY. "
                "Use CURRENT_UNIVERSE_HISTORICAL_BACKFILL."
            )
        elif backfill_mode != BackfillMode.CURRENT_UNIVERSE_HISTORICAL_BACKFILL.value:
            raise ValueError(f"Unknown backfill_mode: {backfill_mode}")

        manifest = self.universe_builder.get_build_manifest(universe_build_id)
        if not manifest:
            raise ValueError(f"Universe build {universe_build_id} not found.")

        spec = SmartMoneyBackfillSpec(
            universe_build_id=universe_build_id,
            universe_content_hash=manifest.member_identity_hash,
            backfill_mode=backfill_mode,
            requested_date_start=requested_date_start,
            requested_date_end=requested_date_end,
            source_contract_version=source_contract_version,
            normalization_version=normalization_version,
            parser_version=parser_version,
            eligibility_policy_version=eligibility_policy_version,
            cik_mapping_policy_version=cik_mapping_policy_version,
            request_policy_version=request_policy_version,
        )
        return spec

    def create_execution(
        self,
        plan: Any,
        target_symbols: Optional[List[str]] = None,
        max_issuers: Optional[int] = None,
        max_filings_per_issuer: Optional[int] = None,
        stage_label: str = "EXECUTION",
    ) -> BackfillExecution:
        """
        Binds plan identity to concrete target scope, batch limits, and stage label.
        Deterministic scope hash guarantees zero identity collisions across stages.
        """
        plan_obj: BackfillPlan = plan.to_plan() if hasattr(plan, "to_plan") else plan
        members = self.universe_builder.get_scannable_members(plan_obj.universe_build_id)
        members = sorted(members, key=lambda m: m["symbol"])

        if target_symbols:
            target_set = {s.upper().strip() for s in target_symbols}
            filtered = [m for m in members if m["symbol"] in target_set]
            scope_type = ExecutionScopeType.EXPLICIT_TARGETS.value
        elif max_issuers is not None:
            filtered = members[:max_issuers]
            scope_type = ExecutionScopeType.TOP_N_UNIVERSE.value
        else:
            filtered = members
            scope_type = ExecutionScopeType.FULL_UNIVERSE.value

        t_symbols = [m["symbol"] for m in filtered]
        t_ciks = [m["cik"] for m in filtered]

        return BackfillExecution(
            plan_id=plan_obj.get_plan_id(),
            plan_hash=plan_obj.compute_plan_hash(),
            scope_type=scope_type,
            target_symbols=t_symbols,
            target_ciks=t_ciks,
            stage_label=stage_label,
            max_issuers=max_issuers,
            max_filings_per_issuer=max_filings_per_issuer,
            execution_policy_version="exec-v2.0",
        )

    def get_completed_issuers_for_backfill(self, backfill_id: str) -> Set[str]:
        """Queries completed issuer CIKs for crash-recovery resumption."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT issuer_cik FROM sm_backfill_checkpoints WHERE backfill_id = ? AND status = 'COMPLETED'",
                (backfill_id,),
            )
            rows = cursor.fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def record_issuer_checkpoint(
        self,
        backfill_id: str,
        issuer_cik: str,
        symbol: str,
        filings_count: int,
        status: str = "COMPLETED",
    ):
        """Persists durable issuer checkpoint for resume safety."""
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sm_backfill_checkpoints
                (backfill_id, issuer_cik, symbol, status, filings_count, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (backfill_id, issuer_cik, symbol, status, filings_count, now_str),
            )
            conn.commit()
        finally:
            conn.close()

    def get_completed_issuers_for_execution(self, execution_id: str) -> Set[str]:
        """Queries completed issuer CIKs for a specific execution."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT issuer_cik FROM sm_execution_checkpoints WHERE execution_id = ? AND status = 'COMPLETED'",
                (execution_id,),
            )
            rows = cursor.fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def record_execution_checkpoint(
        self,
        execution_id: str,
        issuer_cik: str,
        symbol: str,
        filings_count: int,
        status: str = "COMPLETED",
    ):
        """Persists durable execution checkpoint for resume safety."""
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sm_execution_checkpoints
                (execution_id, issuer_cik, symbol, status, filings_count, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (execution_id, issuer_cik, symbol, status, filings_count, now_str),
            )
            conn.commit()
        finally:
            conn.close()

    def save_execution_manifest(self, manifest: SealedExecutionManifest):
        """
        Persists sealed execution manifest to sm_backfill_executions.
        Enforces EXECUTION_MANIFEST_OVERWRITE_COUNT = 0 on sealed manifests.
        """
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM sm_backfill_executions WHERE execution_id = ?",
                (manifest.execution_id,),
            )
            existing = cursor.fetchone()
            if existing:
                existing_status = existing[0]
                if existing_status in (
                    BackfillStatus.SEALED_COMPLETE.value,
                    BackfillStatus.SEALED_PARTIAL.value,
                    BackfillStatus.SEALED_FAILED.value,
                ):
                    raise RuntimeError(
                        f"EXECUTION_MANIFEST_SEALED: Cannot overwrite sealed execution manifest {manifest.execution_id}"
                    )

            cursor.execute(
                """
                INSERT OR REPLACE INTO sm_backfill_executions
                (execution_id, plan_id, scope_hash, stage_label, universe_build_id, status, manifest_json, created_at, sealed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.execution_id,
                    manifest.plan_id,
                    manifest.scope_hash,
                    manifest.stage_label,
                    manifest.universe_build_id,
                    manifest.status,
                    json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")),
                    manifest.started_at,
                    now_str if manifest.is_sealed else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_execution_manifest(self, execution_id: str) -> Optional[SealedExecutionManifest]:
        """Loads a stored execution manifest from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT manifest_json FROM sm_backfill_executions WHERE execution_id = ?", (execution_id,))
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return SealedExecutionManifest(**data)
            return None
        finally:
            conn.close()

    def run_execution(
        self,
        execution: BackfillExecution,
        plan: BackfillPlan,
        resume: bool = True,
        force_refresh: bool = False,
    ) -> SealedExecutionManifest:
        """
        Executes an idempotent, scoped historical backfill execution.
        Preserves checkpoint progression and enforces sealed manifest immutability.
        """
        execution_id = execution.get_execution_id()

        # Idempotency / Immuntability check
        existing = self.get_execution_manifest(execution_id)
        if existing and existing.is_sealed and not force_refresh:
            return existing

        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        legacy_bf_id = f"BF-{plan.compute_plan_hash()[:16]}"
        completed_ciks = (
            self.get_completed_issuers_for_execution(execution_id)
            | self.get_completed_issuers_for_backfill(execution_id)
            | self.get_completed_issuers_for_backfill(legacy_bf_id)
            | self.get_completed_issuers_for_backfill(plan.get_plan_id())
        ) if resume else set()

        manifest = SealedExecutionManifest(
            execution_id=execution_id,
            plan_id=execution.plan_id,
            plan_hash=execution.plan_hash,
            scope_hash=execution.compute_scope_hash(),
            stage_label=execution.stage_label,
            scope_type=execution.scope_type,
            universe_build_id=plan.universe_build_id,
            universe_content_hash=plan.universe_content_hash,
            requested_start=plan.requested_date_start,
            requested_end=plan.requested_date_end,
            started_at=started_at,
            target_symbols=list(execution.target_symbols),
            target_ciks=list(execution.target_ciks),
            issuer_count_requested=len(execution.target_symbols),
            status=BackfillStatus.RUNNING.value,
        )

        disposition_counts = {disp.value: 0 for disp in FilingDisposition}
        seen_accessions: Set[str] = set()
        seen_record_keys: Set[str] = set()

        all_accessions_for_manifest: List[Tuple[str, str]] = []
        all_transactions_for_manifest: List[Tuple[str, str]] = []

        targets = list(zip(execution.target_symbols, execution.target_ciks))

        for symbol, cik in targets:
            manifest.issuer_count_attempted += 1

            if resume and cik in completed_ciks and not force_refresh:
                manifest.issuer_count_completed += 1
                txs = self.ingestion_mgr.get_parsed_transactions(cik=cik, min_date=plan.requested_date_start)
                for tx in txs:
                    if tx.accession_number not in seen_accessions:
                        seen_accessions.add(tx.accession_number)
                        manifest.filings_cached += 1
                        cached_raw = self.ingestion_mgr.get_cached_raw_filing(tx.accession_number)
                        s_hash = cached_raw.get("source_content_hash") if cached_raw else tx.content_hash
                        all_accessions_for_manifest.append((tx.accession_number, s_hash or ""))
                    if tx.record_key not in seen_record_keys:
                        seen_record_keys.add(tx.record_key)
                        manifest.transactions_parsed += 1
                        if tx.is_open_market_purchase_evidence:
                            manifest.open_market_purchase_count += 1
                        all_transactions_for_manifest.append((tx.record_key, tx.content_hash))
                continue

            try:
                filings = self.ingestion_mgr.ingest_issuer(
                    symbol=symbol,
                    cik=cik,
                    date_start=plan.requested_date_start,
                    date_end=plan.requested_date_end,
                    max_filings=execution.max_filings_per_issuer,
                    force_refresh=force_refresh,
                )

                for f in filings:
                    disp = f.disposition if hasattr(f, "disposition") and f.disposition else FilingDisposition.FETCHED_AND_PARSED.value
                    if disp in disposition_counts:
                        disposition_counts[disp] += 1
                    else:
                        disposition_counts[disp] = 1

                    if f.accession_number in seen_accessions:
                        manifest.duplicates_detected += 1
                    else:
                        seen_accessions.add(f.accession_number)
                        manifest.filings_discovered += 1
                        manifest.filings_fetched += 1
                        all_accessions_for_manifest.append((f.accession_number, f.source_content_hash))

                    if f.is_amendment:
                        manifest.amendments_detected += 1

                    for tx in f.transactions:
                        if tx.record_key in seen_record_keys:
                            manifest.duplicates_detected += 1
                        else:
                            seen_record_keys.add(tx.record_key)
                            manifest.transactions_parsed += 1
                            if tx.is_open_market_purchase_evidence:
                                manifest.open_market_purchase_count += 1
                            all_transactions_for_manifest.append((tx.record_key, tx.content_hash))

                self.record_execution_checkpoint(execution_id, cik, symbol, len(filings), status="COMPLETED")
                self.record_issuer_checkpoint(execution_id, cik, symbol, len(filings), status="COMPLETED")
                self.record_issuer_checkpoint(legacy_bf_id, cik, symbol, len(filings), status="COMPLETED")
                self.record_issuer_checkpoint(plan.get_plan_id(), cik, symbol, len(filings), status="COMPLETED")
                manifest.issuer_count_completed += 1

            except Exception as e:
                logger.error(f"Error backfilling issuer {symbol} (CIK {cik}): {e}")
                manifest.issuer_count_failed += 1
                manifest.filings_failed += 1

        all_accessions_for_manifest.sort(key=lambda x: x[0])
        all_transactions_for_manifest.sort(key=lambda x: x[0])

        acc_serialized = json.dumps(all_accessions_for_manifest, sort_keys=True, separators=(",", ":"))
        tx_serialized = json.dumps(all_transactions_for_manifest, sort_keys=True, separators=(",", ":"))

        manifest.source_manifest_hash = hashlib.sha256(acc_serialized.encode("utf-8")).hexdigest()
        manifest.transaction_manifest_hash = hashlib.sha256(tx_serialized.encode("utf-8")).hexdigest()

        manifest.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest.disposition_counts = disposition_counts
        manifest.status = (
            BackfillStatus.SEALED_COMPLETE.value
            if manifest.issuer_count_failed == 0
            else BackfillStatus.SEALED_PARTIAL.value
        )
        manifest.is_sealed = True

        # Persist sealed execution manifest
        self.save_execution_manifest(manifest)
        return manifest

    def run_backfill(
        self,
        spec: SmartMoneyBackfillSpec,
        max_issuers: Optional[int] = None,
        target_symbols: Optional[List[str]] = None,
        max_filings_per_issuer: Optional[int] = None,
        stage_label: Optional[str] = None,
        resume: bool = True,
        force_refresh: bool = False,
    ) -> SmartMoneyBackfillManifest:
        """
        Executes idempotent, staged historical backfill.
        Uses Backfill Identity V2 under the hood to ensure distinct scopes receive distinct IDs.
        """
        plan = spec.to_plan()
        label = stage_label or (f"STAGE_{max_issuers}" if max_issuers else ("EXPLICIT" if target_symbols else "FULL"))
        execution = self.create_execution(
            plan=plan,
            target_symbols=target_symbols,
            max_issuers=max_issuers,
            max_filings_per_issuer=max_filings_per_issuer,
            stage_label=label,
        )

        sealed_exec = self.run_execution(
            execution=execution,
            plan=plan,
            resume=resume,
            force_refresh=force_refresh,
        )

        manifest = SmartMoneyBackfillManifest(
            backfill_id=sealed_exec.execution_id,
            backfill_spec_hash=sealed_exec.plan_hash,
            backfill_mode=spec.backfill_mode,
            universe_build_id=sealed_exec.universe_build_id,
            universe_content_hash=sealed_exec.universe_content_hash,
            requested_start=sealed_exec.requested_start,
            requested_end=sealed_exec.requested_end,
            started_at=sealed_exec.started_at,
            completed_at=sealed_exec.completed_at,
            issuer_count_requested=sealed_exec.issuer_count_requested,
            issuer_count_attempted=sealed_exec.issuer_count_attempted,
            issuer_count_completed=sealed_exec.issuer_count_completed,
            issuer_count_failed=sealed_exec.issuer_count_failed,
            filings_discovered=sealed_exec.filings_discovered,
            filings_fetched=sealed_exec.filings_fetched,
            filings_cached=sealed_exec.filings_cached,
            filings_failed=sealed_exec.filings_failed,
            transactions_parsed=sealed_exec.transactions_parsed,
            transactions_rejected=sealed_exec.transactions_rejected,
            open_market_purchase_count=sealed_exec.open_market_purchase_count,
            duplicates_detected=sealed_exec.duplicates_detected,
            amendments_detected=sealed_exec.amendments_detected,
            source_manifest_hash=sealed_exec.source_manifest_hash,
            transaction_manifest_hash=sealed_exec.transaction_manifest_hash,
            status=(
                BackfillStatus.COMPLETE.value
                if sealed_exec.issuer_count_failed == 0
                else BackfillStatus.PARTIAL.value
            ),
            disposition_counts=sealed_exec.disposition_counts,
            execution_id=sealed_exec.execution_id,
            plan_id=sealed_exec.plan_id,
            scope_hash=sealed_exec.scope_hash,
            stage_label=sealed_exec.stage_label,
        )

        # Also persist to legacy manifests table using unique execution_id
        self._save_manifest_to_db(manifest)
        return manifest

    def _save_manifest_to_db(self, manifest: SmartMoneyBackfillManifest):
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sm_backfill_manifests
                (backfill_id, spec_hash, mode, universe_build_id, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.backfill_id,
                    manifest.backfill_spec_hash,
                    manifest.backfill_mode,
                    manifest.universe_build_id,
                    json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")),
                    now_str,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_manifest(self, backfill_id: str) -> Optional[SmartMoneyBackfillManifest]:
        """Loads a stored manifest from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT manifest_json FROM sm_backfill_manifests WHERE backfill_id = ?", (backfill_id,))
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return SmartMoneyBackfillManifest(**data)
        finally:
            conn.close()
        return None

    def get_transactions_as_of(
        self,
        as_of_time: str,
        symbol: Optional[str] = None,
        cik: Optional[str] = None,
        min_date: Optional[str] = None,
    ) -> List[SmartMoneyTransaction]:
        """
        Point-in-time query: Returns only transactions whose public SEC availability
        occurred at or before as_of_time (public_availability_at <= as_of_time).
        Guarantees zero look-ahead leakage.
        """
        return self.ingestion_mgr.get_transactions_as_of(
            as_of_time=as_of_time,
            symbol=symbol,
            cik=cik,
            min_date=min_date,
        )

    def replay_parser_for_accession(self, accession_number: str) -> Tuple[str, str]:
        """
        Fetches cached raw content from SQLite and re-parses it.
        Returns (source_content_hash, normalized_transactions_hash).
        """
        cached = self.ingestion_mgr.get_cached_raw_filing(accession_number)
        if not cached or not cached.get("raw_content"):
            raise ValueError(f"Filing {accession_number} not found in raw cache.")

        raw_content = cached["raw_content"]
        source_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

        filing = Form4XmlParser.parse_filing(
            raw_content=raw_content,
            accession_number=accession_number,
            filing_date_hint=cached.get("filing_date"),
            acceptance_datetime=cached.get("acceptance_datetime"),
            source_url=cached.get("source_url", ""),
            source_fetch_time=cached.get("fetched_at"),
        )

        tx_hashes = [tx.content_hash for tx in filing.transactions]
        tx_serialized = json.dumps(sorted(tx_hashes), sort_keys=True)
        tx_hash = hashlib.sha256(tx_serialized.encode("utf-8")).hexdigest()
        return source_hash, tx_hash

    def export_historical_corpus_manifest(self, backfill_id: str, output_path: str) -> str:
        """
        Writes SMART_MONEY_WAVE2B_HISTORICAL_CORPUS_MANIFEST.json and returns SHA-256.
        Works seamlessly with both execution IDs (EXEC-...) and backfill specs.
        """
        manifest = self.get_execution_manifest(backfill_id) or self.get_manifest(backfill_id)
        if not manifest:
            raise ValueError(f"Backfill {backfill_id} not found in catalog.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = manifest.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        sha = manifest.compute_manifest_sha256()
        return sha
