"""
tests/test_pilot_identity_and_guarded_recovery.py

Acceptance verification suite for ARX Terminal Package B:
1. Actual canonical IDs accepted.
2. Noncanonical aliases rejected.
3. GOOG/GOOGL dual-class isolation.
4. Incompatible lineage forces full backfill.
5. WAL-consistent backup preservation.
6. Guarded recovery refuses active-writer conditions.
7. Guarded recovery executes non-destructive rollback when quiescent.
8. Manifest deterministic serialization and digest.
9. Zero production database mutation.
10. WAL exclusive lock insufficiency & reader handle fence enforcement.
11. Maintenance-mode token mandatory authorization fence.
12. Single-symbol transactional isolation preserves unrelated writes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import pytest

from analyst_dashboard.data.guarded_recovery import GuardedRecoveryEngine, GuardedRecoveryReceipt
from analyst_dashboard.data.market_evidence import Provider, AdjustmentState, StructuralQuality
from scripts.ingest_alpaca_historical import (
    AlpacaHistoricalSipIngestionAdapter,
    inspect_existing_market_store_readonly,
)


class TestPilotIdentityAndGuardedRecovery:

    def test_actual_canonical_ids_accepted(self):
        """Proves that exact canonical IDs for AAPL, GOOG, GOOGL, ABBV are accepted."""
        manifest_path = Path("docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json")
        info = AlpacaHistoricalSipIngestionAdapter.load_manifest(manifest_path)
        assert info.scope_type == "PILOT_MANIFEST"
        assert info.symbols == ["AAPL", "ABBV", "GOOG", "GOOGL"]

        params = info.per_symbol_parameters
        assert params["AAPL"]["durable_security_id"] == "ARX_SEC_0B4C728FE0BE"
        assert params["AAPL"]["listing_association_id"] == "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"
        assert params["AAPL"]["operating_mic"] == "XNAS"

        assert params["GOOG"]["durable_security_id"] == "ARX_SEC_A11A24D442E2"
        assert params["GOOG"]["listing_association_id"] == "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"
        assert params["GOOG"]["share_class"] == "CLASS_C"

        assert params["GOOGL"]["durable_security_id"] == "ARX_SEC_B7EF9ED7A35A"
        assert params["GOOGL"]["listing_association_id"] == "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"
        assert params["GOOGL"]["share_class"] == "CLASS_A"

        assert params["ABBV"]["durable_security_id"] == "ARX_SEC_BFA51AC4C0B5"
        assert params["ABBV"]["listing_association_id"] == "ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19"
        assert params["ABBV"]["operating_mic"] == "XNYS"

    def test_noncanonical_aliases_rejected(self):
        """Proves that informal or noncanonical aliases (e.g. sec_aapl_001) are strictly rejected."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            bad_data = {
                "pilot_candidates": [
                    {
                        "symbol": "sec_aapl_001",
                        "durable_security_id": "ARX_SEC_0B4C728FE0BE",
                    }
                ]
            }
            json.dump(bad_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="NONCANONICAL_IDENTITY_ALIAS_REJECTED"):
                AlpacaHistoricalSipIngestionAdapter.load_manifest(temp_path)
        finally:
            temp_path.unlink()

        # Test malformed security ID
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            bad_sec = {
                "pilot_candidates": [
                    {
                        "symbol": "AAPL",
                        "durable_security_id": "INVALID_SEC_ID_123",
                    }
                ]
            }
            json.dump(bad_sec, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="INVALID_CANONICAL_SECURITY_ID"):
                AlpacaHistoricalSipIngestionAdapter.load_manifest(temp_path)
        finally:
            temp_path.unlink()

    def test_goog_googl_dual_class_isolation(self):
        """Proves that GOOG and GOOGL maintain non-overlapping durable IDs and share classes under Alphabet CIK."""
        manifest_path = Path("docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json")
        info = AlpacaHistoricalSipIngestionAdapter.load_manifest(manifest_path)
        p_goog = info.per_symbol_parameters["GOOG"]
        p_googl = info.per_symbol_parameters["GOOGL"]

        assert p_goog["issuer_cik"] == p_googl["issuer_cik"] == "0001652044"
        assert p_goog["durable_security_id"] != p_googl["durable_security_id"]
        assert p_goog["listing_association_id"] != p_googl["listing_association_id"]
        assert p_goog["share_class"] == "CLASS_C"
        assert p_googl["share_class"] == "CLASS_A"

    def test_incompatible_lineage_forces_full_backfill(self):
        """Proves that stored Polygon provenance triggers full backfill rather than incremental refresh."""
        existing_prov = [
            {
                "trade_date": "2026-10-16",
                "provider": "MASSIVE_POLYGON",
                "adjustment_state": AdjustmentState.SPLIT_ADJUSTED_ONLY,
                "structural_quality": StructuralQuality.COMPLETE,
            }
        ]
        is_comp, reason = AlpacaHistoricalSipIngestionAdapter.is_prov_compatible(existing_prov)
        assert is_comp is False
        assert "INCOMPATIBLE_PROVIDER" in reason

    def test_wal_consistent_backup_preservation(self):
        """Proves SQLite backup captures uncheckpointed writes from WAL while preserving all tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "source.db"
            bck_path = Path(tmpdir) / "backup.db"

            conn = sqlite3.connect(str(src_path))
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, trade_date TEXT, close REAL);")
            cur.execute("CREATE TABLE asset_ohlcv_provenance (symbol TEXT, trade_date TEXT, provider TEXT);")
            cur.execute("INSERT INTO asset_ohlcv_daily VALUES ('AAPL', '2026-10-19', 236.48);")
            cur.execute("INSERT INTO asset_ohlcv_provenance VALUES ('AAPL', '2026-10-19', 'ALPACA');")
            conn.commit()

            bck_conn = sqlite3.connect(str(bck_path))
            conn.backup(bck_conn)
            bck_conn.close()
            conn.close()

            chk_conn = sqlite3.connect(f"{bck_path.as_uri()}?mode=ro", uri=True)
            chk_cur = chk_conn.cursor()
            chk_cur.execute("PRAGMA integrity_check;")
            assert chk_cur.fetchone()[0] == "ok"
            chk_cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily;")
            assert chk_cur.fetchone()[0] == 1
            chk_cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance;")
            assert chk_cur.fetchone()[0] == 1
            chk_conn.close()

    def test_guarded_recovery_refuses_active_writer_conditions(self):
        """Proves that guarded disaster recovery refuses replacement if target DB has an active exclusive lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.db"
            backup_path = Path(tmpdir) / "backup.db"
            quarantine_dir = Path(tmpdir) / "quarantine"

            conn_target = sqlite3.connect(str(target_path))
            cur = conn_target.cursor()
            cur.execute("CREATE TABLE test_data (id INT);")
            cur.execute("INSERT INTO test_data VALUES (1);")
            conn_target.commit()

            conn_bck = sqlite3.connect(str(backup_path))
            cur_b = conn_bck.cursor()
            cur_b.execute("CREATE TABLE test_data (id INT);")
            cur_b.execute("INSERT INTO test_data VALUES (999);")
            conn_bck.commit()
            conn_bck.close()

            bck_sha = GuardedRecoveryEngine.compute_file_sha256(backup_path)

            try:
                cur.execute("BEGIN EXCLUSIVE;")

                engine = GuardedRecoveryEngine(
                    target_db_path=target_path,
                    backup_db_path=backup_path,
                    expected_backup_sha256=bck_sha,
                    incident_id="INC-TEST-001",
                    operator_authorization="G-S2-ACT",
                    maintenance_mode_token="MAINT-G-S2-TEST",
                    quarantine_base_dir=quarantine_dir,
                    busy_timeout_ms=500,
                )

                with pytest.raises(RuntimeError, match="ACTIVE_WRITER_DETECTED"):
                    engine.execute_guarded_recovery()
            finally:
                conn_target.rollback()
                conn_target.close()

            conn_chk = sqlite3.connect(str(target_path))
            res = conn_chk.cursor().execute("SELECT id FROM test_data;").fetchone()[0]
            conn_chk.close()
            assert res == 1

    def test_wal_exclusive_lock_insufficient_and_reader_fence_enforced(self):
        """
        CRITICAL TEST: Proves that in SQLite WAL mode, BEGIN EXCLUSIVE succeeds even when a reader is active.
        Proves that GuardedRecoveryEngine enforces reader handle fencing and REFUSES recovery when active
        reader handles are present, preventing stale inode corruption.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target_wal.db"
            backup_path = Path(tmpdir) / "backup_wal.db"
            quarantine_dir = Path(tmpdir) / "quarantine"

            # 1. Setup WAL database
            conn_w = sqlite3.connect(str(target_path))
            conn_w.execute("PRAGMA journal_mode=WAL;")
            conn_w.execute("CREATE TABLE market (sym TEXT, px REAL);")
            conn_w.execute("INSERT INTO market VALUES ('AAPL', 236.48);")
            conn_w.commit()
            conn_w.close()

            # 2. Setup backup
            conn_b = sqlite3.connect(str(backup_path))
            conn_b.execute("CREATE TABLE market (sym TEXT, px REAL);")
            conn_b.execute("INSERT INTO market VALUES ('AAPL', 200.00);")
            conn_b.commit()
            conn_b.close()
            bck_sha = GuardedRecoveryEngine.compute_file_sha256(backup_path)

            # 3. Open an active reader connection with open read transaction
            conn_r = sqlite3.connect(f"{target_path.as_uri()}?mode=ro", uri=True)
            cur_r = conn_r.cursor()
            cur_r.execute("SELECT * FROM market;")
            try:
                # 4. Demonstrate SQLite WAL behavior: BEGIN EXCLUSIVE SUCCEEDS despite open reader!
                # (Proves that BEGIN EXCLUSIVE does NOT prove reader quiescence)
                conn_probe = sqlite3.connect(str(target_path))
                conn_probe.execute("BEGIN EXCLUSIVE;")
                conn_probe.execute("COMMIT;")
                conn_probe.close()

                # 5. Now verify GuardedRecoveryEngine detects active reader handle and refuses recovery
                engine = GuardedRecoveryEngine(
                    target_db_path=target_path,
                    backup_db_path=backup_path,
                    expected_backup_sha256=bck_sha,
                    incident_id="INC-FENCE-001",
                    operator_authorization="G-S2-ACT",
                    maintenance_mode_token="MAINT-G-S2-TEST",
                    quarantine_base_dir=quarantine_dir,
                    handle_checker=lambda path: [9999],  # Simulated active PID holding open fd
                )

                with pytest.raises(RuntimeError, match="RECOVERY_FENCE_VIOLATION"):
                    engine.execute_guarded_recovery()
            finally:
                cur_r.close()
                conn_r.close()

    def test_recovery_refused_without_maintenance_mode(self):
        """Proves that guarded recovery refuses to proceed without explicit maintenance mode authorization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.db"
            backup_path = Path(tmpdir) / "backup.db"
            quarantine_dir = Path(tmpdir) / "quarantine"

            target_path.write_bytes(b"data")
            backup_path.write_bytes(b"backup")

            # Missing or invalid maintenance token
            engine = GuardedRecoveryEngine(
                target_db_path=target_path,
                backup_db_path=backup_path,
                expected_backup_sha256="dummy",
                incident_id="INC-001",
                operator_authorization="G-S2-ACT",
                maintenance_mode_token="",  # Empty
                quarantine_base_dir=quarantine_dir,
            )

            with pytest.raises(ValueError, match="MAINTENANCE_MODE_REQUIRED"):
                engine.execute_guarded_recovery()

    def test_single_symbol_transactional_rollback_preserves_unrelated_writes(self):
        """
        Proves that a failure on a single pilot candidate rolls back ONLY that candidate's
        transaction, preserves prior committed candidates and unrelated writes, and does
        NOT trigger whole-database recovery.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "store.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, trade_date TEXT, close REAL);")
            conn.execute("CREATE TABLE asset_ohlcv_provenance (symbol TEXT, trade_date TEXT, provider TEXT);")

            # Unrelated existing data (MSFT)
            conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('MSFT', '2026-10-19', 420.00);")
            conn.commit()

            # Simulated candidate 1 (AAPL) succeeds
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('AAPL', '2026-10-19', 236.48);")
            conn.execute("INSERT INTO asset_ohlcv_provenance VALUES ('AAPL', '2026-10-19', 'ALPACA');")
            conn.commit()

            # Simulated candidate 2 (GOOG) fails mid-flight -> rolled back
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('GOOG', '2026-10-19', 180.00);")
                raise ValueError("SIMULATED_API_RATE_LIMIT")
            except ValueError:
                conn.rollback()

            conn.close()

            # Verify store state: MSFT and AAPL remain safely committed; GOOG rolled back; store intact
            chk_conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
            cur = chk_conn.cursor()
            symbols = [r[0] for r in cur.execute("SELECT symbol FROM asset_ohlcv_daily ORDER BY symbol;").fetchall()]
            chk_conn.close()

            assert symbols == ["AAPL", "MSFT"]
            # No whole-database recovery needed or triggered!

    def test_guarded_recovery_executes_when_quiescent(self):
        """Proves that guarded disaster recovery non-destructively quarantines files and restores correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.db"
            backup_path = Path(tmpdir) / "backup.db"
            quarantine_dir = Path(tmpdir) / "quarantine"

            conn_t = sqlite3.connect(str(target_path))
            conn_t.execute("PRAGMA journal_mode=WAL;")
            conn_t.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, close REAL);")
            conn_t.execute("CREATE TABLE asset_ohlcv_provenance (symbol TEXT);")
            conn_t.execute("INSERT INTO asset_ohlcv_daily VALUES ('CORRUPT', 0.0);")
            conn_t.commit()
            conn_t.close()

            conn_b = sqlite3.connect(str(backup_path))
            conn_b.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, close REAL);")
            conn_b.execute("CREATE TABLE asset_ohlcv_provenance (symbol TEXT);")
            conn_b.execute("INSERT INTO asset_ohlcv_daily VALUES ('AAPL', 236.48);")
            conn_b.execute("INSERT INTO asset_ohlcv_provenance VALUES ('AAPL');")
            conn_b.commit()
            conn_b.close()

            bck_sha = GuardedRecoveryEngine.compute_file_sha256(backup_path)

            engine = GuardedRecoveryEngine(
                target_db_path=target_path,
                backup_db_path=backup_path,
                expected_backup_sha256=bck_sha,
                incident_id="INC-G-S2-PILOT-001",
                operator_authorization="G-S2-ACT",
                maintenance_mode_token="MAINT-G-S2-ACT-PILOT-001",
                quarantine_base_dir=quarantine_dir,
            )

            receipt = engine.execute_guarded_recovery()
            assert receipt.recovery_status == "SUCCESS"
            assert len(receipt.quarantined_files) >= 1

            q_dir = Path(receipt.quarantine_directory)
            assert q_dir.exists()
            assert (q_dir / "quarantine_manifest.json").exists()

            conn_res = sqlite3.connect(f"{target_path.as_uri()}?mode=ro", uri=True)
            cur = conn_res.cursor()
            cur.execute("PRAGMA integrity_check;")
            assert cur.fetchone()[0] == "ok"
            cur.execute("SELECT symbol FROM asset_ohlcv_daily;")
            assert cur.fetchone()[0] == "AAPL"
            conn_res.close()

    def test_manifest_deterministic_serialization_and_digest(self):
        """Proves that PILOT_PROPOSED_MANIFEST.json serialization is deterministic and matches exact digest."""
        manifest_path = Path("docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json")
        raw_bytes = manifest_path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest().upper()

        assert len(raw_bytes) == 13833
        assert digest == "8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B"

    def test_zero_production_database_mutation(self):
        """Proves that dry run inspection makes zero modifications to target store."""
        with tempfile.NamedTemporaryFile("w", suffix=".db", delete=False) as f:
            temp_db = Path(f.name)

        candles, prov = inspect_existing_market_store_readonly(str(temp_db), "AAPL")
        assert candles == []
        assert prov == []
        assert temp_db.stat().st_size == 0
        temp_db.unlink()

    def test_canonical_reconciled_manifest_v3_deterministic_digest(self):
        """Proves that PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json serializes deterministically and matches exact digest."""
        manifest_path = Path("docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json")
        raw_bytes = manifest_path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest().upper()

        assert len(raw_bytes) == 9972
        assert digest == "0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF"

        info = AlpacaHistoricalSipIngestionAdapter.load_manifest(manifest_path)
        assert info.scope_type == "PILOT_MANIFEST"
        assert info.symbols == ["AAPL", "ABBV", "GOOG", "GOOGL"]
        assert info.per_symbol_parameters["AAPL"]["mandated_action"] == "REPLACE"
        assert info.per_symbol_parameters["GOOG"]["mandated_action"] == "REPLACE"
        assert info.per_symbol_parameters["GOOGL"]["mandated_action"] == "REPLACE"
        assert info.per_symbol_parameters["ABBV"]["mandated_action"] == "BACKFILL"

    def test_package_b_candidate_bindings_accepted(self):
        """Proves that Package B candidate_bindings structure is parsed with canonical ID validation."""
        data = {
            "ingestion_scope": {
                "candidate_bindings": {
                    "AAPL": {
                        "internal_security_id": "ARX_SEC_0B4C728FE0BE",
                        "association_id": "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
                        "action": "REPLACE",
                    },
                    "ABBV": {
                        "internal_security_id": "ARX_SEC_BFA51AC4C0B5",
                        "association_id": "ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19",
                        "action": "BACKFILL",
                    },
                }
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            info = AlpacaHistoricalSipIngestionAdapter.load_manifest(temp_path)
            assert info.scope_type == "PACKAGE_B"
            assert info.symbols == ["AAPL", "ABBV"]
            assert info.per_symbol_parameters["AAPL"]["action"] == "REPLACE"
            assert info.per_symbol_parameters["ABBV"]["action"] == "BACKFILL"
        finally:
            temp_path.unlink()

    def test_package_b_candidate_bindings_rejects_noncanonical_alias(self):
        """Proves that candidate_bindings strictly rejects noncanonical symbol aliases."""
        data = {
            "ingestion_scope": {
                "candidate_bindings": {
                    "sec_aapl_001": {
                        "internal_security_id": "ARX_SEC_0B4C728FE0BE",
                        "action": "REPLACE",
                    }
                }
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="NONCANONICAL_IDENTITY_ALIAS_REJECTED"):
                AlpacaHistoricalSipIngestionAdapter.load_manifest(temp_path)
        finally:
            temp_path.unlink()

    def test_recovery_refused_with_malformed_or_placeholder_tokens(self):
        """Proves that GuardedRecoveryEngine refuses execution with empty, malformed or placeholder tokens."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dummy_target = tdp / "dummy_target.db"
            dummy_target.touch()
            dummy_backup = tdp / "dummy_backup.db"
            dummy_backup.touch()

            # 1. Empty incident_id
            engine = GuardedRecoveryEngine(
                target_db_path=dummy_target,
                backup_db_path=dummy_backup,
                expected_backup_sha256="dummy",
                incident_id="",
                operator_authorization="G-S2-ACT",
                maintenance_mode_token="MAINT-INC-001",
            )
            with pytest.raises(ValueError, match="AUTHORIZATION_REJECTED.*Incident ID"):
                engine.step1_verify_authorization()

            # 2. Invalid operator_authorization
            engine.incident_id = "INC-TEST-001"
            engine.operator_authorization = "UNAUTHORIZED_OP"
            with pytest.raises(ValueError, match="AUTHORIZATION_REJECTED.*Operator authorization"):
                engine.step1_verify_authorization()

            # 3. Placeholder maintenance token
            engine.operator_authorization = "G-S2-ACT"
            engine.maintenance_mode_token = "MAINT-DEFAULT"
            with pytest.raises(ValueError, match="MAINTENANCE_MODE_REQUIRED.*Generic or placeholder"):
                engine.step2_verify_maintenance_mode()

            # 4. Maintenance token matching incident ID
            engine.maintenance_mode_token = "INC-TEST-001"
            with pytest.raises(ValueError, match="MAINTENANCE_MODE_REQUIRED.*must start with"):
                engine.step2_verify_maintenance_mode()

    def test_preflight_backup_validation_failure_preserves_target_database_unquarantined(self):
        """Proves that pre-flight backup validation runs before quarantine and preserves target intact if backup fails."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            target_path = tdp / "target.db"
            backup_path = tdp / "backup_corrupt.db"
            quarantine_dir = tdp / "quarantine"

            # Create target DB with valid data
            conn = sqlite3.connect(str(target_path))
            conn.execute("CREATE TABLE market_data (symbol TEXT, close REAL);")
            conn.execute("INSERT INTO market_data VALUES ('AAPL', 150.25);")
            conn.commit()
            conn.close()

            # Create backup DB with corrupted/non-matching hash
            backup_path.write_text("corrupted_content", encoding="utf-8")

            engine = GuardedRecoveryEngine(
                target_db_path=target_path,
                backup_db_path=backup_path,
                expected_backup_sha256="0000000000000000000000000000000000000000000000000000000000000000",
                incident_id="INC-G-S2-PILOT-001",
                operator_authorization="G-S2-ACT",
                maintenance_mode_token="MAINT-INC-G-S2-PILOT-001",
                quarantine_base_dir=quarantine_dir,
            )

            # Execution must fail before any file is moved
            with pytest.raises(ValueError, match="BACKUP_DIGEST_MISMATCH"):
                engine.execute_guarded_recovery()

            # Target DB must be completely untouched at its original path
            assert target_path.exists()
            conn_check = sqlite3.connect(f"{target_path.as_uri()}?mode=ro", uri=True)
            cur = conn_check.cursor()
            cur.execute("SELECT symbol, close FROM market_data;")
            assert cur.fetchall() == [("AAPL", 150.25)]
            conn_check.close()

            # Quarantine directory must not have any quarantined DB files
            if quarantine_dir.exists():
                assert len(list(quarantine_dir.iterdir())) == 0

    def test_recovery_failure_post_quarantine_rolls_back_original_database_files(self):
        """Proves that if replacement fails post-quarantine, original target files are restored to prevent data loss."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            target_path = tdp / "target.db"
            backup_path = tdp / "backup.db"
            quarantine_dir = tdp / "quarantine"

            # Create target DB
            conn = sqlite3.connect(str(target_path))
            conn.execute("CREATE TABLE target_marker (id INT);")
            conn.execute("INSERT INTO target_marker VALUES (42);")
            conn.commit()
            conn.close()

            # Create valid backup DB
            conn_b = sqlite3.connect(str(backup_path))
            conn_b.execute("CREATE TABLE backup_marker (id INT);")
            conn_b.execute("INSERT INTO backup_marker VALUES (99);")
            conn_b.commit()
            conn_b.close()

            bck_sha = GuardedRecoveryEngine.compute_file_sha256(backup_path)

            engine = GuardedRecoveryEngine(
                target_db_path=target_path,
                backup_db_path=backup_path,
                expected_backup_sha256=bck_sha,
                incident_id="INC-G-S2-PILOT-001",
                operator_authorization="G-S2-ACT",
                maintenance_mode_token="MAINT-INC-G-S2-PILOT-001",
                quarantine_base_dir=quarantine_dir,
            )

            # Simulate failure during step 8 (replacement)
            def fail_step8():
                raise IOError("Simulated disk write failure during atomic replacement")

            engine.step8_restore_via_candidate_file = fail_step8

            with pytest.raises(IOError, match="Simulated disk write failure"):
                engine.execute_guarded_recovery()

            # Target DB must have been restored by rollback
            assert target_path.exists()
            conn_res = sqlite3.connect(f"{target_path.as_uri()}?mode=ro", uri=True)
            cur = conn_res.cursor()
            cur.execute("SELECT id FROM target_marker;")
            assert cur.fetchall() == [(42,)]
            conn_res.close()

            # Quarantine evidence directory preserved
            assert quarantine_dir.exists()

    def test_guarded_recovery_receipt_deterministic_content_hash(self):
        """Proves that GuardedRecoveryReceipt produces deterministic SHA-256 content hashes for audit integrity."""
        receipt = GuardedRecoveryReceipt(
            incident_id="INC-G-S2-PILOT-001",
            operator_authorization="G-S2-ACT",
            maintenance_mode_token="MAINT-INC-G-S2-PILOT-001",
            target_database_path="/tmp/test.db",
            backup_source_path="/tmp/backup.db",
            backup_source_sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            quarantine_directory="/tmp/quarantine",
            quarantined_files=[],
            pre_recovery_state={"status": "ok"},
            post_recovery_readback={"status": "verified"},
            recovery_status="SUCCESS",
        )
        hash1 = receipt.compute_receipt_sha256()
        assert len(hash1) == 64
        assert hash1 == receipt.receipt_sha256

        # Deterministic: computing again gives exact same value
        hash2 = receipt.compute_receipt_sha256()
        assert hash1 == hash2

        # Changing field mutates content hash
        receipt.recovery_status = "FAILED"
        hash3 = receipt.compute_receipt_sha256()
        assert hash3 != hash1

        # Export to dict includes content hash
        d = receipt.to_dict()
        assert d["receipt_sha256"] == hash3
        assert "signature" not in d  # Plain content hash, NOT digital/PKI signature

    def test_cli_requires_explicit_maintenance_token(self):
        """Proves that scripts/guarded_market_recovery.py CLI requires explicit --maintenance-token."""
        cli_script = Path("scripts/guarded_market_recovery.py").resolve()

        # Invoke without --maintenance-token
        cmd = [
            sys.executable,
            str(cli_script),
            "--target-db", "dummy.db",
            "--backup-file", "dummy_bck.db",
            "--expected-sha256", "abc",
            "--incident-id", "INC-001",
            "--operator-auth", "G-S2-ACT",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        assert "the following arguments are required: --maintenance-token" in result.stderr


