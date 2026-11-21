"""
ARX Terminal — Smart Money Backup Retention Bounded Adversarial Test Suite.

Validates fail-closed safety and bounded deadline enforcement under realistic offline failure paths:
1. 55+ backup sets with slow/trickling storage responses;
2. Strict monotonic cycle deadline compliance (<= 60-second contract);
3. Zero unverified or abandoned destructive work on deadline expiry;
4. Legacy schema contract enforcement (tampered hash, size mismatch, missing reference);
5. Explicit unverified / conflicting flag rejection;
6. Interrupted deletion containment;
7. Dry-run retention planning without storage mutation.
"""

import os
import time
import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest

from analyst_dashboard.smart_money.storage.backup.retention import (
    BackupRetentionManager,
    RetentionOutcome,
)


class RealisticSimulatedStorageAdapter:
    """Simulates realistic off-volume or S3/R2 storage with latency injection and tracking."""

    def __init__(self, root: Path, per_call_delay_sec: float = 0.0, delete_delay_sec: float = 0.0):
        self.root = Path(root)
        self.per_call_delay_sec = per_call_delay_sec
        self.delete_delay_sec = delete_delay_sec
        self.deleted: List[str] = []
        self.entries: Dict[str, str] = {}
        self.download_count = 0
        self.delete_count = 0

    def add(
        self,
        backup_id: str,
        when: datetime,
        *,
        valid: bool = True,
        dump_present: bool = True,
        legacy: bool = False,
        dump_size_bytes: Optional[int] = None,
        tamper_dump_hash: bool = False,
        tamper_dump_size: bool = False,
        malformed_json: bool = False,
        explicit_false_verified: bool = False,
    ):
        dump = b"database-binary-dump-payload-" + backup_id.encode()
        actual_size = len(dump)
        digest = hashlib.sha256(dump).hexdigest()

        if tamper_dump_hash:
            manifest_hash = "f" * 64
        else:
            manifest_hash = digest

        if legacy:
            manifest = {
                "backup_id": backup_id,
                "dump_filename": f"smart_money_{backup_id}.dump",
                "dump_sha256": manifest_hash,
                "dump_size_bytes": actual_size if not tamper_dump_size else actual_size + 999,
                "created_at": when.isoformat(),
                "include_globals": False,
            }
        else:
            upload_val = False if explicit_false_verified else valid
            manifest = {
                "backup_id": backup_id,
                "upload_verified": upload_val,
                "database_artifact": f"smart_money_{backup_id}.dump",
                "artifact_sha256": manifest_hash,
                "include_globals": False,
            }

        name = f"manifest_{backup_id}.json"
        if malformed_json:
            (self.root / name).write_text("{malformed_json: true, unterminated", encoding="utf-8")
        else:
            (self.root / name).write_text(json.dumps(manifest), encoding="utf-8")

        self.entries[name] = when.isoformat()
        dump_name = manifest.get("database_artifact") or manifest.get("dump_filename")
        if dump_present:
            (self.root / dump_name).write_bytes(dump)

    def list_backups(self, deadline_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("STORAGE_DEADLINE_EXCEEDED: list_backups")
        if self.per_call_delay_sec > 0:
            time.sleep(self.per_call_delay_sec)
        return [
            {"remote_filename": name, "modified_at": timestamp}
            for name, timestamp in self.entries.items()
        ]

    def download_file(self, remote_filename: str, local_dest_path: str, deadline_monotonic: Optional[float] = None) -> str:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED: download_file {remote_filename}")
        self.download_count += 1
        if self.per_call_delay_sec > 0:
            time.sleep(self.per_call_delay_sec)
        source = self.root / remote_filename
        if not source.exists():
            raise FileNotFoundError(f"Remote object not found: {remote_filename}")
        Path(local_dest_path).write_bytes(source.read_bytes())
        return str(local_dest_path)

    def delete_backup(self, remote_filename: str, deadline_monotonic: Optional[float] = None) -> bool:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"STORAGE_DEADLINE_EXCEEDED: delete_backup {remote_filename}")
        self.delete_count += 1
        delay = self.delete_delay_sec if self.delete_delay_sec > 0 else self.per_call_delay_sec
        if delay > 0:
            time.sleep(delay)
        self.deleted.append(remote_filename)
        (self.root / remote_filename).unlink(missing_ok=True)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: 55+ Backup Sets, Slow Storage, Deadline Bounding
# ─────────────────────────────────────────────────────────────────────────────

def test_55_plus_backups_slow_storage_respects_deadline(tmp_path):
    """
    Simulates production catalog (55 backups) with network latency on Cloudflare R2.
    Verifies that when a deadline is set, pruning defers cleanly before the deadline,
    deletes 0 unverified files, and preserves the protected newest recovery point.
    """
    adapter = RealisticSimulatedStorageAdapter(tmp_path, per_call_delay_sec=0.02)
    now = datetime.now(timezone.utc)

    # Populate 55 backups: newest today, oldest 54 days ago
    for day in range(55):
        b_id = f"sm-backup-202611{20-day:02d}T000000Z-item{day:03d}" if day < 20 else f"sm-backup-202610{50-day:02d}T000000Z-item{day:03d}"
        adapter.add(b_id, now - timedelta(days=day), valid=True)

    # Set a tight monotonic deadline: only 0.25 seconds from now
    deadline = time.monotonic() + 0.25
    manager = BackupRetentionManager(adapter, retention_count=10, retention_days=7)

    start = time.monotonic()
    pruned = manager.prune_backups(deadline_monotonic=deadline, safety_margin_sec=0.1)
    duration = time.monotonic() - start

    # Assertions
    assert pruned == [], "Must return empty list on deadline deferral"
    assert adapter.deleted == [], "Must NEVER delete any files when verification is incomplete"
    assert manager.last_outcome == RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
    # Duration must respect deadline and safety margin
    assert duration <= 0.35, f"Execution must return within budget, took {duration:.3f}s"
    # Verify all 55 manifests and dumps survive
    assert len(adapter.list_backups()) == 55


def test_60_second_cycle_budget_simulation_offline(tmp_path):
    """
    Simulates a full cycle where Steps 1-9 consumed 4.0s of a 60s total budget.
    Ensures that Step 10 respects the remaining 56s budget under slow storage
    without exceeding the 60s contract.
    """
    adapter = RealisticSimulatedStorageAdapter(tmp_path, per_call_delay_sec=0.01)
    now = datetime.now(timezone.utc)

    for idx in range(55):
        adapter.add(f"sm-backup-item-{idx:03d}", now - timedelta(days=idx), valid=True)

    cycle_start = time.monotonic()
    # In continuous_pipeline.py, cycle_deadline_monotonic = start + 60.0
    # Simulate tight budget remaining: 0.15s left in cycle
    cycle_deadline = cycle_start + 0.15

    manager = BackupRetentionManager(adapter, retention_count=10, retention_days=7)
    pruned = manager.prune_backups(deadline_monotonic=cycle_deadline, safety_margin_sec=0.05)
    cycle_duration = time.monotonic() - cycle_start

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
    assert cycle_duration <= 0.25, f"Cycle return must respect 60s contract, took {cycle_duration}s"


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Legacy Schema Contract & Tamper Rejection
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_manifest_strict_schema_contract_valid(tmp_path):
    """Verifies that an authentic legacy pre-SM-ACT-020 manifest prunes safely when fully verified."""
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old_legacy = "sm-backup-20261025T152219Z-post-ingestion"
    new_modern = "sm-backup-20261120T141500Z-post-ingestion"

    adapter.add(old_legacy, now - timedelta(days=26), legacy=True)
    adapter.add(new_modern, now - timedelta(hours=1), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    pruned = manager.prune_backups()

    assert pruned == [old_legacy]
    assert adapter.deleted == [f"manifest_{old_legacy}.json", f"smart_money_{old_legacy}.dump"]
    assert manager.last_outcome == RetentionOutcome.SUCCESS
    assert (tmp_path / f"manifest_{new_modern}.json").exists()


def test_legacy_manifest_with_tampered_hash_blocks_all_deletion(tmp_path):
    """Tampered dump hash in legacy manifest MUST fail closed and block all deletion."""
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old_legacy = "sm-backup-20261025T152219Z-post-ingestion"
    new_modern = "sm-backup-20261120T141500Z-post-ingestion"

    adapter.add(old_legacy, now - timedelta(days=26), legacy=True, tamper_dump_hash=True)
    adapter.add(new_modern, now - timedelta(hours=1), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    pruned = manager.prune_backups()

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DISABLED_UNVERIFIED_RECOVERY_POINT


def test_legacy_manifest_with_size_mismatch_blocks_all_deletion(tmp_path):
    """Dump size mismatch in legacy manifest MUST fail closed."""
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old_legacy = "sm-backup-20261025T152219Z-post-ingestion"
    new_modern = "sm-backup-20261120T141500Z-post-ingestion"

    adapter.add(old_legacy, now - timedelta(days=26), legacy=True, tamper_dump_size=True)
    adapter.add(new_modern, now - timedelta(hours=1), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    pruned = manager.prune_backups()

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DISABLED_UNVERIFIED_RECOVERY_POINT


def test_explicit_false_upload_verified_in_legacy_format_blocks_deletion(tmp_path):
    """An explicit upload_verified: false MUST NEVER be forgiven as a legacy manifest."""
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old = "sm-backup-20261025T152219Z-unverified"
    new = "sm-backup-20261120T141500Z-valid"

    adapter.add(old, now - timedelta(days=26), explicit_false_verified=True)
    adapter.add(new, now - timedelta(hours=1), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    pruned = manager.prune_backups()

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DISABLED_UNVERIFIED_RECOVERY_POINT


def test_malformed_json_manifest_blocks_all_deletion(tmp_path):
    """A corrupted or truncated JSON manifest MUST fail closed immediately."""
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old_corrupt = "sm-backup-20261025T152219Z-corrupt"
    new_valid = "sm-backup-20261120T141500Z-valid"

    adapter.add(old_corrupt, now - timedelta(days=26), malformed_json=True)
    adapter.add(new_valid, now - timedelta(hours=1), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    pruned = manager.prune_backups()

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DISABLED_UNVERIFIED_RECOVERY_POINT


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Interrupted Deletion & Dry-Run Planning
# ─────────────────────────────────────────────────────────────────────────────

def test_interrupted_deletion_stops_gracefully_without_corrupting_state(tmp_path):
    """
    If the deadline expires during the deletion phase itself, pruning halts
    immediately, returns the safely deleted subset, and leaves remaining files intact.
    The outcome MUST be recorded as PARTIAL_DEFERRED, never unconditional SUCCESS.
    """
    # Fast verification (0 delay), slow deletion (0.05s per delete call)
    adapter = RealisticSimulatedStorageAdapter(tmp_path, per_call_delay_sec=0.0, delete_delay_sec=0.05)
    now = datetime.now(timezone.utc)

    # Add 1 newest (keep) + 5 expired candidates
    newest = "sm-backup-item-00"
    adapter.add(newest, now - timedelta(hours=1), valid=True)
    for idx in range(1, 6):
        adapter.add(f"sm-backup-item-{idx:02d}", now - timedelta(days=10 + idx), valid=True)

    # Verification of 6 items takes ~0.005s.
    # Each candidate deletion calls delete_backup twice (manifest + dump) = 0.10s per item.
    # Giving 0.15s budget allows deleting exactly 1 item before deadline hits on 2nd item!
    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    t0 = time.monotonic()
    deadline = t0 + 0.15
    pruned = manager.prune_backups(deadline_monotonic=deadline, safety_margin_sec=0.02)
    elapsed = time.monotonic() - t0

    # Must have deleted at least 1 item before stopping
    assert len(pruned) >= 1
    assert len(pruned) < 5
    assert manager.last_outcome == RetentionOutcome.PARTIAL_DEFERRED
    assert manager.last_details["interrupted_by_deadline"] is True
    assert len(manager.last_details["fully_pruned_candidates"]) == len(pruned)
    assert len(manager.last_details["remaining_candidates"]) >= 1
    # Measured elapsed time must be bounded by deadline (+ small margin)
    assert elapsed < 0.25, f"Deletion phase exceeded bounded deadline: elapsed={elapsed:.3f}s"
    # Newest item must NEVER be touched
    assert f"manifest_{newest}.json" not in adapter.deleted
    assert (tmp_path / f"manifest_{newest}.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite: Real Adapter Boundary (Slow Headers, Trickling Bodies, Retries)
# ─────────────────────────────────────────────────────────────────────────────

import threading


class AdversarialBoundaryStorageAdapter(RealisticSimulatedStorageAdapter):
    """Adapter simulating specific TCP/HTTP adversarial conditions at the network boundary."""

    def __init__(
        self,
        root: Path,
        slow_header_delay_sec: float = 0.0,
        trickle_chunk_delay_sec: float = 0.0,
        blocked_listing_delay_sec: float = 0.0,
        fail_deletions_for: Optional[List[str]] = None,
        max_allowed_attempts: int = 1,
    ):
        super().__init__(root)
        self.slow_header_delay_sec = slow_header_delay_sec
        self.trickle_chunk_delay_sec = trickle_chunk_delay_sec
        self.blocked_listing_delay_sec = blocked_listing_delay_sec
        self.fail_deletions_for = set(fail_deletions_for or [])
        self.max_allowed_attempts = max_allowed_attempts
        self.active_workers: List[threading.Thread] = []

    def list_backups(self, deadline_monotonic: Optional[float] = None) -> List[Dict[str, Any]]:
        if self.blocked_listing_delay_sec > 0:
            t0 = time.monotonic()
            while (time.monotonic() - t0) < self.blocked_listing_delay_sec:
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    raise TimeoutError("ADAPTER_BLOCKED_LISTING_DEADLINE_EXCEEDED")
                time.sleep(0.01)
        return super().list_backups()

    def download_file(self, remote_filename: str, local_dest_path: str, deadline_monotonic: Optional[float] = None) -> str:
        self.download_count += 1
        source = self.root / remote_filename
        if not source.exists():
            raise FileNotFoundError(f"Remote object not found: {remote_filename}")

        # Simulate slow header response (blocking before first byte)
        if self.slow_header_delay_sec > 0:
            t0 = time.monotonic()
            while (time.monotonic() - t0) < self.slow_header_delay_sec:
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    raise TimeoutError(f"ADAPTER_SLOW_HEADERS_TIMEOUT: {remote_filename}")
                time.sleep(0.01)

        # Simulate trickling chunks (slow stream delivery)
        if self.trickle_chunk_delay_sec > 0:
            data = source.read_bytes()
            chunk_size = max(1, len(data) // 5)
            with open(local_dest_path, "wb") as f_out:
                for idx in range(0, len(data), chunk_size):
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        if os.path.exists(local_dest_path):
                            os.remove(local_dest_path)
                        raise TimeoutError(f"ADAPTER_TRICKLE_BODY_TIMEOUT: {remote_filename}")
                    time.sleep(self.trickle_chunk_delay_sec)
                    f_out.write(data[idx:idx + chunk_size])
            return str(local_dest_path)

        return super().download_file(remote_filename, local_dest_path)

    def delete_backup(self, remote_filename: str, deadline_monotonic: Optional[float] = None) -> bool:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"ADAPTER_DELETE_DEADLINE_EXCEEDED: {remote_filename}")
        if remote_filename in self.fail_deletions_for:
            return False
        return super().delete_backup(remote_filename)


def test_adapter_slow_headers_timeout_aborts_within_budget(tmp_path):
    """
    Verifies that slow server headers trigger bounded timeout within budget,
    deferring pruning safely with 0 files deleted and measured elapsed time bounded.
    """
    adapter = AdversarialBoundaryStorageAdapter(tmp_path, slow_header_delay_sec=0.5)
    now = datetime.now(timezone.utc)
    for idx in range(5):
        adapter.add(f"sm-backup-item-{idx:02d}", now - timedelta(days=idx), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=2, retention_days=2)
    t0 = time.monotonic()
    deadline = t0 + 0.10  # 100ms budget vs 500ms header delay
    pruned = manager.prune_backups(deadline_monotonic=deadline, safety_margin_sec=0.01)
    elapsed = time.monotonic() - t0

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
    assert elapsed < 0.20, f"Slow headers call took too long: elapsed={elapsed:.3f}s"


def test_adapter_trickling_body_aborts_mid_stream_within_budget(tmp_path):
    """
    Verifies that trickling body stream aborts mid-stream as soon as deadline is reached,
    cleaning up partial local files and leaving storage unmutated.
    """
    adapter = AdversarialBoundaryStorageAdapter(tmp_path, trickle_chunk_delay_sec=0.05)
    now = datetime.now(timezone.utc)
    for idx in range(3):
        adapter.add(f"sm-backup-item-{idx:02d}", now - timedelta(days=idx), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=1)
    t0 = time.monotonic()
    deadline = t0 + 0.08  # ~80ms budget vs ~250ms total trickle
    pruned = manager.prune_backups(deadline_monotonic=deadline, safety_margin_sec=0.01)
    elapsed = time.monotonic() - t0

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
    assert elapsed < 0.18, f"Trickling stream took too long: elapsed={elapsed:.3f}s"


def test_adapter_blocked_listing_aborts_within_budget(tmp_path):
    """
    Verifies that blocked/hanging S3 listing calls abort within deadline budget.
    """
    adapter = AdversarialBoundaryStorageAdapter(tmp_path, blocked_listing_delay_sec=0.5)
    now = datetime.now(timezone.utc)
    adapter.add("sm-backup-item-00", now, valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=1)
    t0 = time.monotonic()
    deadline = t0 + 0.10
    pruned = manager.prune_backups(deadline_monotonic=deadline, safety_margin_sec=0.01)
    elapsed = time.monotonic() - t0

    assert pruned == []
    assert adapter.deleted == []
    assert manager.last_outcome == RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED
    assert elapsed < 0.20, f"Blocked listing took too long: elapsed={elapsed:.3f}s"


def test_adapter_failed_deletion_reports_partial_deferred_not_success(tmp_path):
    """
    Verifies that if any artifact deletion fails (e.g. storage error, missing permission),
    the outcome is PARTIAL_DEFERRED, never unconditional SUCCESS, and newest recovery point is preserved.
    """
    # Fail deletion of dump file for item-01
    adapter = AdversarialBoundaryStorageAdapter(
        tmp_path,
        fail_deletions_for=["smart_money_sm-backup-item-01.dump"]
    )
    now = datetime.now(timezone.utc)
    adapter.add("sm-backup-item-00", now - timedelta(hours=1), valid=True)
    adapter.add("sm-backup-item-01", now - timedelta(days=10), valid=True)
    adapter.add("sm-backup-item-02", now - timedelta(days=12), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=7)
    pruned = manager.prune_backups()

    # item-02 should be fully pruned, item-01 had 1 failed file deletion
    assert "sm-backup-item-02" in pruned
    assert "sm-backup-item-01" not in pruned
    assert manager.last_outcome == RetentionOutcome.PARTIAL_DEFERRED
    assert "smart_money_sm-backup-item-01.dump" in manager.last_details["failed_file_deletions"]
    assert "manifest_sm-backup-item-00.json" not in adapter.deleted
    assert (tmp_path / "manifest_sm-backup-item-00.json").exists()


def test_no_work_continues_after_return_and_no_threads_abandoned(tmp_path):
    """
    Asserts that measured elapsed time returns promptly and no background threads
    are spawned or left running after prune_backups returns.
    """
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    for idx in range(10):
        adapter.add(f"sm-backup-item-{idx:02d}", now - timedelta(days=idx), valid=True)

    threads_before = {t.ident for t in threading.enumerate() if t.is_alive()}
    manager = BackupRetentionManager(adapter, retention_count=3, retention_days=5)
    t0 = time.monotonic()
    pruned = manager.prune_backups()
    elapsed = time.monotonic() - t0

    threads_after = {t.ident for t in threading.enumerate() if t.is_alive()}
    new_threads = threads_after - threads_before

    assert len(pruned) == 7
    assert manager.last_outcome == RetentionOutcome.SUCCESS
    assert elapsed < 1.0
    assert len(new_threads) == 0, f"Detected abandoned background threads: {new_threads}"


def test_dry_run_retention_inventory_generates_plan_without_mutations(tmp_path):
    """
    Verifies dry_run=True identifies the exact deletion candidates
    without deleting a single file from storage.
    """
    adapter = RealisticSimulatedStorageAdapter(tmp_path)
    now = datetime.now(timezone.utc)

    # Add 15 backups: all within 30 days, so items 0-9 kept by count=10, 10-14 pruned by count
    for idx in range(15):
        adapter.add(f"sm-backup-item-{idx:02d}", now - timedelta(days=idx), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=10, retention_days=30)
    plan_pruned = manager.prune_backups(dry_run=True)

    # Items 0 through 9 are protected (10 count window)
    # Items 10 through 14 must be identified for deletion
    expected_candidates = [f"sm-backup-item-{idx:02d}" for idx in range(10, 15)]
    assert plan_pruned == expected_candidates
    assert adapter.deleted == [], "Dry run must NOT delete any files!"
    assert manager.last_outcome == RetentionOutcome.SUCCESS
    assert manager.last_details["protected_newest"] == "sm-backup-item-00"
    assert manager.last_details["candidates_to_prune"] == expected_candidates


def test_adapter_typeerror_does_not_fall_back_to_unbounded_call(tmp_path):
    """
    Verifies that when an adapter raises TypeError due to missing deadline_monotonic support,
    it NEVER falls back to an unbounded call during deadline-bound execution.
    """
    class LegacyUnsupportedAdapter(RealisticSimulatedStorageAdapter):
        def list_backups(self, **kwargs):
            if "deadline_monotonic" in kwargs:
                raise TypeError("list_backups() got an unexpected keyword argument 'deadline_monotonic'")
            return super().list_backups()

    adapter = LegacyUnsupportedAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    for idx in range(5):
        adapter.add(f"sm-backup-item-{idx:02d}", now - timedelta(days=idx), valid=True)

    manager = BackupRetentionManager(adapter, retention_count=2, retention_days=10)
    # When deadline_monotonic is supplied, it must defer and NOT execute unbounded calls
    pruned = manager.prune_backups(deadline_monotonic=time.monotonic() + 10.0)
    assert pruned == []
    assert manager.last_outcome == RetentionOutcome.DEFERRED_DEADLINE_EXCEEDED


