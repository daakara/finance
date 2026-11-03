"""Fail-closed retention regression tests; no database, R2, or SEC access."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager


class FakeAdapter:
    def __init__(self, root):
        self.root = Path(root)
        self.deleted = []
        self.entries = {}

    def add(self, backup_id, when, *, valid=True, dump_present=True):
        dump = b"valid-recovery-dump-" + backup_id.encode()
        digest = hashlib.sha256(dump).hexdigest()
        manifest = {
            "backup_id": backup_id,
            "upload_verified": valid,
            "database_artifact": f"smart_money_{backup_id}.dump",
            "artifact_sha256": digest,
            "include_globals": False,
        }
        name = f"manifest_{backup_id}.json"
        (self.root / name).write_text(json.dumps(manifest), encoding="utf-8")
        self.entries[name] = when.isoformat()
        if dump_present:
            (self.root / manifest["database_artifact"]).write_bytes(dump)

    def list_backups(self):
        return [
            {"remote_filename": name, "modified_at": timestamp}
            for name, timestamp in self.entries.items()
        ]

    def download_file(self, remote_filename, local_dest_path):
        source = self.root / remote_filename
        Path(local_dest_path).write_bytes(source.read_bytes())
        return str(local_dest_path)

    def delete_backup(self, remote_filename):
        self.deleted.append(remote_filename)
        (self.root / remote_filename).unlink(missing_ok=True)
        return True


def test_invalid_newest_cannot_cause_older_valid_deletion(tmp_path):
    adapter = FakeAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old = "sm-backup-20261001T000000Z-valid"
    newer = "sm-backup-20261002T000000Z-invalid"
    adapter.add(old, now - timedelta(days=15))
    adapter.add(newer, now, valid=False)
    manager = BackupRetentionManager(adapter, retention_count=1, retention_days=1)
    assert manager.prune_backups() == []
    assert adapter.deleted == []


def test_missing_dump_never_enables_pruning(tmp_path):
    adapter = FakeAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    adapter.add("sm-backup-20261001T000000Z-valid", now - timedelta(days=10))
    adapter.add("sm-backup-20261002T000000Z-incomplete", now, dump_present=False)
    assert BackupRetentionManager(adapter, 1, 1).prune_backups() == []
    assert adapter.deleted == []


def test_verified_sets_allow_normal_pruning_and_protect_latest(tmp_path):
    adapter = FakeAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old = "sm-backup-20261001T000000Z-valid"
    new = "sm-backup-20261002T000000Z-valid"
    adapter.add(old, now - timedelta(days=12))
    adapter.add(new, now - timedelta(days=1))
    pruned = BackupRetentionManager(adapter, 1, 1).prune_backups()
    assert pruned == [old]
    assert adapter.deleted == [f"manifest_{old}.json", f"smart_money_{old}.dump"]
    assert (tmp_path / f"manifest_{new}.json").exists()


def test_corrupt_dump_blocks_all_deletion(tmp_path):
    adapter = FakeAdapter(tmp_path)
    now = datetime.now(timezone.utc)
    old = "sm-backup-20261001T000000Z-valid"
    new = "sm-backup-20261002T000000Z-invalid"
    adapter.add(old, now - timedelta(days=12))
    adapter.add(new, now)
    (tmp_path / f"smart_money_{new}.dump").write_bytes(b"wrong")
    assert BackupRetentionManager(adapter, 1, 1).prune_backups() == []
    assert adapter.deleted == []
