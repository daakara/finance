"""Adversarial ratification tests for RQ-10 production authority (ARX-039)."""
import json
import shutil
from pathlib import Path

import pytest

from scripts.release import rq10_config_integrity as rq10


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "FROZEN_ENGINE_MANIFEST_V2_5_3.json"
CANDIDATE = "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json"
RECORD = "docs/governance/ARX_RC5_ENGINE_MANIFEST_V253_SUCCESSION_RECORD.md"


@pytest.fixture
def authority_tree(tmp_path, monkeypatch):
    for filename in (MANIFEST, CANDIDATE, RECORD):
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, target)
    for name in ("confluence_engine", "optimal_execution", "decision_hierarchy"):
        rel = Path("analyst_dashboard") / "analyzers" / (name + ".py")
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    monkeypatch.setattr(rq10, "PROJECT_ROOT", str(tmp_path))
    return tmp_path


def alter_manifest(root, mutation):
    path = root / MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    mutation(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_ratified_three_engine_manifest_passes(authority_tree):
    result = rq10.verify_engine_manifest(enforce_production=True)
    assert result["ratified"] is True
    assert len(result["engine_hashes"]) == 3


def test_unratified_candidate_rejected_in_production(authority_tree):
    with pytest.raises(AssertionError, match="exact frozen"):
        rq10.verify_engine_manifest(enforce_production=True, manifest_override=CANDIDATE)


def test_unratified_candidate_permitted_in_explicit_development_mode(authority_tree):
    result = rq10.verify_engine_manifest(enforce_production=False)
    assert result["ratified"] is False
    assert len(result["engine_hashes"]) == 3


def test_forged_flag_false_rejected(authority_tree):
    alter_manifest(authority_tree, lambda d: d.update(isProductionRatified=False))
    with pytest.raises(AssertionError, match="not ratified"):
        rq10.verify_engine_manifest()


def test_forged_release_authority_binding_rejected(authority_tree):
    alter_manifest(authority_tree, lambda d: d.update(releaseAuthorityBinding="FORGED"))
    with pytest.raises(AssertionError, match="authority binding"):
        rq10.verify_engine_manifest()


def test_owner_authorization_token_missing_rejected(authority_tree):
    path = authority_tree / RECORD
    path.write_text("No ratification authorization provided", encoding="utf-8")
    with pytest.raises(AssertionError, match="authorization token"):
        rq10.verify_engine_manifest()


def test_changed_engine_hash_rejected(authority_tree):
    alter_manifest(authority_tree, lambda d: d["engines"]["confluence_engine"].update(sha256="0" * 64))
    with pytest.raises(AssertionError, match="hash mismatch"):
        rq10.verify_engine_manifest()


def test_missing_engine_is_rejected(authority_tree):
    alter_manifest(authority_tree, lambda d: d["engines"].pop("optimal_execution"))
    with pytest.raises(AssertionError, match="mandatory engines"):
        rq10.verify_engine_manifest()
