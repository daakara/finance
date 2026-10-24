"""ARX Terminal — Candidate 004 / 005 / 006 Reproducibility & Dual Execution Suite.

Proves:
0. Fixture package integrity matches cryptographic manifest.
1. Default instantiation of VCPClassifier selects Candidate 004.
2. Numeric contract defaults preserve Candidate 004 monotonic tightening rules.
3. Candidate 004 produces 100% bit-exact outputs matching frozen baseline records on HVA-00 development cases.
4. Candidate 004 and Candidate 005 can execute side-by-side against identical inputs without state leakage.
5. Candidate 005 produces 100% bit-exact outputs matching frozen baseline records on HVA-00 development cases.
6. Tri-candidate side-by-side execution (C004, C005, C006) exhibits zero state leakage and order invariance.
7. Candidate 006 produces 100% bit-exact outputs matching frozen baseline records on HVA-00 development cases.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import pytest

from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar


def _resolve_vcp_fixture(filename: str) -> Path:
    """Resolves VCP research fixtures from tests/fixtures/vcp/ with fallback to data/research/."""
    worktree_root = Path(__file__).resolve().parent.parent
    primary = worktree_root / "tests" / "fixtures" / "vcp" / filename
    if primary.exists():
        return primary
    fallback = worktree_root / "data" / "research" / filename
    if fallback.exists():
        return fallback
    return primary


def test_00_fixture_manifest_integrity():
    """Validates that packaged VCP research fixtures exist and match cryptographic hashes."""
    manifest_path = _resolve_vcp_fixture("FIXTURE_MANIFEST.json")
    if not manifest_path.exists():
        pytest.skip("FIXTURE_MANIFEST.json not present in fixtures directory.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    worktree_root = Path(__file__).resolve().parent.parent
    fixture_dir = manifest_path.parent

    for item in meta["fixtures"]:
        fpath = fixture_dir / item["filename"]
        assert fpath.exists(), f"Packaged fixture missing: {item['filename']}"
        assert fpath.stat().st_size == item["size_bytes"], f"Size mismatch for {item['filename']}"

        h = hashlib.sha256()
        with open(fpath, "rb") as bf:
            while chunk := bf.read(8192):
                h.update(chunk)
        assert h.hexdigest() == item["sha256"], f"SHA-256 mismatch for {item['filename']}"


def test_01_default_classifier_is_candidate_004():
    """Default classifier instantiation must be Candidate 004 to preserve production behavior."""
    clf = VCPClassifier()
    assert clf.algorithm_version == VCPClassifier.VERSION_CANDIDATE_004
    assert clf.algorithm_version == "ARX_VCP_CLASSIFIER_V1_CANDIDATE_004"


def test_02_numeric_contract_defaults_preserve_candidate_004():
    """VCPNumericContract.verify_progressive_tightening default arguments must enforce strict monotonic decrease."""
    nc = VCPNumericContract()
    # Shallow depths without 5% reduction: 0.0400 to 0.0399 (decrease of 0.25%, not 5%)
    # Under Candidate 004 (relative_reduction=0.0): PASS
    assert nc.verify_progressive_tightening([0.0400, 0.0399]) is True
    # Under Candidate 005 (relative_reduction=0.05): FAIL (0.0400 * 0.95 = 0.0380, 0.0399 > 0.0380)
    assert nc.verify_progressive_tightening([0.0400, 0.0399], relative_reduction=0.05) is False


def test_03_candidate_004_bit_exact_reproducibility_on_frozen_cases():
    """Proves Candidate 004 yields bit-exact outputs matching vcp_hva00_frozen_algorithm_results.json."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = _resolve_vcp_fixture("vcp_hva00_case_manifest.json")
    frozen_path = _resolve_vcp_fixture("vcp_hva00_frozen_algorithm_results.json")

    if not (db_path.exists() and manifest_path.exists() and frozen_path.exists()):
        pytest.skip("Research data files not present in worktree; skipping frozen replay.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    frozen_records = {r["case_id"]: r for r in frozen["algorithm_records"]}
    pos_cases = [c for c in manifest["cases"] if c.get("stratum") == "CLEAR_POSITIVE_VCP"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    compiler = VCPTemporalCaseCompiler()
    clf_c004 = VCPClassifier()  # default Candidate 004

    discrepancies = []

    for defn in pos_cases:
        cid = defn["case_id"]
        sym = defn["symbol"]
        cutoff = defn["evaluation_cutoff_date"]
        cutoff_iso = f"{cutoff}T21:00:00Z"

        rows = cursor.execute(
            """
            SELECT trade_date, open, high, low, close, volume
            FROM asset_ohlcv_daily
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date ASC
            """,
            (sym, cutoff),
        ).fetchall()

        bars = [
            DailyOHLCVBar(
                symbol=sym,
                bar_date=r[0],
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=int(r[5]),
                valid_time=f"{r[0]}T21:00:00Z",
                known_at=f"{r[0]}T22:00:00Z",
            )
            for r in rows
        ]

        pkg = compiler.compile_case(
            case_id=cid,
            security_id=f"SEC-{sym}",
            symbol=sym,
            evaluation_as_of=cutoff_iso,
            raw_bars=bars,
            reference_data={"rs_rank": 85.0},
            corporate_actions=[],
        )

        assessment = clf_c004.classify_case(pkg)
        obs = clf_c004.extract_observations(pkg)
        frz = frozen_records.get(cid, {})

        qual_match = (assessment.vcp_qualified == frz.get("vcp_qualified"))
        class_match = (assessment.vcp_classification == frz.get("vcp_classification"))
        depths_match = (obs.contraction_depths == frz.get("contraction_depths"))
        tight_match = (obs.is_progressive_tightening == frz.get("is_progressive_tightening"))
        vol_match = (obs.final_wave_volume_ratio == frz.get("volume_dry_up_ratio"))

        if not (qual_match and class_match and depths_match and tight_match and vol_match):
            discrepancies.append((cid, {
                "qual": (assessment.vcp_qualified, frz.get("vcp_qualified")),
                "class": (assessment.vcp_classification, frz.get("vcp_classification")),
                "depths": (obs.contraction_depths, frz.get("contraction_depths")),
                "tight": (obs.is_progressive_tightening, frz.get("is_progressive_tightening")),
                "vol": (obs.final_wave_volume_ratio, frz.get("volume_dry_up_ratio")),
            }))

    conn.close()
    assert len(discrepancies) == 0, f"Discrepancies found against frozen baseline: {discrepancies}"


def test_04_dual_candidate_isolated_side_by_side_execution():
    """Instantiating both Candidate 004 and Candidate 005 executes cleanly without shared state contamination."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = _resolve_vcp_fixture("vcp_hva00_case_manifest.json")

    if not (db_path.exists() and manifest_path.exists()):
        pytest.skip("Database or manifest not present in worktree.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    case_defn = manifest["cases"][0]
    sym = case_defn["symbol"]
    cutoff = case_defn["evaluation_cutoff_date"]
    cutoff_iso = f"{cutoff}T21:00:00Z"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    rows = cursor.execute(
        """
        SELECT trade_date, open, high, low, close, volume
        FROM asset_ohlcv_daily
        WHERE symbol = ? AND trade_date <= ?
        ORDER BY trade_date ASC
        """,
        (sym, cutoff),
    ).fetchall()
    conn.close()

    bars = [
        DailyOHLCVBar(
            symbol=sym,
            bar_date=r[0],
            open=float(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            volume=int(r[5]),
            valid_time=f"{r[0]}T21:00:00Z",
            known_at=f"{r[0]}T22:00:00Z",
        )
        for r in rows
    ]

    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id=case_defn["case_id"],
        security_id=f"SEC-{sym}",
        symbol=sym,
        evaluation_as_of=cutoff_iso,
        raw_bars=bars,
        reference_data={"rs_rank": 85.0},
        corporate_actions=[],
    )

    clf_004 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_004)
    clf_005 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)

    obs_004 = clf_004.extract_observations(pkg)
    obs_005 = clf_005.extract_observations(pkg)

    # Candidate 004 does not populate macro metadata
    assert obs_004.metadata == {}
    # Candidate 005 populates macro metadata
    assert "algorithm_version" in obs_005.metadata
    assert obs_005.metadata["algorithm_version"] == VCPClassifier.VERSION_CANDIDATE_005
    assert "anchor_selected" in obs_005.metadata

    # Running them again does not alter version identities
    assert clf_004.algorithm_version == VCPClassifier.VERSION_CANDIDATE_004
    assert clf_005.algorithm_version == VCPClassifier.VERSION_CANDIDATE_005


def test_05_candidate_005_bit_exact_reproducibility_on_frozen_cases():
    """Proves Candidate 005 yields bit-exact outputs matching vcp_hva02_candidate_005_results.json."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = _resolve_vcp_fixture("vcp_hva00_case_manifest.json")
    frozen_c005_path = _resolve_vcp_fixture("vcp_hva02_candidate_005_results.json")

    if not (db_path.exists() and manifest_path.exists() and frozen_c005_path.exists()):
        pytest.skip("Research data files not present in worktree; skipping frozen replay.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(frozen_c005_path, "r", encoding="utf-8") as f:
        frozen_c005 = json.load(f)

    frozen_records = {r["case_id"]: r["candidate_005"] for r in frozen_c005["case_records"]}
    pos_cases = [c for c in manifest["cases"] if c.get("stratum") == "CLEAR_POSITIVE_VCP"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    compiler = VCPTemporalCaseCompiler()
    clf_c005 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)

    for case_defn in pos_cases:
        cid = case_defn["case_id"]
        sym = case_defn["symbol"]
        cutoff = case_defn["evaluation_cutoff_date"]
        cutoff_iso = f"{cutoff}T21:00:00Z"

        rows = cursor.execute(
            """
            SELECT trade_date, open, high, low, close, volume
            FROM asset_ohlcv_daily
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date ASC
            """,
            (sym, cutoff),
        ).fetchall()

        bars = [
            DailyOHLCVBar(
                symbol=sym,
                bar_date=r[0],
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=int(r[5]),
                valid_time=f"{r[0]}T21:00:00Z",
                known_at=f"{r[0]}T22:00:00Z",
            )
            for r in rows
        ]

        pkg = compiler.compile_case(
            case_id=cid,
            security_id=f"SEC-{sym}",
            symbol=sym,
            evaluation_as_of=cutoff_iso,
            raw_bars=bars,
            reference_data={"rs_rank": 85.0},
            corporate_actions=[],
        )

        obs = clf_c005.extract_observations(pkg)
        assessment = clf_c005.classify_case(pkg)

        frozen_rec = frozen_records[cid]
        assert assessment.vcp_classification == frozen_rec["vcp_classification"], f"{cid} classification mismatch"
        assert obs.contraction_count == frozen_rec["contraction_count"], f"{cid} count mismatch"
        assert obs.contraction_depths == frozen_rec["contraction_depths"], f"{cid} depths mismatch"

    conn.close()


def test_06_tri_candidate_isolated_side_by_side_and_order_invariance():
    """Instantiating Candidate 004, Candidate 005, and Candidate 006 executes cleanly with zero state leakage and order invariance."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = _resolve_vcp_fixture("vcp_hva00_case_manifest.json")

    if not (db_path.exists() and manifest_path.exists()):
        pytest.skip("Database or manifest not present in worktree.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    case_defn = manifest["cases"][0]
    sym = case_defn["symbol"]
    cutoff = case_defn["evaluation_cutoff_date"]
    cutoff_iso = f"{cutoff}T21:00:00Z"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    rows = cursor.execute(
        """
        SELECT trade_date, open, high, low, close, volume
        FROM asset_ohlcv_daily
        WHERE symbol = ? AND trade_date <= ?
        ORDER BY trade_date ASC
        """,
        (sym, cutoff),
    ).fetchall()
    conn.close()

    bars = [
        DailyOHLCVBar(
            symbol=sym,
            bar_date=r[0],
            open=float(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            volume=int(r[5]),
            valid_time=f"{r[0]}T21:00:00Z",
            known_at=f"{r[0]}T22:00:00Z",
        )
        for r in rows
    ]

    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id=case_defn["case_id"],
        security_id=f"SEC-{sym}",
        symbol=sym,
        evaluation_as_of=cutoff_iso,
        raw_bars=bars,
        reference_data={"rs_rank": 85.0},
        corporate_actions=[],
    )

    # Forward order
    clf_004_fwd = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_004)
    clf_005_fwd = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    clf_006_fwd = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)

    obs_004_fwd = clf_004_fwd.extract_observations(pkg)
    obs_005_fwd = clf_005_fwd.extract_observations(pkg)
    obs_006_fwd = clf_006_fwd.extract_observations(pkg)

    # Reverse order
    clf_006_rev = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    clf_005_rev = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    clf_004_rev = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_004)

    obs_006_rev = clf_006_rev.extract_observations(pkg)
    obs_005_rev = clf_005_rev.extract_observations(pkg)
    obs_004_rev = clf_004_rev.extract_observations(pkg)

    # Verify provenance and distinct metadata
    assert obs_004_fwd.metadata == {}
    assert obs_005_fwd.metadata["algorithm_version"] == VCPClassifier.VERSION_CANDIDATE_005
    assert obs_006_fwd.metadata["algorithm_version"] == VCPClassifier.VERSION_CANDIDATE_006

    # Verify order invariance
    assert obs_004_fwd.contraction_count == obs_004_rev.contraction_count
    assert obs_004_fwd.contraction_depths == obs_004_rev.contraction_depths
    assert obs_005_fwd.contraction_count == obs_005_rev.contraction_count
    assert obs_005_fwd.contraction_depths == obs_005_rev.contraction_depths
    assert obs_006_fwd.contraction_count == obs_006_rev.contraction_count
    assert obs_006_fwd.contraction_depths == obs_006_rev.contraction_depths


def test_07_candidate_006_bit_exact_reproducibility_on_frozen_cases():
    """Proves Candidate 006 yields bit-exact outputs matching vcp_hva02_candidate_006_results.json."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = _resolve_vcp_fixture("vcp_hva00_case_manifest.json")
    frozen_c006_path = _resolve_vcp_fixture("vcp_hva02_candidate_006_results.json")

    if not (db_path.exists() and manifest_path.exists() and frozen_c006_path.exists()):
        pytest.skip("Research data files not present in worktree; skipping frozen replay.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(frozen_c006_path, "r", encoding="utf-8") as f:
        frozen_c006 = json.load(f)

    frozen_records = {r["case_id"]: r["candidate_006"] for r in frozen_c006["case_records"]}
    pos_cases = [c for c in manifest["cases"] if c.get("stratum") == "CLEAR_POSITIVE_VCP"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    compiler = VCPTemporalCaseCompiler()
    clf_c006 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)

    for case_defn in pos_cases:
        cid = case_defn["case_id"]
        sym = case_defn["symbol"]
        cutoff = case_defn["evaluation_cutoff_date"]
        cutoff_iso = f"{cutoff}T21:00:00Z"

        rows = cursor.execute(
            """
            SELECT trade_date, open, high, low, close, volume
            FROM asset_ohlcv_daily
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date ASC
            """,
            (sym, cutoff),
        ).fetchall()

        bars = [
            DailyOHLCVBar(
                symbol=sym,
                bar_date=r[0],
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=int(r[5]),
                valid_time=f"{r[0]}T21:00:00Z",
                known_at=f"{r[0]}T22:00:00Z",
            )
            for r in rows
        ]

        pkg = compiler.compile_case(
            case_id=cid,
            security_id=f"SEC-{sym}",
            symbol=sym,
            evaluation_as_of=cutoff_iso,
            raw_bars=bars,
            reference_data={"rs_rank": 85.0},
            corporate_actions=[],
        )

        obs = clf_c006.extract_observations(pkg)
        assessment = clf_c006.classify_case(pkg)

        frozen_rec = frozen_records[cid]
        assert assessment.vcp_classification == frozen_rec["vcp_classification"], f"{cid} classification mismatch"
        assert obs.contraction_count == frozen_rec["contraction_count"], f"{cid} count mismatch"
        assert obs.contraction_depths == frozen_rec["contraction_depths"], f"{cid} depths mismatch"
        assert obs.is_progressive_tightening == frozen_rec["is_progressive_tightening"], f"{cid} tightening mismatch"
        if frozen_rec["final_wave_volume_ratio"] is not None:
            assert obs.final_wave_volume_ratio == frozen_rec["final_wave_volume_ratio"], f"{cid} volume ratio mismatch"
        assert assessment.vcp_qualified == frozen_rec["vcp_qualified"], f"{cid} qualification mismatch"

    conn.close()
