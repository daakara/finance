"""ARX Terminal — Candidate 006 Replay Harness.

Evaluates Candidate 006 against the 60 development cases using the packaged
fixtures in tests/fixtures/vcp/ and the committed market database.
Zero external network calls, zero state mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar


def resolve_fixture(filename: str) -> Path:
    p1 = REPO_ROOT / "tests" / "fixtures" / "vcp" / filename
    if p1.exists():
        return p1
    p2 = REPO_ROOT / "data" / "research" / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Fixture {filename} not found in tests/fixtures/vcp or data/research")


def main() -> int:
    manifest_path = resolve_fixture("vcp_hva00_case_manifest.json")
    db_path = REPO_ROOT / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"

    if not db_path.exists():
        print(f"ERROR: Market database not found at {db_path}", file=sys.stderr)
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = manifest["cases"]
    print(f"Loaded manifest with {len(cases)} cases.")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    compiler = VCPTemporalCaseCompiler()
    clf_c006 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)

    qualified_cases = []
    non_qualified_count = 0
    abstention_count = 0
    controls_qualified = 0

    for c in cases:
        cid = c["case_id"]
        sym = c["symbol"]
        cutoff = c["evaluation_cutoff_date"]
        cutoff_iso = c.get("evaluation_cutoff_iso", f"{cutoff}T21:00:00Z")
        is_control = not cid.startswith("HVA00-POS")

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

        ass = clf_c006.classify_case(pkg)
        obs = clf_c006.extract_observations(pkg)

        if ass.vcp_qualified:
            if is_control:
                controls_qualified += 1
            else:
                qualified_cases.append({
                    "case_id": cid,
                    "symbol": sym,
                    "cutoff": cutoff,
                    "count": obs.contraction_count,
                    "depths": obs.contraction_depths,
                    "volume_ratio": obs.final_wave_volume_ratio
                })
        elif ass.vcp_classification == "VCP_INSUFFICIENT_DATA":
            abstention_count += 1
        else:
            non_qualified_count += 1

    conn.close()

    print("\n=== Candidate 006 Replay Summary ===")
    print(f"Total Cases Evaluated: {len(cases)}")
    print(f"Positive Cases Qualified: {len(qualified_cases)} / 15")
    for q in qualified_cases:
        print(f"  - {q['case_id']} ({q['symbol']}) cutoff={q['cutoff']} waves={q['count']} depths={q['depths']} vol_ratio={q['volume_ratio']}")
    print(f"Control False Positives: {controls_qualified} / 45")
    print(f"Total Non-Qualified: {non_qualified_count}")
    print(f"Total Abstentions: {abstention_count}")

    assert len(qualified_cases) == 3, f"Expected 3 qualified positive cases, got {len(qualified_cases)}"
    assert controls_qualified == 0, f"Expected 0 control false positives, got {controls_qualified}"
    assert abstention_count == 8, f"Expected 8 total abstentions (7 control + 1 positive RTX), got {abstention_count}"

    print("\nCandidate 006 Replay Verified: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
