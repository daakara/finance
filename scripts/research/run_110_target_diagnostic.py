"""110-target diagnostic re-run for Gate v1.0.3 Section 16.

Runs only the 110 targets in PRE_REMEDIATION_REGRESSION_CORPUS_V2.json sublist_A
through the V1.3.0/V1.4.0/V1.5.0 engine stack and reports post-implementation outcomes.

NOT a scratch file — this diagnostic re-run is part of the gate verification.
"""
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.research.document_index_engine import (
    DocumentIndex, DocumentIdentity, INDEX_ENGINE_VERSION,
)
from scripts.research.series_prospectus_mapper import (
    SeriesProspectusMapper, SeriesMetadata, SERIES_RESOLVER_VERSION,
)

CORPUS_PATH = Path("docs/research/PRE_REMEDIATION_REGRESSION_CORPUS_V2.json")
SELECTOR_RESULTS_PATH = Path("docs/research/STATUTORY_SELECTOR_DRY_RUN_RESULTS.json")
PROSPECTUS_DIR = Path("data/research/cache/sec_prospectus")
ALIAS_AUTHORITY_PATH = Path("docs/research/ETF_HISTORICAL_IDENTITY_ALIAS_AUTHORITY_V1_1.json")


def run_diagnostic():
    print("=" * 80)
    print("SECTION 16: 110-TARGET DIAGNOSTIC RE-RUN")
    print(f"Engine: {INDEX_ENGINE_VERSION} / {SERIES_RESOLVER_VERSION}")
    print("=" * 80)

    corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
    targets_a = corpus["sublist_A_actionable_defects"]["targets"]
    print(f"Loaded {len(targets_a)} sublist_A targets")

    # Build acc -> cache_filename mapping directly from corpus entries
    # (STATUTORY_SELECTOR_DRY_RUN_RESULTS.json lacks document_filename)
    acc_to_file: Dict[str, str] = {}
    for t in targets_a:
        acc = t["selected_accession"]
        fname = t.get("document_filename", "")
        if acc and fname:
            cache_fname = f"{acc}_{fname}"
            acc_to_file[acc] = cache_fname

    # Load alias authority
    alias_authority: Dict[str, List[str]] = {}
    if ALIAS_AUTHORITY_PATH.exists():
        aa = json.load(open(ALIAS_AUTHORITY_PATH, encoding="utf-8"))
        for entry in aa.get("aliases", []):
            sym = entry.get("symbol", "")
            alias_name = entry.get("historical_source_legal_name", "")
            if sym and alias_name:
                alias_authority.setdefault(sym, []).append(alias_name)
        print(f"Loaded alias authority: {len(alias_authority)} entries")

    # Group targets by document
    doc_to_targets = defaultdict(list)
    for t in targets_a:
        acc = t["selected_accession"]
        cache_fname = acc_to_file.get(acc, "")
        if cache_fname:
            doc_to_targets[cache_fname].append(t)
        else:
            print(f"[WARN] No cache file found for {t['symbol']} acc={acc}")

    print(f"Grouped across {len(doc_to_targets)} unique documents")
    print()

    outcomes = []
    outcome_counts = Counter()

    for doc_num, (cache_fname, targets_in_doc) in enumerate(sorted(doc_to_targets.items()), 1):
        fpath = PROSPECTUS_DIR / cache_fname
        if not fpath.exists():
            for t in targets_in_doc:
                outcome_counts["FILE_NOT_FOUND"] += 1
                outcomes.append({
                    "symbol": t["symbol"],
                    "pre_remediation_status": t["pre_remediation_resolution_status"],
                    "post_remediation_status": "FILE_NOT_FOUND",
                    "movement": "NO_CHANGE",
                })
            continue

        try:
            raw_bytes = fpath.read_bytes()
        except Exception as e:
            print(f"Failed to read {cache_fname}: {e}")
            continue

        acc = cache_fname.split("_")[0]
        doc_filename = cache_fname.split("_", 1)[1] if "_" in cache_fname else cache_fname
        cik = str(targets_in_doc[0].get("cik", "")).zfill(10)

        known_meta = [{"legal_name": t["legal_name"]} for t in targets_in_doc]
        doc_alias_names: List[str] = []
        for t in targets_in_doc:
            sym = t["symbol"]
            for alias in alias_authority.get(sym, []):
                if alias not in doc_alias_names:
                    doc_alias_names.append(alias)

        identity = DocumentIdentity(
            cik=cik,
            accession=acc,
            form=targets_in_doc[0].get("selected_form", "485BPOS"),
            filing_date="2026-09-01",
            document_filename=doc_filename,
            source_byte_length=len(raw_bytes),
        )

        doc_index = DocumentIndex(
            identity, raw_bytes, known_meta,
            alias_legal_names=doc_alias_names if doc_alias_names else None,
        )

        for t in targets_in_doc:
            sym = t["symbol"]
            target = SeriesMetadata(
                symbol=sym,
                cik=cik,
                series_id=t.get("series_id", ""),
                class_id=t.get("class_id", ""),
                legal_name=t["legal_name"],
            )
            res = SeriesProspectusMapper.map_series(target, doc_index, [])
            post_status = res.mapping_outcome
            pre_status = t.get("pre_remediation_resolution_status", "")
            movement = "FIXED" if post_status != pre_status else "NO_CHANGE"
            if post_status != pre_status and "NOT_FOUND" in pre_status and "MAPPED" not in post_status:
                movement = "CHANGE_UNKNOWN"

            outcome_counts[post_status] += 1
            outcomes.append({
                "symbol": sym,
                "causal_owner": t.get("causal_owner", ""),
                "pre_remediation_status": pre_status,
                "post_remediation_status": post_status,
                "movement": movement,
            })

        print(f"  [{doc_num:3d}/{len(doc_to_targets)}] {cache_fname[:60]} -> {[t['symbol'] for t in targets_in_doc]}")

    print()
    print("=" * 80)
    print("POST-RUN OUTCOME SUMMARY")
    print("=" * 80)
    fixed = [r for r in outcomes if r["post_remediation_status"] != r["pre_remediation_status"]]
    unchanged = [r for r in outcomes if r["post_remediation_status"] == r["pre_remediation_status"]]
    print(f"Total targets: {len(outcomes)}")
    print(f"FIXED (status changed): {len(fixed)}")
    print(f"UNCHANGED: {len(unchanged)}")
    print()
    print("Outcome distribution:")
    for status, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {status:40s}: {count}")
    print()
    print("Still-failing targets (no change):")
    for r in unchanged:
        print(f"  {r['symbol']:10s}  causal={r['causal_owner']:35s}  status={r['post_remediation_status']}")
    print()
    print("Fixed targets:")
    for r in sorted(fixed, key=lambda x: x["symbol"]):
        print(f"  {r['symbol']:10s}  {r['pre_remediation_status']} -> {r['post_remediation_status']}")

    # Write output
    output_path = Path("docs/research/SECTION16_110TARGET_DIAGNOSTIC_RERUN.json")
    output = {
        "engine_versions": {
            "doc_index": INDEX_ENGINE_VERSION,
            "series_resolver": SERIES_RESOLVER_VERSION,
        },
        "total": len(outcomes),
        "fixed": len(fixed),
        "unchanged": len(unchanged),
        "outcome_counts": dict(outcome_counts),
        "results": sorted(outcomes, key=lambda x: x["symbol"]),
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    run_diagnostic()
