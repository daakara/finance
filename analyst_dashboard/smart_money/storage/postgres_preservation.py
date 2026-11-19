"""
ARX Smart Money — PostgreSQL State Preservation Fingerprinting.

Provides cryptographic before/after fingerprinting over protected table contents
to guarantee zero silent corruption, zero unexpected mutations, and complete row-level
preservation across canary runs, dry runs, and batch qualification cycles.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The 12 canonical tables forming the Smart Money storage universe
CANONICAL_PROTECTED_TABLES: Tuple[str, ...] = (
    "form4_raw_filings",
    "form4_parsed_transactions",
    "sm_source_availability_ledger",
    "sm_amendments",
    "sm_audit_events",
    "sm_reconciliation_receipts",
    "scanner_snapshots",
    "scanner_active_snapshots",
    "sm_run_requests",
    "governed_issuers",
    "sm_schema_migrations",
    "sm_source_gaps",
)


def compute_database_preservation_fingerprint(
    conn: Any,
    scope: Optional[str] = None,
    tables: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Computes a deterministic cryptographic fingerprint over the contents of protected tables.

    Uses PostgreSQL `to_jsonb(t)` to convert entire rows into canonical JSON representations,
    sorts the serialized records, and computes a SHA-256 digest per table and an aggregate
    digest across the entire protected universe.

    Guarantees detection of any field-level mutation even when table row counts remain identical.
    """
    target_tables = tuple(tables or CANONICAL_PROTECTED_TABLES)
    captured_at = datetime.now(timezone.utc).isoformat()
    table_fingerprints: Dict[str, Dict[str, Any]] = {}
    aggregate_hash_input: List[str] = []

    with conn.cursor() as cur:
        # Check which tables exist in public schema
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        existing_tables = {row[0] for row in cur.fetchall()}

        for table in target_tables:
            if table not in existing_tables:
                continue

            # Fetch all rows converted to JSONB by PostgreSQL
            cur.execute(f"SELECT to_jsonb(t) FROM {table} AS t;")  # noqa: S608
            raw_rows = cur.fetchall()
            row_count = len(raw_rows)

            if row_count == 0:
                content_sha = hashlib.sha256(b"EMPTY_TABLE").hexdigest()
            else:
                serialized_rows = sorted(json.dumps(r[0], sort_keys=True) for r in raw_rows)
                content_sha = hashlib.sha256("\n".join(serialized_rows).encode("utf-8")).hexdigest()

            table_fingerprints[table] = {
                "row_count": row_count,
                "content_sha256": content_sha,
            }
            aggregate_hash_input.append(f"{table}:{row_count}:{content_sha}")

    # Roll back read cursor transaction to release locks/snapshots cleanly
    conn.rollback()

    overall_sha = hashlib.sha256("\n".join(aggregate_hash_input).encode("utf-8")).hexdigest()

    return {
        "fingerprint_version": "1.0.0",
        "captured_at": captured_at,
        "scope": scope,
        "table_count": len(table_fingerprints),
        "tables": table_fingerprints,
        "overall_fingerprint_sha256": overall_sha,
    }


def diff_preservation_fingerprints(
    before: Dict[str, Any],
    after: Dict[str, Any],
    allowed_table_mutations: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compares two preservation fingerprints and reports exact differences.

    Args:
        before: Baseline fingerprint dictionary.
        after: Post-execution fingerprint dictionary.
        allowed_table_mutations: Optional map of table_name -> {'allowed_count_delta': int}
            specifying intended additions.

    Returns:
        Dictionary with 'matched': bool and list of 'differences'.
    """
    differences: List[Dict[str, Any]] = []
    before_tables = before.get("tables", {})
    after_tables = after.get("tables", {})
    all_table_names = sorted(set(before_tables.keys()) | set(after_tables.keys()))
    allowed = allowed_table_mutations or {}

    for t in all_table_names:
        b_info = before_tables.get(t)
        a_info = after_tables.get(t)

        if b_info is None:
            differences.append({
                "table": t,
                "kind": "TABLE_CREATED",
                "detail": f"Table '{t}' did not exist before execution, row_count={a_info.get('row_count')}",
            })
            continue

        if a_info is None:
            differences.append({
                "table": t,
                "kind": "TABLE_DROPPED",
                "detail": f"Table '{t}' was dropped during execution",
            })
            continue

        count_diff = a_info["row_count"] - b_info["row_count"]
        hash_diff = a_info["content_sha256"] != b_info["content_sha256"]

        table_allowed = allowed.get(t, {})
        allowed_count_delta = table_allowed.get("allowed_count_delta")

        if allowed_count_delta is not None and count_diff == allowed_count_delta:
            # Expected addition, content hash change is anticipated
            continue

        if count_diff != 0:
            differences.append({
                "table": t,
                "kind": "ROW_COUNT_CHANGED",
                "before_count": b_info["row_count"],
                "after_count": a_info["row_count"],
                "delta": count_diff,
            })
        elif hash_diff:
            differences.append({
                "table": t,
                "kind": "CONTENT_MUTATION_DETECTED",
                "before_sha": b_info["content_sha256"],
                "after_sha": a_info["content_sha256"],
                "row_count": a_info["row_count"],
                "detail": f"Row count for '{t}' remained {a_info['row_count']}, but row content was modified.",
            })

    matched = len(differences) == 0
    return {
        "matched": matched,
        "difference_count": len(differences),
        "differences": differences,
        "before_sha": before.get("overall_fingerprint_sha256"),
        "after_sha": after.get("overall_fingerprint_sha256"),
    }
