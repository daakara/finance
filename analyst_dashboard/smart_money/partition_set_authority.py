"""
Partition Set Authority for ARX Terminal Smart Money Wave 2C.
Establishes the immutable required partition set for the 106 governed common-equity issuers.
Enforces the denominator lock: no missing partition can ever disappear from the denominator,
and observed partition sets cannot substitute for the required set.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash

DEFAULT_UNIVERSE_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "smart_money_universe.db",
)

GOVERNED_UNIVERSE_BUILD_ID = "UB_SM_20261010_R1_19ef857fec1f"
REQUIRED_PARTITION_COUNT = 106
GOVERNED_UNIVERSE_CONTENT_HASH = "ccd99943c46d1a9892c6b25cf712ceee73bb14c6941fe9ab8e34968220fd822c"


class _HybridMethod:
    def __init__(self, f):
        self.f = f

    def __get__(self, inst, owner):
        target = inst if inst is not None else owner()
        return lambda *a, **k: self.f(target, *a, **k)


class PartitionSetAuthority:
    """
    Immutable authority over the required partition set for Wave 2C certification.
    """

    def __init__(self, universe_db_path: str = DEFAULT_UNIVERSE_DB):
        self.universe_db_path = universe_db_path
        self._cached_partitions: Optional[List[Dict[str, str]]] = None
        self._cached_hash: Optional[str] = None

    @_HybridMethod
    def get_required_partitions(self) -> List[Dict[str, str]]:
        """
        Loads and returns the canonical sorted list of 106 required partitions.
        Sorted strictly ascending by 10-digit zero-padded issuer_cik.
        """
        if self._cached_partitions is not None:
            return self._cached_partitions

        conn = sqlite3.connect(self.universe_db_path)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT scannable_members_json FROM sm_universe_builds WHERE universe_build_id = ?",
                (GOVERNED_UNIVERSE_BUILD_ID,),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"CANONICAL_UNIVERSE_NOT_FOUND: {GOVERNED_UNIVERSE_BUILD_ID} missing from {self.universe_db_path}")

            raw_members = json.loads(row["scannable_members_json"])
            if len(raw_members) != REQUIRED_PARTITION_COUNT:
                raise RuntimeError(
                    f"DENOMINATOR_MUTATION_BREACH: Expected {REQUIRED_PARTITION_COUNT} scannable members, found {len(raw_members)}"
                )

            partitions = []
            for m in raw_members:
                cik = m["cik"].strip().zfill(10)
                sec_id = m.get("security_id", f"US_EQ_{m.get('symbol', 'UNKNOWN')}_{cik}")
                partitions.append({
                    "canonical_security_id": sec_id,
                    "issuer_cik": cik,
                    "symbol": m["symbol"],
                    "universe_build_id": GOVERNED_UNIVERSE_BUILD_ID,
                })

            # Canonical sort: issuer_cik ascending
            partitions.sort(key=lambda p: p["issuer_cik"])
            self._cached_partitions = partitions
            return self._cached_partitions
        finally:
            conn.close()

    @_HybridMethod
    def get_required_partition_ciks(self) -> List[str]:
        """Returns sorted list of canonical 10-digit zero-padded CIKs."""
        return [p["issuer_cik"] for p in self.get_required_partitions()]

    @_HybridMethod
    def compute_required_partition_set_hash(self) -> str:
        """Computes deterministic SHA256 of canonical partition set under SM_CERT_CANON_V1."""
        if self._cached_hash is not None:
            return self._cached_hash
        partitions = self.get_required_partitions()
        self._cached_hash = compute_canonical_hash(partitions)
        return self._cached_hash

    def validate_partition_set_completeness(self, observed_ciks: List[str]) -> Dict[str, Any]:
        """
        Validates an observed partition set against the immutable required set.
        Enforces:
        - MISSING_REQUIRED_PARTITION_CAN_DISAPPEAR_FROM_DENOMINATOR = NO
        - OBSERVED_PARTITION_SET_SUBSTITUTES_FOR_REQUIRED_PARTITION_SET = NO
        """
        required_set = set(self.get_required_partition_ciks())
        observed_set = set([c.strip().zfill(10) for c in observed_ciks])

        missing = sorted(list(required_set - observed_set))
        unexpected = sorted(list(observed_set - required_set))

        is_complete = (len(missing) == 0 and len(unexpected) == 0)

        return {
            "is_complete": is_complete,
            "required_count": len(required_set),
            "observed_count": len(observed_set),
            "missing_ciks": missing,
            "unexpected_ciks": unexpected,
            "required_partition_set_hash": self.compute_required_partition_set_hash(),
        }

    def export_canonical_artifact(self, output_path: str):
        """Exports the immutable canonical partition set to disk."""
        partitions = self.get_required_partitions()
        payload = {
            "universe_build_id": GOVERNED_UNIVERSE_BUILD_ID,
            "required_partition_count": len(partitions),
            "required_partition_set_hash": self.compute_required_partition_set_hash(),
            "partitions": partitions,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(canonicalize_json(payload))
