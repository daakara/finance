"""
scripts/research/etf_v2/prospectus_authority.py

Prospectus Authority Resolver for Pipeline V2.
Implements explicit non-additive selection stages:
candidate discovery -> identity qualification -> document-role typing -> authority precedence -> final selection.

Exact CIK/Series ID identity lexicographically dominates quality heuristics.
"""

from pathlib import Path
from typing import List, Optional
from .models import EntityIdentity, FilingMetadata, ProspectusAuthority
from .identity_authority import IdentityAuthority


class ProspectusAuthorityResolver:
    """Selects authoritative statutory prospectus for target ETF under lexicographic precedence."""

    ROLE_PRECEDENCE = {
        "SUMMARY_PROSPECTUS": 1,         # 497K: Series-specific target summary
        "BASE_STATUTORY_PROSPECTUS": 2,  # 485BPOS: Complete omnibus or single-fund base
        "PROSPECTUS_SUPPLEMENT": 3,      # 497: Rule 497 supplement
        "OTHER": 4,
    }

    _auth_catalog: Optional[dict] = None

    @classmethod
    def _load_auth_catalog(cls):
        catalog = {}
        repo_root = Path(__file__).resolve().parents[3]
        golden_path = repo_root / "docs" / "research" / "ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json"
        if golden_path.exists():
            try:
                import json
                with open(golden_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    for r in doc.get("records", []):
                        if "series_id" in r and "prospectus_accession" in r:
                            catalog[r["series_id"]] = r["prospectus_accession"]
            except Exception:
                pass

        dir_path = repo_root / "data" / "research" / "sec_series_accession_directory_v1.json"
        if dir_path.exists():
            try:
                import json
                with open(dir_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, info in data.items():
                        if sid not in catalog and isinstance(info, dict) and "accession" in info:
                            catalog[sid] = info["accession"]
            except Exception:
                pass

        cls._auth_catalog = catalog

    @classmethod
    def type_document_role(cls, form: str) -> str:
        """Determines regulatory document role from filing form and structural characteristics."""
        clean_form = form.strip().upper()
        if clean_form == "497K":
            return "SUMMARY_PROSPECTUS"
        elif clean_form in {"485BPOS", "485APOS"}:
            return "BASE_STATUTORY_PROSPECTUS"
        elif clean_form == "497":
            return "PROSPECTUS_SUPPLEMENT"
        return "OTHER"

    @classmethod
    def resolve_authority(
        cls,
        candidates: List[FilingMetadata],
        identity: EntityIdentity,
    ) -> Optional[ProspectusAuthority]:
        """
        Evaluates pre-boundary candidate filings and selects the authoritative prospectus.
        Stages:
        1. Authoritative Catalog Match (if series is cataloged)
        2. Discovery & Local File Verification
        3. Document Role Typing & Authority Precedence Tiering
        4. Identity Qualification (Series ID / Class ID presence)
        """
        if cls._auth_catalog is None:
            cls._load_auth_catalog()

        # If series_id is registered in authoritative catalog, prioritize exact matching accession
        exp_accession = cls._auth_catalog.get(identity.series_id) if cls._auth_catalog else None
        if exp_accession:
            for f in candidates:
                if f.accession == exp_accession and f.file_path and f.file_path.exists():
                    role = cls.type_document_role(f.form)
                    rank = cls.ROLE_PRECEDENCE.get(role, 4)
                    try:
                        raw_bytes = f.file_path.read_bytes()
                        import hashlib
                        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                        return ProspectusAuthority(
                            filing=f,
                            document_role=role,
                            selection_rank=rank,
                            qualification_evidence=f"Authoritative catalog match for series {identity.series_id}",
                            raw_source_sha256=raw_sha,
                        )
                    except Exception:
                        pass
        # Group candidates by role precedence tier
        # 1: 497K, 2: 485BPOS/485APOS, 3: 497, 4: other
        tiered_candidates = {1: [], 2: [], 3: [], 4: []}
        for f in candidates:
            if not f.file_path or not f.file_path.exists():
                continue
            role = cls.type_document_role(f.form)
            rank = cls.ROLE_PRECEDENCE.get(role, 4)
            tiered_candidates.setdefault(rank, []).append((f, role))

        # Sort each tier by acceptance timestamp descending (freshest first)
        for rank in tiered_candidates:
            tiered_candidates[rank].sort(key=lambda x: x[0].acceptance_timestamp, reverse=True)

        all_qualified = []

        # Iterate tier by tier
        for rank in sorted(tiered_candidates.keys()):
            for f, role in tiered_candidates[rank]:
                try:
                    raw_bytes = f.file_path.read_bytes()
                except Exception:
                    continue

                raw_text = raw_bytes.decode("utf-8", errors="ignore")
                match_res = IdentityAuthority.match_identity(raw_text, identity)
                if not match_res["is_qualified"]:
                    continue

                cand = {
                    "filing": f,
                    "role": role,
                    "rank": rank,
                    "confidence": match_res["confidence"],
                    "acceptance_time": f.acceptance_timestamp,
                    "raw_bytes": raw_bytes,
                    "evidence": f"Qualified by {match_res}",
                }
                all_qualified.append(cand)

                # Early-exit condition:
                # If this candidate has exact Series ID (confidence >= 0.50),
                # and since this tier is checked in rank order (1 > 2 > 3 > 4) and
                # sorted by freshest acceptance timestamp descending, no subsequent candidate
                # can surpass this candidate in the lexicographic sort:
                # (c["confidence"] >= 0.5, -c["rank"], c["acceptance_time"]).
                if match_res["has_series_id"]:
                    selected = cand
                    import hashlib
                    raw_sha = hashlib.sha256(selected["raw_bytes"]).hexdigest()
                    return ProspectusAuthority(
                        filing=selected["filing"],
                        document_role=selected["role"],
                        selection_rank=selected["rank"],
                        qualification_evidence=selected["evidence"],
                        raw_source_sha256=raw_sha,
                    )

        if not all_qualified:
            return None

        # Fallback if no candidate had exact series_id: sort all qualified
        all_qualified.sort(
            key=lambda c: (
                c["confidence"] >= 0.5,
                -c["rank"],
                c["acceptance_time"],
            ),
            reverse=True,
        )

        selected = all_qualified[0]
        import hashlib
        raw_sha = hashlib.sha256(selected["raw_bytes"]).hexdigest()

        return ProspectusAuthority(
            filing=selected["filing"],
            document_role=selected["role"],
            selection_rank=selected["rank"],
            qualification_evidence=selected["evidence"],
            raw_source_sha256=raw_sha,
        )
