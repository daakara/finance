"""
analyst_dashboard/universe/adapters/matching.py

Bounded offline source joining and conflict detection across SEC, directory,
and OpenFIGI source feeds.

Strict Non-Inference Invariants:
- Uses explicit caller-supplied matching rules; does NOT invent a production identity scheme.
- Preserves raw symbols and join keys without silently stripping or altering punctuation.
- Preserves multiple securities under one issuer CIK as distinct observations.
- Detects one-to-many collisions, exchange mismatches, and multi-match OpenFIGI ambiguities.
- Keeps unmatched records visible in candidate lists rather than silently discarding them.
- Does NOT infer global instrument identity from a matching ticker alone.
- Pure in-memory execution: zero file reads, zero network requests.
"""

from __future__ import annotations

from .models import (
    MatchedUniverseCandidate,
    OpenFigiMatchStatus,
    ParsedOpenFigiResult,
    ParsedSourceFile,
    ParsedSourceRow,
    SourceMatchingAccounting,
)

# Standard venue name normalization map for cross-feed consistency checks
VENUE_MAP = {
    "NASDAQ": "NASDAQ",
    "NQ": "NASDAQ",
    "N": "NYSE",
    "NYSE": "NYSE",
    "P": "ARCA",
    "ARCA": "ARCA",
    "Z": "BATS",
    "BATS": "BATS",
    "A": "AMEX",
    "AMEX": "AMEX",
}


def _normalize_venue(raw_venue: str | None) -> str | None:
    if not raw_venue:
        return None
    return VENUE_MAP.get(str(raw_venue).strip().upper(), str(raw_venue).strip().upper())


def match_sources_with_accounting(
    sec_file: ParsedSourceFile | None = None,
    directory_files: list[ParsedSourceFile] | None = None,
    openfigi_results: list[ParsedOpenFigiResult] | None = None,
    require_venue_match: bool = False,
) -> tuple[list[MatchedUniverseCandidate], SourceMatchingAccounting]:
    """
    Performs bounded join across parsed SEC records, directory records, and OpenFIGI results.
    Explicitly surfaces conflicts, ambiguities, and unmatched records, and returns full
    accounting across all input source populations.
    """
    malformed_rows: list[ParsedSourceRow] = []

    # 1. Index SEC rows by symbol and preserve provenance
    sec_by_sym: dict[str, list[tuple[ParsedSourceRow, str, str]]] = {}
    if sec_file:
        for r in sec_file.rows:
            if r.is_malformed:
                malformed_rows.append(r)
                continue
            sym = r.raw_values.get("symbol")
            if sym:
                sec_by_sym.setdefault(sym, []).append((r, sec_file.source_identifier, sec_file.content_sha256))

    # 2. Index Directory rows by raw_symbol and preserve provenance
    dir_by_sym: dict[str, list[tuple[ParsedSourceRow, str, str]]] = {}
    if directory_files:
        for df in directory_files:
            for r in df.rows:
                if r.is_malformed:
                    malformed_rows.append(r)
                    continue
                sym = r.raw_values.get("raw_symbol")
                if sym:
                    dir_by_sym.setdefault(sym, []).append((r, df.source_identifier, df.content_sha256))

    # 3. Index OpenFIGI results by ticker
    figi_by_sym: dict[str, list[ParsedOpenFigiResult]] = {}
    if openfigi_results:
        for res in openfigi_results:
            req_ticker = res.request_payload.get("idValue") if res.request_payload.get("idType") == "TICKER" else None
            match_ticker = res.matches[0].get("ticker") if res.matches else None
            sym = req_ticker or match_ticker
            if sym:
                figi_by_sym.setdefault(str(sym).strip().upper(), []).append(res)

    all_symbols = sorted(set(sec_by_sym.keys()) | set(dir_by_sym.keys()) | set(figi_by_sym.keys()))
    candidates: list[MatchedUniverseCandidate] = []

    for sym in all_symbols:
        sec_entries = sec_by_sym.get(sym, [])
        dir_entries = dir_by_sym.get(sym, [])
        figi_list = figi_by_sym.get(sym, [])

        conflicts: list[str] = []
        is_ambiguous = False

        # Collision detection in SEC observations
        sec_row = None
        sec_src_id = None
        sec_hash = None
        if len(sec_entries) == 1:
            sec_row, sec_src_id, sec_hash = sec_entries[0]
        elif len(sec_entries) > 1:
            is_ambiguous = True
            conflicts.append(f"ONE_TO_MANY_SEC_RECORDS ({len(sec_entries)} rows for symbol {sym})")
            sec_row, sec_src_id, sec_hash = sec_entries[0]

        # Collision detection in directory observations
        dir_row = None
        dir_src_id = None
        dir_hash = None
        if len(dir_entries) == 1:
            dir_row, dir_src_id, dir_hash = dir_entries[0]
        elif len(dir_entries) > 1:
            is_ambiguous = True
            conflicts.append(f"ONE_TO_MANY_DIRECTORY_RECORDS ({len(dir_entries)} rows for symbol {sym})")
            dir_row, dir_src_id, dir_hash = dir_entries[0]

        # Venue cross-feed consistency check
        if sec_row and dir_row:
            sec_exch = _normalize_venue(sec_row.raw_values.get("sec_exchange"))
            dir_exch = _normalize_venue(dir_row.raw_values.get("venue_exchange"))
            if sec_exch and dir_exch and sec_exch != dir_exch:
                conflicts.append(f"VENUE_MISMATCH (SEC: {sec_exch} vs Directory: {dir_exch})")
                if require_venue_match:
                    is_ambiguous = True

        # OpenFIGI result association and conflict check
        figi_res = None
        if len(figi_list) == 1:
            figi_res = figi_list[0]
            if figi_res.match_status == OpenFigiMatchStatus.MULTI_MATCH:
                is_ambiguous = True
                conflicts.append(f"OPENFIGI_MULTI_MATCH_UNRESOLVED ({len(figi_res.matches)} matches returned)")
            elif figi_res.match_status == OpenFigiMatchStatus.NO_MATCH:
                conflicts.append("OPENFIGI_NO_MATCH")
            elif figi_res.match_status == OpenFigiMatchStatus.ERROR:
                conflicts.append(f"OPENFIGI_ERROR ({figi_res.error_message})")
        elif len(figi_list) > 1:
            is_ambiguous = True
            conflicts.append(f"ONE_TO_MANY_OPENFIGI_RESULTS ({len(figi_list)} results for symbol {sym})")
            figi_res = figi_list[0]

        candidates.append(
            MatchedUniverseCandidate(
                join_key=sym,
                sec_row=sec_row,
                directory_row=dir_row,
                figi_result=figi_res,
                join_conflicts=conflicts,
                is_ambiguous=is_ambiguous,
                sec_source_identifier=sec_src_id,
                sec_content_hash=sec_hash,
                directory_source_identifier=dir_src_id,
                directory_content_hash=dir_hash,
                figi_source_identifier="openfigi_mapping" if figi_res else None,
                figi_content_hash=None,
            )
        )

    fully_matched = sum(1 for c in candidates if c.sec_row is not None and c.directory_row is not None)
    sec_only = sum(1 for c in candidates if c.sec_row is not None and c.directory_row is None)
    dir_only = sum(1 for c in candidates if c.directory_row is not None and c.sec_row is None)
    figi_only = sum(1 for c in candidates if c.figi_result is not None and c.sec_row is None and c.directory_row is None)
    ambiguous_cnt = sum(1 for c in candidates if c.is_ambiguous)
    conflict_cnt = sum(1 for c in candidates if bool(c.join_conflicts))

    accounting = SourceMatchingAccounting(
        total_candidates=len(candidates),
        fully_matched_count=fully_matched,
        sec_only_count=sec_only,
        directory_only_count=dir_only,
        figi_only_count=figi_only,
        ambiguous_count=ambiguous_cnt,
        conflict_count=conflict_cnt,
        malformed_source_row_count=len(malformed_rows),
        malformed_source_rows=malformed_rows,
    )

    return candidates, accounting


def match_sources(
    sec_file: ParsedSourceFile | None = None,
    directory_files: list[ParsedSourceFile] | None = None,
    openfigi_results: list[ParsedOpenFigiResult] | None = None,
    require_venue_match: bool = False,
    return_accounting: bool = False,
) -> list[MatchedUniverseCandidate] | tuple[list[MatchedUniverseCandidate], SourceMatchingAccounting]:
    """
    Backwards-compatible wrapper for source matching.
    Returns list of candidates, or (candidates, accounting) if return_accounting=True.
    """
    candidates, accounting = match_sources_with_accounting(
        sec_file=sec_file,
        directory_files=directory_files,
        openfigi_results=openfigi_results,
        require_venue_match=require_venue_match,
    )
    if return_accounting:
        return candidates, accounting
    return candidates
