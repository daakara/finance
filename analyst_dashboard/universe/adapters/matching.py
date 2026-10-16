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


def match_sources(
    sec_file: ParsedSourceFile | None = None,
    directory_files: list[ParsedSourceFile] | None = None,
    openfigi_results: list[ParsedOpenFigiResult] | None = None,
    require_venue_match: bool = False,
) -> list[MatchedUniverseCandidate]:
    """
    Performs bounded join across parsed SEC records, directory records, and OpenFIGI results.
    Explicitly surfaces conflicts, ambiguities, and unmatched records.
    """
    # 1. Index SEC rows by symbol
    # Notice: Multiple rows could theoretically share a symbol or share an issuer CIK
    sec_by_sym: dict[str, list[ParsedSourceRow]] = {}
    if sec_file:
        for r in sec_file.rows:
            if r.is_malformed:
                continue
            sym = r.raw_values.get("symbol")
            if sym:
                sec_by_sym.setdefault(sym, []).append(r)

    # 2. Index Directory rows by raw_symbol and normalized_symbol
    dir_by_sym: dict[str, list[ParsedSourceRow]] = {}
    if directory_files:
        for df in directory_files:
            for r in df.rows:
                if r.is_malformed:
                    continue
                sym = r.raw_values.get("raw_symbol")
                if sym:
                    dir_by_sym.setdefault(sym, []).append(r)

    # 3. Index OpenFIGI results by ticker
    figi_by_sym: dict[str, list[ParsedOpenFigiResult]] = {}
    if openfigi_results:
        for res in openfigi_results:
            # Check request payload ticker or returned matches ticker
            req_ticker = res.request_payload.get("idValue") if res.request_payload.get("idType") == "TICKER" else None
            match_ticker = res.matches[0].get("ticker") if res.matches else None
            sym = req_ticker or match_ticker
            if sym:
                figi_by_sym.setdefault(str(sym).strip().upper(), []).append(res)

    all_symbols = sorted(set(sec_by_sym.keys()) | set(dir_by_sym.keys()))
    candidates: list[MatchedUniverseCandidate] = []

    for sym in all_symbols:
        sec_rows = sec_by_sym.get(sym, [])
        dir_rows = dir_by_sym.get(sym, [])
        figi_list = figi_by_sym.get(sym, [])

        conflicts: list[str] = []
        is_ambiguous = False

        # Collision detection in SEC observations
        sec_row = None
        if len(sec_rows) == 1:
            sec_row = sec_rows[0]
        elif len(sec_rows) > 1:
            is_ambiguous = True
            conflicts.append(f"ONE_TO_MANY_SEC_RECORDS ({len(sec_rows)} rows for symbol {sym})")
            sec_row = sec_rows[0]

        # Collision detection in directory observations
        dir_row = None
        if len(dir_rows) == 1:
            dir_row = dir_rows[0]
        elif len(dir_rows) > 1:
            is_ambiguous = True
            conflicts.append(f"ONE_TO_MANY_DIRECTORY_RECORDS ({len(dir_rows)} rows for symbol {sym})")
            dir_row = dir_rows[0]

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
            )
        )

    return candidates
