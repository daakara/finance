"""Adversarial Verification Suite for CHECKPOINT_STORE_V1_1_0.

Asserts Section 8:
Proves a checkpoint record is invalidated when ANY one of these changes:
- source SHA256
- source accession
- selector version
- resolver version
- parser version
- policy version
- snapshot boundary

Also proves identical inputs resume successfully.
Requires: STALE_CHECKPOINT_REUSE = 0.
"""

import pytest
from pathlib import Path
from scripts.research.checkpoint_store import (
    CheckpointStore,
    CheckpointRecord,
    CheckpointIdentity,
    CHECKPOINT_STORE_VERSION,
)


def test_checkpoint_store_version():
    assert CHECKPOINT_STORE_VERSION == "CHECKPOINT_STORE_V1_1_0"


def test_checkpoint_adversarial_invalidation_matrix(tmp_path):
    checkpoint_file = tmp_path / "test_adversarial_checkpoints.jsonl"
    manifest_sha = "764363abedf51dd40365cf26d17d429fe4596619bd7e8e648cca17502286635a"
    run_id = "RUN_ADVERSARIAL_001"

    store = CheckpointStore(checkpoint_file, run_id, manifest_sha)

    # Base valid inputs
    base_sym = "IVV"
    base_cik = "0001100663"
    base_series = "S000002871"
    base_class = "C000007882"
    base_acc = "0001193125-26-000001"
    base_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    base_sel_ver = "STATUTORY_FILING_SELECTOR_V1_3_1"
    base_idx_ver = "DOC_INDEX_V1_1_0"
    base_res_ver = "SERIES_RESOLVER_V1_2_0"
    base_parse_ver = "MANDATE_PARSER_V1_2_0_FROZEN"
    base_pol_ver = "ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_1"
    base_boundary = "2026-09-24T23:59:59Z"

    store.save_record(
        symbol=base_sym,
        cik=base_cik,
        series_id=base_series,
        class_id=base_class,
        document_index_key="doc_key_ivv",
        series_resolution_key="res_key_ivv",
        mapping_outcome="MAPPED_EXACT_SERIES_ID",
        section_hash="sec_hash_ivv",
        mandate_parse_status="PASS",
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        document_index_version=base_idx_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )

    # 1. Identical inputs resume successfully
    identical_id = CheckpointIdentity(
        symbol=base_sym,
        cik=base_cik,
        series_id=base_series,
        class_id=base_class,
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        document_index_version=base_idx_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=identical_id) is True

    # 2. Invalidation upon source SHA change
    mutated_sha = CheckpointIdentity(
        symbol=base_sym,
        source_accession=base_acc,
        source_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        selector_version=base_sel_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=mutated_sha) is False

    # 3. Invalidation upon accession change
    mutated_acc = CheckpointIdentity(
        symbol=base_sym,
        source_accession="0001193125-26-999999",
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=mutated_acc) is False

    # 4. Invalidation upon selector version change
    mutated_sel = CheckpointIdentity(
        symbol=base_sym,
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version="STATUTORY_FILING_SELECTOR_V1_4_0",
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=mutated_sel) is False

    # 5. Invalidation upon resolver version change
    mutated_res = CheckpointIdentity(
        symbol=base_sym,
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        series_resolver_version="SERIES_RESOLVER_V1_3_0",
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=mutated_res) is False

    # 6. Invalidation upon parser version change
    mutated_parse = CheckpointIdentity(
        symbol=base_sym,
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version="MANDATE_PARSER_V1_3_0",
        policy_version=base_pol_ver,
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=mutated_parse) is False

    # 7. Invalidation upon policy version change
    mutated_pol = CheckpointIdentity(
        symbol=base_sym,
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version="ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_2",
        snapshot_boundary=base_boundary,
    )
    assert store.is_completed(base_sym, identity=mutated_pol) is False

    # 8. Invalidation upon snapshot boundary change
    mutated_boundary = CheckpointIdentity(
        symbol=base_sym,
        source_accession=base_acc,
        source_sha256=base_sha,
        selector_version=base_sel_ver,
        series_resolver_version=base_res_ver,
        mandate_parser_version=base_parse_ver,
        policy_version=base_pol_ver,
        snapshot_boundary="2026-09-27T23:59:59Z",
    )
    assert store.is_completed(base_sym, identity=mutated_boundary) is False

    # 9. Verify keyword arguments checking
    assert store.is_completed(base_sym, mandate_parser_version="MANDATE_PARSER_V1_3_0") is False
    assert store.is_completed(base_sym, mandate_parser_version=base_parse_ver) is True
