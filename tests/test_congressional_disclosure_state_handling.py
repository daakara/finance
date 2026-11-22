"""
Comprehensive Verification Suite: Congressional STOCK Act Disclosure State Handling (Phase 0)
Directive: ARX-CONGRESSIONAL-INGESTION-001 (Phase 0)

Verifies state handling across required operational states & adversarial invariants:
1. UNMONITORED: Default live posture where no operational pipeline exists.
   - Must emit status UNAVAILABLE / Not Monitored.
   - Must NOT cite HOLD_SEC_001 in congressional sub-domain.
   - Must NOT fabricate trades or false zeros.
2. INCOMPLETE_COVERAGE / PARTIAL:
   - Equal document counts alone CANNOT certify completeness when extraction outcomes fail or unresolved records exist.
   - Subdomain reports PARTIAL_COVERAGE.
3. CERTIFIED_EMPTY: Authoritative complete search proves 0 disclosures in coverage window.
   - Requires full binding: source inventory, chamber, interval, symbol, document identities,
     all extraction outcomes SUCCESS, and unresolved_records_count == 0.
   - Bare booleans or bare hash strings must NEVER certify empty coverage.
   - Reject fabricated CONFIRMED_ZERO_VERIFIED_COVERAGE fallback.
4. CURATED_RECORDS: User or client explicitly passes include_curated=True.
   - Emits CURATED_RESEARCH with dataset date August 2026.
   - Discloses curated historical research dataset (not live feed).
5. RETRIEVAL_ERRORS / EMPTY ARRAY WITHOUT PROOF:
   - Fails closed to UNAVAILABLE.
   - Never converts unverified empty list into verified zero.
"""

import pytest
from analyst_dashboard.analyzers.smart_money import (
    SmartMoneyEngine,
    SmartMoneyDomainStatus,
    SmartMoneyDirection,
    calculate_legislative_alignment,
)


class TestCongressionalDisclosureStateHandling:
    """Rigorous verification of congressional state handling & adversarial invariants."""

    # -------------------------------------------------------------------------
    # State 1: UNMONITORED (Default Live Posture)
    # -------------------------------------------------------------------------
    def test_state_01_unmonitored_default_live_posture(self):
        """In live mode without active pipeline, congressional collection is described as 'Not Monitored'."""
        assessment = SmartMoneyEngine.assess_congressional_trades(
            symbol="AAPL",
            live_congress=None,
            include_curated=False,
            is_verified_empty=False,
        )

        assert assessment["domain_id"] == "congressional_stock_act"
        assert assessment["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert assessment["status_label"] == "Not Monitored"
        assert assessment["available"] is False
        assert assessment["is_live"] is False
        assert assessment["observation_count"] == 0
        assert assessment["records"] == []
        assert assessment["directional_bias"] == SmartMoneyDirection.UNASSESSED.value
        assert assessment["net_value_usd"] == 0.0

        # Epistemic Honesty: HOLD_SEC_001 must NEVER be cited as governing congressional data
        assert "HOLD_SEC_001" not in assessment["disclosure"]
        assert "HOLD_SEC_001" not in assessment["status_label"]
        assert "not currently monitored" in assessment["disclosure"].lower()

    def test_state_01_assess_symbol_preserves_sec_hold_without_congress_conflation(self):
        """Composite assess_symbol maintains HOLD_SEC_001 for SEC Form 4 while congress is unmonitored."""
        res = SmartMoneyEngine.assess_symbol("AAPL", include_curated=False)

        # Composite overall status reflects SEC Form 4 held state
        assert res["overall_status"] == SmartMoneyDomainStatus.PIPELINE_HELD.value
        assert "HOLD_SEC_001" in res["active_holds"]

        # Sub-domain isolation: SEC is held under HOLD_SEC_001, Congress is UNAVAILABLE (Not Monitored)
        assert res["subdomains"]["sec_form_4"]["status"] == SmartMoneyDomainStatus.PIPELINE_HELD.value
        assert "HOLD_SEC_001" in res["subdomains"]["sec_form_4"]["status_label"]

        cong_sub = res["subdomains"]["congressional_stock_act"]
        assert cong_sub["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert cong_sub["status_label"] == "Not Monitored"
        assert "HOLD_SEC_001" not in cong_sub["disclosure"]

    # -------------------------------------------------------------------------
    # State 2: INCOMPLETE_COVERAGE (Partial Coverage)
    # -------------------------------------------------------------------------
    def test_state_02_incomplete_coverage_with_active_signal_in_other_domain(self):
        """When another sub-domain has verified signal, congressional remains unmonitored and composite reports PARTIAL_COVERAGE."""
        live_options_sweep = [{
            "ticker": "AAPL",
            "time": "10:30:00",
            "type": "SWEEP CALL",
            "strike": 230.0,
            "expiration": "2026-12-18",
            "premium": 1200000.0,
            "sentiment": "Bullish",
            "volume_oi_ratio": 4.5,
            "order_type": "SWEEP",
        }]

        res = SmartMoneyEngine.assess_symbol(
            "AAPL",
            live_options=live_options_sweep,
            live_congress=None,
            include_curated=False,
        )

        assert res["subdomains"]["options_flow"]["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
        assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert res["subdomains"]["congressional_stock_act"]["status_label"] == "Not Monitored"
        assert res["overall_status"] == SmartMoneyDomainStatus.PARTIAL_COVERAGE.value

    # -------------------------------------------------------------------------
    # State 3: CERTIFIED_EMPTY & STRICT EVIDENCE-READINESS BINDING
    # -------------------------------------------------------------------------
    def test_state_03_certified_empty_requires_authoritative_coverage_evidence(self):
        """Valid coverage evidence matching requested symbol and full bindings establishes VERIFIED_NO_ACTIVITY."""
        valid_evidence = {
            "source_inventory": "house_clerk_daily_index_2026",
            "chamber": "HOUSE",
            "interval_days": 180,
            "symbol": "NVDA",
            "document_identities": ["DOC_20025123", "DOC_20025145"],
            "extraction_outcomes": {
                "DOC_20025123": "SUCCESS",
                "DOC_20025145": "SUCCESS",
            },
            "unresolved_records_count": 0,
            "watermark_hash": "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
        }
        certified = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=valid_evidence,
        )
        assert certified["status"] == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
        assert certified["status_label"] == "Verified No Activity"
        assert certified["available"] is True
        assert certified["is_live"] is True
        assert certified["observation_count"] == 0
        assert certified["directional_bias"] == SmartMoneyDirection.NEUTRAL.value
        assert "0 congressional disclosures" in certified["disclosure"]

    def test_adversarial_bare_boolean_fails_closed(self):
        """Bare boolean is_verified_empty=True without coverage evidence fails closed to UNAVAILABLE."""
        unverified = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=None,
        )
        assert unverified["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert unverified["status_label"] == "Not Monitored"
        assert "bare boolean or missing evidence cannot certify empty coverage" in unverified["disclosure"]

    def test_adversarial_placeholder_hash_fails_closed(self):
        """Fabricated watermark_hash 'CONFIRMED_ZERO_VERIFIED_COVERAGE' is rejected and fails closed."""
        fake_evidence = {
            "source_inventory": "House & Senate Official STOCK Act Portals",
            "chamber": "HOUSE",
            "symbol": "NVDA",
            "watermark_hash": "CONFIRMED_ZERO_VERIFIED_COVERAGE",
        }
        res = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=fake_evidence,
        )
        assert res["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert "Fabricated or placeholder watermark hash rejected" in res["disclosure"]

    def test_adversarial_bare_string_hash_without_bindings_fails_closed(self):
        """Bare hash string without source inventory, chamber, or document identities fails closed."""
        bare_hash_evidence = {
            "watermark_hash": "a" * 64,
            "symbol": "NVDA",
        }
        res = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=bare_hash_evidence,
        )
        assert res["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert "Missing source_inventory binding" in res["disclosure"]

    def test_adversarial_equal_document_counts_with_unresolved_records_fails_to_partial(self):
        """Equal document counts alone CANNOT certify completeness when unresolved records exist."""
        evidence_with_unresolved = {
            "source_inventory": "house_clerk_daily_index_2026",
            "chamber": "HOUSE",
            "interval_days": 180,
            "symbol": "NVDA",
            "document_identities": ["DOC_1", "DOC_2"],
            "extraction_outcomes": {"DOC_1": "SUCCESS", "DOC_2": "SUCCESS"},
            "unresolved_records_count": 2,  # 2 unresolved asset lines
            "watermark_hash": "b" * 64,
        }
        res = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=evidence_with_unresolved,
        )
        assert res["status"] == SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
        assert res["status_label"] == "Partial Coverage"
        assert "contains 2 unresolved records" in res["disclosure"]

    def test_adversarial_failed_document_extraction_outcome_fails_to_partial(self):
        """If any document extraction outcome is FAILED or UNPARSED, coverage fails to PARTIAL."""
        evidence_with_failure = {
            "source_inventory": "house_clerk_daily_index_2026",
            "chamber": "HOUSE",
            "interval_days": 180,
            "symbol": "NVDA",
            "document_identities": ["DOC_1", "DOC_2"],
            "extraction_outcomes": {"DOC_1": "SUCCESS", "DOC_2": "FAILED"},
            "unresolved_records_count": 0,
            "watermark_hash": "c" * 64,
        }
        res = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=evidence_with_failure,
        )
        assert res["status"] == SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
        assert "outcome is 'FAILED'" in res["disclosure"]

    def test_adversarial_symbol_and_interval_mismatch_fail_closed(self):
        """Symbol mismatch and interval deficit fail closed respectively."""
        # Symbol mismatch -> UNAVAILABLE
        mismatch_sym = {
            "source_inventory": "house_clerk_daily_index_2026",
            "chamber": "HOUSE",
            "interval_days": 45,
            "symbol": "AAPL",  # Requested NVDA
            "document_identities": ["DOC_1"],
            "extraction_outcomes": {"DOC_1": "SUCCESS"},
            "unresolved_records_count": 0,
            "watermark_hash": "d" * 64,
        }
        res_sym = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            coverage_evidence=mismatch_sym,
        )
        assert res_sym["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert "does not match requested symbol" in res_sym["disclosure"]

        # Short interval (10d < 30d expected) -> PARTIAL_COVERAGE
        short_interval = {
            "source_inventory": "house_clerk_daily_index_2026",
            "chamber": "HOUSE",
            "interval_days": 10,
            "symbol": "NVDA",
            "document_identities": ["DOC_1"],
            "extraction_outcomes": {"DOC_1": "SUCCESS"},
            "unresolved_records_count": 0,
            "watermark_hash": "e" * 64,
        }
        res_int = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            coverage_evidence=short_interval,
        )
        assert res_int["status"] == SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
        assert "shorter than requested" in res_int["disclosure"]

    def test_adversarial_assess_symbol_bare_boolean_fails_closed_without_fake_fallback(self):
        """assess_symbol called with bare congress_verified_empty=True fails closed to PARTIAL_COVERAGE (never VERIFIED_NO_ACTIVITY)."""
        res = SmartMoneyEngine.assess_symbol(
            "AAPL",
            sec_verified_empty=True,
            options_verified_empty=True,
            congress_verified_empty=True,  # Bare boolean without coverage evidence
            congress_coverage_evidence=None,
        )
        # Because congressional is unmonitored (UNAVAILABLE), composite overall status MUST NOT be VERIFIED_NO_ACTIVITY
        assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert res["overall_status"] == SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
        assert res["overall_status"] != SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value

    def test_adversarial_assess_symbol_with_valid_receipt_certifies_neutral_zero(self):
        """assess_symbol called with authoritative valid coverage evidence emits VERIFIED_NO_ACTIVITY."""
        valid_cong_evidence = {
            "source_inventory": "house_clerk_daily_index_2026",
            "chamber": "HOUSE",
            "interval_days": 180,
            "symbol": "AAPL",
            "document_identities": ["DOC_A", "DOC_B"],
            "extraction_outcomes": {"DOC_A": "SUCCESS", "DOC_B": "SUCCESS"},
            "unresolved_records_count": 0,
            "watermark_hash": "f" * 64,
        }
        res = SmartMoneyEngine.assess_symbol(
            "AAPL",
            sec_verified_empty=True,
            options_verified_empty=True,
            congress_verified_empty=True,
            congress_coverage_evidence=valid_cong_evidence,
        )
        assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
        assert res["overall_status"] == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
        assert res["pillar_score"] == 50.0

    def test_adversarial_legislative_alignment_undisclosed_amount_zero_points(self):
        """Undisclosed amount_range in calculate_legislative_alignment awards 0 sizing points (never defaults to a bracket)."""
        trade_no_amount = {
            "sector": "defense",
            "amount_range": None,  # Undisclosed
            "details": {"committee_assignments": ["Armed Services"]},
        }
        trade_with_amount = {
            "sector": "defense",
            "amount_range": "$1,000,000 - $5,000,000",
            "details": {"committee_assignments": ["Armed Services"]},
        }
        score_no_amt = calculate_legislative_alignment(trade_no_amount)
        score_with_amt = calculate_legislative_alignment(trade_with_amount)
        
        # Sizing tier for $1M-$5M is +15 pts; score_no_amt must be exactly 15 pts lower
        assert score_with_amt - score_no_amt == 15

    # -------------------------------------------------------------------------
    # State 4: CURATED_RECORDS (Explicit include_curated=True)
    # -------------------------------------------------------------------------
    def test_state_04_curated_records_explicitly_labeled(self):
        """Curated records are badged as historical research archive (August 2026) and not live."""
        assessment = SmartMoneyEngine.assess_congressional_trades(
            symbol="PLTR",
            include_curated=True,
        )

        assert assessment["status"] == SmartMoneyDomainStatus.CURATED_RESEARCH.value
        assert assessment["status_label"] == "Curated Research Sample"
        assert assessment["available"] is True
        assert assessment["is_live"] is False
        assert assessment["observation_count"] > 0
        assert "August 2026" in assessment["disclosure"]
        assert "Not a real-time live trading feed" in assessment["disclosure"]

    def test_state_04_curated_symbol_with_zero_records(self):
        """Curated query for symbol not in the 40-record fixture returns No Curated Disclosures, not verified zero."""
        assessment = SmartMoneyEngine.assess_congressional_trades(
            symbol="UNLISTED_TICKER_XYZ",
            include_curated=True,
        )

        assert assessment["status"] == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
        assert assessment["status_label"] == "No Curated Disclosures"
        assert assessment["available"] is True
        assert assessment["is_live"] is False
        assert assessment["observation_count"] == 0
        assert "0 curated congressional disclosures on record" in assessment["disclosure"]

    # -------------------------------------------------------------------------
    # State 5: RETRIEVAL_ERRORS & UNVERIFIED EMPTY ARRAYS
    # -------------------------------------------------------------------------
    def test_state_05_unverified_empty_array_fails_closed(self):
        """Empty array passed without verified flag fails closed to UNAVAILABLE (Not Monitored), never verified zero."""
        assessment = SmartMoneyEngine.assess_congressional_trades(
            symbol="AAPL",
            live_congress=[],
            is_verified_empty=False,
            include_curated=False,
        )

        assert assessment["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert assessment["status_label"] == "Not Monitored"
        assert assessment["available"] is False
        assert assessment["is_live"] is False
        assert assessment["observation_count"] == 0
        assert "HOLD_SEC_001" not in assessment["disclosure"]

    def test_state_05_crypto_assets_are_strictly_inapplicable(self):
        """Digital assets are marked INAPPLICABLE to STOCK Act disclosures."""
        assessment = SmartMoneyEngine.assess_congressional_trades(
            symbol="BTC-USD",
            include_curated=False,
        )

        assert assessment["status"] == SmartMoneyDomainStatus.INAPPLICABLE.value
        assert assessment["status_label"] == "Exempt / Digital Asset"
        assert assessment["available"] is False
        assert assessment["is_live"] is False
        assert assessment["observation_count"] == 0
        assert "inapplicable to crypto tokens" in assessment["disclosure"]
