"""
Comprehensive Verification Suite: Congressional STOCK Act Disclosure State Handling (Phase 0)
Directive: ARX-CONGRESSIONAL-INGESTION-001 (Phase 0)

Verifies state handling across 5 required operational states:
1. UNMONITORED: Default live posture where no operational pipeline exists.
   - Must emit status UNAVAILABLE / Not Monitored.
   - Must NOT cite HOLD_SEC_001 in congressional sub-domain.
   - Must NOT fabricate trades or false zeros.
2. INCOMPLETE_COVERAGE: Partial coverage where one feed has verified signal while congressional is unmonitored.
   - Subdomain preserves unmonitored disclosure.
   - Overall composite handles partial coverage without score contamination.
3. CERTIFIED_EMPTY: Authoritative complete search proves 0 disclosures in coverage window.
   - Emits VERIFIED_NO_ACTIVITY with NEUTRAL direction.
   - Legitimate certified empty presentation.
4. CURATED_RECORDS: User or client explicitly passes include_curated=True.
   - Emits CURATED_RESEARCH with dataset date 2026-08-28.
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
)


class TestCongressionalDisclosureStateHandling:
    """Rigorous verification of the 5 required congressional state handling contracts."""

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
    # State 3: CERTIFIED_EMPTY (Explicit Verification with Coverage Evidence)
    # -------------------------------------------------------------------------
    def test_state_03_certified_empty_requires_authoritative_coverage_evidence(self):
        """Explicit is_verified_empty=True without coverage evidence fails closed to UNAVAILABLE."""
        unverified = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=None,
        )
        assert unverified["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert unverified["status_label"] == "Not Monitored"
        assert "lacks certified interval coverage" in unverified["disclosure"]

        # Valid coverage evidence matching requested symbol and interval establishes VERIFIED_NO_ACTIVITY
        valid_evidence = {
            "source": "House Clerk & Senate EFD Portals",
            "interval_days": 45,
            "symbol": "NVDA",
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

        # Mismatching symbol filter fails closed
        mismatch_evidence = {
            "source": "House Clerk & Senate EFD Portals",
            "interval_days": 45,
            "symbol": "AAPL",  # mismatched symbol
            "watermark_hash": "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
        }
        mismatched = SmartMoneyEngine.assess_congressional_trades(
            symbol="NVDA",
            is_verified_empty=True,
            include_curated=False,
            coverage_evidence=mismatch_evidence,
        )
        assert mismatched["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value

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
