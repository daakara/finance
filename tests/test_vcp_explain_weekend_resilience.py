import pytest
from unittest.mock import patch, MagicMock
from api.routes.screener import explain_vcp_classification
from fastapi import HTTPException

def test_explain_vcp_classification_weekend_resilience():
    """
    ARX-VCP-EXPLAIN-001: Non-trading session snapshot dates (e.g. Saturday 2026-11-07)
    must resolve to the preceding valid trading session (2026-11-06) without raising HTTP 422.
    """
    mock_snapshot = {
        "status": "AVAILABLE",
        "data_as_of": "2026-11-07",  # Saturday
        "snapshot_id": "test-snap-weekend",
        "results": []
    }

    with patch("api.routes.screener.vcp_scanner_runner.get_active_or_latest_snapshot", return_value=mock_snapshot):
        # Must not raise HTTPException 422
        res = explain_vcp_classification(symbol="DHLGY")
        assert res["symbol"] == "DHLGY"
        assert "evaluationContext" in res
        # Evaluation cutoff must resolve to a valid session
        assert res["evaluationContext"]["evaluationAsOf"].startswith("2026-11-06")
        assert "classification" in res
        assert "primaryRejection" in res["classification"]
        assert "predicates" in res

def test_explain_vcp_classification_explicit_as_of():
    """
    Explicit valid as_of date must be preserved.
    """
    res = explain_vcp_classification(symbol="NVDA", as_of="2026-11-03")
    assert res["symbol"] == "NVDA"
    assert res["evaluationContext"]["asOfDate"] == "2026-11-03"
    assert res["evaluationContext"]["evaluationAsOf"].startswith("2026-11-03")
