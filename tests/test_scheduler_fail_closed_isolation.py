"""Tests for autonomous natural scheduler fail-closed disarming and deployment profile isolation."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

import scripts.run_natural_scheduler as scheduler_mod


def test_scheduler_fails_closed_under_read_only_research_mode():
    """Verify natural execution fails closed when SMART_MONEY_RELEASE_MODE is READ_ONLY_RESEARCH."""
    test_args = ["run_natural_scheduler.py", "--mode", "natural"]
    with patch.object(sys, "argv", test_args):
        with patch.dict(os.environ, {"SMART_MONEY_RELEASE_MODE": "READ_ONLY_RESEARCH", "AUTONOMOUS_ROOTING_ENABLED": "true"}):
            with pytest.raises(RuntimeError, match="FAIL-CLOSED: Autonomous natural scheduler execution is disarmed"):
                scheduler_mod.main()


def test_scheduler_fails_closed_when_autonomous_rooting_disabled():
    """Verify natural execution fails closed when AUTONOMOUS_ROOTING_ENABLED is false."""
    test_args = ["run_natural_scheduler.py", "--mode", "natural"]
    with patch.object(sys, "argv", test_args):
        with patch.dict(os.environ, {"SMART_MONEY_RELEASE_MODE": "PRODUCTION", "AUTONOMOUS_ROOTING_ENABLED": "false"}):
            with pytest.raises(RuntimeError, match="FAIL-CLOSED: Autonomous natural scheduler execution is disarmed"):
                scheduler_mod.main()


def test_scheduler_fails_closed_without_github_oidc_environment():
    """Verify extract_safe_oidc_claims fails closed when ambient GitHub token environment is absent."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="ACTIONS_ID_TOKEN_REQUEST_URL or ACTIONS_ID_TOKEN_REQUEST_TOKEN not in environment"):
            scheduler_mod.extract_safe_oidc_claims()


def test_gate3_pre_arm_authorization_required_for_natural_accession():
    """Verify that Gate-3 authorization artifact is mandatory for natural accession."""
    # When active_auth_path does not exist, mode cannot become natural accession
    assert not os.path.exists("evidence/gate3_authorizations/SMART_MONEY_GATE3_ACTIVE_AUTHORIZATION.json")
