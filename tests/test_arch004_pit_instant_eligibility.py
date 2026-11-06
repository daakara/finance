"""ARCH-004 point-in-time evidence must compare normalized instants, not text."""
import pytest
from analyst_dashboard.governance.decision_context import is_pit_eligible

pytestmark = pytest.mark.tier2b


@pytest.mark.parametrize(
    "available,evaluated,expected",
    [
        ("2026-11-06T08:00:00Z", "2026-11-06T08:00:00Z", True),
        ("2026-11-06T08:00:00Z", "2026-11-06T08:00:01Z", True),
        ("2026-11-06T08:00:01Z", "2026-11-06T08:00:00Z", False),
        # Lexical order says available hour 08 precedes 09, but after UTC
        # normalization the publication at 08:00Z is AFTER 09:00+02 (07:00Z).
        ("2026-11-06T08:00:00Z", "2026-11-06T09:00:00+02:00", False),
        # Lexical order says available hour 09 follows 08, but 09:00+02
        # means publication 07:00Z, available by evaluation 08:00Z.
        ("2026-11-06T09:00:00+02:00", "2026-11-06T08:00:00Z", True),
        ("2026-11-06T09:00:00-05:00", "2026-11-06T15:00:00+01:00", True),
        ("2026-11-06T09:00:00-05:00", "2026-11-06T14:59:59+01:00", False),
        ("2026-11-06T08:00:00.100Z", "2026-11-06T08:00:00.09Z", False),
        ("2026-11-06T08:00:00.090+00:00", "2026-11-06T08:00:00.1Z", True),
        ("2026-11-06", "2026-11-07T00:00:00Z", False),
        ("2026-11-06T09:00:00", "2026-11-06T10:00:00Z", False),
        ("2026-11-06T09:00:00Z", "2026-11-06T10:00:00", False),
        ("", "2026-11-06T10:00:00Z", False),
        ("2026-11-06T10:00:00Z", "", False),
        (None, "2026-11-06T10:00:00Z", False),
        ("not-a-date", "2026-11-06T10:00:00Z", False),
        ("2026-11-06T09:00:00+99:00", "2026-11-06T10:00:00Z", False),
        ("2026-11-06T09:00:00Z", "2026-11-06T10:00:00+99:00", False),
    ],
)
def test_point_in_time_eligibility_uses_real_instants(available, evaluated, expected):
    assert is_pit_eligible(available, evaluated) is expected
