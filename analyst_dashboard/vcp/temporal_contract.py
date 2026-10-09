"""ARX VCP Domain Temporal Contract & Bitemporal Admissibility.

Sprint 2B Domain-Authority Resolution.
Freezes market timezone, session close semantics, partial-bar policy,
and bitemporal admissibility invariants:
valid_time <= evaluation_as_of AND known_at <= evaluation_as_of.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DailyOHLCVBar:
    symbol: str
    bar_date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: int
    valid_time: str  # ISO-8601 UTC or ET
    known_at: str  # ISO-8601 UTC or ET
    is_session_closed: bool = True
    split_factor: float = 1.0  # As of known_at


class VCPTemporalContract:
    """Authoritative temporal contract for VCP domain evaluation."""

    CONTRACT_ID = "ARX_VCP_TEMPORAL_CONTRACT"
    VERSION = "1.0.0"

    TIMEZONE = "America/New_York"
    SESSION_OPEN_TIME = "09:30:00"
    SESSION_CLOSE_TIME = "16:00:00"
    BAR_FREQUENCY = "1D"

    @staticmethod
    def parse_iso(ts_str: str) -> datetime.datetime:
        """Parses ISO-8601 string or YYYY-MM-DD date."""
        clean = ts_str.strip()
        if "T" in clean:
            # Handle possible Z or offset
            clean_iso = clean.replace("Z", "+00:00")
            return datetime.datetime.fromisoformat(clean_iso)
        else:
            # Date only - default to session close at 16:00:00 ET (21:00 UTC / 20:00 UTC depending on DST)
            dt = datetime.date.fromisoformat(clean)
            return datetime.datetime(dt.year, dt.month, dt.day, 16, 0, 0, tzinfo=datetime.timezone.utc)

    @classmethod
    def is_bar_admissible(
        cls, bar: DailyOHLCVBar, evaluation_as_of: str
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether a bar satisfies the strict bitemporal cutoff.

        Conditions:
        1. valid_time <= evaluation_as_of
        2. known_at <= evaluation_as_of
        3. is_session_closed is True (partial-bar safety)
        """
        cutoff_dt = cls.parse_iso(evaluation_as_of)
        valid_dt = cls.parse_iso(bar.valid_time)
        known_dt = cls.parse_iso(bar.known_at)

        if valid_dt > cutoff_dt:
            return False, f"POST_CUTOFF_VALID_TIME: valid {bar.valid_time} > as_of {evaluation_as_of}"

        if known_dt > cutoff_dt:
            return False, f"POST_CUTOFF_KNOWN_AT: known {bar.known_at} > as_of {evaluation_as_of}"

        if not bar.is_session_closed:
            # If bar is partial and cutoff is before session close
            return False, f"PARTIAL_BAR_UNCLOSED: session on {bar.bar_date} not closed at {evaluation_as_of}"

        return True, None

    def compute_contract_hash(self) -> str:
        serialized = {
            "contract_id": self.CONTRACT_ID,
            "version": self.VERSION,
            "timezone": self.TIMEZONE,
            "session_open": self.SESSION_OPEN_TIME,
            "session_close": self.SESSION_CLOSE_TIME,
            "bar_frequency": self.BAR_FREQUENCY,
            "admissibility_rule": "valid_time <= evaluation_as_of AND known_at <= evaluation_as_of AND is_session_closed",
        }
        data_bytes = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def export_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.CONTRACT_ID,
            "version": self.VERSION,
            "hash": self.compute_contract_hash(),
            "timezone": self.TIMEZONE,
            "session_open": self.SESSION_OPEN_TIME,
            "session_close": self.SESSION_CLOSE_TIME,
            "bar_frequency": self.BAR_FREQUENCY,
        }
