"""
analyst_dashboard/universe/acquisition/config.py

Explicit configuration for supported listing directory sources:
- SEC EDGAR company tickers & exchanges
- Nasdaq Trader nasdaqlisted.txt
- Nasdaq Trader otherlisted.txt

Strict Access & Operational Boundaries:
- Enforces HTTPS with TLS verification by default.
- Rejects insecure HTTP downgrades.
- Enforces bounded response payload limits (max_bytes).
- Provides honest, non-fabricated User-Agent headers.
- Explicitly bounds retry counts and timeouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_USER_AGENT = "ArxTerminal/1.0 (research@arxterminal.org)"


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for an individual public directory source."""
    source_id: str
    url: str
    headers: Mapping[str, str] = field(default_factory=lambda: {"User-Agent": DEFAULT_USER_AGENT})
    timeout_seconds: float = 15.0
    max_bytes: int = 25 * 1024 * 1024  # 25 MB default ceiling
    max_retries: int = 2
    retry_backoff_factor: float = 0.5
    enforce_https: bool = True
    rate_limit_delay_seconds: float = 0.15  # Default spacing floor (respects SEC <= 10 req/s)


# Canonical production-source definitions (read-only, official endpoints)
CANONICAL_SOURCES: dict[str, SourceConfig] = {
    "sec_company_tickers_exchange": SourceConfig(
        source_id="sec_company_tickers_exchange",
        url="https://www.sec.gov/files/company_tickers_exchange.json",
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        },
        timeout_seconds=20.0,
        max_bytes=30 * 1024 * 1024,
        max_retries=2,
        enforce_https=True,
        rate_limit_delay_seconds=0.2,
    ),
    "nasdaq_listed_directory": SourceConfig(
        source_id="nasdaq_listed_directory",
        url="https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/plain",
            "Accept-Encoding": "gzip, deflate",
        },
        timeout_seconds=20.0,
        max_bytes=15 * 1024 * 1024,
        max_retries=2,
        enforce_https=True,
        rate_limit_delay_seconds=0.1,
    ),
    "other_listed_directory": SourceConfig(
        source_id="other_listed_directory",
        url="https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/plain",
            "Accept-Encoding": "gzip, deflate",
        },
        timeout_seconds=20.0,
        max_bytes=20 * 1024 * 1024,
        max_retries=2,
        enforce_https=True,
        rate_limit_delay_seconds=0.1,
    ),
}
