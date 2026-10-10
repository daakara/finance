"""ARX Terminal — VCP Sprint 3 Natural Trigger Remediation Service.

Provides an authoritative, backend-governed natural production trigger mechanism for
market-wide Radar VCP scans, satisfying Section 14 of Sprint 3 Production Shadow Governance.

GOVERNING INVARIANTS:
1. BACKEND-GOVERNED ORIGIN: Execution triggered via this service is strictly classified as
   origin_class = NATURAL_PRODUCTION.
2. OPERATOR SEPARATION: Operator manual invocations via POST /vcp/scan remain ADMIN_FORCED.
3. CALLER INTEGRITY: Callers cannot self-declare origin_class.
4. DETERMINISTIC IDEMPOTENCY: Scheduled executions bind deterministic job keys:
   vcp:scheduled:{scheduled_date}:{universe_build_id}.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("arx.vcp.natural_trigger")

NATURAL_PRODUCTION_TRIGGER_TYPE: str = "SCHEDULED_MARKET_WIDE_VCP_SCAN"
TRIGGER_OWNER: str = "analyst_dashboard.vcp.natural_trigger:NaturalVCPTriggerService"
TRIGGER_FREQUENCY_OR_EVENT: str = "DAILY_EOD_AND_BOOT_WARMUP"
FAILURE_RETRY_POLICY: str = "FAIL_CLOSED_NO_SYNTHETIC_FALLBACK"
IDEMPOTENCY_KEY_SOURCE: str = "vcp:scheduled:{scheduled_date}:{universe_build_id}"
ORIGIN_CLASSIFICATION: str = "NATURAL_PRODUCTION"


class NaturalVCPTriggerService:
    """Authoritative natural production trigger service for Minervini VCP market-wide scans."""

    def __init__(self, scanner_runner: Optional[Any] = None) -> None:
        self._scanner_runner = scanner_runner

    def _get_runner(self) -> Any:
        if self._scanner_runner is not None:
            return self._scanner_runner
        # Lazy import to avoid circular dependency
        from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
        from analyst_dashboard.data.market_db import MarketDatabaseEngine
        from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
        db = MarketDatabaseEngine()
        confluence = ConfluenceEngine()
        self._scanner_runner = VCPScannerRunner(market_db=db, confluence_engine=confluence)
        return self._scanner_runner

    def trigger_natural_scan(
        self,
        reason: str = "SCHEDULED_CADENCE",
        scheduled_date: Optional[str] = None,
        universe_build_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes a genuine natural production market-wide VCP scan.

        Origin class is unconditionally set to NATURAL_PRODUCTION by backend authority.
        """
        from analyst_dashboard.coordination import TriggerType

        runner = self._get_runner()
        today_str = scheduled_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        effective_build_id = universe_build_id or "ARX_CANONICAL_V1"
        logical_key = f"vcp:scheduled:{today_str}:{effective_build_id}"

        logger.info(
            f"Executing natural production VCP scan: reason={reason}, key={logical_key}, "
            f"origin_class={ORIGIN_CLASSIFICATION}"
        )

        # Call execute_market_wide_scan with TriggerType.SCHEDULED (maps to NATURAL_PRODUCTION)
        return runner.execute_market_wide_scan(
            universe_build_id=effective_build_id,
            logical_job_key=logical_key,
            trigger_type=TriggerType.SCHEDULED,
            scheduled_for=today_str,
            operator_request_id=None,  # Not an operator request
        )


_DEFAULT_NATURAL_TRIGGER_SERVICE: Optional[NaturalVCPTriggerService] = None


def get_natural_vcp_trigger_service() -> NaturalVCPTriggerService:
    global _DEFAULT_NATURAL_TRIGGER_SERVICE
    if _DEFAULT_NATURAL_TRIGGER_SERVICE is None:
        _DEFAULT_NATURAL_TRIGGER_SERVICE = NaturalVCPTriggerService()
    return _DEFAULT_NATURAL_TRIGGER_SERVICE
