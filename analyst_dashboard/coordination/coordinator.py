"""
analyst_dashboard/coordination/coordinator.py

High-level Coordinator Facade managing Lease Lifecycles,
Heartbeat Background Worker, and Reconciliations.
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable

from .contracts import (
    TriggerType,
    AcquisitionStatus,
    CurrentLease,
    LeaseAcquisitionResult,
    LeasePolicy,
    PRODUCTION_LEASE_POLICY,
    RESOURCE_KEY_VCP_PIPELINE,
)
from .store import CoordinationStore, COORDINATION_DB_PATH

logger = logging.getLogger(__name__)


class DurableRunCoordinator:
    """
    Coordinates distributed run execution, heartbeat renewals,
    and fail-closed lease loss notification.
    """

    def __init__(
        self,
        resource_key: str = RESOURCE_KEY_VCP_PIPELINE,
        db_path: str = COORDINATION_DB_PATH,
        policy: LeasePolicy = PRODUCTION_LEASE_POLICY,
    ):
        self.resource_key = resource_key
        self.store = CoordinationStore(db_path=db_path)
        self.policy = policy
        self._active_lease: Optional[CurrentLease] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._on_lease_lost_callback: Optional[Callable[[], None]] = None

    @property
    def active_lease(self) -> Optional[CurrentLease]:
        return self._active_lease

    def acquire(
        self,
        logical_job_key: str,
        trigger_type: TriggerType,
        owner_instance_id: str,
        implementation_release_sha: str,
        scheduled_for: Optional[str] = None,
        operator_request_id: Optional[str] = None,
        on_lease_lost: Optional[Callable[[], None]] = None,
    ) -> LeaseAcquisitionResult:
        """Atomically acquire the lease or return conflict/already succeeded."""
        res = self.store.acquire_or_join(
            resource_key=self.resource_key,
            logical_job_key=logical_job_key,
            trigger_type=trigger_type,
            owner_instance_id=owner_instance_id,
            implementation_release_sha=implementation_release_sha,
            policy=self.policy,
            scheduled_for=scheduled_for,
            operator_request_id=operator_request_id,
        )

        if res.status == AcquisitionStatus.ACQUIRED and res.lease is not None:
            self._active_lease = res.lease
            self._on_lease_lost_callback = on_lease_lost
            self._start_heartbeat_worker(res.lease)

        return res

    def heartbeat(self, lease: Optional[CurrentLease] = None) -> bool:
        """Manual synchronous heartbeat renewal."""
        target_lease = lease or self._active_lease
        if not target_lease:
            return False
        renewed = self.store.renew_lease(target_lease, self.policy)
        if not renewed:
            self._handle_lease_lost()
        return renewed

    def release(self, lease: Optional[CurrentLease] = None) -> bool:
        """Release active lease and stop heartbeat thread."""
        self._stop_heartbeat_worker()
        target_lease = lease or self._active_lease
        if not target_lease:
            return False
        res = self.store.release_lease(target_lease)
        self._active_lease = None
        return res

    def reconcile_abandoned_runs(self) -> int:
        """Reconcile abandoned runs for this resource key."""
        return self.store.reconcile_abandoned_runs(self.resource_key)

    def reset_coordination_epoch(self, new_epoch: str, reason: str, operator_id: str):
        """Explicit administrative epoch reset."""
        self._stop_heartbeat_worker()
        self._active_lease = None
        self.store.reset_coordination_epoch(
            resource_key=self.resource_key,
            new_epoch=new_epoch,
            reason=reason,
            updated_by=operator_id,
        )

    def _start_heartbeat_worker(self, lease: CurrentLease):
        """Launch background heartbeat thread."""
        self._stop_heartbeat.clear()

        def _worker():
            interval = self.policy.heartbeat_interval_seconds
            while not self._stop_heartbeat.wait(timeout=interval):
                renewed = self.store.renew_lease(lease, self.policy)
                if not renewed:
                    logger.warning(f"Heartbeat renewal failed for lease {lease.lease_id}. Marking lease lost.")
                    self._handle_lease_lost()
                    break

        self._heartbeat_thread = threading.Thread(
            target=_worker,
            name=f"heartbeat-{lease.lease_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_worker(self):
        """Signal and wait for heartbeat thread to stop."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def _handle_lease_lost(self):
        """Transition worker into lease-lost posture."""
        self._active_lease = None
        self._stop_heartbeat.set()
        if self._on_lease_lost_callback:
            try:
                self._on_lease_lost_callback()
            except Exception as e:
                logger.error(f"Error in on_lease_lost callback: {e}")
