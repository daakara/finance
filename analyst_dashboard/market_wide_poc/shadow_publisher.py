"""
ARX Terminal — Directive ARX-MW-001
Shadow Publisher for Market-Wide Discovery POC (Section 9).

Generates and persists the immutable snapshot artifact in isolated locations:
- docs/research/market_wide_poc/market_wide_radar_snapshot.json
- frontend/public/data/poc/market_wide_radar_snapshot.json
Enforces strict shadow-only isolation from production prospective-learning denominators.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from .contracts import MarketWidePublicationSnapshot
from .discovery_orchestrator import MarketWideDiscoveryOrchestrator


class MarketWideShadowPublisher:
    """Publishes verified POC discovery snapshots to isolated research and frontend preview targets."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    def publish_snapshot(
        self,
        snapshot: MarketWidePublicationSnapshot,
    ) -> Dict[str, str]:
        """Writes snapshot JSON to isolated research and frontend public directories."""
        research_dir = os.path.join(self.base_dir, "docs", "research", "market_wide_poc")
        frontend_dir = os.path.join(self.base_dir, "frontend", "public", "data", "poc")

        os.makedirs(research_dir, exist_ok=True)
        os.makedirs(frontend_dir, exist_ok=True)

        research_path = os.path.join(research_dir, "market_wide_radar_snapshot.json")
        frontend_path = os.path.join(frontend_dir, "market_wide_radar_snapshot.json")

        payload = snapshot.to_dict()
        serialized = json.dumps(payload, indent=2)

        with open(research_path, "w", encoding="utf-8") as f:
            f.write(serialized)

        with open(frontend_path, "w", encoding="utf-8") as f:
            f.write(serialized)

        return {
            "research_path": research_path,
            "frontend_path": frontend_path,
            "snapshot_id": snapshot.snapshot_id,
            "candidates_digest": snapshot.candidates_digest,
            "candidates_count": str(len(snapshot.candidates)),
        }


def run_and_publish():
    orchestrator = MarketWideDiscoveryOrchestrator()
    res = orchestrator.run_discovery_pipeline()
    publisher = MarketWideShadowPublisher()
    pub_res = publisher.publish_snapshot(res["snapshot"])
    print("Published Market-Wide POC Snapshot:")
    for k, v in pub_res.items():
        print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    run_and_publish()
