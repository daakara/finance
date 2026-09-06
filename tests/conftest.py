"""ARX Terminal — 4-Tier Test Gate Configuration.

Marker Definitions:
  tier1: Core Invariant Gate (always runs, <60s)
  tier2a: Presentation & UX Gate (path-filtered)
  tier2b: State & Solver Gate (path-filtered)
  tier2c: Provenance & Data Gate (path-filtered)
  tier3: Pre-Flight Release Gate (merge to main only)
"""
