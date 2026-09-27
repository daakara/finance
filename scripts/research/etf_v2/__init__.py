"""
scripts/research/etf_v2/__init__.py

ARX Terminal — ETF Research Pipeline V2.
Clean multi-source regulatory evidence architecture.
"""

from .models import (
    EntityIdentity,
    FilingMetadata,
    ProspectusAuthority,
    DocumentStructure,
    SeriesBoundary,
    MandateSection,
    NPORTMetrics,
    NCENIndexStatus,
    PolicyEvidence,
    ClassificationDecision,
    PopulationRecord,
)

__version__ = "2.0.0"
