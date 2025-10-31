"""
Dashboard Package - Modular Financial Dashboard Components
Provides modular dashboard architecture with specialized components
"""

from .core import DashboardManager
from .workflows import SingleAssetWorkflow, ComparativeWorkflow
from .renderers import (
    SummaryRenderer, TechnicalRenderer, RiskRenderer, MacroRenderer,
    FundamentalRenderer, PortfolioRenderer, ForecastingRenderer,
    CommentaryRenderer, ComparisonRenderer
)

__all__ = [
    'DashboardManager',
    'SingleAssetWorkflow',
    'ComparativeWorkflow',
    'SummaryRenderer',
    'TechnicalRenderer', 
    'RiskRenderer',
    'MacroRenderer',
    'FundamentalRenderer',
    'PortfolioRenderer',
    'ForecastingRenderer',
    'CommentaryRenderer',
    'ComparisonRenderer'
]
