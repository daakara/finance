"""
Dashboard Renderers Package - Modular rendering components
Provides specialized renderers for different analysis sections
"""

from .summary_renderer import SummaryRenderer
from .technical_renderer import TechnicalRenderer
from .risk_renderer import RiskRenderer
from .macro_renderer import MacroRenderer
from .fundamental_renderer import FundamentalRenderer
from .portfolio_renderer import PortfolioRenderer, ForecastingRenderer
from .commentary_renderer import CommentaryRenderer, ComparisonRenderer

__all__ = [
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
