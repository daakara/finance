"""
Analysis Engine Module - Streamlined engine orchestrator
Imports modular engines and provides backward compatibility
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import all engines from the modular engines package
from engines import (
    technical_engine, TechnicalAnalysisEngine,
    risk_engine, RiskAnalysisEngine,
    fundamental_engine, FundamentalAnalysisEngine,
    performance_engine, PerformanceAnalysisEngine,
    macro_engine, MacroeconomicAnalysisEngine,
    portfolio_engine, PortfolioStrategyEngine,
    forecasting_engine, ForecastingEngine
)

logger = logging.getLogger(__name__)

# Export all engines and classes for backward compatibility
__all__ = [
    'technical_engine', 'TechnicalAnalysisEngine',
    'risk_engine', 'RiskAnalysisEngine', 
    'fundamental_engine', 'FundamentalAnalysisEngine',
    'performance_engine', 'PerformanceAnalysisEngine',
    'macro_engine', 'MacroeconomicAnalysisEngine',
    'portfolio_engine', 'PortfolioStrategyEngine',
    'forecasting_engine', 'ForecastingEngine'
]

# Log successful import
logger.info("Analysis engines imported successfully from modular components")
