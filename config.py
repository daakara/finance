"""
Configuration settings and constants for the Financial Analysis Platform.
All API keys and sensitive data should be stored in environment variables.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppSettings:
    """Validated environment settings container."""
    def __init__(self):
        self.alpha_vantage_api_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        self.news_api_key: str = os.getenv('NEWS_API_KEY', '')
        self.twitter_bearer_token: str = os.getenv('TWITTER_BEARER_TOKEN', '')
        self.cache_ttl_seconds: int = int(os.getenv('CACHE_TTL_SECONDS', '300'))
        self.max_cache_size: int = int(os.getenv('MAX_CACHE_SIZE', '1000'))
        self.default_period: str = os.getenv('DEFAULT_PERIOD', '1y')
        self.default_interval: str = os.getenv('DEFAULT_INTERVAL', '1d')
        self.max_symbols_per_request: int = int(os.getenv('MAX_SYMBOLS_PER_REQUEST', '10'))
        self.disable_ssl_verify: bool = os.getenv('DISABLE_SSL_VERIFY', 'false').lower() == 'true'
        self.use_sample_data: bool = os.getenv('USE_SAMPLE_DATA', 'false').lower() == 'true'

_settings = AppSettings()

class Config:
    """Main configuration class for the application."""
    
    # API Configuration
    ALPHA_VANTAGE_API_KEY = _settings.alpha_vantage_api_key
    NEWS_API_KEY = _settings.news_api_key
    TWITTER_BEARER_TOKEN = _settings.twitter_bearer_token
    
    # Cache Configuration
    CACHE_TTL_SECONDS = _settings.cache_ttl_seconds
    MAX_CACHE_SIZE = _settings.max_cache_size
    
    # Data Configuration
    DEFAULT_PERIOD = _settings.default_period
    DEFAULT_INTERVAL = _settings.default_interval
    MAX_SYMBOLS_PER_REQUEST = _settings.max_symbols_per_request
    
    # Network Configuration
    DISABLE_SSL_VERIFY = _settings.disable_ssl_verify
    USE_SAMPLE_DATA = _settings.use_sample_data
    
    # Chart Configuration
    CHART_HEIGHT = 600
    CHART_WIDTH = 1000
    
    # Color Scheme
    COLORS = {
        'bullish': '#00C851',      # Green
        'bearish': '#FF4444',      # Red
        'neutral': '#33b5e5',      # Blue
        'background': '#1e1e1e',   # Dark background
        'text': '#ffffff',         # White text
        'grid': '#404040'          # Gray grid
    }

class MarketConstants:
    """Market-specific constants and configurations."""
    
    # Trading hours (24-hour format)
    MARKET_HOURS = {
        'NYSE': {'open': '09:30', 'close': '16:00', 'timezone': 'America/New_York'},
        'NASDAQ': {'open': '09:30', 'close': '16:00', 'timezone': 'America/New_York'},
        'LSE': {'open': '08:00', 'close': '16:30', 'timezone': 'Europe/London'},
        'TSE': {'open': '09:00', 'close': '15:00', 'timezone': 'Asia/Tokyo'},
    }
    
    # Major market indices
    MAJOR_INDICES = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT',
        'FTSE 100': '^FTSE',
        'Nikkei 225': '^N225',
        'DAX': '^GDAXI'
    }
    
    # Popular ETFs
    POPULAR_ETFS = {
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
        'EFA': 'iShares MSCI EAFE ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF',
        'GLD': 'SPDR Gold Shares',
        'TLT': 'iShares 20+ Year Treasury Bond ETF'
    }
    
    # Cryptocurrency pairs
    CRYPTO_PAIRS = {
        'BTC/USD': 'Bitcoin',
        'ETH/USD': 'Ethereum',
        'ADA/USD': 'Cardano',
        'SOL/USD': 'Solana',
        'DOT/USD': 'Polkadot',
        'LINK/USD': 'Chainlink',
        'MATIC/USD': 'Polygon',
        'AVAX/USD': 'Avalanche'
    }
    
    # Risk-free rate proxies
    RISK_FREE_RATES = {
        'US': '^TNX',      # 10-Year Treasury
        'UK': '^TNXUK',    # UK 10-Year Gilt
        'DE': '^TNX-DE',   # German 10-Year Bund
        'JP': '^TNX-JP'    # Japan 10-Year Bond
    }

class TechnicalIndicators:
    """Configuration for technical indicators."""
    
    # Default parameters for common indicators
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Moving average periods
    MA_PERIODS = [10, 20, 50, 100, 200]
    
    # Volume indicators
    VOLUME_MA_PERIOD = 20
    OBV_PERIOD = 14
    
    # Momentum indicators
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    WILLIAMS_R_PERIOD = 14

# Export configuration instances
config = Config()
market_constants = MarketConstants()
technical_config = TechnicalIndicators()
