"""
Multi-Asset Data Pipeline for Hidden Gems Scanner
Fetches comprehensive data from multiple sources for screening analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
import warnings
import json
import re
from dataclasses import dataclass
import urllib3

# Disable SSL warnings for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for data sources"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    rate_limit: int = 100  # Requests per minute
    timeout: int = 30      # Seconds

class MultiAssetDataPipeline:
    """Comprehensive data fetching pipeline for multi-asset analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with data source configuration"""
        self.config = config or {}
        
        # Configure data sources
        self.sources = {
            'yfinance': DataSource('yfinance', enabled=True),
            'alpha_vantage': DataSource('alpha_vantage', enabled=False, 
                                      api_key=self.config.get('alpha_vantage_key')),
            'polygon': DataSource('polygon', enabled=False,
                                api_key=self.config.get('polygon_key')),
            'sec_edgar': DataSource('sec_edgar', enabled=True),
            'reddit': DataSource('reddit', enabled=False),
            'twitter': DataSource('twitter', enabled=False)
        }
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HiddenGemsScanner/1.0 (Educational/Research)'
        })
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch comprehensive stock data including price, fundamentals, and metadata.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1y, 2y, 5y, etc.)
            
        Returns:
            Dictionary with comprehensive stock data
        """
        try:
            data = {
                'ticker': ticker,
                'asset_type': 'stock',
                'timestamp': datetime.now(),
                'price_data': pd.DataFrame(),
                'fundamentals': {},
                'market_data': {},
                'institutional_data': {},
                'sentiment_data': {},
                'error': None
            }
            
            # Fetch from yfinance
            if self.sources['yfinance'].enabled:
                yf_data = self._fetch_yfinance_data(ticker, period)
                data.update(yf_data)
            
            # Fetch additional fundamental data
            if self.sources['alpha_vantage'].enabled:
                av_data = self._fetch_alpha_vantage_data(ticker)
                data['fundamentals'].update(av_data.get('fundamentals', {}))
            
            # Fetch SEC filings data
            if self.sources['sec_edgar'].enabled:
                sec_data = self._fetch_sec_filings_data(ticker)
                data['institutional_data'].update(sec_data)
            
            # Fetch sentiment data
            sentiment_data = self._fetch_sentiment_data(ticker)
            data['sentiment_data'] = sentiment_data
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'asset_type': 'stock',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def fetch_etf_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch comprehensive ETF data including holdings, flows, and thematic exposure.
        
        Args:
            ticker: ETF ticker symbol
            period: Data period
            
        Returns:
            Dictionary with comprehensive ETF data
        """
        try:
            data = {
                'ticker': ticker,
                'asset_type': 'etf',
                'timestamp': datetime.now(),
                'price_data': pd.DataFrame(),
                'holdings': {},
                'flows': {},
                'thematic_exposure': {},
                'error': None
            }
            
            # Fetch basic ETF data from yfinance
            if self.sources['yfinance'].enabled:
                yf_data = self._fetch_yfinance_data(ticker, period)
                data.update(yf_data)
                
                # ETF-specific data
                etf_info = yf_data.get('info', {})
                data['holdings'] = self._extract_etf_holdings(etf_info)
                data['thematic_exposure'] = self._analyze_thematic_exposure(etf_info)
            
            # Fetch ETF flows data (placeholder - would integrate with ETF flow APIs)
            data['flows'] = self._fetch_etf_flows_data(ticker)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching ETF data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'asset_type': 'etf',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def fetch_crypto_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch comprehensive cryptocurrency data including on-chain metrics.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            period: Data period
            
        Returns:
            Dictionary with comprehensive crypto data
        """
        try:
            data = {
                'symbol': symbol,
                'asset_type': 'crypto',
                'timestamp': datetime.now(),
                'price_data': pd.DataFrame(),
                'on_chain_metrics': {},
                'defi_metrics': {},
                'sentiment_data': {},
                'error': None
            }
            
            # Fetch price data from yfinance (crypto pairs)
            crypto_ticker = f"{symbol}-USD"
            if self.sources['yfinance'].enabled:
                yf_data = self._fetch_yfinance_data(crypto_ticker, period)
                data['price_data'] = yf_data.get('price_data', pd.DataFrame())
            
            # Fetch on-chain metrics (placeholder - would integrate with Glassnode, etc.)
            data['on_chain_metrics'] = self._fetch_onchain_metrics(symbol)
            
            # Fetch DeFi metrics if applicable
            if symbol in ['ETH', 'BNB', 'AVAX', 'MATIC']:
                data['defi_metrics'] = self._fetch_defi_metrics(symbol)
            
            # Fetch crypto sentiment
            data['sentiment_data'] = self._fetch_crypto_sentiment(symbol)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'asset_type': 'crypto',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def fetch_alternative_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch alternative data including social sentiment, news, and insider activity.
        
        Args:
            ticker: Asset ticker/symbol
            
        Returns:
            Dictionary with alternative data
        """
        try:
            data = {
                'ticker': ticker,
                'news_sentiment': {},
                'social_sentiment': {},
                'insider_activity': {},
                'google_trends': {},
                'options_flow': {},
                'timestamp': datetime.now()
            }
            
            # Fetch news sentiment
            data['news_sentiment'] = self._fetch_news_sentiment(ticker)
            
            # Fetch social media sentiment
            data['social_sentiment'] = self._fetch_social_sentiment(ticker)
            
            # Fetch insider trading activity
            data['insider_activity'] = self._fetch_insider_activity(ticker)
            
            # Fetch Google Trends data
            data['google_trends'] = self._fetch_google_trends(ticker)
            
            # Fetch options flow data
            data['options_flow'] = self._fetch_options_flow(ticker)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching alternative data for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def _fetch_yfinance_data(self, ticker: str, period: str) -> Dict[str, Any]:
        """Fetch data from yfinance with comprehensive SSL and error handling"""
        import os
        import ssl
        import certifi
        
        try:
            # Set SSL environment variables
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['CURL_CA_BUNDLE'] = certifi.where()
            os.environ['CURL_DISABLE_SSL_VERIFY'] = '1'
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Suppress yfinance logging
                yf_logger = logging.getLogger('yfinance')
                original_level = yf_logger.level
                yf_logger.setLevel(logging.CRITICAL)
                
                try:
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Price data with timeout
                    hist = ticker_obj.history(period=period, timeout=60)
                    
                    # Company info
                    info = {}
                    try:
                        info = ticker_obj.info or {}
                    except:
                        logger.debug(f"Could not fetch info for {ticker}")
                    
                    # Financial data
                    financials = {}
                    try:
                        if hasattr(ticker_obj, 'financials'):
                            financials_df = ticker_obj.financials
                            if not financials_df.empty:
                                financials['income_statement'] = financials_df.to_dict()
                        
                        if hasattr(ticker_obj, 'balance_sheet'):
                            balance_sheet_df = ticker_obj.balance_sheet
                            if not balance_sheet_df.empty:
                                financials['balance_sheet'] = balance_sheet_df.to_dict()
                        
                        if hasattr(ticker_obj, 'cashflow'):
                            cashflow_df = ticker_obj.cashflow
                            if not cashflow_df.empty:
                                financials['cash_flow'] = cashflow_df.to_dict()
                    except:
                        logger.debug(f"Could not fetch financials for {ticker}")
                    
                    return {
                        'price_data': hist,
                        'info': info,
                        'financials': financials,
                        'market_data': self._extract_market_data(info),
                        'fundamentals': self._extract_fundamental_metrics(info, financials)
                    }
                    
                finally:
                    # Restore logging level
                    yf_logger.setLevel(original_level)
                
        except Exception as e:
            logger.error(f"yfinance fetch error for {ticker}: {e}")
            return {'price_data': pd.DataFrame(), 'info': {}, 'financials': {}}
    
    def _extract_market_data(self, info: Dict) -> Dict[str, Any]:
        """Extract market-related data from ticker info"""
        try:
            return {
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'shares_short': info.get('sharesShort', 0),
                'short_ratio': info.get('shortRatio', 0),
                'held_percent_institutions': info.get('heldPercentInstitutions', 0),
                'held_percent_insiders': info.get('heldPercentInsiders', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'avg_volume_10d': info.get('averageVolume10days', 0)
            }
        except Exception as e:
            logger.error(f"Error extracting market data: {e}")
            return {}
    
    def _extract_fundamental_metrics(self, info: Dict, financials: Dict) -> Dict[str, Any]:
        """Extract fundamental metrics from ticker data"""
        try:
            metrics = {
                'revenue_growth_yoy': 0,
                'gross_margin': 0,
                'operating_margin': 0,
                'net_margin': 0,
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'enterprise_to_revenue': info.get('enterpriseToRevenue', 0),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda', 0)
            }
            
            # Calculate additional metrics from financials
            if 'income_statement' in financials:
                income_stmt = financials['income_statement']
                if income_stmt:
                    # Get most recent year's data
                    recent_data = {}
                    for key, value in income_stmt.items():
                        if isinstance(value, dict) and value:
                            recent_key = max(value.keys())
                            recent_data[key] = value[recent_key]
                    
                    # Calculate margins
                    revenue = recent_data.get('Total Revenue', 0)
                    if revenue > 0:
                        gross_profit = recent_data.get('Gross Profit', 0)
                        metrics['gross_margin'] = gross_profit / revenue if gross_profit else 0
                        
                        operating_income = recent_data.get('Operating Income', 0)
                        metrics['operating_margin'] = operating_income / revenue if operating_income else 0
                        
                        net_income = recent_data.get('Net Income', 0)
                        metrics['net_margin'] = net_income / revenue if net_income else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting fundamental metrics: {e}")
            return {}
    
    def _fetch_alpha_vantage_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage API (placeholder)"""
        # This would integrate with Alpha Vantage API for enhanced fundamentals
        return {'fundamentals': {}}
    
    def _fetch_sec_filings_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch SEC filings data for institutional holdings and insider activity"""
        try:
            # This is a simplified version - real implementation would parse SEC EDGAR filings
            return {
                'insider_transactions': [],
                'institutional_holdings': {},
                '13f_filings': {},
                'form4_filings': []
            }
        except Exception as e:
            logger.error(f"Error fetching SEC data for {ticker}: {e}")
            return {}
    
    def _fetch_sentiment_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch sentiment data from various sources"""
        try:
            return {
                'news_sentiment_score': 0.5,  # Neutral
                'social_sentiment_score': 0.5,
                'analyst_sentiment': 'Hold',
                'sentiment_trend': 'Stable',
                'mention_volume': 100,
                'sentiment_sources': ['news', 'social_media']
            }
        except Exception as e:
            logger.error(f"Error fetching sentiment for {ticker}: {e}")
            return {}
    
    def _extract_etf_holdings(self, etf_info: Dict) -> Dict[str, Any]:
        """Extract ETF holdings information"""
        try:
            return {
                'top_holdings': etf_info.get('holdings', []),
                'sector_weightings': etf_info.get('sectorWeightings', {}),
                'holdings_count': etf_info.get('holdingsCount', 0),
                'turnover_rate': etf_info.get('annualHoldingsTurnover', 0)
            }
        except Exception as e:
            logger.error(f"Error extracting ETF holdings: {e}")
            return {}
    
    def _analyze_thematic_exposure(self, etf_info: Dict) -> Dict[str, Any]:
        """Analyze ETF's thematic exposure to emerging sectors"""
        try:
            # This would analyze holdings and categorize by themes
            category = etf_info.get('category', '')
            fund_name = etf_info.get('longName', '')
            
            themes = []
            theme_keywords = {
                'AI/ML': ['artificial intelligence', 'machine learning', 'ai', 'robotics'],
                'Clean Energy': ['clean energy', 'solar', 'wind', 'renewable', 'green'],
                'Blockchain': ['blockchain', 'bitcoin', 'crypto', 'digital assets'],
                'Biotech': ['biotech', 'genomics', 'pharmaceutical', 'healthcare'],
                'Fintech': ['fintech', 'digital payments', 'financial technology'],
                'Space': ['space', 'satellite', 'aerospace'],
                'Cybersecurity': ['cybersecurity', 'security', 'privacy']
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in category.lower() or keyword in fund_name.lower() 
                      for keyword in keywords):
                    themes.append(theme)
            
            return {
                'primary_themes': themes,
                'theme_purity': len(themes) == 1,  # Pure play vs diversified
                'emerging_sector_exposure': len(themes) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing thematic exposure: {e}")
            return {}
    
    def _fetch_etf_flows_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch ETF flows data (placeholder for real ETF flow APIs)"""
        try:
            # This would integrate with ETF flow data providers
            return {
                'net_flows_1m': 0,
                'net_flows_3m': 0,
                'net_flows_1y': 0,
                'flow_trend': 'neutral',
                'aum_change_pct': 0
            }
        except Exception as e:
            logger.error(f"Error fetching ETF flows for {ticker}: {e}")
            return {}
    
    def _fetch_onchain_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch on-chain metrics for cryptocurrencies"""
        try:
            # This would integrate with Glassnode, Messari, or similar APIs
            base_metrics = {
                'active_addresses': 0,
                'transaction_count': 0,
                'network_value_to_transactions': 0,
                'mvrv_ratio': 0,
                'holder_distribution': {},
                'exchange_flows': {
                    'inflows': 0,
                    'outflows': 0,
                    'net_flows': 0
                }
            }
            
            # Symbol-specific metrics
            if symbol == 'BTC':
                base_metrics.update({
                    'hash_rate': 0,
                    'mining_difficulty': 0,
                    'puell_multiple': 0,
                    'stock_to_flow': 0
                })
            elif symbol == 'ETH':
                base_metrics.update({
                    'gas_price': 0,
                    'total_value_locked': 0,
                    'staking_ratio': 0,
                    'burn_rate': 0
                })
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error fetching on-chain metrics for {symbol}: {e}")
            return {}
    
    def _fetch_defi_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch DeFi-related metrics for applicable cryptocurrencies"""
        try:
            return {
                'total_value_locked': 0,
                'protocol_count': 0,
                'top_protocols': [],
                'yield_farming_apy': 0,
                'governance_activity': {}
            }
        except Exception as e:
            logger.error(f"Error fetching DeFi metrics for {symbol}: {e}")
            return {}
    
    def _fetch_crypto_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch cryptocurrency-specific sentiment data"""
        try:
            return {
                'fear_greed_index': 50,  # Neutral
                'social_dominance': 0,
                'developer_activity': 0,
                'github_commits': 0,
                'reddit_sentiment': 0.5,
                'twitter_sentiment': 0.5
            }
        except Exception as e:
            logger.error(f"Error fetching crypto sentiment for {symbol}: {e}")
            return {}
    
    def _fetch_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Fetch news sentiment analysis"""
        try:
            # This would integrate with news APIs and NLP for sentiment analysis
            return {
                'sentiment_score': 0.5,
                'article_count_7d': 5,
                'article_count_30d': 20,
                'sentiment_trend': 'stable',
                'key_topics': [],
                'sources': []
            }
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {ticker}: {e}")
            return {}
    
    def _fetch_social_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Fetch social media sentiment analysis"""
        try:
            # This would integrate with Reddit, Twitter, Discord APIs
            return {
                'reddit_sentiment': 0.5,
                'twitter_sentiment': 0.5,
                'discord_activity': 0,
                'mention_volume': 100,
                'trending_score': 0,
                'influencer_mentions': []
            }
        except Exception as e:
            logger.error(f"Error fetching social sentiment for {ticker}: {e}")
            return {}
    
    def _fetch_insider_activity(self, ticker: str) -> Dict[str, Any]:
        """Fetch insider trading activity"""
        try:
            # This would parse SEC Form 4 filings
            return {
                'recent_purchases': [],
                'recent_sales': [],
                'net_insider_activity': 0,
                'insider_confidence_score': 0.5,
                'key_insider_transactions': []
            }
        except Exception as e:
            logger.error(f"Error fetching insider activity for {ticker}: {e}")
            return {}
    
    def _fetch_google_trends(self, ticker: str) -> Dict[str, Any]:
        """Fetch Google Trends data"""
        try:
            # This would integrate with Google Trends API
            return {
                'search_volume_trend': 0,
                'relative_interest': 50,
                'trending_queries': [],
                'geographic_interest': {}
            }
        except Exception as e:
            logger.error(f"Error fetching Google Trends for {ticker}: {e}")
            return {}
    
    def _fetch_options_flow(self, ticker: str) -> Dict[str, Any]:
        """Fetch options flow data"""
        try:
            # This would integrate with options data providers
            return {
                'unusual_options_activity': False,
                'call_put_ratio': 1.0,
                'options_volume': 0,
                'large_trades': [],
                'gamma_exposure': 0,
                'implied_volatility': 0
            }
        except Exception as e:
            logger.error(f"Error fetching options flow for {ticker}: {e}")
            return {}
    
    def get_comprehensive_data(self, ticker: str, asset_type: str = 'stock') -> Dict[str, Any]:
        """
        Get comprehensive data for any asset type.
        
        Args:
            ticker: Asset ticker/symbol
            asset_type: Type of asset ('stock', 'etf', 'crypto')
            
        Returns:
            Comprehensive data dictionary
        """
        try:
            if asset_type.lower() == 'stock':
                base_data = self.fetch_stock_data(ticker)
            elif asset_type.lower() == 'etf':
                base_data = self.fetch_etf_data(ticker)
            elif asset_type.lower() == 'crypto':
                base_data = self.fetch_crypto_data(ticker)
            else:
                raise ValueError(f"Unsupported asset type: {asset_type}")
            
            # Add alternative data
            alt_data = self.fetch_alternative_data(ticker)
            base_data['alternative_data'] = alt_data
            
            return base_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'asset_type': asset_type,
                'error': str(e),
                'timestamp': datetime.now()
            }


# Utility functions for data processing
def calculate_revenue_growth(financials: Dict) -> float:
    """Calculate year-over-year revenue growth from financial data"""
    try:
        if 'income_statement' not in financials:
            return 0.0
        
        income_stmt = financials['income_statement']
        revenues = income_stmt.get('Total Revenue', {})
        
        if len(revenues) < 2:
            return 0.0
        
        # Get last two years
        years = sorted(revenues.keys(), reverse=True)
        current_revenue = revenues[years[0]]
        previous_revenue = revenues[years[1]]
        
        if previous_revenue and previous_revenue != 0:
            growth = (current_revenue - previous_revenue) / previous_revenue
            return growth
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating revenue growth: {e}")
        return 0.0

def calculate_cash_runway(financials: Dict, current_burn_rate: float) -> float:
    """Calculate cash runway in years"""
    try:
        if 'balance_sheet' not in financials:
            return 0.0
        
        balance_sheet = financials['balance_sheet']
        cash_items = balance_sheet.get('Cash And Cash Equivalents', {})
        
        if not cash_items:
            return 0.0
        
        # Get most recent cash position
        recent_date = max(cash_items.keys())
        cash_position = cash_items[recent_date]
        
        if current_burn_rate <= 0:
            return float('inf')  # No burn or positive cash flow
        
        runway_years = cash_position / (current_burn_rate * 4)  # Quarterly burn rate
        return max(runway_years, 0)
        
    except Exception as e:
        logger.error(f"Error calculating cash runway: {e}")
        return 0.0

def detect_sector_rotation_signals(etf_flows: Dict[str, Dict]) -> Dict[str, Any]:
    """Detect sector rotation signals from ETF flow data"""
    try:
        rotation_signals = {
            'sectors_in_favor': [],
            'sectors_out_of_favor': [],
            'rotation_strength': 0,
            'rotation_direction': 'neutral'
        }
        
        # Analyze flow trends across sector ETFs
        for etf, flow_data in etf_flows.items():
            net_flows_3m = flow_data.get('net_flows_3m', 0)
            
            if net_flows_3m > 100e6:  # $100M+ inflows
                rotation_signals['sectors_in_favor'].append(etf)
            elif net_flows_3m < -100e6:  # $100M+ outflows
                rotation_signals['sectors_out_of_favor'].append(etf)
        
        # Calculate rotation strength
        total_rotation = len(rotation_signals['sectors_in_favor']) + len(rotation_signals['sectors_out_of_favor'])
        rotation_signals['rotation_strength'] = min(total_rotation / 10.0, 1.0)  # 0-1 scale
        
        return rotation_signals
        
    except Exception as e:
        logger.error(f"Error detecting sector rotation: {e}")
        return {'sectors_in_favor': [], 'sectors_out_of_favor': [], 'rotation_strength': 0}


# Example usage
if __name__ == "__main__":
    # Initialize data pipeline
    pipeline = MultiAssetDataPipeline()
    
    # Test data fetching
    print("🔄 Multi-Asset Data Pipeline - Testing")
    print("=" * 40)
    
    # Test stock data
    stock_data = pipeline.fetch_stock_data('AAPL')
    print(f"Stock data keys: {list(stock_data.keys())}")
    
    # Test ETF data
    etf_data = pipeline.fetch_etf_data('SPY')
    print(f"ETF data keys: {list(etf_data.keys())}")
    
    # Test crypto data
    crypto_data = pipeline.fetch_crypto_data('BTC')
    print(f"Crypto data keys: {list(crypto_data.keys())}")
    
    print("\n✅ Data pipeline test completed")
