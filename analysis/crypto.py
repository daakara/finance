"""
Cryptocurrency analysis module.
Specialized analysis for cryptocurrencies including volatility, market metrics, and DeFi data.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data.cache import cache_result
from data.fetchers import crypto_fetcher

logger = logging.getLogger(__name__)

class CryptoAnalyzer:
    """Main cryptocurrency analysis class."""
    
    @staticmethod
    @cache_result(ttl=1800)  # Cache for 30 minutes (crypto moves fast)
    def get_crypto_data(
        symbol: str,
        period: str = '1y'
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Get comprehensive cryptocurrency data and analysis.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            period: Time period for analysis
        
        Returns:
            Dict containing crypto data and analysis
        """
        results = {}
        
        try:
            # Convert single symbol to trading pair if needed
            trading_symbol = CryptoAnalyzer._convert_to_trading_pair(symbol)
            
            # Convert period to timeframe and limit
            timeframe, limit = CryptoAnalyzer._convert_period_to_timeframe(period)
            
            # Get price data
            price_data = crypto_fetcher.get_crypto_data(trading_symbol, timeframe, limit)
            results['price_data'] = price_data
            
            # Get crypto info
            crypto_info = crypto_fetcher.get_crypto_info(trading_symbol)
            results['crypto_info'] = crypto_info
            
            if not price_data.empty:
                # Calculate crypto-specific metrics
                results['volatility_metrics'] = CryptoAnalyzer._calculate_crypto_volatility(price_data)
                results['performance_metrics'] = CryptoAnalyzer._calculate_crypto_performance(price_data)
                results['risk_metrics'] = CryptoAnalyzer._calculate_crypto_risk_metrics(price_data)
                results['momentum_metrics'] = CryptoAnalyzer._calculate_momentum_metrics(price_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing crypto {symbol}: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def _calculate_crypto_volatility(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate cryptocurrency volatility metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}
        
        close_prices = price_data['Close']
        daily_returns = close_prices.pct_change().dropna()
        
        # Realized volatility (annualized)
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(365) * 100  # 365 days for crypto
        
        # Rolling volatilities
        vol_7d = daily_returns.rolling(7).std().iloc[-1] * np.sqrt(365) * 100
        vol_30d = daily_returns.rolling(30).std().iloc[-1] * np.sqrt(365) * 100
        vol_90d = daily_returns.rolling(90).std().iloc[-1] * np.sqrt(365) * 100
        
        # Volatility of volatility
        rolling_vol = daily_returns.rolling(30).std()
        vol_of_vol = rolling_vol.std() * np.sqrt(365) * 100
        
        # Extreme move frequency
        extreme_moves = abs(daily_returns) > 0.05  # 5%+ moves
        extreme_frequency = extreme_moves.sum() / len(daily_returns) * 100
        
        return {
            'annual_volatility': annual_vol,
            'volatility_7d': vol_7d,
            'volatility_30d': vol_30d,
            'volatility_90d': vol_90d,
            'volatility_of_volatility': vol_of_vol,
            'extreme_move_frequency': extreme_frequency
        }
    
    @staticmethod
    def _calculate_crypto_performance(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate cryptocurrency performance metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}
        
        close_prices = price_data['Close']
        daily_returns = close_prices.pct_change().dropna()
        
        # Basic performance
        total_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
        
        # Period-specific returns
        if len(close_prices) >= 7:
            weekly_return = (close_prices.iloc[-1] / close_prices.iloc[-8] - 1) * 100
        else:
            weekly_return = 0
        
        if len(close_prices) >= 30:
            monthly_return = (close_prices.iloc[-1] / close_prices.iloc[-31] - 1) * 100
        else:
            monthly_return = 0
        
        if len(close_prices) >= 90:
            quarterly_return = (close_prices.iloc[-1] / close_prices.iloc[-91] - 1) * 100
        else:
            quarterly_return = 0
        
        # All-time high analysis
        all_time_high = close_prices.max()
        current_price = close_prices.iloc[-1]
        drawdown_from_ath = ((current_price - all_time_high) / all_time_high) * 100
        
        # Best and worst performing periods
        rolling_7d_returns = close_prices.pct_change(7).dropna()
        best_week = rolling_7d_returns.max() * 100 if not rolling_7d_returns.empty else 0
        worst_week = rolling_7d_returns.min() * 100 if not rolling_7d_returns.empty else 0
        
        return {
            'total_return': total_return,
            'weekly_return': weekly_return,
            'monthly_return': monthly_return,
            'quarterly_return': quarterly_return,
            'all_time_high': all_time_high,
            'drawdown_from_ath': drawdown_from_ath,
            'best_week': best_week,
            'worst_week': worst_week
        }
    
    @staticmethod
    def _calculate_crypto_risk_metrics(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate cryptocurrency risk metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}
        
        close_prices = price_data['Close']
        daily_returns = close_prices.pct_change().dropna()
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Value at Risk
        var_1 = np.percentile(daily_returns, 1) * 100
        var_5 = np.percentile(daily_returns, 5) * 100
        
        # Expected Shortfall (Conditional VaR)
        es_1 = daily_returns[daily_returns <= np.percentile(daily_returns, 1)].mean() * 100
        es_5 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
        
        # Skewness and Kurtosis
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        
        # Downside deviation
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(365) * 100
        
        return {
            'max_drawdown': max_drawdown,
            'var_1_percent': var_1,
            'var_5_percent': var_5,
            'expected_shortfall_1': es_1,
            'expected_shortfall_5': es_5,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_deviation': downside_deviation
        }
    
    @staticmethod
    def _calculate_momentum_metrics(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate cryptocurrency momentum metrics."""
        if price_data.empty or 'Close' not in price_data.columns:
            return {}
        
        close_prices = price_data['Close']
        
        # Simple momentum
        momentum_7d = (close_prices.iloc[-1] / close_prices.iloc[-8] - 1) * 100 if len(close_prices) >= 8 else 0
        momentum_30d = (close_prices.iloc[-1] / close_prices.iloc[-31] - 1) * 100 if len(close_prices) >= 31 else 0
        
        # Momentum strength
        daily_returns = close_prices.pct_change().dropna()
        
        # Up/Down day ratios
        up_days = (daily_returns > 0).sum()
        down_days = (daily_returns < 0).sum()
        up_down_ratio = up_days / down_days if down_days > 0 else float('inf')
        
        # Average up/down magnitudes
        avg_up_move = daily_returns[daily_returns > 0].mean() * 100 if (daily_returns > 0).any() else 0
        avg_down_move = abs(daily_returns[daily_returns < 0].mean()) * 100 if (daily_returns < 0).any() else 0
        
        # Consecutive moves
        consecutive_ups = CryptoAnalyzer._max_consecutive(daily_returns > 0)
        consecutive_downs = CryptoAnalyzer._max_consecutive(daily_returns < 0)
        
        return {
            'momentum_7d': momentum_7d,
            'momentum_30d': momentum_30d,
            'up_down_ratio': min(up_down_ratio, 10),  # Cap at 10 for display
            'avg_up_move': avg_up_move,
            'avg_down_move': avg_down_move,
            'max_consecutive_ups': consecutive_ups,
            'max_consecutive_downs': consecutive_downs
        }
    
    @staticmethod
    def _max_consecutive(series: pd.Series) -> int:
        """Calculate maximum consecutive True values."""
        if series.empty:
            return 0
        
        max_count = 0
        current_count = 0
        
        for value in series:
            if value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    @staticmethod
    def _convert_period_to_timeframe(period: str) -> tuple:
        """Convert yfinance-style period to crypto exchange timeframe and limit."""
        period_mapping = {
            '1d': ('1h', 24),      # 1 day = 24 hours
            '5d': ('1h', 120),     # 5 days = 120 hours  
            '1mo': ('1d', 30),     # 1 month = 30 days
            '3mo': ('1d', 90),     # 3 months = 90 days
            '6mo': ('1d', 180),    # 6 months = 180 days
            '1y': ('1d', 365),     # 1 year = 365 days
            '2y': ('1d', 730),     # 2 years = 730 days
            '5y': ('1d', 1825),    # 5 years = 1825 days
            'ytd': ('1d', 365),    # Year to date ~365 days
            'max': ('1d', 2000)    # Max available ~5+ years
        }
        return period_mapping.get(period, ('1d', 365))  # Default to 1 year
    
    @staticmethod
    def _convert_to_trading_pair(symbol: str) -> str:
        """Convert single crypto symbol to trading pair format for exchanges."""
        # If already a trading pair, return as is
        if '/' in symbol:
            return symbol
        
        # Convert single symbol to USDT pair (most liquid on most exchanges)
        # Special cases for stablecoins
        if symbol in ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'FRAX', 'TUSD', 'USDP']:
            return f"{symbol}/USD"  # Stablecoins against USD
        else:
            return f"{symbol}/USDT"  # All others against USDT
    
    @staticmethod
    def get_top_cryptocurrencies() -> Dict[str, Dict[str, str]]:
        """Get list of top cryptocurrencies by category."""
        return {
            'Major Coins': {
                'BTC': 'Bitcoin',
                'ETH': 'Ethereum',
                'BNB': 'Binance Coin',
                'ADA': 'Cardano',
                'SOL': 'Solana',
                'XRP': 'Ripple',
                'DOT': 'Polkadot',
                'AVAX': 'Avalanche'
            },
            'DeFi Tokens': {
                'UNI': 'Uniswap',
                'AAVE': 'Aave',
                'COMP': 'Compound',
                'MKR': 'Maker',
                'SNX': 'Synthetix',
                'CRV': 'Curve DAO Token',
                'SUSHI': 'SushiSwap',
                '1INCH': '1inch'
            },
            'Smart Contract Platforms': {
                'ETH': 'Ethereum',
                'ADA': 'Cardano',
                'SOL': 'Solana',
                'AVAX': 'Avalanche',
                'MATIC': 'Polygon',
                'ALGO': 'Algorand',
                'ATOM': 'Cosmos',
                'NEAR': 'NEAR Protocol'
            },
            'Stablecoins': {
                'USDT': 'Tether',
                'USDC': 'USD Coin',
                'BUSD': 'Binance USD',
                'DAI': 'Dai',
                'UST': 'TerraUSD',
                'FRAX': 'Frax',
                'TUSD': 'TrueUSD',
                'USDP': 'Pax Dollar'
            },
            'Layer 2 Solutions': {
                'MATIC': 'Polygon',
                'LRC': 'Loopring',
                'OMG': 'OMG Network',
                'METIS': 'Metis',
                'ARB': 'Arbitrum',
                'OP': 'Optimism'
            },
            'Meme Coins': {
                'DOGE': 'Dogecoin',
                'SHIB': 'Shiba Inu',
                'FLOKI': 'Floki Inu',
                'PEPE': 'Pepe',
                'BONK': 'Bonk'
            }
        }
    
    @staticmethod
    def compare_cryptocurrencies(
        crypto_symbols: List[str],
        period: str = '1y'
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Compare multiple cryptocurrencies across various metrics.
        
        Args:
            crypto_symbols: List of crypto symbols to compare
            period: Time period for comparison
        
        Returns:
            Dict containing comparison data
        """
        comparison_data = {}
        price_data = {}
        
        # Fetch data for all cryptos
        for symbol in crypto_symbols:
            try:
                crypto_data = CryptoAnalyzer.get_crypto_data(symbol, period)
                if 'price_data' in crypto_data and not crypto_data['price_data'].empty:
                    price_data[symbol] = crypto_data['price_data']['Close']
                    comparison_data[symbol] = {
                        'performance': crypto_data.get('performance_metrics', {}),
                        'volatility': crypto_data.get('volatility_metrics', {}),
                        'risk': crypto_data.get('risk_metrics', {}),
                        'momentum': crypto_data.get('momentum_metrics', {})
                    }
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        # Create comparison tables
        results = {
            'price_data': pd.DataFrame(price_data),
            'comparison_data': comparison_data
        }
        
        if comparison_data:
            # Performance comparison
            perf_metrics = ['total_return', 'monthly_return', 'quarterly_return']
            perf_table = []
            
            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in perf_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('performance', {}).get(metric, 0)
                perf_table.append(row)
            
            results['performance_comparison'] = pd.DataFrame(perf_table)
            
            # Volatility comparison
            vol_metrics = ['annual_volatility', 'volatility_30d', 'extreme_move_frequency']
            vol_table = []
            
            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in vol_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('volatility', {}).get(metric, 0)
                vol_table.append(row)
            
            results['volatility_comparison'] = pd.DataFrame(vol_table)
            
            # Risk comparison
            risk_metrics = ['max_drawdown', 'var_5_percent', 'skewness', 'kurtosis']
            risk_table = []
            
            for symbol, data in comparison_data.items():
                row = {'Symbol': symbol}
                for metric in risk_metrics:
                    row[metric.replace('_', ' ').title()] = data.get('risk', {}).get(metric, 0)
                risk_table.append(row)
            
            results['risk_comparison'] = pd.DataFrame(risk_table)
        
        return results
    
    @staticmethod
    def get_market_sentiment_indicators() -> Dict[str, Union[float, str]]:
        """
        Get cryptocurrency market sentiment indicators.
        (This would connect to real sentiment APIs in production)
        """
        # Simulated sentiment data
        import random
        
        fear_greed_index = random.randint(0, 100)
        
        if fear_greed_index <= 25:
            sentiment = 'Extreme Fear'
            color = 'red'
        elif fear_greed_index <= 45:
            sentiment = 'Fear'
            color = 'orange'
        elif fear_greed_index <= 55:
            sentiment = 'Neutral'
            color = 'yellow'
        elif fear_greed_index <= 75:
            sentiment = 'Greed'
            color = 'lightgreen'
        else:
            sentiment = 'Extreme Greed'
            color = 'green'
        
        return {
            'fear_greed_index': fear_greed_index,
            'sentiment': sentiment,
            'sentiment_color': color,
            'social_volume': random.randint(50, 150),  # Relative social mentions
            'google_trends': random.randint(30, 100),  # Google search interest
            'reddit_sentiment': random.choice(['Bullish', 'Bearish', 'Neutral'])
        }

# Create instance for easy importing
crypto_analyzer = CryptoAnalyzer()
