"""
Portfolio analytics and optimization functions.
Implements Modern Portfolio Theory and risk management calculations.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from data.cache import cache_result
from data.processors import DataProcessor

logger = logging.getLogger(__name__)

class PortfolioMetrics:
    """Calculate portfolio performance and risk metrics."""
    
    @staticmethod
    @cache_result(ttl=600)  # Cache for 10 minutes
    def calculate_portfolio_returns(
        returns_data: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate weighted portfolio returns.
        
        Args:
            returns_data: DataFrame with individual asset returns
            weights: Dictionary mapping asset symbols to weights
        
        Returns:
            pd.Series: Portfolio returns
        """
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0, index=returns_data.index)
        
        for symbol, weight in weights.items():
            if symbol in returns_data.columns:
                portfolio_returns += returns_data[symbol] * weight
            else:
                logger.warning(f"Symbol {symbol} not found in returns data")
        
        return portfolio_returns
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
        
        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # Convert risk-free rate to period rate
        period_rf_rate = risk_free_rate / periods_per_year
        
        # Calculate excess returns
        excess_returns = returns - period_rf_rate
        
        # Annualized Sharpe ratio
        sharpe = (excess_returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
        
        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0
        
        # Convert risk-free rate to period rate
        period_rf_rate = risk_free_rate / periods_per_year
        
        # Calculate excess returns
        excess_returns = returns - period_rf_rate
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return float('inf')
        
        sortino = (excess_returns.mean() * periods_per_year) / downside_deviation
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            returns: Return series
        
        Returns:
            Dict containing max drawdown metrics
        """
        if returns.empty:
            return {'max_drawdown': 0, 'drawdown_duration': 0, 'recovery_time': 0}
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown duration
        max_dd_date = drawdown.idxmin()
        
        # Find start of drawdown period
        start_date = running_max[:max_dd_date].idxmax()
        
        # Find recovery date (if any)
        recovery_date = None
        post_drawdown = cumulative[max_dd_date:]
        peak_value = running_max.loc[max_dd_date]
        
        recovery_candidates = post_drawdown[post_drawdown >= peak_value]
        if not recovery_candidates.empty:
            recovery_date = recovery_candidates.index[0]
        
        # Calculate durations
        drawdown_duration = (max_dd_date - start_date).days if isinstance(max_dd_date, pd.Timestamp) else 0
        recovery_time = (recovery_date - max_dd_date).days if recovery_date else None
        
        return {
            'max_drawdown': abs(max_drawdown),
            'drawdown_start': start_date,
            'drawdown_end': max_dd_date,
            'recovery_date': recovery_date,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time
        }
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
        Returns:
            VaR value
        """
        if returns.empty:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        if returns.empty:
            return 0.0
        
        var = PortfolioMetrics.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    @staticmethod
    def calculate_beta(
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate beta coefficient.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
        
        Returns:
            Beta coefficient
        """
        if asset_returns.empty or market_returns.empty:
            return 0.0
        
        # Align the series
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return 0.0
        
        return covariance / market_variance

class PortfolioOptimization:
    """Modern Portfolio Theory optimization."""
    
    @staticmethod
    def calculate_correlation_matrix(returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for assets.
        
        Args:
            returns_data: DataFrame with asset returns
        
        Returns:
            Correlation matrix
        """
        return returns_data.corr()
    
    @staticmethod
    def calculate_covariance_matrix(returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix for assets.
        
        Args:
            returns_data: DataFrame with asset returns
        
        Returns:
            Covariance matrix
        """
        return returns_data.cov()
    
    @staticmethod
    def efficient_frontier_point(
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate optimal portfolio for a target return using analytical solution.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            target_return: Target portfolio return
        
        Returns:
            Dict containing optimal weights and portfolio metrics
        """
        try:
            n_assets = len(expected_returns)
            
            # Create constraint matrices
            # Constraint 1: weights sum to 1
            # Constraint 2: expected return equals target
            A = np.vstack([
                np.ones(n_assets),
                expected_returns.values
            ])
            
            b = np.array([1.0, target_return])
            
            # Solve using matrix algebra
            # w = inv(Sigma) * A^T * inv(A * inv(Sigma) * A^T) * b
            inv_cov = np.linalg.inv(cov_matrix.values)
            temp1 = A @ inv_cov @ A.T
            temp2 = np.linalg.inv(temp1)
            optimal_weights = inv_cov @ A.T @ temp2 @ b
            
            # Create weights series
            weights = pd.Series(optimal_weights, index=expected_returns.index)
            
            # Calculate portfolio metrics
            portfolio_return = (weights * expected_returns).sum()
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'variance': portfolio_variance
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def minimum_variance_portfolio(
        cov_matrix: pd.DataFrame
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate minimum variance portfolio.
        
        Args:
            cov_matrix: Covariance matrix
        
        Returns:
            Dict containing optimal weights and portfolio metrics
        """
        try:
            n_assets = len(cov_matrix)
            
            # Minimum variance portfolio: w = inv(Sigma) * 1 / (1^T * inv(Sigma) * 1)
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones(n_assets)
            
            numerator = inv_cov @ ones
            denominator = ones.T @ inv_cov @ ones
            
            optimal_weights = numerator / denominator
            
            # Create weights series
            weights = pd.Series(optimal_weights, index=cov_matrix.index)
            
            # Calculate portfolio metrics
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            return {
                'weights': weights,
                'volatility': portfolio_volatility,
                'variance': portfolio_variance
            }
            
        except Exception as e:
            logger.error(f"Error calculating minimum variance portfolio: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def equal_weight_portfolio(asset_list: List[str]) -> Dict[str, float]:
        """
        Create equal-weight portfolio.
        
        Args:
            asset_list: List of asset symbols
        
        Returns:
            Dict with equal weights
        """
        if not asset_list:
            return {}
        
        weight = 1.0 / len(asset_list)
        return {asset: weight for asset in asset_list}

class RiskManager:
    """Risk management and position sizing utilities."""
    
    @staticmethod
    def kelly_criterion(
        win_probability: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_probability: Probability of winning trade
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive value)
        
        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_probability, q = 1 - p
        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at reasonable maximum (25% of capital)
        return max(0, min(kelly_fraction, 0.25))
    
    @staticmethod
    def position_sizing_fixed_fractional(
        account_value: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float
    ) -> int:
        """
        Calculate position size using fixed fractional risk.
        
        Args:
            account_value: Total account value
            risk_percentage: Percentage of account to risk (0-1)
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
        
        Returns:
            Number of shares to trade
        """
        if entry_price <= 0 or abs(entry_price - stop_loss_price) <= 0:
            return 0
        
        risk_amount = account_value * risk_percentage
        risk_per_share = abs(entry_price - stop_loss_price)
        
        shares = int(risk_amount / risk_per_share)
        
        return max(0, shares)
    
    @staticmethod
    def calculate_portfolio_risk_metrics(
        returns_data: pd.DataFrame,
        weights: Dict[str, float],
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            returns_data: DataFrame with asset returns
            weights: Portfolio weights
            risk_free_rate: Risk-free rate
        
        Returns:
            Dict with risk metrics
        """
        # Calculate portfolio returns
        portfolio_returns = PortfolioMetrics.calculate_portfolio_returns(returns_data, weights)
        
        if portfolio_returns.empty:
            return {'error': 'Unable to calculate portfolio returns'}
        
        # Calculate various risk metrics
        metrics = {}
        
        # Basic statistics
        metrics['annualized_return'] = portfolio_returns.mean() * 252
        metrics['annualized_volatility'] = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = PortfolioMetrics.calculate_sharpe_ratio(
            portfolio_returns, risk_free_rate
        )
        metrics['sortino_ratio'] = PortfolioMetrics.calculate_sortino_ratio(
            portfolio_returns, risk_free_rate
        )
        
        # Drawdown metrics
        drawdown_metrics = PortfolioMetrics.calculate_max_drawdown(portfolio_returns)
        metrics.update(drawdown_metrics)
        
        # Value at Risk
        metrics['var_5%'] = PortfolioMetrics.calculate_var(portfolio_returns, 0.05)
        metrics['cvar_5%'] = PortfolioMetrics.calculate_cvar(portfolio_returns, 0.05)
        
        # Annualize VaR metrics
        metrics['annual_var_5%'] = metrics['var_5%'] * np.sqrt(252)
        metrics['annual_cvar_5%'] = metrics['cvar_5%'] * np.sqrt(252)
        
        return metrics

class PortfolioAnalyzer:
    """Main portfolio analysis class."""
    
    def __init__(self):
        self.metrics = PortfolioMetrics()
        self.optimization = PortfolioOptimization()
        self.risk_manager = RiskManager()
    
    def analyze_portfolio(
        self,
        price_data: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        benchmark_symbol: str = 'SPY'
    ) -> Dict[str, Union[float, pd.Series, Dict]]:
        """
        Perform comprehensive portfolio analysis.
        
        Args:
            price_data: Dictionary of symbol -> price DataFrame
            weights: Portfolio weights
            benchmark_symbol: Benchmark for comparison
        
        Returns:
            Complete portfolio analysis
        """
        analysis = {}
        
        try:
            # Calculate returns for all assets
            returns_data = {}
            for symbol, df in price_data.items():
                if not df.empty and 'Close' in df.columns:
                    returns_data[symbol] = DataProcessor.calculate_returns(df['Close'])
            
            if not returns_data:
                return {'error': 'No valid price data provided'}
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            # Portfolio metrics
            portfolio_returns = self.metrics.calculate_portfolio_returns(returns_df, weights)
            analysis['portfolio_returns'] = portfolio_returns
            
            # Risk metrics
            risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(
                returns_df, weights
            )
            analysis['risk_metrics'] = risk_metrics
            
            # Correlation analysis
            analysis['correlation_matrix'] = self.optimization.calculate_correlation_matrix(returns_df)
            
            # Benchmark comparison
            if benchmark_symbol in returns_df.columns:
                benchmark_returns = returns_df[benchmark_symbol]
                analysis['portfolio_beta'] = self.metrics.calculate_beta(
                    portfolio_returns, benchmark_returns
                )
                
                # Alpha calculation
                risk_free_rate = 0.02 / 252  # Daily risk-free rate
                expected_return = risk_free_rate + analysis['portfolio_beta'] * (
                    benchmark_returns.mean() - risk_free_rate
                )
                analysis['alpha'] = portfolio_returns.mean() - expected_return
                analysis['annualized_alpha'] = analysis['alpha'] * 252
            
            # Portfolio composition
            analysis['weights'] = weights
            analysis['number_of_assets'] = len(weights)
            analysis['concentration'] = max(weights.values())  # Largest position
            
            logger.info("Portfolio analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {str(e)}")
            analysis['error'] = str(e)
        
        return analysis
    
    def suggest_rebalancing(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict[str, Union[bool, Dict[str, float]]]:
        """
        Suggest portfolio rebalancing if drift exceeds threshold.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Rebalancing threshold
        
        Returns:
            Rebalancing recommendation
        """
        needs_rebalancing = False
        adjustments = {}
        
        for symbol in target_weights:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights[symbol]
            
            difference = abs(current_weight - target_weight)
            if difference > threshold:
                needs_rebalancing = True
                adjustments[symbol] = target_weight - current_weight
        
        return {
            'needs_rebalancing': needs_rebalancing,
            'threshold_breached': threshold,
            'suggested_adjustments': adjustments
        }

# Create instances for easy importing
portfolio_metrics = PortfolioMetrics()
portfolio_optimization = PortfolioOptimization()
risk_manager = RiskManager()
portfolio_analyzer = PortfolioAnalyzer()
