"""
Utility functions for the financial analysis platform.
General-purpose helper functions used across modules.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataHelpers:
    """Data manipulation and utility functions."""
    
    @staticmethod
    def format_currency(amount: float, currency: str = "USD") -> str:
        """
        Format number as currency string.
        
        Args:
            amount: Amount to format
            currency: Currency code
        
        Returns:
            Formatted currency string
        """
        if pd.isna(amount) or amount is None:
            return "N/A"
        
        if currency == "USD":
            if abs(amount) >= 1e12:
                return f"${amount/1e12:.2f}T"
            elif abs(amount) >= 1e9:
                return f"${amount/1e9:.2f}B"
            elif abs(amount) >= 1e6:
                return f"${amount/1e6:.2f}M"
            elif abs(amount) >= 1e3:
                return f"${amount/1e3:.2f}K"
            else:
                return f"${amount:.2f}"
        else:
            return f"{amount:.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 2) -> str:
        """
        Format number as percentage string.
        
        Args:
            value: Value to format (0.05 = 5%)
            decimal_places: Number of decimal places
        
        Returns:
            Formatted percentage string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def format_large_number(number: float) -> str:
        """
        Format large numbers with appropriate suffixes.
        
        Args:
            number: Number to format
        
        Returns:
            Formatted number string
        """
        if pd.isna(number) or number is None:
            return "N/A"
        
        if abs(number) >= 1e12:
            return f"{number/1e12:.2f}T"
        elif abs(number) >= 1e9:
            return f"{number/1e9:.2f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.2f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.2f}K"
        else:
            return f"{number:.2f}"
    
    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """
        Clean and validate stock symbol.
        
        Args:
            symbol: Raw symbol string
        
        Returns:
            Cleaned symbol
        """
        if not symbol:
            return ""
        
        # Remove whitespace and convert to uppercase
        cleaned = symbol.strip().upper()
        
        # Remove invalid characters (keep only letters, numbers, dots, hyphens)
        cleaned = re.sub(r'[^A-Z0-9.-]', '', cleaned)
        
        return cleaned
    
    @staticmethod
    def validate_date_range(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Tuple[datetime, datetime]:
        """
        Validate and convert date range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Tuple of validated datetime objects
        """
        # Convert strings to datetime if necessary
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Ensure start is before end
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Ensure dates are not in the future
        today = datetime.now()
        if end_date > today:
            end_date = today
        
        return start_date, end_date
    
    @staticmethod
    def calculate_business_days(
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """
        Calculate number of business days between dates.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Number of business days
        """
        return pd.bdate_range(start_date, end_date).size
    
    @staticmethod
    def get_market_calendar_info(date: datetime) -> Dict[str, bool]:
        """
        Get market calendar information for a given date.
        
        Args:
            date: Date to check
        
        Returns:
            Dict with market status information
        """
        # Simplified market calendar (US markets)
        weekday = date.weekday()  # Monday = 0, Sunday = 6
        
        # Basic holidays (simplified)
        holidays = [
            # New Year's Day
            datetime(date.year, 1, 1),
            # Independence Day
            datetime(date.year, 7, 4),
            # Christmas
            datetime(date.year, 12, 25),
        ]
        
        is_weekend = weekday >= 5  # Saturday or Sunday
        is_holiday = date.date() in [h.date() for h in holidays]
        is_trading_day = not (is_weekend or is_holiday)
        
        return {
            'is_trading_day': is_trading_day,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'weekday': weekday
        }

class MathHelpers:
    """Mathematical utility functions."""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero
        
        Returns:
            Division result or default
        """
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    
    @staticmethod
    def calculate_compound_return(
        start_value: float,
        end_value: float,
        periods: int
    ) -> float:
        """
        Calculate compound annual growth rate.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            periods: Number of periods
        
        Returns:
            Compound return rate
        """
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return 0.0
        
        return (end_value / start_value) ** (1 / periods) - 1
    
    @staticmethod
    def interpolate_missing_values(
        series: pd.Series,
        method: str = 'linear'
    ) -> pd.Series:
        """
        Interpolate missing values in a series.
        
        Args:
            series: Series with missing values
            method: Interpolation method
        
        Returns:
            Series with interpolated values
        """
        if series.empty:
            return series
        
        return series.interpolate(method=method)
    
    @staticmethod
    def calculate_z_score(values: pd.Series) -> pd.Series:
        """
        Calculate z-scores for a series.
        
        Args:
            values: Value series
        
        Returns:
            Z-score series
        """
        if values.empty or values.std() == 0:
            return pd.Series(0, index=values.index)
        
        return (values - values.mean()) / values.std()
    
    @staticmethod
    def winsorize_outliers(
        series: pd.Series,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ) -> pd.Series:
        """
        Winsorize outliers in a series.
        
        Args:
            series: Input series
            lower_percentile: Lower percentile threshold
            upper_percentile: Upper percentile threshold
        
        Returns:
            Winsorized series
        """
        if series.empty:
            return series
        
        lower_bound = series.quantile(lower_percentile)
        upper_bound = series.quantile(upper_percentile)
        
        return series.clip(lower=lower_bound, upper=upper_bound)

class ValidationHelpers:
    """Input validation functions."""
    
    @staticmethod
    def validate_symbol_list(symbols: List[str]) -> List[str]:
        """
        Validate and clean a list of stock symbols.
        
        Args:
            symbols: List of symbols to validate
        
        Returns:
            List of valid symbols
        """
        if not symbols:
            return []
        
        valid_symbols = []
        for symbol in symbols:
            cleaned = DataHelpers.clean_symbol(symbol)
            if cleaned and len(cleaned) <= 10:  # Reasonable symbol length
                valid_symbols.append(cleaned)
            else:
                logger.warning(f"Invalid symbol skipped: {symbol}")
        
        return list(set(valid_symbols))  # Remove duplicates
    
    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate portfolio weights.
        
        Args:
            weights: Dictionary of symbol -> weight
        
        Returns:
            Validated weights dictionary
        """
        if not weights:
            return {}
        
        # Remove invalid weights
        valid_weights = {}
        for symbol, weight in weights.items():
            if isinstance(weight, (int, float)) and 0 <= weight <= 1:
                valid_weights[symbol] = float(weight)
            else:
                logger.warning(f"Invalid weight for {symbol}: {weight}")
        
        # Check if weights sum to approximately 1
        total_weight = sum(valid_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, consider normalizing")
        
        return valid_weights
    
    @staticmethod
    def validate_numeric_input(
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default: Optional[float] = None
    ) -> Optional[float]:
        """
        Validate numeric input with optional bounds.
        
        Args:
            value: Input value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            default: Default value if validation fails
        
        Returns:
            Validated numeric value or default
        """
        try:
            numeric_value = float(value)
            
            if min_value is not None and numeric_value < min_value:
                return default
            if max_value is not None and numeric_value > max_value:
                return default
            
            return numeric_value
            
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def validate_period_string(period: str) -> bool:
        """
        Validate period string format.
        
        Args:
            period: Period string (e.g., '1y', '6mo', '5d')
        
        Returns:
            True if valid period format
        """
        valid_periods = {
            '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        }
        return period in valid_periods

class CacheHelpers:
    """Cache management utilities."""
    
    @staticmethod
    def generate_cache_key(*args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Cache key string
        """
        import hashlib
        
        # Convert arguments to string
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Create hash
        key_string = f"{args_str}_{kwargs_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def is_cache_valid(timestamp: datetime, ttl_seconds: int) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            timestamp: Cache timestamp
            ttl_seconds: Time to live in seconds
        
        Returns:
            True if cache is still valid
        """
        age = (datetime.now() - timestamp).total_seconds()
        return age <= ttl_seconds

class ErrorHandlers:
    """Error handling utilities."""
    
    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Function arguments
            default: Default return value on error
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            return default
    
    @staticmethod
    def log_api_error(api_name: str, error: Exception, symbol: str = None):
        """
        Log API-related errors with context.
        
        Args:
            api_name: Name of the API
            error: Exception object
            symbol: Symbol being processed (if applicable)
        """
        symbol_info = f" for symbol {symbol}" if symbol else ""
        logger.error(f"{api_name} API error{symbol_info}: {str(error)}")
    
    @staticmethod
    def create_error_response(message: str, error_type: str = "general") -> Dict[str, str]:
        """
        Create standardized error response.
        
        Args:
            message: Error message
            error_type: Type of error
        
        Returns:
            Error response dictionary
        """
        return {
            'error': True,
            'error_type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

# Create helper instances for easy importing
data_helpers = DataHelpers()
math_helpers = MathHelpers()
validation_helpers = ValidationHelpers()
cache_helpers = CacheHelpers()
error_handlers = ErrorHandlers()
