"""
Enhanced Error Handling Utilities
Provides robust error handling with retry logic and user-friendly messages
"""

import time
import logging
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors"""
    pass


class NetworkError(APIError):
    """Network connectivity errors"""
    pass


class RateLimitError(APIError):
    """API rate limit exceeded"""
    pass


class DataValidationError(Exception):
    """Data validation errors"""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
    
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def fetch_data(ticker):
            return api.get_data(ticker)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                                 f"retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    delay *= backoff_factor
            
            raise last_exception
        
        return wrapper
    return decorator


def safe_api_call(
    func: Callable,
    *args,
    default: Any = None,
    error_message: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Safely execute an API call with error handling
    
    Args:
        func: Function to call
        *args: Positional arguments for function
        default: Default value to return on error
        error_message: Custom error message
        **kwargs: Keyword arguments for function
    
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Error in {func.__name__}"
        logger.error(f"{msg}: {e}")
        return default


def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format
    
    Args:
        ticker: Ticker symbol to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Remove common suffixes
    clean = ticker.upper().strip()
    
    # Basic validation: alphanumeric plus dash and dot
    return clean.replace('-', '').replace('.', '').isalnum() and len(clean) <= 10


def format_user_error(error: Exception, context: str = "") -> str:
    """
    Format error message for end users
    
    Args:
        error: The exception that occurred
        context: Additional context about where error occurred
    
    Returns:
        User-friendly error message
    """
    error_str = str(error).lower()
    
    # Network/SSL errors
    if any(keyword in error_str for keyword in ['ssl', 'certificate', 'connection', 'timeout']):
        return (
            f"Network connection issue{': ' + context if context else ''}\n"
            "Possible solutions:\n"
            "  - Check your internet connection\n"
            "  - Verify firewall/proxy settings\n"
            "  - Try again in a few moments\n"
            "  - Contact IT if in corporate environment"
        )
    
    # Rate limiting
    if 'rate limit' in error_str or '429' in error_str:
        return (
            f"API rate limit exceeded{': ' + context if context else ''}\n"
            "Possible solutions:\n"
            "  - Wait a few minutes before retrying\n"
            "  - Reduce the number of symbols screened\n"
            "  - Data is cached for 5 minutes to reduce API calls"
        )
    
    # Invalid ticker
    if 'invalid' in error_str or 'not found' in error_str:
        return (
            f"Invalid ticker or data not available{': ' + context if context else ''}\n"
            "Possible solutions:\n"
            "  - Verify the ticker symbol is correct\n"
            "  - Check if the asset is still trading\n"
            "  - Try a different ticker"
        )
    
    # Generic error
    return (
        f"An error occurred{': ' + context if context else ''}\n"
        f"Error details: {str(error)[:100]}"
    )


# Example usage
if __name__ == "__main__":
    # Test retry decorator
    @retry_with_backoff(max_retries=3, initial_delay=0.1)
    def flaky_function():
        import random
        if random.random() < 0.7:
            raise NetworkError("Connection failed")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test ticker validation
    print(f"AAPL valid: {validate_ticker('AAPL')}")
    print(f"Invalid ticker: {validate_ticker('INVALID@@@')}")
    print(f"Empty ticker: {validate_ticker('')}")
