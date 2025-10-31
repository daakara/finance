"""
Caching mechanisms for financial data to improve performance and respect API limits.
"""

from typing import Any, Dict, Optional, Union
import pandas as pd
import pickle
import hashlib
import time
from pathlib import Path
import logging
from functools import wraps

from config import config

logger = logging.getLogger(__name__)

class MemoryCache:
    """In-memory cache with TTL support."""
    
    def __init__(self, max_size: int = config.MAX_CACHE_SIZE, default_ttl: int = config.CACHE_TTL_SECONDS):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_string = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key not in self.cache:
            return None
        
        # Check if expired
        if time.time() - self.timestamps[key] > self.default_ttl:
            self.delete(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        # If cache is full, remove oldest item
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()
    
    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if not self.timestamps:
            return
        
        oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
        self.delete(oldest_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
        }

class FileCache:
    """File-based cache for persistent storage."""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = config.CACHE_TTL_SECONDS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_string = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from file cache if not expired."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        # Check if file is expired
        file_age = time.time() - file_path.stat().st_mtime
        if file_age > self.default_ttl:
            file_path.unlink(missing_ok=True)
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache file {file_path}: {str(e)}")
            file_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in file cache."""
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error saving cache file {file_path}: {str(e)}")
    
    def delete(self, key: str) -> None:
        """Delete item from file cache."""
        file_path = self._get_file_path(key)
        file_path.unlink(missing_ok=True)
    
    def clear(self) -> None:
        """Clear all cache files."""
        for file_path in self.cache_dir.glob("*.pkl"):
            file_path.unlink(missing_ok=True)
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files."""
        removed_count = 0
        current_time = time.time()
        
        for file_path in self.cache_dir.glob("*.pkl"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > self.default_ttl:
                file_path.unlink(missing_ok=True)
                removed_count += 1
        
        return removed_count

class CacheManager:
    """Main cache manager combining memory and file caching."""
    
    def __init__(self):
        self.memory_cache = MemoryCache()
        self.file_cache = FileCache()
    
    def cached_function(self, ttl: Optional[int] = None, use_file_cache: bool = False):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds (uses default if None)
            use_file_cache: Whether to use file cache in addition to memory cache
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Try memory cache first
                result = self.memory_cache.get(cache_key)
                if result is not None:
                    logger.debug(f"Memory cache hit for {func.__name__}")
                    return result
                
                # Try file cache if enabled
                if use_file_cache:
                    result = self.file_cache.get(cache_key)
                    if result is not None:
                        logger.debug(f"File cache hit for {func.__name__}")
                        # Store in memory cache for faster access
                        self.memory_cache.set(cache_key, result)
                        return result
                
                # Execute function and cache result
                logger.debug(f"Cache miss for {func.__name__}, executing function")
                result = func(*args, **kwargs)
                
                # Store in caches
                self.memory_cache.set(cache_key, result)
                if use_file_cache:
                    self.file_cache.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key for function call."""
        # Convert pandas DataFrames to string representation for hashing
        str_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                str_args.append(f"df_{hash(str(arg.values.tobytes()))}")
            else:
                str_args.append(str(arg))
        
        str_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame):
                str_kwargs[k] = f"df_{hash(str(v.values.tobytes()))}"
            else:
                str_kwargs[k] = str(v)
        
        key_string = f"{func_name}_{str(str_args)}_{str(sorted(str_kwargs.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear_all_caches(self) -> None:
        """Clear both memory and file caches."""
        self.memory_cache.clear()
        self.file_cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        memory_stats = self.memory_cache.get_stats()
        
        # Count file cache entries
        file_count = len(list(self.file_cache.cache_dir.glob("*.pkl")))
        
        return {
            'memory_cache': memory_stats,
            'file_cache': {
                'size': file_count,
                'cache_dir': str(self.file_cache.cache_dir)
            }
        }
    
    def cleanup_expired_files(self) -> int:
        """Clean up expired file cache entries."""
        return self.file_cache.cleanup_expired()

# Global cache manager instance
cache_manager = CacheManager()

# Convenient decorators
def cache_result(ttl: Optional[int] = None):
    """Decorator for caching function results in memory."""
    return cache_manager.cached_function(ttl=ttl, use_file_cache=False)

def cache_to_file(ttl: Optional[int] = None):
    """Decorator for caching function results to file."""
    return cache_manager.cached_function(ttl=ttl, use_file_cache=True)
