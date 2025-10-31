# Contributing to Financial Market Analysis Platform

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- **Clear title** describing the issue
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Screenshots or logs** if applicable

### Suggesting Features

Feature requests are welcome! Please:
- Check existing issues to avoid duplicates
- Clearly describe the feature and its benefits
- Explain use cases
- Be open to discussion and feedback

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding standards** (see below)
3. **Write tests** for new features
4. **Update documentation** as needed
5. **Ensure all tests pass** before submitting
6. **Write clear commit messages**

## 📋 Development Setup

### Prerequisites
- Python 3.12 or higher
- Git
- Virtual environment tool

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/yourusername/finance.git
cd finance

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov pytest-mock black flake8 mypy
```

## 🎨 Coding Standards

### Python Style Guide

Follow **PEP 8** with these specifics:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Docstrings**: Use Google style docstrings

### Example:

```python
"""
Module for data fetching operations.

This module provides classes and functions for fetching real-time market data
from various sources including Yahoo Finance and cryptocurrency exchanges.
"""

import os
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import config


class DataFetcher:
    """Fetches market data from external APIs.
    
    This class implements data fetching with error handling, caching,
    and retry logic for reliable data retrieval.
    
    Attributes:
        cache_enabled: Whether caching is enabled
        retry_count: Number of retry attempts for failed requests
    """
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize the data fetcher.
        
        Args:
            cache_enabled: Enable or disable caching
        """
        self.cache_enabled = cache_enabled
        self.retry_count = 3
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "1y"
    ) -> pd.DataFrame:
        """Fetch historical stock data.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Time period for data ('1d', '5d', '1mo', '1y', etc.)
        
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ConnectionError: If unable to fetch data after retries
            ValueError: If symbol is invalid
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        # Implementation here...
        pass
```

### Formatting Tools

Use these tools to maintain code quality:

```bash
# Auto-format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

## 🧪 Testing

### Writing Tests

- **Location**: Place tests in `tests/` directory
- **Naming**: Test files should start with `test_`
- **Structure**: Use descriptive test function names

Example test:

```python
import pytest
from data.fetchers import DataFetcher


class TestDataFetcher:
    """Test suite for DataFetcher class."""
    
    def test_initialization(self):
        """Test that DataFetcher initializes correctly."""
        fetcher = DataFetcher()
        assert fetcher.cache_enabled is True
        assert fetcher.retry_count == 3
    
    def test_fetch_stock_data_valid_symbol(self):
        """Test fetching data for valid stock symbol."""
        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data("AAPL", "1mo")
        
        assert not data.empty
        assert "Close" in data.columns
        assert len(data) > 0
    
    def test_fetch_stock_data_invalid_symbol(self):
        """Test that invalid symbol raises ValueError."""
        fetcher = DataFetcher()
        
        with pytest.raises(ValueError):
            fetcher.fetch_stock_data("", "1mo")
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_data_fetchers.py -v

# Run specific test
pytest tests/test_data_fetchers.py::TestDataFetcher::test_initialization -v
```

### Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Critical modules**: 90% coverage required
- **Focus on**: Edge cases, error handling, validation

## 🔒 Data Integrity Policy

### Critical Rule: LIVE DATA ONLY

**Never implement or commit code that generates sample/fake market data.**

This project follows a **strict live-data-only policy**:

✅ **DO:**
- Fetch data from real market sources (Yahoo Finance, exchanges)
- Raise clear exceptions when data is unavailable
- Show transparent error messages to users
- Use mocking in tests only

❌ **DON'T:**
- Generate fake stock prices
- Create sample market data for fallbacks
- Return dummy data on errors
- Commit `_generate_sample_*` methods

### Rationale

Users trust financial applications with important decisions. Showing fake data, even with warnings, can:
- Lead to poor investment decisions
- Damage user trust
- Create legal liability
- Undermine platform credibility

See [SAMPLE_DATA_REMOVAL_REPORT.md](SAMPLE_DATA_REMOVAL_REPORT.md) for details.

## 📝 Documentation

### Code Documentation

- **All public functions** must have docstrings
- **All classes** must have docstrings
- **Complex logic** should have inline comments
- **Type hints** required for function parameters and returns

### README Updates

Update README.md when adding:
- New features
- New dependencies
- Configuration changes
- Breaking changes

### Documentation Files

Maintain these documentation files:
- `README.md` - Project overview
- `CONTRIBUTING.md` - This file
- `CHANGELOG.md` - Version history
- Technical guides in `docs/`

## 🔄 Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Urgent fixes
- `docs/description` - Documentation only
- `refactor/description` - Code refactoring

### Commit Messages

Follow the Conventional Commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(dashboard): add real-time price updates

Implement WebSocket connection for live price streaming.
Includes connection management and auto-reconnect logic.

Closes #123
```

```
fix(data): handle SSL certificate errors

Add certifi configuration to resolve corporate proxy issues.
See SSL_TROUBLESHOOTING_GUIDE.md for details.

Fixes #456
```

### Pull Request Process

1. **Create a descriptive PR title**
2. **Fill out the PR template** (if available)
3. **Link related issues** (e.g., "Closes #123")
4. **Add labels** (e.g., `enhancement`, `bug`, `documentation`)
5. **Request review** from maintainers
6. **Address feedback** promptly
7. **Keep PR focused** - one feature/fix per PR

## 🚫 What NOT to Contribute

Please don't submit PRs for:

- Sample/fake data generation
- Commented-out code
- Temporary debugging code
- Personal configuration files
- IDE-specific settings (unless general benefit)
- Breaking changes without discussion
- Features without tests
- Code without documentation

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an Issue
- **Feature Requests**: Create an Issue with [Feature Request] prefix
- **Security Issues**: Email privately (see SECURITY.md)

## 🏆 Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- Project documentation

Thank you for helping make this project better! 🎉
