"""
SSL Configuration for yfinance and financial data fetching in corporate environments
"""

import os
import ssl
import certifi
import warnings
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

def configure_ssl_environment():
    """Configure SSL environment variables and settings for corporate networks."""
    
    # Set certificate paths
    cert_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['CURL_CA_BUNDLE'] = cert_path
    
    # Disable SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecurePlatformWarning)
        urllib3.disable_warnings(urllib3.exceptions.SNIMissingWarning)
    except AttributeError:
        pass  # Some versions may not have these warnings
    
    # Disable SSL verification environment overrides
    os.environ.pop('CURL_DISABLE_SSL_VERIFY', None)
    os.environ.pop('PYTHONHTTPSVERIFY', None)

def create_ssl_context():
    """Create a secure default SSL context."""
    context = ssl.create_default_context(cafile=certifi.where())
    return context

def create_session_with_retries():
    """Create a requests session with retry strategy and secure SSL verification."""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Secure SSL verification using certifi CA bundle
    session.verify = certifi.where()
    
    return session

# Configure SSL on import
configure_ssl_environment()
