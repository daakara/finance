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
    
    # Set environment variables for curl_cffi
    os.environ['CURL_DISABLE_SSL_VERIFY'] = '1'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # Suppress all SSL-related warnings
    warnings.filterwarnings('ignore', message='.*SSL.*')
    warnings.filterwarnings('ignore', message='.*certificate.*')
    warnings.filterwarnings('ignore', message='.*curl_cffi.*')
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)

def create_ssl_context():
    """Create a permissive SSL context for corporate environments."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

def create_session_with_retries():
    """Create a requests session with retry strategy and SSL configuration."""
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
    
    # Disable SSL verification
    session.verify = False
    
    return session

# Configure SSL on import
configure_ssl_environment()
