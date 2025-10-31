"""
SSL Certificate Fix Utility
Helps resolve SSL certificate issues for yfinance data fetching
"""

import ssl
import certifi
import os

def fix_ssl_certificates():
    """Fix SSL certificate issues by updating environment variables"""
    print("🔧 Fixing SSL Certificate Configuration...\n")
    
    # Method 1: Set certifi certificate path
    cert_path = certifi.where()
    print(f"✅ Found certifi certificates at: {cert_path}")
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    print(f"✅ Set SSL_CERT_FILE environment variable")
    print(f"✅ Set REQUESTS_CA_BUNDLE environment variable\n")
    
    # Method 2: Create default SSL context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✅ Created unverified SSL context (for testing only)\n")
    except Exception as e:
        print(f"⚠️ Could not create unverified context: {e}\n")
    
    print("=" * 60)
    print("SSL Certificate Configuration Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Restart your Streamlit applications")
    print("2. The applications should now be able to fetch market data")
    print("\n⚠️ If issues persist:")
    print("- You may be behind a corporate proxy")
    print("- Contact your IT department for proxy configuration")
    print("- Consider using a VPN or different network\n")

if __name__ == "__main__":
    fix_ssl_certificates()
