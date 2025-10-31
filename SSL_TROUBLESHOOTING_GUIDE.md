# SSL Certificate Issues - Troubleshooting Guide

## Issue Summary

You encountered SSL certificate errors when trying to fetch market data from Yahoo Finance:
```
Failed to perform, curl: (60) SSL certificate problem: unable to get local issuer certificate
```

This is a **common issue** in corporate environments, behind proxies, or with certain network configurations.

---

## ✅ Solutions Implemented

### 1. **SSL Certificate Fix at Application Startup**
Both `app.py` and `gem_dashboard.py` now include SSL certificate configuration at the very top:

```python
# SSL Certificate Fix - Must be at the very top
import ssl
import os
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception as e:
    print(f"Warning: Could not configure SSL certificates: {e}")
```

### 2. **Startup Scripts Created**

**Windows Users:** `start_apps.bat`
- Sets SSL environment variables
- Starts both applications automatically
- Opens in separate windows

**Linux/Mac Users:** `start_apps.sh`
- Sets SSL environment variables
- Starts both applications in background
- Usage: `bash start_apps.sh`

### 3. **Certifi Package Installed**
The `certifi` package provides Mozilla's trusted CA certificate bundle for Python.

---

## 🔧 Additional Troubleshooting Steps

If SSL errors persist after restarting the applications, try these solutions:

### Option 1: Corporate Proxy Configuration (Most Common)

If you're behind a corporate proxy, you need to configure it:

```bash
# Set proxy environment variables (replace with your proxy details)
export HTTP_PROXY="http://your-proxy:port"
export HTTPS_PROXY="http://your-proxy:port"
export NO_PROXY="localhost,127.0.0.1"
```

For Windows:
```cmd
set HTTP_PROXY=http://your-proxy:port
set HTTPS_PROXY=http://your-proxy:port
```

### Option 2: Download CA Certificates

Download and install updated CA certificates:

```bash
# Using pip
.venv/Scripts/python.exe -m pip install --upgrade certifi

# Manual certificate update
.venv/Scripts/python.exe -m certifi
```

### Option 3: Disable SSL Verification (TESTING ONLY)

⚠️ **Not recommended for production** - use only for testing:

Add to your Python code:
```python
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
```

### Option 4: System Certificate Store (Windows)

Install certificates to Windows certificate store:

1. Download certificate from https://curl.se/ca/cacert.pem
2. Right-click → Install Certificate
3. Select "Trusted Root Certification Authorities"
4. Complete the wizard

### Option 5: Alternative Network

If none of the above work:
- Try a different network (home WiFi vs. corporate)
- Use a VPN to bypass proxy restrictions
- Use mobile hotspot temporarily
- Contact IT department for proxy configuration

---

## 🎯 What Changed With Live-Data-Only Policy

**Before (with sample data):**
```
SSL Error → Application shows fake data with orange warning
```

**After (live-data-only):**
```
SSL Error → Application shows clear error message, no fake data
```

**This is the CORRECT behavior!** The transparent error helps you:
1. Identify the real problem (SSL certificates)
2. Fix the underlying issue
3. Trust that displayed data is always authentic

---

## 📊 Testing Your Fix

After implementing solutions, test with these steps:

### 1. Restart Applications
```bash
# Kill existing processes
# Windows: Close terminal windows or use Task Manager
# Linux/Mac: pkill -f streamlit

# Start with new configuration
.\start_apps.bat  # Windows
bash start_apps.sh  # Linux/Mac
```

### 2. Test Valid Symbols
Try fetching data for: AAPL, MSFT, GOOGL, AMZN, TSLA

**Expected Results:**
- ✅ Real market data displays
- ✅ Charts render correctly
- ✅ Company information loads
- ✅ No error messages

### 3. Test Invalid Symbols
Try fetching data for: INVALID123, FAKE999

**Expected Results:**
- ❌ Clear error message displayed
- ❌ No fake/sample data shown
- ✅ Error suggests checking symbol or network

### 4. Test Network Disconnection
Disconnect from internet and try fetching data

**Expected Results:**
- ❌ Connection error displayed
- ❌ No fake/sample data shown
- ✅ Error suggests checking network connection

---

## 🔍 Verifying SSL Configuration

Run this diagnostic script to check your SSL setup:

```python
# Save as check_ssl.py
import ssl
import certifi
import requests
import os

print("SSL Configuration Check")
print("=" * 60)
print(f"Certifi CA Bundle: {certifi.where()}")
print(f"SSL_CERT_FILE: {os.environ.get('SSL_CERT_FILE', 'Not set')}")
print(f"REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', 'Not set')}")
print(f"Default SSL Context: {ssl._create_default_https_context}")
print("=" * 60)

# Test connection
try:
    response = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/AAPL')
    print(f"✅ Connection test: SUCCESS (status {response.status_code})")
except Exception as e:
    print(f"❌ Connection test: FAILED - {str(e)}")
```

Run it:
```bash
.venv/Scripts/python.exe check_ssl.py
```

---

## 📝 Common Error Messages Explained

### 1. "SSL certificate problem: unable to get local issuer certificate"
**Cause:** Python can't verify the SSL certificate chain
**Solution:** Use certifi package and set environment variables (already done)

### 2. "CERTIFICATE_VERIFY_FAILED"
**Cause:** Similar to #1, certificate verification failing
**Solution:** Same as #1, or temporarily disable verification for testing

### 3. "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:xxx)"
**Cause:** OpenSSL certificate verification failure
**Solution:** Update certifi, check system certificates, configure proxy

### 4. "Failed to perform, curl: (60)"
**Cause:** curl library (used by yfinance) can't verify certificates
**Solution:** Set CURL_CA_BUNDLE environment variable to certifi path

---

## 🚀 Best Practices for Production

1. **Always use proper SSL verification** in production
2. **Never commit SSL-disabled code** to version control
3. **Configure proxy settings** properly if in corporate environment
4. **Keep certifi updated**: `pip install --upgrade certifi`
5. **Monitor SSL errors** in logs for security issues
6. **Use environment variables** for proxy configuration
7. **Document your network requirements** for users

---

## 📞 Getting Help

If you continue to experience SSL issues:

1. **Check with your IT Department:**
   - Ask for proxy configuration details
   - Request firewall exceptions for finance APIs
   - Confirm if SSL inspection is enabled

2. **Yahoo Finance Status:**
   - Check https://finance.yahoo.com in browser
   - Verify the API is accessible from your network
   - Try alternative data sources if Yahoo is blocked

3. **Network Configuration:**
   - Test from different network (home vs. work)
   - Check if VPN is required/causing issues
   - Verify DNS resolution works properly

4. **Python Environment:**
   - Ensure you're using the correct virtual environment
   - Check Python version (3.8+ recommended)
   - Verify all dependencies are installed

---

## 📚 Additional Resources

- **Certifi Documentation:** https://github.com/certifi/python-certifi
- **yfinance SSL Issues:** https://github.com/ranaroussi/yfinance/issues
- **Requests SSL Verification:** https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
- **Corporate Proxy Guide:** https://stackoverflow.com/questions/28521535/requests-how-to-disable-ssl-verification

---

## ✅ Quick Command Reference

```bash
# Check Python version
.venv/Scripts/python.exe --version

# Update certifi
.venv/Scripts/python.exe -m pip install --upgrade certifi

# Find certifi path
.venv/Scripts/python.exe -c "import certifi; print(certifi.where())"

# Start applications
.\start_apps.bat  # Windows
bash start_apps.sh  # Linux/Mac

# Set proxy (if needed)
set HTTP_PROXY=http://proxy:port  # Windows
export HTTP_PROXY=http://proxy:port  # Linux/Mac

# Test connection
.venv/Scripts/python.exe check_ssl.py
```

---

**Last Updated:** October 31, 2025  
**Related Files:**
- `app.py` - Main Financial Platform (SSL fix added)
- `analyst_dashboard/visualizers/gem_dashboard.py` - Hidden Gems Scanner (SSL fix added)
- `fix_ssl_certificates.py` - SSL configuration utility
- `start_apps.bat` / `start_apps.sh` - Startup scripts with SSL config
- `SAMPLE_DATA_REMOVAL_REPORT.md` - Live-data-only implementation details
