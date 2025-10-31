@echo off
REM Startup script for both Financial Platform applications

echo ==================================
echo 🚀 Starting Financial Applications
echo ==================================
echo.

REM Set SSL environment variables
set SSL_CERT_FILE=.venv\Lib\site-packages\certifi\cacert.pem
set REQUESTS_CA_BUNDLE=.venv\Lib\site-packages\certifi\cacert.pem

echo ✅ SSL certificates configured
echo.

REM Start Main Financial Platform
echo 🎯 Starting Main Financial Platform on port 8501...
start "Main Financial Platform" .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501

timeout /t 3 /nobreak > nul

REM Start Hidden Gems Scanner
echo 💎 Starting Hidden Gems Scanner on port 8502...
start "Hidden Gems Scanner" .venv\Scripts\python.exe -m streamlit run analyst_dashboard\visualizers\gem_dashboard.py --server.port 8502

timeout /t 3 /nobreak > nul

echo.
echo ==================================
echo ✅ Both Applications Started!
echo ==================================
echo.
echo 📊 Main Financial Platform:  http://localhost:8501
echo 💎 Hidden Gems Scanner:      http://localhost:8502
echo.
echo Both applications are running in separate windows
echo Close the terminal windows to stop the applications
echo ==================================
echo.

pause
