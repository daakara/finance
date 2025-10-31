#!/bin/bash
# Startup script for both Financial Platform applications

echo "=================================="
echo "🚀 Starting Financial Applications"
echo "=================================="
echo ""

# Set SSL environment variables
export SSL_CERT_FILE=".venv/Lib/site-packages/certifi/cacert.pem"
export REQUESTS_CA_BUNDLE=".venv/Lib/site-packages/certifi/cacert.pem"

echo "✅ SSL certificates configured"
echo ""

# Start Main Financial Platform
echo "🎯 Starting Main Financial Platform on port 8501..."
.venv/Scripts/python.exe -m streamlit run app.py --server.port 8501 &
MAIN_APP_PID=$!

sleep 2

# Start Hidden Gems Scanner
echo "💎 Starting Hidden Gems Scanner on port 8502..."
.venv/Scripts/python.exe -m streamlit run analyst_dashboard/visualizers/gem_dashboard.py --server.port 8502 &
GEM_APP_PID=$!

sleep 3

echo ""
echo "=================================="
echo "✅ Both Applications Started!"
echo "=================================="
echo ""
echo "📊 Main Financial Platform:  http://localhost:8501"
echo "💎 Hidden Gems Scanner:      http://localhost:8502"
echo ""
echo "Process IDs:"
echo "  - Main App: $MAIN_APP_PID"
echo "  - Gem App:  $GEM_APP_PID"
echo ""
echo "To stop applications, press Ctrl+C"
echo "=================================="
echo ""

# Wait for user interrupt
wait
