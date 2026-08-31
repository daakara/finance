#!/bin/bash
# Startup script for ARX Terminal (FastAPI Backend + Next.js Frontend)

echo "=================================="
echo "🚀 Starting ARX Terminal Platform"
echo "=================================="
echo ""

# Set SSL environment variables
export SSL_CERT_FILE=".venv/Lib/site-packages/certifi/cacert.pem"
export REQUESTS_CA_BUNDLE=".venv/Lib/site-packages/certifi/cacert.pem"

echo "✅ SSL certificates configured"
echo ""

# 1. Start FastAPI Backend Microservice
echo "⚙️ Starting FastAPI Backend on port 8000..."
.venv/Scripts/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

# 2. Start Next.js Frontend Development Server
echo "🖥️ Starting Next.js 14 Frontend on port 3000..."
cd frontend && npm run dev &
FRONTEND_PID=$!

sleep 3

echo ""
echo "=================================="
echo "✅ ARX Terminal Services Launched!"
echo "=================================="
echo ""
echo "🖥️ Frontend Dashboard: http://localhost:3000"
echo "⚙️ FastAPI Swagger:   http://localhost:8000/docs"
echo ""
echo "Process IDs:"
echo "  - Backend:  $BACKEND_PID"
echo "  - Frontend: $FRONTEND_PID"
echo ""
echo "To stop applications, press Ctrl+C"
echo "=================================="
echo ""

# Wait for user interrupt
wait
