@echo off
REM Startup script for ARX Terminal (FastAPI Backend + Next.js Frontend)

echo ==================================
echo 🚀 Starting ARX Terminal Platform
echo ==================================
echo.

REM Set SSL environment variables
set SSL_CERT_FILE=.venv\Lib\site-packages\certifi\cacert.pem
set REQUESTS_CA_BUNDLE=.venv\Lib\site-packages\certifi\cacert.pem

echo ✅ SSL certificates configured
echo.

REM 1. Start FastAPI Backend Microservice
echo ⚙️ Starting FastAPI Backend on port 8000...
start "ARX Backend API" .venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

timeout /t 3 /nobreak > nul

REM 2. Start Next.js Frontend Development Server
echo 🖥️ Starting Next.js 14 Frontend on port 3000...
start "ARX Frontend" cmd /c "cd frontend && npm run dev"

echo.
echo ==================================
echo ✅ ARX Terminal Services Launched!
echo ==================================
echo.
echo 🖥️ Frontend Dashboard: http://localhost:3000
echo ⚙️ FastAPI Swagger:   http://localhost:8000/docs
echo.
echo Both services are running in separate terminal windows.
echo ==================================
echo.

pause
