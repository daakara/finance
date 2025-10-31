@echo off
REM Initialize Git repository and prepare for GitHub publication

echo ==================================
echo 📦 GitHub Publication Setup
echo ==================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)

echo ✅ Git is installed
echo.

REM Initialize git repository if not already initialized
if not exist ".git" (
    echo 🔧 Initializing Git repository...
    git init
    echo ✅ Git repository initialized
) else (
    echo ✅ Git repository already initialized
)
echo.

REM Check if remote exists
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Remote 'origin' already configured:
    git remote -v
) else (
    echo ⚠️  No remote configured. Please add your GitHub repository:
    echo    git remote add origin https://github.com/yourusername/finance.git
)
echo.

REM Set main branch
echo 🔧 Setting main branch...
git branch -M main
echo ✅ Main branch set
echo.

REM Show what will be committed
echo 📋 Files to be committed:
git add -n .
echo.

REM Stage all files
echo 🔧 Staging files...
git add .
echo ✅ Files staged
echo.

REM Show status
echo 📊 Repository Status:
git status
echo.

echo ==================================
echo ✅ Setup Complete!
echo ==================================
echo.
echo 📋 Next Steps:
echo.
echo 1. Create a new repository on GitHub
echo 2. Add remote (if not done):
echo    git remote add origin https://github.com/yourusername/finance.git
echo.
echo 3. Commit your changes:
echo    git commit -m "Initial commit: Financial Platform v1.0.0"
echo.
echo 4. Push to GitHub:
echo    git push -u origin main
echo.
echo See GITHUB_PUBLICATION_GUIDE.md for detailed instructions
echo ==================================
echo.

pause
