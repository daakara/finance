#!/bin/bash
# Initialize Git repository and prepare for GitHub publication

echo "=================================="
echo "📦 GitHub Publication Setup"
echo "=================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already initialized"
fi
echo ""

# Check if remote exists
if git remote get-url origin &> /dev/null; then
    echo "✅ Remote 'origin' already configured:"
    git remote -v
else
    echo "⚠️  No remote configured. Please add your GitHub repository:"
    echo "   git remote add origin https://github.com/yourusername/finance.git"
fi
echo ""

# Set main branch
echo "🔧 Setting main branch..."
git branch -M main
echo "✅ Main branch set"
echo ""

# Show what will be committed
echo "📋 Files to be committed:"
git add -n .
echo ""

# Stage all files
echo "🔧 Staging files..."
git add .
echo "✅ Files staged"
echo ""

# Show status
echo "📊 Repository Status:"
git status
echo ""

echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Create a new repository on GitHub"
echo "2. Add remote (if not done):"
echo "   git remote add origin https://github.com/yourusername/finance.git"
echo ""
echo "3. Commit your changes:"
echo "   git commit -m 'Initial commit: Financial Platform v1.0.0'"
echo ""
echo "4. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "See GITHUB_PUBLICATION_GUIDE.md for detailed instructions"
echo "=================================="
