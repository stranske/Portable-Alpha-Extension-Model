#!/bin/bash
# Codespaces post-create setup script

set -e

echo "🚀 Setting up Portable Alpha Extension Model development environment..."

# Create and activate virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Make scripts executable
chmod +x dev.sh
chmod +x setup.sh
chmod +x setup_deps.sh

# Install pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
fi

# Create sample configuration if it doesn't exist
if [ ! -f "parameters.csv" ] && [ -f "config/parameters_template.csv" ]; then
    echo "📄 Creating sample configuration..."
    cp config/parameters_template.csv parameters.csv
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  ./dev.sh demo      - Run demo with sample data"
echo "  ./dev.sh test      - Run test suite"
echo "  ./dev.sh dashboard - Start Streamlit dashboard"
echo "  ./dev.sh ci        - Run full CI pipeline"
echo ""
echo "📊 Dashboard will be available at:"
echo "  http://localhost:8501 (when running ./dev.sh dashboard)"
echo ""
