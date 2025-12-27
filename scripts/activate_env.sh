#!/bin/bash
# Quick script to activate the virtual environment

echo "🐍 Activating Portable Alpha Extension Model virtual environment..."
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python version: $(python --version)"
echo "📍 Python location: $(which python)"
echo "🧪 Package location: $(python -c 'import pa_core; print(pa_core.__file__)')"

echo ""
echo "💡 You can now run:"
echo "   python -m pa_core --help"
echo "   python -m pytest tests/"
echo "   streamlit run dashboard/app.py"
echo ""
