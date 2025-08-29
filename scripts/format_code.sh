#!/bin/bash

# Auto-format code for the RAG Chatbot project
# Applies Black formatting and isort import organization

set -e

echo "🎨 Formatting code..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

echo "📝 Formatting code with Black..."
uv run black backend/ main.py

echo "🔧 Organizing imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"