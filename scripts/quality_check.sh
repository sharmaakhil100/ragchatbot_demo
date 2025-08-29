#!/bin/bash

# Quality check script for the RAG Chatbot project
# Run all code quality checks

set -e

echo "🔍 Running code quality checks..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

echo "📝 Checking code formatting with Black..."
uv run black --check --diff backend/ main.py

echo "🔧 Checking import sorting with isort..."
uv run isort --check-only --diff backend/ main.py

echo "🐍 Running flake8 linting..."
uv run flake8 backend/ main.py

echo "✅ All quality checks passed!"