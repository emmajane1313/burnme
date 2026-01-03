#!/bin/bash

# Publish script for burnmewhileimhot to TestPyPI
# This script uploads the built distribution to TestPyPI

set -e  # Exit on any error

echo "🚀 Publishing burnmewhileimhot to TestPyPI..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ Error: dist directory not found. Please run build.sh first."
    exit 1
fi

# Check if distribution files exist
if ! ls dist/*.whl 1> /dev/null 2>&1; then
    echo "❌ Error: Distribution files not found. Please run build.sh first."
    exit 1
fi

# Check if credentials are set
if [ -z "$TWINE_USERNAME" ] || [ -z "$TWINE_PASSWORD" ]; then
    echo "❌ Error: TWINE_USERNAME and TWINE_PASSWORD environment variables must be set."
    echo "Please run:"
    echo "  export TWINE_USERNAME=__token__"
    echo "  export TWINE_PASSWORD=your_testpypi_token_here"
    exit 1
fi

echo "📦 Uploading to TestPyPI..."
uv run --group dev twine upload --repository testpypi dist/*

echo "✅ Upload completed successfully!"
echo "🔗 Your package is now available at: https://test.pypi.org/project/burnmewhileimhot/"
echo ""
echo "To install from TestPyPI, users can run:"
echo "  pip install --index-url https://test.pypi.org/simple/ burnmewhileimhot"
