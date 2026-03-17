#!/bin/bash
# Build and upload moxing to PyPI
# Usage: ./uploadpypi.sh

set -e

echo "============================================================"
echo "  moxing PyPI Upload Script"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Clean old builds
echo "Cleaning old builds..."
rm -rf build/ dist/ *.egg-info/
echo ""

# Build package
echo "Building package..."
$PYTHON_CMD -m build
echo ""

# Check package
echo "Checking package..."
$PYTHON_CMD -m twine check dist/*
echo ""

# Upload to PyPI
echo "Uploading to PyPI..."
$PYTHON_CMD -m twine upload dist/*
echo ""

echo "============================================================"
echo "  Upload Complete!"
echo "============================================================"
echo ""
echo "View at: https://pypi.org/project/moxing/"
echo ""