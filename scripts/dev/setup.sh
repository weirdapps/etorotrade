#!/bin/bash
# One-command development environment setup
set -e

echo "🚀 Setting up eToro Trade Analysis development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --only-binary :all: --upgrade pip
pip install --only-binary :all: -r requirements.txt

# Install dev dependencies
echo "Installing development dependencies..."
pip install --only-binary :all: pytest-cov pytest-xdist hypothesis mutmut autoflake radon

# Setup pre-commit hooks if config exists
if [ -f ".config/ci/.pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pip install --only-binary :all: pre-commit
    pre-commit install --config .config/ci/.pre-commit-config.yaml
fi

# Copy environment template if needed
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "📝 Created .env file (please configure)"
fi

echo ""
echo "✅ Development environment ready!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Available commands:"
echo "  scripts/dev/test.sh      - Run tests with coverage"
echo "  scripts/dev/lint.sh      - Run all linters"
echo "  scripts/dev/format.sh    - Auto-format code"
