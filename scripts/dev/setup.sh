#!/bin/bash
# One-command development environment setup
set -e

echo "Setting up eToro Trade Analysis development environment..."

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating venv/ ..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# Install from the CHECKED-IN, pinned + SHA256-hashed lockfile (same path CI
# uses). pip's `--only-binary :all: --require-hashes` blocks setup-script
# execution from sdists and verifies every artifact's SHA256.
echo "Installing dependencies (only-binary, hashed)..."
pip install --only-binary :all: --upgrade pip
pip install --only-binary :all: --require-hashes -r requirements-dev-lock.txt

# Setup pre-commit hooks if config exists
if [ -f ".config/ci/.pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install --config .config/ci/.pre-commit-config.yaml
fi

# Copy environment template if needed
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env file (please configure)"
fi

echo ""
echo "Development environment ready."
echo ""
echo "To activate the venv:"
echo "  source venv/bin/activate"
echo ""
echo "Available helper scripts:"
echo "  scripts/dev/test.sh      - Run tests with coverage"
echo "  scripts/dev/lint.sh      - Run all linters"
echo "  scripts/dev/format.sh    - Auto-format code"
