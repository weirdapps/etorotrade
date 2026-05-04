#!/bin/bash
# One-command development environment setup
set -e

echo "Setting up eToro Trade Analysis development environment..."

# Verify Poetry is available
if ! command -v poetry > /dev/null 2>&1; then
    echo "ERROR: Poetry is not installed. Install it first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    echo "  (or: pipx install poetry==2.4.0)"
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating venv/ ..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# Export pinned, SHA256-hashed deps from poetry.lock, then install via pip with
# --only-binary :all: --require-hashes. This matches CI behaviour and avoids
# the SonarCloud S8541 "Poetry can run setup scripts" finding by going through
# pip's safer install path.
echo "Exporting pinned hashed requirements..."
poetry export --extras dev -f requirements.txt -o /tmp/etorotrade-req.txt

echo "Installing dependencies (only-binary, hashed)..."
pip install --upgrade pip
pip install --only-binary :all: --require-hashes -r /tmp/etorotrade-req.txt
rm -f /tmp/etorotrade-req.txt

# Setup pre-commit hooks if config exists
if [ -f ".config/ci/.pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pip install pre-commit
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
