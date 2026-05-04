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

# Install all dependencies (production + dev extra) into a Poetry-managed venv
echo "Installing dependencies via Poetry..."
poetry install --extras dev --no-interaction

# Setup pre-commit hooks if config exists
if [ -f ".config/ci/.pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    poetry run pip install pre-commit
    poetry run pre-commit install --config .config/ci/.pre-commit-config.yaml
fi

# Copy environment template if needed
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env file (please configure)"
fi

echo ""
echo "Development environment ready."
echo ""
echo "To run commands inside the Poetry venv:"
echo "  poetry run python trade.py ..."
echo "  poetry run pytest"
echo "Or activate the venv shell:"
echo "  poetry env activate    # prints the activation command"
echo ""
echo "Available helper scripts:"
echo "  scripts/dev/test.sh      - Run tests with coverage"
echo "  scripts/dev/lint.sh      - Run all linters"
echo "  scripts/dev/format.sh    - Auto-format code"
