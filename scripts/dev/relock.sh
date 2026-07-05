#!/bin/bash
# Regenerate poetry.lock + the three checked-in requirements-*-lock.txt files
# from pyproject.toml. Run this after editing dependencies in pyproject.toml.
#
# CI (lockfile-sync job) re-exports the requirements files and diffs against
# the committed copies; running this keeps that check green.
#
# Usage:
#   scripts/dev/relock.sh                 # full: poetry lock + export (default)
#   scripts/dev/relock.sh --export-only   # re-export requirements from the
#                                         # EXISTING poetry.lock (no re-resolve)
# --export-only is used by the dependabot-relock CI workflow: Dependabot already
# maintains a consistent poetry.lock, so re-resolving could undo its (transitive)
# bumps — there we only refresh the committed requirements-*-lock.txt exports.
set -e

EXPORT_ONLY=0
if [ "${1:-}" = "--export-only" ]; then
    EXPORT_ONLY=1
fi

if ! command -v poetry > /dev/null 2>&1; then
    echo "ERROR: Poetry is not installed. Install it first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    echo "  (or: pipx install poetry==2.4.0)"
    exit 1
fi

if ! poetry self show plugins 2>/dev/null | grep -q poetry-plugin-export; then
    echo "Installing poetry-plugin-export ..."
    poetry self add poetry-plugin-export
fi

if [ "$EXPORT_ONLY" = 0 ]; then
    echo "1/2 Regenerating poetry.lock from pyproject.toml ..."
    poetry lock --no-interaction
else
    echo "1/2 --export-only: skipping 'poetry lock' (re-export from existing poetry.lock)"
fi

echo "2/2 Exporting committed requirements files ..."
poetry export --only main         -f requirements.txt -o requirements-lock.txt
poetry export --extras dev        -f requirements.txt -o requirements-dev-lock.txt
poetry export --extras smoketest  -f requirements.txt -o requirements-smoketest-lock.txt

echo ""
echo "Done. Commit the regenerated files:"
echo "  git add pyproject.toml poetry.lock requirements-lock.txt requirements-dev-lock.txt requirements-smoketest-lock.txt"
