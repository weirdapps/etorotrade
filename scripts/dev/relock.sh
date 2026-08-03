#!/bin/bash
# Regenerate poetry.lock + the three checked-in requirements-*-lock.txt files
# from pyproject.toml. Run this after editing dependencies in pyproject.toml.
#
# CI (lockfile-sync job) re-exports the requirements files and diffs against
# the committed copies; running this keeps that check green.
#
# Usage:
#   scripts/dev/relock.sh                 # poetry lock + export (default)
#   scripts/dev/relock.sh --regenerate    # re-resolve EVERYTHING to the newest
#                                         # allowed versions, then export
#   scripts/dev/relock.sh --export-only   # re-export requirements from the
#                                         # EXISTING poetry.lock (no re-resolve)
#
# --export-only is used by the dependabot-relock CI workflow: Dependabot already
# maintains a consistent poetry.lock, so re-resolving could undo its (transitive)
# bumps, there we only refresh the committed requirements-*-lock.txt exports.
#
# --regenerate is used by deps-refresh.yml (monthly). Plain `poetry lock` in
# Poetry 2.x deliberately keeps every already-locked package at its current
# version and only resolves what pyproject.toml forces to move, so on its own it
# will NOT pull transitive dependencies forward. Only --regenerate discards the
# existing lock and re-resolves from scratch.
#
# NOTE: requirements-universe-lock.txt is NOT produced here. It is a tiny,
# hand-maintained hash list for weekly-universe-refresh.yml (5 packages, Python
# 3.11 / manylinux2014_x86_64). Bump it by hand, see the header of that file.
set -e

EXPORT_ONLY=0
REGENERATE=0
case "${1:-}" in
    --export-only) EXPORT_ONLY=1 ;;
    --regenerate)  REGENERATE=1 ;;
    "")            ;;
    *) echo "ERROR: unknown option '$1' (expected --regenerate or --export-only)"; exit 2 ;;
esac

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

if [ "$EXPORT_ONLY" = 1 ]; then
    echo "1/2 --export-only: skipping 'poetry lock' (re-export from existing poetry.lock)"
elif [ "$REGENERATE" = 1 ]; then
    echo "1/2 --regenerate: re-resolving every dependency to its newest allowed version ..."
    poetry lock --regenerate --no-interaction
else
    echo "1/2 Regenerating poetry.lock from pyproject.toml ..."
    poetry lock --no-interaction
fi

echo "2/2 Exporting committed requirements files ..."
poetry export --only main         -f requirements.txt -o requirements-lock.txt
poetry export --extras dev        -f requirements.txt -o requirements-dev-lock.txt
poetry export --extras smoketest  -f requirements.txt -o requirements-smoketest-lock.txt

echo ""
echo "Done. Commit the regenerated files:"
echo "  git add pyproject.toml poetry.lock requirements-lock.txt requirements-dev-lock.txt requirements-smoketest-lock.txt"
