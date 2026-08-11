#!/usr/bin/env python3
"""Pull version pins that only exist in the committed exports back into poetry.lock.

Dependabot treats requirements-lock.txt / requirements-dev-lock.txt /
requirements-smoketest-lock.txt as pip manifests, so it rewrites the pins in
those files directly and never touches poetry.lock. Re-exporting from an
untouched poetry.lock would simply undo the bump, and the CI `lockfile sync`
job fails either way.

This script closes that gap. For every package whose export pin disagrees with
poetry.lock, it drops that package's `[[package]]` block from poetry.lock and
re-runs `poetry lock`, which re-resolves exactly those packages to the newest
version the rest of the constraint set allows. Packages held back by an
upstream pin (e.g. safety pinning safety-schemas==0.0.16) resolve back to where
they were, which is the correct answer, not a failure.

Run from the repository root, then run `scripts/dev/relock.sh --export-only`.
Exits 0 and does nothing when poetry.lock already agrees with the exports.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LOCK_PATH = Path("poetry.lock")
EXPORT_PATHS = (
    Path("requirements-lock.txt"),
    Path("requirements-dev-lock.txt"),
    Path("requirements-smoketest-lock.txt"),
)

# `packaging==26.3 ; python_version >= "3.10" ...` at the start of a line.
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
BLOCK_SPLIT_RE = re.compile(r"(?m)^(?=\[\[package\]\])")
BLOCK_NAME_RE = re.compile(r'\[\[package\]\]\nname = "([^"]+)"')
BLOCK_VERSION_RE = re.compile(r'(?m)^version = "([^"]+)"')


def normalize(name: str) -> str:
    """PEP 503 name normalisation, so `types_setuptools` == `types-setuptools`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock_text: str) -> dict[str, str]:
    versions = {}
    for block in BLOCK_SPLIT_RE.split(lock_text):
        name_match = BLOCK_NAME_RE.match(block)
        version_match = BLOCK_VERSION_RE.search(block)
        if name_match and version_match:
            versions[normalize(name_match.group(1))] = version_match.group(1)
    return versions


def exported_pins() -> dict[str, str]:
    pins = {}
    for path in EXPORT_PATHS:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            match = PIN_RE.match(line)
            if match:
                pins[normalize(match.group(1))] = match.group(2)
    return pins


def drop_blocks(lock_text: str, names: set[str]) -> str:
    kept = []
    for block in BLOCK_SPLIT_RE.split(lock_text):
        name_match = BLOCK_NAME_RE.match(block)
        if name_match and normalize(name_match.group(1)) in names:
            continue
        kept.append(block)
    return "".join(kept)


def main() -> int:
    if not LOCK_PATH.exists():
        print("ERROR: poetry.lock not found; run from the repository root.", file=sys.stderr)
        return 2

    lock_text = LOCK_PATH.read_text()
    locked = locked_versions(lock_text)
    stale = {
        name: (locked[name], pin)
        for name, pin in exported_pins().items()
        if name in locked and locked[name] != pin
    }

    if not stale:
        print("poetry.lock already agrees with the committed exports; nothing to adopt.")
        return 0

    for name, (lock_version, export_version) in sorted(stale.items()):
        print(f"adopting {name}: poetry.lock {lock_version} -> export asks {export_version}")

    LOCK_PATH.write_text(drop_blocks(lock_text, set(stale)))
    subprocess.run(["poetry", "lock", "--no-interaction"], check=True)

    resolved = locked_versions(LOCK_PATH.read_text())
    for name, (_, export_version) in sorted(stale.items()):
        got = resolved.get(name, "MISSING")
        note = "" if got == export_version else f"  (not {export_version}; other constraints won)"
        print(f"resolved {name}: {got}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
