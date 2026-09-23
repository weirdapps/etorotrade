#!/usr/bin/env python3
"""MOVED. The daily brief's number validator lives in plessas-trading-stack now.

    plessas-trading-stack/plugins/etoro-social/scripts/validate_brief.py

Ported there on 2026-09-24 (plessas-trading-stack social-posting-2) so the repo that publishes
the post also tests the gate in front of it. This copy passed wrong numbers: its display-name
map knew " es ", " nq " and " ym " while the prompt writes "S&P futures", "Nasdaq" and "Dow",
so those three bound to nothing and matched ANY snapshot move within 0.35pp, and a percentage
after a $TICKER was never checked at all.

This file is a POINTER, not a second copy: it forwards its arguments to the ported validator
and holds no logic, so no fork survives. When the ported file is absent (a checkout of the
stack older than the port) it exits 2, the validator's own usage-error code, which every caller
already treats as "did not pass". It never passes on its own.
"""

import os
import sys
from pathlib import Path

TARGET = (
    Path.home()
    / "SourceCode"
    / "plessas-trading-stack"
    / "plugins"
    / "etoro-social"
    / "scripts"
    / "validate_brief.py"
)


def main() -> int:
    if not TARGET.is_file():
        print(
            f"validate_brief.py moved to plessas-trading-stack, and {TARGET} is not there. "
            f"Refusing to validate (exit 2, never a pass).",
            file=sys.stderr,
        )
        return 2
    os.execv(sys.executable, [sys.executable, str(TARGET), *sys.argv[1:]])
    return 2  # unreachable: execv replaces this process


if __name__ == "__main__":
    sys.exit(main())
