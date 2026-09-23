"""scripts/validate_brief.py is a pointer to the ported validator in plessas-trading-stack.

The validator's logic, and the regression cases this file used to pin (2026-05-12 ticker and
market-cap false positives, 2026-08-07 directional claims), moved with it to
plessas-trading-stack/plugins/etoro-social/tests/test_validate_brief.py, where that repo's CI
runs them. What is left to test here is the pointer's one promise: it never passes on its own.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "validate_brief.py"


def test_the_pointer_holds_no_validation_logic():
    src = SCRIPT_PATH.read_text()
    assert "DISPLAY_TO_TICKER" not in src and "def validate(" not in src


def test_with_the_stack_absent_the_pointer_REFUSES_with_exit_2(tmp_path):
    """Exit 2 is the validator's usage-error code; every caller reads non-zero as not passed."""
    post, snap = tmp_path / "post.txt", tmp_path / "snap.json"
    post.write_text("S&P futures +0.90%.")
    snap.write_text('{"instruments": {}}')
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(post), str(snap)],
        env={"HOME": str(tmp_path / "no-stack-here"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 2
    assert "moved to plessas-trading-stack" in r.stderr


def test_the_pointer_forwards_its_arguments_and_exit_code(tmp_path):
    target = (
        tmp_path
        / "SourceCode"
        / "plessas-trading-stack"
        / "plugins"
        / "etoro-social"
        / "scripts"
        / "validate_brief.py"
    )
    target.parent.mkdir(parents=True)
    target.write_text("import sys; print('forwarded', sys.argv[1:]); sys.exit(1)\n")
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "a.txt", "b.json"],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 1
    assert "forwarded ['a.txt', 'b.json']" in r.stdout
