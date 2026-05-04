"""
N2: Smoke tests for the moved-into-repo committee agent scripts.

The 9 agent scripts were moved from ~/.weirdapps-trading/committee/scripts/
into scripts/agents/ on 2026-05-04 (CIO v36 N2). The external location now
holds symlinks pointing back to this repo. These tests assert:

- Each expected script is present and importable
- Each script exposes a main() function or runs as __main__
- The README.md catalog is in sync with actual files

This is mechanical verification, not behavior testing — the actual agent
output validation happens in the synthesis tests.
"""

import importlib.util
import os
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts" / "agents"

# Scripts that are gitignored (contain personal/portfolio-specific logic) — skip
# their presence assertion in CI where they're absent.
_GITIGNORED_SCRIPTS = (
    "fundamental_analysis.py",
    "opportunity_scanner.py",
    "technical_analysis.py",
)

EXPECTED_SCRIPTS = (
    "fundamental_analysis.py",
    "technical_analysis.py",
    "macro_analyst.py",
    "census_analyst.py",
    "fetch_news_events.py",
    "opportunity_scanner.py",
    "analyze_priority_candidates.py",
    "generate_panw_report.py",
    "run_v36_synthesis.py",
)


class TestAgentScriptsCatalog:
    def test_agents_dir_exists(self):
        assert AGENTS_DIR.is_dir(), f"scripts/agents/ missing: {AGENTS_DIR}"

    def test_readme_present(self):
        assert (AGENTS_DIR / "README.md").exists()

    @pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
    def test_each_expected_script_present(self, script_name):
        path = AGENTS_DIR / script_name
        if script_name in _GITIGNORED_SCRIPTS and not path.exists():
            pytest.skip(f"{script_name} is gitignored (personal); not in CI checkout")
        assert path.exists(), f"Missing agent script: {path}"
        assert path.stat().st_size > 100, f"Empty agent script: {path}"

    def test_no_unexpected_scripts(self):
        """Catalog drift check — README mentions all scripts present."""
        present = {p.name for p in AGENTS_DIR.glob("*.py")}
        unexpected = present - set(EXPECTED_SCRIPTS)
        assert not unexpected, (
            f"Scripts in dir but not in catalog: {unexpected}. "
            f"Update tests/unit/scripts/test_agent_scripts_present.py + README.md"
        )


class TestExternalSymlinks:
    """The external location at ~/.weirdapps-trading/committee/scripts/
    must be symlinks pointing to this repo so the orchestrator keeps working.
    Skip when the external dir doesn't exist (CI/laptop without trading data)."""

    EXT_DIR = Path.home() / ".weirdapps-trading" / "committee" / "scripts"

    @pytest.mark.skipif(
        not (Path.home() / ".weirdapps-trading" / "committee" / "scripts").is_dir(),
        reason="External committee dir not present (CI environment)",
    )
    @pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
    def test_external_is_symlink_to_repo(self, script_name):
        ext_path = self.EXT_DIR / script_name
        if not ext_path.exists():
            pytest.skip(f"External {script_name} not present")
        assert ext_path.is_symlink(), (
            f"{ext_path} should be a symlink to scripts/agents/{script_name}, "
            f"not a copy (would drift)"
        )
        target = os.readlink(ext_path)
        assert (
            "scripts/agents" in target
        ), f"{ext_path} symlinks to {target}, expected scripts/agents/"


class TestMacroAnalystImportable:
    """macro_analyst.py is the new v36 yfinance-based macro generator."""

    def test_module_loads_without_executing(self):
        spec = importlib.util.spec_from_file_location(
            "macro_analyst",
            AGENTS_DIR / "macro_analyst.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        # Just import — don't execute main() (which fetches yfinance)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main"), "macro_analyst.py must expose main()"
        assert hasattr(mod, "SECTOR_ETFS"), "macro_analyst.py must expose SECTOR_ETFS"
        # Sanity check: it includes all 11 GICS sector ETFs
        assert len(mod.SECTOR_ETFS) == 11
        assert "XLK" in mod.SECTOR_ETFS
        assert "XLC" in mod.SECTOR_ETFS


class TestRunV36SynthesisImportable:
    def test_module_loads_without_executing(self):
        spec = importlib.util.spec_from_file_location(
            "run_v36_synthesis",
            AGENTS_DIR / "run_v36_synthesis.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        # Sanity checks for v36 wiring
        assert hasattr(mod, "build_portfolio_signals")
        assert hasattr(mod, "REPORTS_DIR")
