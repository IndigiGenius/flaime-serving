"""Tests for the vendor tamper gate and LoC budget gate (26Q3-REPO-01).

Each test copies the repo's gate inputs into a sandbox so tampering never
touches the real tree.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

# tests/ is a gate input too: vendored test copies (REPO-14+) get manifest
# entries, and the tamper gate hashes every manifest path.
GATE_INPUTS = ["scripts", "flaime_serving", "tests", "VENDORED_FROM.json", "LOC_BUDGET"]


@pytest.fixture
def sandbox(tmp_path: Path, pytestconfig: pytest.Config) -> Path:
    root = pytestconfig.rootpath
    for name in GATE_INPUTS:
        source = root / name
        if source.is_dir():
            shutil.copytree(source, tmp_path / name)
        else:
            shutil.copy2(source, tmp_path / name)
    return tmp_path


def run_gate(sandbox: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(sandbox / "scripts" / script)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestVendorTamperGate:
    def test_clean_tree_passes(self, sandbox: Path) -> None:
        result = run_gate(sandbox, "check_vendored.sh")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_edited_vendored_file_fails(self, sandbox: Path) -> None:
        target = sandbox / "flaime_serving" / "vendored" / "packing_config.py"
        target.write_text(target.read_text() + "\n# sneaky edit\n")
        result = run_gate(sandbox, "check_vendored.sh")
        assert result.returncode != 0
        assert "TAMPERED" in result.stdout

    def test_unlisted_vendored_file_fails(self, sandbox: Path) -> None:
        (sandbox / "flaime_serving" / "vendored" / "rogue.py").write_text("x = 1\n")
        result = run_gate(sandbox, "check_vendored.sh")
        assert result.returncode != 0
        assert "UNLISTED" in result.stdout


class TestLocBudgetGate:
    def test_within_budget_passes(self, sandbox: Path) -> None:
        result = run_gate(sandbox, "check_loc_budget.sh")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "LoC budget OK" in result.stdout

    def test_over_budget_fails(self, sandbox: Path) -> None:
        (sandbox / "LOC_BUDGET").write_text("1\n")
        result = run_gate(sandbox, "check_loc_budget.sh")
        assert result.returncode != 0
        assert "exceeded" in result.stdout

    def test_vendored_files_are_exempt(self, sandbox: Path) -> None:
        """Vendored code must not count toward the budget (tamper gate owns it)."""
        vendored_loc = sum(
            len(p.read_text().splitlines())
            for p in (sandbox / "flaime_serving" / "vendored").glob("*.py")
        )
        result = run_gate(sandbox, "check_loc_budget.sh")
        counted = int(result.stdout.rsplit(":", 1)[1].split("/")[0])
        assert counted < vendored_loc, "vendored LoC appears to be counted"
