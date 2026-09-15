"""The console scripts must exist and run. Both were declared broken in the first release."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]


@pytest.mark.parametrize("target", sorted(SCRIPTS.values()))
def test_every_declared_console_script_resolves(target: str) -> None:
    """pyproject declared two entry points before either module existed; nothing caught it."""
    module_path, _, attribute = target.partition(":")
    module = __import__(module_path, fromlist=[attribute])
    assert callable(getattr(module, attribute))


def test_every_subcommand_has_a_handler() -> None:
    from toolchoice.cli import build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    assert actions, "the parser exposes no subcommands"
    for action in actions:
        for name, sub in action.choices.items():
            has_handler = sub.get_default("handler") is not None
            nested = [a for a in sub._actions if hasattr(a, "choices") and a.choices]
            assert has_handler or nested, f"subcommand {name!r} has no handler and no children"


def test_no_subcommand_prints_help_and_exits_nonzero() -> None:
    from toolchoice.cli import main

    assert main([]) == 2


def test_verify_with_no_population_is_blocked_not_passed() -> None:
    """Nothing to verify is not a pass. It is exit 2 with a named reason."""
    from toolchoice.verification.gates import main as gates_main

    assert gates_main(["--population", str(ROOT / "does-not-exist.jsonl")]) == 2


def test_gates_main_has_a_system_exit_guard() -> None:
    """`python -m ...` that imports and exits 0 is indistinguishable from a passing gate."""
    source = (ROOT / "src/toolchoice/verification/gates.py").read_text(encoding="utf-8")
    assert "raise SystemExit(main())" in source


def test_core_package_imports_with_no_third_party_packages() -> None:
    """`dependencies = []` has to be true, not aspirational."""
    code = "import toolchoice, opengrad.evaluation.routing; print('ok')"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
        cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src"), "PYTHONNOUSERSITE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
