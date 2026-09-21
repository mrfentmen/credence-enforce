"""
test_examples.py — Every shipped example must actually run.

This exists because the examples/ directory had already rotted once: all four
files imported ``from esm import ...``, a module belonging to an abandoned
sibling project that is not in this repository. Nothing executed them, so
nothing noticed — a user following the examples hit ModuleNotFoundError.

An example that is not run by CI is a claim, not a feature. These tests run
each one as a subprocess and require exit 0.

Coverage:
  X1  examples/quickstart.py runs clean
  X2  examples/hook_demo.py runs clean (drives the real hooks)
  X3  No example imports a module that does not exist in this repo
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
EXAMPLES = ROOT / "examples"

# Walked, not globbed at import time, so a new example is covered the moment
# it lands. tests/ is excluded deliberately: it has its own runner.
_EXAMPLES = sorted(p for p in EXAMPLES.glob("*.py"))


def test_examples_directory_is_not_empty():
    assert _EXAMPLES, f"no examples found in {EXAMPLES}"


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: p.name)
def test_example_runs(example):
    """Run the example the way a user would: a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=180,
    )
    assert result.returncode == 0, (
        f"{example.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: p.name)
def test_example_imports_resolve(example):
    """Statically check every imported top-level module exists.

    Catches the precise rot that happened: an example importing a package
    that is not in this repository, before the example is ever run.
    """
    import importlib.util

    tree = ast.parse(example.read_text())
    missing = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # relative import
                continue
            if node.module:
                names = [node.module.split(".")[0]]
        for name in names:
            if name in sys.builtin_module_names:
                continue
            spec = importlib.util.find_spec(name)
            if spec is None or (name == "credence" and str(ROOT) not in sys.path):
                # "credence" resolves because the example inserts the repo root
                # on sys.path; anything else missing is a real problem.
                if name == "credence":
                    continue
                missing.append(name)
    assert not missing, f"{example.name} imports modules that do not exist: {missing}"
