"""
test_rust_gate_parity.py — the Rust gate is a second implementation of the
blocking decision, in another language.

`credence_gate/` is offered as a faster drop-in for `credence/hooks.py` (the
README cites 3.4ms against a Python hook). That makes it a fourth enforcement
path, and like the other three it was written independently: its own tokeniser,
its own stopword list, its own synonym clusters. Nothing compared them, and the
matching path in this file is the only place that would notice if they diverged
again.

This test needs the release binary, so it skips when the crate has not been
built. Build it with:

    cd credence_gate && cargo build --release

Note for CI: the workflow's `rust-gate` job builds the crate but runs in a
separate job from the one that runs pytest, so this file skips there. Wiring it
up would mean adding the build (and a Rust toolchain) to the test job. That is a
change to `.github/workflows/ci.yml` and has deliberately not been made.

Coverage:
  X1 The binary is found, or the file skips with instructions
  X2 It blocks the same writes the Python hook blocks
  X3 It allows the same writes the Python hook allows
  X4 It honours CREDENCE_DB — the variable hooks.py, observer.py, and the
     README all use [regression]
"""

import json
import subprocess
from pathlib import Path

import pytest

from credence import matching
from credence.registry import CredenceRegistry
from tests.unit.test_matcher_parity import CORPUS

ROOT = Path(__file__).resolve().parent.parent.parent
RUST_BIN = ROOT / "credence_gate" / "target" / "release" / "credence-gate"

CONSTRAINT = "I think the Stripe rate limit is around 100 req/min"


def _binary() -> Path:
    if not RUST_BIN.exists():
        pytest.skip(
            f"Rust gate not built at {RUST_BIN.relative_to(ROOT)} — "
            "run `cd credence_gate && cargo build --release` to exercise it"
        )
    return RUST_BIN


def _run_gate(binary: Path, payload: dict, env_extra: dict, cwd: Path):
    """Invoke the Rust gate the way Claude Code does: JSON on stdin, exit code
    as the verdict."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "CREDENCE_NO_LOG": "1",
    }
    env.update(env_extra)
    return subprocess.run(
        [str(binary)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=60,
    )


# ── X1 ───────────────────────────────────────────────────────────────────────

def test_rust_binary_present_or_skipped():
    """Makes the requirement explicit rather than silently passing."""
    if not RUST_BIN.exists():
        pytest.skip("Rust gate not built (see module docstring)")


# ── X2 / X3 ──────────────────────────────────────────────────────────────────

def test_rust_gate_matches_python_on_the_corpus(tmp_path):
    binary = _binary()
    db = tmp_path / "registry.db"
    sid = "rust-parity"
    CredenceRegistry(db_path=str(db)).register(CONSTRAINT, sid, j_score=0.3, zone="LOW")
    constraints = [{"constraint_id": "c1", "content": CONSTRAINT}]

    for action, _, _ in CORPUS:
        tool_name, _, summary = action.partition(" ")
        expected = bool(matching.evaluate_constraints(action, constraints))

        proc = _run_gate(
            binary,
            {
                "tool_name": tool_name,
                "session_id": sid,
                "tool_input": {"file_path": summary, "content": summary},
            },
            {"CREDENCE_DB": str(db)},
            tmp_path,
        )
        rust_blocks = proc.returncode == 2
        assert rust_blocks == expected, (
            f"Rust gate disagrees with the Python hook on {action!r}: "
            f"rust={rust_blocks} (exit {proc.returncode}) python={expected}"
            f"\nstderr: {proc.stderr[:400]}"
        )


# ── X4 ───────────────────────────────────────────────────────────────────────

def test_rust_gate_honours_credence_db(tmp_path):
    """Regression: the gate read only CREDENCE_DB_PATH.

    `credence/hooks.py` and `credence/observer.py` resolve `CREDENCE_DB`, and
    README.md tells users to set it to keep the registry out of the working
    directory. A user who followed that advice had this gate open a different,
    empty database, find no constraints, and allow every write — enforcement
    looked installed and was inert.

    Run from a directory with no registry in it, with no CREDENCE_DB_PATH set,
    so the only way to find the constraint is to read CREDENCE_DB.
    """
    binary = _binary()
    db = tmp_path / "global-registry.db"
    sid = "rust-env"
    CredenceRegistry(db_path=str(db)).register(CONSTRAINT, sid, j_score=0.3, zone="LOW")

    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    assert not (elsewhere / "epistemic_registry.db").exists()

    proc = _run_gate(
        binary,
        {
            "tool_name": "Write",
            "session_id": sid,
            "tool_input": {"file_path": "a.py", "content": "RATE_LIMIT = 100"},
        },
        {"CREDENCE_DB": str(db)},
        elsewhere,
    )
    assert proc.returncode == 2, (
        "the Rust gate did not find the constraint via CREDENCE_DB — it "
        f"exited {proc.returncode} instead of blocking"
    )
