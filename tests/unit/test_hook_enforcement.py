"""
test_hook_enforcement.py — End-to-end tests for the PreToolUse gate.

credence/hooks.py is the ONLY layer that actually blocks a tool call. It had
no execution coverage at all: tests/unit/test_gate.py exercises
ContextManager._direct_constraint_matches, a different scorer used by a
different layer. That is how three defects survived in the enforcing path:

  1. It tokenised RATE_LIMIT as one term, so it never matched the prose
     "rate limit" and stayed silent on real code.
  2. It required CREDENCE_SESSION_ID and returned 0 — allow — when unset, so
     enforcement was a no-op in the documented default setup.
  3. The observer never registered anything until something else created the
     registry database.

These tests invoke the hook the way Claude Code does: a subprocess, JSON on
stdin, exit code as the verdict.

Coverage:
  H1  Blocks a write embedding an unverified value (the README's case)
  H2  Blocks on a snake_case identifier
  H3  Allows an unrelated write (no false positive)
  H4  Allows once the constraint is verified
  H5  Derives the session id when CREDENCE_SESSION_ID is unset  [regression]
  H6  Passes through when no registry exists
  H7  Passes through on malformed stdin and on an empty payload
  H8  The block message names the offending constraint, on stderr
  H9  Exit-code contract: 0 allow, 2 block
  H10 Observer creates the registry on first use            [regression]
  H11 Observer and hook agree on the session id when unset   [regression]
  H12 File-path-only overlap does not block
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from credence.registry import CredenceRegistry

# Repo root — the hook is invoked as a subprocess with this as cwd, the way
# Claude Code invokes it. `credence` itself is importable via tests/conftest.py.
ROOT = Path(__file__).resolve().parent.parent.parent

CONSTRAINT = "I think the Stripe rate limit is around 100 req/min"


def _run_hook(payload, db, session_id=None, timeout=60):
    """Invoke the hook as a subprocess, exactly as Claude Code does."""
    env = {
        # minimal env — do NOT inherit CREDENCE_SESSION_ID
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "CREDENCE_DB": str(db),
        "CREDENCE_NO_LOG": "1",
    }
    if session_id is not None:
        env["CREDENCE_SESSION_ID"] = session_id
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload)
    return subprocess.run(
        [sys.executable, "-m", "credence.hooks"],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=timeout,
    )


def _run_observer(prompt, db, session_id=None):
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "CREDENCE_DB": str(db),
        "CREDENCE_NO_LOG": "1",
    }
    if session_id is not None:
        env["CREDENCE_SESSION_ID"] = session_id
    return subprocess.run(
        [sys.executable, "-m", "credence.observer"],
        input=json.dumps({"prompt": prompt}),
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=60,
    )


def _derived_session_id():
    """Ask the real resolver what session the hook will use. Uses the real
    function rather than re-deriving the formula in the test."""
    out = subprocess.run(
        [sys.executable, "-c",
         "from credence.matching import derive_session_id; print(derive_session_id())"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.fixture
def registry(tmp_path):
    """A registry holding one unverified constraint, under an explicit
    session id so the hook can find it without relying on derivation."""
    db = tmp_path / "epistemic_registry.db"
    reg = CredenceRegistry(db_path=str(db))
    cid = reg.register(CONSTRAINT, "test_session", j_score=0.3, zone="LOW")
    return db, reg, cid


# ── H1: the headline case ────────────────────────────────────────────────────

def test_blocks_write_embedding_unverified_value(registry):
    db, _, _ = registry
    result = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "stripe_client.py",
                        "new_string": "RATE_LIMIT = 100"}},
        db, session_id="test_session",
    )
    assert result.returncode == 2, (
        f"README's own example must block. rc={result.returncode}\n"
        f"stderr={result.stderr}"
    )


# ── H2: snake_case ───────────────────────────────────────────────────────────

def test_blocks_snake_case_identifier(registry):
    db, _, _ = registry
    result = _run_hook(
        {"tool_name": "Write",
         "tool_input": {"file_path": "client.py",
                        "content": "rate_limit = 100"}},
        db, session_id="test_session",
    )
    assert result.returncode == 2


def test_blocks_identifier_match_without_a_shared_number(tmp_path):
    """The enforcing path must match on identifier terms too, not only on
    values. No number is shared here, so only RATE_LIMIT -> {rate, limit}
    can produce the block."""
    db = tmp_path / "terms.db"
    CredenceRegistry(db_path=str(db)).register(
        "the rate limit value is still unverified", "test_session"
    )
    result = _run_hook(
        {"tool_name": "Write",
         "tool_input": {"file_path": "client.py",
                        "content": "RATE_LIMIT = settings.get('RATE_LIMIT')"}},
        db, session_id="test_session",
    )
    assert result.returncode == 2, (
        f"identifier-only match must block. rc={result.returncode}"
    )


# ── H3: precision ────────────────────────────────────────────────────────────

def test_allows_unrelated_write(registry):
    db, _, _ = registry
    result = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "theme.py",
                        "new_string": "COLOR_SCHEME = 'dark'"}},
        db, session_id="test_session",
    )
    assert result.returncode == 0, f"false positive: {result.stderr}"


# ── H4: verified clears the gate ─────────────────────────────────────────────

def test_allows_after_verify(registry):
    db, reg, cid = registry
    reg.verify(cid, verified_value="confirmed 100 req/min per stripe docs")
    result = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "stripe_client.py",
                        "new_string": "RATE_LIMIT = 100"}},
        db, session_id="test_session",
    )
    assert result.returncode == 0, "a verified constraint must not block"


# ── H5: session identity regression ──────────────────────────────────────────

def test_blocks_without_explicit_session_id(tmp_path):
    """Regression: with CREDENCE_SESSION_ID unset the hook used to return 0
    unconditionally, so the gate was a no-op in the documented default setup.
    It must now derive the same id the observer uses.

    Uses its own registry because CredenceRegistry dedupes identical content:
    registering the same sentence under a second session id is a no-op, so the
    constraint would stay owned by the first session.
    """
    db = tmp_path / "derived.db"
    derived = _derived_session_id()
    CredenceRegistry(db_path=str(db)).register(
        CONSTRAINT, derived, j_score=0.3, zone="LOW"
    )

    result = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "stripe_client.py",
                        "new_string": "RATE_LIMIT = 100"}},
        db, session_id=None,          # <- the whole point
    )
    assert result.returncode == 2, (
        "the gate must fire with no CREDENCE_SESSION_ID set, deriving the "
        f"observer's session id. rc={result.returncode} stderr={result.stderr}"
    )


# ── H6: no registry ──────────────────────────────────────────────────────────

def test_passes_through_when_no_registry(tmp_path):
    result = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "a.py", "new_string": "x = 1"}},
        tmp_path / "does_not_exist.db", session_id="test_session",
    )
    assert result.returncode == 0


# ── H7: malformed input never breaks the user's tool call ────────────────────

@pytest.mark.parametrize("payload", ["", "not json at all", "{", "[]", "{}"])
def test_passes_through_on_bad_input(registry, payload):
    db, _, _ = registry
    result = _run_hook(payload, db, session_id="test_session")
    assert result.returncode == 0, (
        f"malformed hook input must never block a tool call. payload={payload!r}"
    )


# ── H8: the message ──────────────────────────────────────────────────────────

def test_block_message_names_the_constraint_on_stderr(registry):
    db, _, _ = registry
    result = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "stripe_client.py",
                        "new_string": "RATE_LIMIT = 100"}},
        db, session_id="test_session",
    )
    assert "blocked" in result.stderr
    assert "rate limit" in result.stderr.lower()
    assert result.stdout == "", "the gate must not write to stdout"


# ── H9: exit-code contract ───────────────────────────────────────────────────

def test_exit_codes_are_only_zero_or_two(registry):
    db, _, _ = registry
    allowed = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "a.py", "new_string": "y = 7"}},
        db, session_id="test_session",
    )
    blocked = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "stripe_client.py", "new_string": "RATE_LIMIT = 100"}},
        db, session_id="test_session",
    )
    assert allowed.returncode == 0
    assert blocked.returncode == 2


# ── H10/H11: observer and hook agree ─────────────────────────────────────────

def test_observer_creates_registry_on_first_use(tmp_path):
    """Regression: the observer used to require the database to pre-exist, so
    a fresh install registered nothing, ever."""
    db = tmp_path / "fresh.db"
    assert not db.exists()
    result = _run_observer("I think the rate limit is around 100 req/min", db)
    assert result.returncode == 0
    assert db.exists(), "the observer must create the registry on first real use"
    reg = CredenceRegistry(db_path=str(db))
    assert len(reg.list_uncertain(_derived_session_id())) == 1


def test_observer_then_hook_end_to_end_with_no_env(tmp_path):
    """The documented default setup: install the two hooks, set nothing else.
    A value the user hedged must reach the gate and block the write."""
    db = tmp_path / "e2e.db"
    _run_observer("I think the Stripe rate limit is around 100 req/min", db)
    result = _run_hook(
        {"tool_name": "Edit",
         "tool_input": {"file_path": "stripe_client.py",
                        "new_string": "RATE_LIMIT = 100"}},
        db, session_id=None,
    )
    assert result.returncode == 2, (
        "observer -> gate must work with no CREDENCE_SESSION_ID. "
        f"rc={result.returncode} stderr={result.stderr}"
    )


@pytest.mark.parametrize("payload", ["[]", "\"a string\"", "123", "null"])
def test_observer_survives_non_object_json(tmp_path, payload):
    """The observer promises always to exit 0. Valid JSON that is not an
    object used to raise AttributeError and exit 1."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "CREDENCE_DB": str(tmp_path / "p.db"),
        "CREDENCE_NO_LOG": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "credence.observer"],
        input=payload, capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=60,
    )
    assert result.returncode == 0, f"payload={payload!r} stderr={result.stderr}"


def test_observer_does_not_litter_short_prompts(tmp_path):
    """The observer fires on every message. A prompt with nothing to register
    must not create a database."""
    db = tmp_path / "litter.db"
    _run_observer("hello", db)
    assert not db.exists()


# ── H12: file path alone ─────────────────────────────────────────────────────

def test_file_path_alone_does_not_block(registry):
    """Touching a file whose *name* shares vocabulary is not embedding the
    value. Only content and values count."""
    db, _, _ = registry
    result = _run_hook(
        {"tool_name": "Read", "tool_input": {"file_path": "stripe_client.py"}},
        db, session_id="test_session",
    )
    assert result.returncode == 0
