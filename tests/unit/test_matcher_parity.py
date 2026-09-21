"""
test_matcher_parity.py — every enforcement path must give the same answer.

The codebase grew four independent implementations of "do these two texts
concern the same thing?":

  1. credence/matching.py            — the canonical scorer
  2. credence/hooks.py               — the PreToolUse gate (route via matching)
  3. credence/mcp_server.py          — the credence_gate MCP tool
  4. credence/mcp_server.py          — the credence_autoverify MCP tool

They disagreed, and the disagreement was the reported bug: on the README's own
example ("RATE_LIMIT = 100" against "I think the Stripe rate limit is around
100 req/min") the hook BLOCKED and the MCP gate answered PROCEED. The MCP gate
had lower-cased before splitting, so `RATE_LIMIT` stayed a single token and
never matched the prose "rate"/"limit"; it found one shared term and required
two. The autoverifier was worse in kind: it carried a 20-word stopword list
against the gate's ~190, and it decides what gets marked VERIFIED — the state
that makes the gate allow a write. A more generous matcher there silences
enforcement.

A test per implementation cannot catch this. Two paths each behaving as
documented can still contradict each other, and only a test that runs them
against the same input notices. That is what this file is.

Coverage:
  P1 The MCP gate agrees with the canonical scorer on every corpus row
  P2 The MCP gate and the real PreToolUse hook agree (subprocess, exit code)
  P3 The autoverifier and the gate are consistent — one matcher, both directions
  P4 The Consistency Enforcer is never narrower than the blocking path
  P5 The enforcement modules all share one matcher object, not four copies
  P6 The deleted duplicates have not come back
  P7 Every layer resolves the same registry file from the same environment
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from credence import hooks, matching, mcp_server
from credence.context_manager import ContextManager
from credence.matching import resolve_db_path
from credence.registry import CredenceRegistry

ROOT = Path(__file__).resolve().parent.parent.parent

CONSTRAINT = "I think the Stripe rate limit is around 100 req/min"

# (action, constraint, should_block) — the third field is what a human reading
# the README would expect, and is asserted against every path.
CORPUS = [
    # The README's case: identifiers against prose.
    ("Write RATE_LIMIT = 100", CONSTRAINT, True),
    ("Write stripe_client.py: RATE_LIMIT = 100", CONSTRAINT, True),
    ("Edit config rate_limit = 100", CONSTRAINT, True),
    ("Write stripeClientRateLimit = 100", CONSTRAINT, True),
    # Shared numeric value.
    ("Bash python -c 'print(100)'", CONSTRAINT, True),
    ("Write TIMEOUT = 100", CONSTRAINT, True),
    # Genuinely unrelated — must not block.
    ("Write body { color: rebeccapurple }", CONSTRAINT, False),
    ("Write README.md describing the build", CONSTRAINT, False),
    ("Write docker-compose.yml with postgres", CONSTRAINT, False),
    # Same domain, different value — must not block.
    ("Write MAX_RETRIES = 3", CONSTRAINT, False),
    ("Write CACHE_TTL = 900", CONSTRAINT, False),
]


def _run_hook(payload, db, session_id, env_extra=None, db_var="CREDENCE_DB"):
    """Invoke the hook as a subprocess, the way Claude Code does."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        db_var: str(db),
        "CREDENCE_NO_LOG": "1",
        "CREDENCE_SESSION_ID": session_id,
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "credence.hooks"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=60,
    )


# ── P1 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("action,constraint,should_block", CORPUS)
def test_mcp_gate_agrees_with_canonical(action, constraint, should_block):
    constraints = [{"constraint_id": "c1", "content": constraint}]
    expected = bool(matching.evaluate_constraints(action, constraints))
    assert expected is should_block, "corpus expectation disagrees with canonical"

    blocked = mcp_server.blocking_constraints("", action, constraints)
    assert bool(blocked) is expected


# ── P2 ───────────────────────────────────────────────────────────────────────

def test_mcp_gate_and_pretooluse_hook_agree(tmp_path):
    """The strongest parity check available: the real hook, as a subprocess,
    against the real MCP-gate decision, on the same corpus."""
    db = tmp_path / "parity.db"
    sid = "parity-session"
    reg = CredenceRegistry(db_path=str(db))
    reg.register(CONSTRAINT, sid, j_score=0.3, zone="LOW")
    constraints = [{"constraint_id": "c1", "content": CONSTRAINT}]

    for action, _, _ in CORPUS:
        tool_name, _, summary = action.partition(" ")
        mcp_blocks = bool(
            mcp_server.blocking_constraints(tool_name, summary, constraints)
        )

        proc = _run_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": sid,
                "tool_name": tool_name,
                "tool_input": {"file_path": summary, "content": summary},
            },
            db,
            sid,
        )
        hook_blocks = proc.returncode == 2
        assert hook_blocks == mcp_blocks, (
            f"enforcement paths disagree on {action!r}: "
            f"hook={hook_blocks} (exit {proc.returncode}) mcp_gate={mcp_blocks}"
            f"\nstderr: {proc.stderr[:400]}"
        )


# ── P3 ───────────────────────────────────────────────────────────────────────

def test_autoverify_agrees_with_the_gate_it_disarms():
    """Confirmation and blocking must use the same matcher.

    `confirmable_constraints` removes a constraint from the uncertain set, and
    the gate only blocks on uncertain constraints — so confirming is the act of
    switching enforcement off for that constraint. If confirmation matched more
    generously than blocking, a text the gate would not have honoured could
    disarm it.
    """
    constraints = [{"constraint_id": "c1", "content": CONSTRAINT}]

    confirmer = "confirmed: rate limit is 100 per stripe docs"
    assert mcp_server.confirmable_constraints(confirmer, constraints) == ["c1"]
    # The same text blocks, because it shares the value and the domain terms.
    assert mcp_server.blocking_constraints("", confirmer, constraints)

    for vague in (
        "confirmed the deploy finished",
        "actually the tests pass",
        "turns out we shipped it",
    ):
        assert mcp_server.confirmable_constraints(vague, constraints) == [], (
            f"{vague!r} must not verify a rate-limit constraint"
        )


@pytest.mark.parametrize("action,constraint,should_block", CORPUS)
def test_confirmation_and_blocking_share_one_verdict(action, constraint, should_block):
    """Whatever the canonical scorer decides, both MCP paths must see it."""
    constraints = [{"constraint_id": "c1", "content": constraint}]
    verdict = matching.evaluate(action, constraint)["block"]
    assert verdict is bool(mcp_server.blocking_constraints("", action, constraints))
    assert verdict is bool(mcp_server.confirmable_constraints(action, constraints))


# ── P4 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("action,constraint,should_block", CORPUS)
def test_consistency_enforcer_is_never_narrower(action, constraint, should_block):
    """The warning path must fire wherever the blocking path fires.

    The Consistency Enforcer does not block anything — it decides whether to
    tell the model to hedge. So it is allowed to be *broader* than the gate
    (it runs with expand_synonyms=True, deliberately; see the module docstring
    in credence/matching.py). It is not allowed to be narrower, because
    "the gate would have blocked this but nothing warned the model" is exactly
    the false-certainty failure the project exists to prevent.

    This is the assertion that catches the old enforcer: it tokenised with
    text.lower().split(), so `RATE_LIMIT = 100` became the single term
    `rate_limit`, never matched `rate`/`limit`, and the enforcer stayed silent
    on every code-shaped query.
    """
    constraints = [{"constraint_id": "c1", "content": constraint}]
    blocks = bool(matching.evaluate_constraints(action, constraints))
    fires = bool(ContextManager.__new__(ContextManager)._direct_constraint_matches(
        action, constraints
    ))
    if blocks:
        assert fires, (
            f"the gate blocks {action!r} but the enforcer would not warn about it"
        )


def test_consistency_enforcer_handles_disputed_escalation():
    """A disputed constraint escalates regardless of the query, by contract.

    Verified-then-contradicted is the highest-risk state the registry tracks;
    the user must be told whether or not their message mentions it.
    """
    cm = ContextManager.__new__(ContextManager)
    disputed = [{
        "constraint_id": "c1",
        "content": "the timeout is 30 seconds",
        "validation_status": "disputed",
    }]
    matches = cm._direct_constraint_matches("what colour is the button?", disputed)
    assert len(matches) == 1
    assert matches[0]["_overlap"] == ["DISPUTED"]


# ── P5 ───────────────────────────────────────────────────────────────────────

def test_enforcement_modules_share_one_matcher_object():
    """Not a copy of the canonical function — the same function object.

    A re-implementation with identical behaviour today is a divergence
    tomorrow. Identity is the property worth pinning.
    """
    assert hooks.evaluate_constraints is matching.evaluate_constraints
    assert mcp_server.evaluate_constraints is matching.evaluate_constraints
    assert mcp_server.blocking_constraints.__module__ == "credence.mcp_server"


# ── P7 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, "epistemic_registry.db"),
        ({"CREDENCE_DB": "a.db"}, "a.db"),
        ({"CREDENCE_DB_PATH": "b.db"}, "b.db"),
        ({"CREDENCE_DB_PATH": "b.db", "CREDENCE_DB": "a.db"}, "b.db"),
        ({"CREDENCE_DB": "a.db", "CREDENCE_REGISTRY_PATH": "c.db"}, "a.db"),
        ({"CREDENCE_REGISTRY_PATH": "c.db"}, "c.db"),
    ],
)
def test_registry_path_resolution(monkeypatch, env, expected):
    """One chain, one answer, regardless of which name a user sets."""
    for var in ("CREDENCE_DB_PATH", "CREDENCE_DB", "CREDENCE_REGISTRY_PATH"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    assert resolve_db_path() == expected


def test_hook_honours_credence_db_path(tmp_path):
    """Regression: the hook read only CREDENCE_DB.

    CREDENCE_DB_PATH is the name mcp_server.py calls canonical. Setting it used
    to point the MCP tools at one database while the hook and observer opened
    `epistemic_registry.db` in the working directory — so the layer that
    REGISTERS and the layer that ENFORCES disagreed, the gate found no
    constraints, and every write was allowed. The Rust gate had the same split.
    """
    db = tmp_path / "via-db-path.db"
    sid = "cfg-split"
    CredenceRegistry(db_path=str(db)).register(CONSTRAINT, sid, j_score=0.3, zone="LOW")

    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    assert not (elsewhere / "epistemic_registry.db").exists()

    proc = _run_hook(
        {
            "hook_event_name": "PreToolUse",
            "session_id": sid,
            "tool_name": "Write",
            "tool_input": {"file_path": "a.py", "content": "RATE_LIMIT = 100"},
        },
        db,
        sid,
        env_extra={"CREDENCE_DB_PATH": str(db)},
        db_var="CREDENCE_DB_PATH",
    )
    assert proc.returncode == 2, (
        "the hook did not find the constraint via CREDENCE_DB_PATH — it exited "
        f"{proc.returncode} instead of blocking\nstderr: {proc.stderr[:400]}"
    )


# ── P6 ───────────────────────────────────────────────────────────────────────

def test_private_matcher_copies_have_not_returned():
    """The removed duplicate must stay removed.

    `_expand_tokens` and `_AUTOVERIFY_STOPWORDS` lived in mcp_server.py as
    private second implementations. If either reappears, a future edit can
    silently reintroduce the split that caused this bug.
    """
    assert not hasattr(mcp_server, "_expand_tokens"), (
        "mcp_server._expand_tokens is back — enforcement must use "
        "credence.matching.expand, not a private copy"
    )
    assert not hasattr(mcp_server, "_AUTOVERIFY_STOPWORDS"), (
        "mcp_server._AUTOVERIFY_STOPWORDS is back — a second stopword list "
        "means a second definition of 'same thing'"
    )
