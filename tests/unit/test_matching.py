"""
test_matching.py — Unit tests for the canonical overlap scorer.

credence/matching.py is the single scorer used by the enforcing hook
(hooks.py), the observer, and — via the same tables — the MCP gate. These
tests pin its behaviour and pin it to context_manager's tables so the
duplicate copies cannot silently drift apart again.

Coverage:
  M1  Identifier splitting: snake_case, SCREAMING_CASE, camelCase, kebab
  M2  Code and prose produce the same scored term set
  M3  The README's headline case blocks (regression: it did not)
  M4  Value agreement blocks on a shared numeric literal
  M5  Term agreement blocks at MIN_OVERLAP
  M6  Precision: same domain, different value -> NOT blocked
  M7  Unrelated action -> NOT blocked
  M8  Numeric literals below the length floor are ignored
  M9  Session identity: stable, cwd-derived, env override honoured
  M10 Table parity with context_manager (_CE_STOPWORDS, _CE_DOMAIN_SYNONYMS)
  M11 explain() names shared values, and is empty when nothing is shared
"""

import pytest

from credence import matching
from credence.matching import (
    MIN_OVERLAP,
    evaluate,
    evaluate_constraints,
    explain,
    numbers,
    resolve_session_id,
    split_identifier,
    tokenize,
)


# ── M1: Identifier splitting ─────────────────────────────────────────────────

@pytest.mark.parametrize("token,expected", [
    ("rate_limit",  ["rate", "limit"]),
    ("RATE_LIMIT",  ["rate", "limit"]),
    ("rateLimit",   ["rate", "limit"]),
    ("RateLimit",   ["rate", "limit"]),
    ("max-retries", ["max", "retries"]),
    ("TOKEN_EXPIRY_SECONDS", ["token", "expiry", "seconds"]),
    ("stripe",      ["stripe"]),
    ("",            []),
])
def test_split_identifier(token, expected):
    assert split_identifier(token) == expected


def test_camel_case_split_survives_lowercasing():
    """The regression that bit the first draft: lowering before splitting
    collapses rateLimit into ratelimit and loses the boundary."""
    assert "limit" in tokenize("rateLimit")
    assert "rate" in tokenize("rateLimit")


# ── M2: Code and prose agree ─────────────────────────────────────────────────

def test_code_and_prose_tokenize_identically():
    code = tokenize("RATE_LIMIT = 100")
    prose = tokenize("the rate limit is 100")
    assert {"rate", "limit", "100"} <= code
    assert {"rate", "limit", "100"} <= prose


def test_file_path_contributes_terms():
    assert "stripe" in tokenize("stripe_client.py")


# ── M3: The README headline case ─────────────────────────────────────────────

def test_readme_headline_case_blocks():
    """README: 'I think the rate limit is around 100 req/min' must block a
    write of 'RATE_LIMIT = 100'. Before this change it did not: the gate
    tokenised RATE_LIMIT as one token and found a single shared term."""
    constraint = "I think the Stripe rate limit is around 100 req/min"
    action = "Edit stripe_client.py RATE_LIMIT = 100"
    verdict = evaluate(action, constraint)
    assert verdict["block"], "README's own example must be blocked"
    assert verdict["reason"] == "value"
    assert "100" in verdict["shared_values"]


def test_snake_case_write_blocks():
    constraint = "I think the rate limit is around 100 req/min"
    assert evaluate("Edit client.py rate_limit = 100", constraint)["block"]


def test_identifier_splitting_is_load_bearing_without_a_shared_value():
    """Term agreement must work on identifiers alone. This case shares no
    numeric literal, so only splitting RATE_LIMIT into {rate, limit} can
    block it — the value rule cannot rescue it."""
    constraint = "the rate limit value is still unverified"
    action = "RATE_LIMIT = settings.get('RATE_LIMIT')"
    assert not numbers(action) & numbers(constraint), "must share no value"
    verdict = evaluate(action, constraint)
    assert verdict["block"]
    assert verdict["reason"] == "terms"


# ── M4: Value agreement ──────────────────────────────────────────────────────

def test_shared_value_blocks_even_with_one_shared_term():
    constraint = "the token expiry is probably 3600"
    action = "TOKEN_TTL = 3600"
    verdict = evaluate(action, constraint)
    assert verdict["block"]
    assert verdict["reason"] == "value"
    assert verdict["shared_values"] == ["3600"]


# ── M5: Term agreement ───────────────────────────────────────────────────────

def test_term_agreement_blocks_at_min_overlap():
    constraint = "the rate limit is around 100"
    action = "rate limit"
    verdict = evaluate(action, constraint)
    assert verdict["block"]
    assert verdict["reason"] == "terms"
    assert len(verdict["shared_terms"]) >= MIN_OVERLAP


# ── M6: Precision — same domain, different value ─────────────────────────────

def test_same_domain_different_value_does_not_block():
    """A shared domain word is not evidence. Writing a *different* timeout
    must not be blocked by an unverified timeout constraint, or the gate
    becomes noise. Synonym expansion is deliberately not a trigger."""
    constraint = "the webhook timeout is probably 30 seconds"
    action = "TIMEOUT_MS = 5000"
    verdict = evaluate(action, constraint)
    assert not verdict["block"], (
        "domain proximity alone must not block; only a shared value or "
        f"{MIN_OVERLAP} shared terms may. got {verdict}"
    )


def test_same_domain_same_value_does_block():
    constraint = "the webhook timeout is probably 30 seconds"
    assert evaluate("TIMEOUT_MS = 30", constraint)["block"]


# ── M7: Unrelated ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("action", [
    "Edit theme.py COLOR_SCHEME = 'dark'",
    "Edit readme.md add a usage section",
    "Write tests/unit/test_foo.py with two cases",
])
def test_unrelated_action_does_not_block(action):
    constraint = "I think the Stripe rate limit is around 100 req/min"
    assert not evaluate(action, constraint)["block"], f"false positive: {action}"


# ── M8: Numeric floor ────────────────────────────────────────────────────────

def test_single_digit_numbers_are_not_values():
    """Every file contains a 3 or a 5. Only >= 2 digits counts as a value."""
    assert numbers("retries 3 and timeout 5") == set()
    assert numbers("rate limit 100") == {"100"}


def test_single_digit_shared_number_alone_does_not_block():
    constraint = "we set max retries to 3"
    assert not evaluate("x = 3", constraint)["block"]


# ── M9: Session identity ─────────────────────────────────────────────────────

def test_derive_session_id_is_stable_and_cwd_derived():
    a = matching.derive_session_id()
    b = matching.derive_session_id()
    assert a == b
    assert a.endswith("_auto")


def test_resolve_session_id_prefers_env(monkeypatch):
    monkeypatch.setenv("CREDENCE_SESSION_ID", "my-project")
    assert resolve_session_id() == "my-project"


def test_resolve_session_id_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("CREDENCE_SESSION_ID", raising=False)
    assert resolve_session_id() == matching.derive_session_id()


def test_resolve_session_id_falls_back_when_empty(monkeypatch):
    monkeypatch.setenv("CREDENCE_SESSION_ID", "")
    assert resolve_session_id() == matching.derive_session_id()


# ── M10: Table parity with context_manager ───────────────────────────────────

def test_stopwords_match_context_manager():
    """matching.STOPWORDS is the canonical copy. If context_manager's copy is
    edited without mirroring it here, this fails rather than letting the two
    scorers drift."""
    from credence.context_manager import _CE_STOPWORDS
    assert matching.STOPWORDS == _CE_STOPWORDS


def test_domain_synonyms_match_context_manager():
    from credence.context_manager import _CE_DOMAIN_SYNONYMS
    assert matching.DOMAIN_SYNONYMS == _CE_DOMAIN_SYNONYMS


def test_min_overlap_matches_context_manager():
    from credence.context_manager import _CE_MIN_OVERLAP
    assert MIN_OVERLAP == _CE_MIN_OVERLAP


# ── M11: explain() ───────────────────────────────────────────────────────────

def test_explain_names_the_shared_value():
    why = explain("RATE_LIMIT = 100", "I think the rate limit is around 100")
    assert "100" in why


def test_explain_falls_back_to_synonym_cluster_for_a_term_match():
    """Two shared terms block with no shared value, so the explanation has no
    number to name. It should fall back to the expanded domain vocabulary."""
    verdict = evaluate("TOKEN_TTL = abc", "the token ttl is unverified")
    assert verdict["block"] and verdict["reason"] == "terms"
    assert explain("TOKEN_TTL = abc", "the token ttl is unverified")


def test_single_shared_term_does_not_block():
    """One shared word is not enough — that is the precision floor."""
    verdict = evaluate("TOKEN_EXPIRY = abc", "the token ttl is unverified")
    assert not verdict["block"]
    assert verdict["shared_terms"] == ["token"]


def test_explain_empty_when_not_blocking():
    assert explain("COLOR = 'blue'", "the rate limit is 100") == []


# ── evaluate_constraints() ───────────────────────────────────────────────────

def test_evaluate_constraints_returns_only_blocking_ones():
    constraints = [
        {"constraint_id": "c1", "content": "the rate limit is probably 100"},
        {"constraint_id": "c2", "content": "the logo colour is blue"},
    ]
    blocking = evaluate_constraints("RATE_LIMIT = 100", constraints)
    assert [c["constraint_id"] for c in blocking] == ["c1"]
    assert blocking[0]["_verdict"]["reason"] == "value"


def test_evaluate_constraints_handles_missing_content():
    assert evaluate_constraints("x = 1", [{"constraint_id": "c1"}]) == []
