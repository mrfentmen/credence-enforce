"""
credence/matching.py
====================
The one scorer. Zero dependencies, no heavy imports.

Why this module exists
----------------------
Overlap scoring was implemented three times, independently:

  credence/hooks.py           the PreToolUse gate — the path that BLOCKS
  credence/mcp_server.py      the voluntary credence_gate tool
  credence/context_manager.py the Consistency Enforcer

They did not agree. The enforcing copy was the weakest of the three: it
did no synonym expansion and, because ``\\w`` matches ``_``, it tokenised
``RATE_LIMIT`` as the single token ``rate_limit`` — which can never match
the ``rate`` and ``limit`` that appear in the constraint's prose. So the
gate fired on prose and stayed silent on code, including the example in
the README:

    "I think the rate limit is around 100 req/min"   (constraint)
    RATE_LIMIT = 100                                 (action)
    -> overlap {100} = 1 < 2 -> ALLOWED. Gate silent.

This module is the single implementation. ``hooks.py`` and ``observer.py``
use it directly. ``tests/unit/test_matching.py`` pins it to
``context_manager``'s tables so the three copies cannot drift apart again
without a test failing.

Scoring rule (deliberately narrow)
----------------------------------
An action is blocked by a constraint when EITHER holds:

1. **Value agreement** — the action contains a numeric literal that the
   constraint also claims (both at least 2 digits). Writing the specific
   number into code is the threat this tool exists to catch, so a shared
   value is enough on its own.

2. **Term agreement** — at least ``MIN_OVERLAP`` literal terms are shared,
   after identifier splitting and stopword removal.

Synonym clusters (``DOMAIN_SYNONYMS``) are deliberately NOT part of the
trigger condition. Expanding both sides means any single shared cluster
key satisfies the threshold on its own (one shared key contributes the
whole cluster to the intersection), which would block a write of
``TIMEOUT_MS = 5000`` merely because an unverified constraint mentions a
different timeout. Domain proximity is not evidence; a shared value is.
Clusters remain available via :func:`explain` for annotating *why* a
write was blocked.

Two modes, one implementation
-----------------------------
``evaluate(..., expand_synonyms=True)`` opts into counting cluster agreement
as evidence. It is off by default and must stay off for anything that BLOCKS a
write, for the reason above. It exists for the opposite kind of caller: the
Consistency Enforcer, which decides whether to *warn* the model to hedge. There
a spurious warning costs one sentence, and a missed warning costs the false
certainty this project exists to prevent — so recall is the right trade, and
that path's behaviour is pinned by tests/unit/test_gate.py.

The alternative was a second matcher in context_manager.py, which is what this
module was created to eliminate. Two behaviours, one code path, an explicit
argument at the call site.
"""

from __future__ import annotations

import hashlib
import os
import re

# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

MIN_OVERLAP = 2   # minimum shared literal terms to trigger a block

# Minimum length for a numeric literal to count as a shared value. Guards
# against every ``3`` and ``5`` in a file matching every constraint that
# happens to contain a small integer. Matches _GTS_NUM_PATTERN usage in
# context_manager.py, which also requires len(n) >= 2.
_MIN_NUM_LEN = 2

# ---------------------------------------------------------------------------
# Stopwords — canonical copy. Kept identical to context_manager._CE_STOPWORDS
# (pinned by tests/unit/test_matching.py).
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "about", "what", "how",
    "when", "where", "which", "who", "i", "we", "you", "it", "my", "our",
    "your", "its", "that", "this", "and", "or", "but", "if", "so", "use",
    "used", "using", "get", "set", "now", "just", "also", "need", "want",
    "tell", "know", "think", "make", "give", "take", "see", "say", "go",
    # Cross-domain words excluded to prevent false enforcement:
    # "size" spans UI (font size) vs API (batch size); "error" spans general
    # programming errors vs API retry/error handling. Both are too ambiguous
    # to participate in overlap scoring — context-specific clusters cover them.
    "size", "error",
})

# ---------------------------------------------------------------------------
# Domain-aware synonym clusters — canonical copy.
# Kept identical to context_manager._CE_DOMAIN_SYNONYMS (pinned by tests).
# Used for explaining a block, not for triggering one. See module docstring.
# ---------------------------------------------------------------------------

DOMAIN_SYNONYMS: dict[str, frozenset[str]] = {
    # ---- rate / throughput ------------------------------------------------
    "rate":       frozenset({"limit", "throttle", "quota", "rps", "rpm", "qps",
                              "frequency", "speed", "fast", "slow", "calls",
                              "requests", "throughput", "bandwidth"}),
    "limit":      frozenset({"rate", "cap", "max", "maximum", "ceiling",
                              "quota", "throttle", "threshold"}),
    "throttle":   frozenset({"rate", "limit", "quota", "cap", "restrict",
                              "slow", "backoff"}),
    "quota":      frozenset({"rate", "limit", "cap", "allowance", "budget"}),
    "fast":       frozenset({"rate", "speed", "frequency", "quickly", "rapid",
                              "throughput"}),
    "slow":       frozenset({"rate", "throttle", "delay", "latency", "backoff"}),
    "requests":   frozenset({"rate", "calls", "rps", "rpm", "qps", "invocations",
                              "hits", "traffic"}),
    "calls":      frozenset({"requests", "invocations", "hits", "rate", "rps"}),
    "endpoint":   frozenset({"api", "url", "route", "path", "service",
                              "resource", "host"}),
    # ---- authentication / tokens ------------------------------------------
    "token":      frozenset({"auth", "jwt", "bearer", "credential", "secret",
                              "key", "access", "refresh", "expiry", "expire",
                              "ttl", "session", "oauth"}),
    "auth":       frozenset({"token", "credential", "login", "authenticate",
                              "authorization", "permission", "access", "oauth",
                              "jwt", "bearer"}),
    "expiry":     frozenset({"expire", "expires", "ttl", "timeout", "duration",
                              "lifetime", "valid", "renew", "token"}),
    "expire":     frozenset({"expiry", "ttl", "timeout", "lifetime", "valid"}),
    "credential": frozenset({"token", "auth", "secret", "key", "password",
                              "apikey", "cert"}),
    "secret":     frozenset({"token", "key", "credential", "password", "apikey"}),
    # "refresh" intentionally omitted: auth token refresh != cache refresh —
    # keeping it in both clusters created false CE positives (cache queries
    # matching auth constraints via {refresh, ttl, expiry} overlap).
    "session":    frozenset({"token", "auth", "cookie", "login", "timeout"}),
    # ---- pagination / batching --------------------------------------------
    "page":       frozenset({"pagination", "paging", "offset", "cursor",
                              "batch", "chunk", "size", "per", "results"}),
    "pagination": frozenset({"page", "paging", "offset", "cursor", "batch",
                              "limit", "size", "results"}),
    "batch":      frozenset({"chunk", "bulk", "page", "size", "limit", "group"}),
    # "size" removed as standalone key — too cross-domain (font size != batch size).
    # Reachable as a value from "page", "pagination", "batch", "memory" clusters.
    # ---- infrastructure / error -------------------------------------------
    "timeout":    frozenset({"latency", "delay", "slow", "wait", "deadline",
                              "response", "expiry", "ttl", "out", "times"}),
    "request":    frozenset({"requests", "call", "invocation", "hit", "api"}),
    "latency":    frozenset({"timeout", "delay", "slow", "response",
                              "performance", "speed"}),
    "retry":      frozenset({"backoff", "attempt", "reconnect", "throttle",
                              "fail", "retrying", "retries"}),
    "backoff":    frozenset({"retry", "wait", "delay", "throttle", "slow"}),
    "wait":       frozenset({"retry", "backoff", "delay", "pause", "hold"}),
    # "error" removed as standalone key — too cross-domain (programming error !=
    # API retry error). Reachable as a value from "retry" and "fail" clusters.
    "fail":       frozenset({"exception", "crash", "bug", "issue", "retry"}),
    # ---- configuration / deployment ---------------------------------------
    "config":     frozenset({"setting", "option", "parameter", "value",
                              "configure", "setup", "env", "environment"}),
    "deploy":     frozenset({"release", "push", "rollout", "ship", "launch",
                              "prod", "production", "staging"}),
    "memory":     frozenset({"ram", "heap", "allocation", "buffer",
                              "cache", "storage", "size"}),
    "cpu":        frozenset({"processor", "compute", "cores", "performance",
                              "load", "utilization", "capacity"}),
    # cache cluster deliberately excludes "expiry" and "refresh" to prevent
    # cross-domain bleed with auth/token constraints that also use those terms.
    "cache":      frozenset({"ttl", "invalidate", "stale", "memory", "storage"}),
    # ---- financial / billing ------------------------------------------------
    "cost":       frozenset({"price", "fee", "charge", "billing", "spend", "budget",
                              "rate", "tier", "plan", "pricing", "pay", "invoice"}),
    "price":      frozenset({"cost", "fee", "charge", "billing", "rate", "pricing"}),
    "budget":     frozenset({"cost", "spend", "limit", "cap", "allocation",
                              "forecast", "estimate"}),
    "payment":    frozenset({"charge", "invoice", "billing", "fee", "subscription",
                              "renewal", "refund"}),
    "threshold":  frozenset({"limit", "cap", "max", "minimum", "floor", "ceiling",
                              "cutoff", "trigger"}),
    # ---- medical / dosing / clinical ----------------------------------------
    "dose":       frozenset({"dosage", "dosing", "mg", "ml", "concentration",
                              "amount", "quantity", "regimen", "prescription"}),
    "dosage":     frozenset({"dose", "dosing", "mg", "ml", "frequency",
                              "schedule", "concentration"}),
    "frequency":  frozenset({"dose", "schedule", "interval", "period", "rate",
                              "daily", "weekly", "hourly"}),
    "interval":   frozenset({"frequency", "period", "schedule", "gap",
                              "delay", "duration", "between"}),
    "duration":   frozenset({"period", "length", "interval", "window", "lifetime",
                              "expiry", "ttl"}),
    "contraindication": frozenset({"warning", "restriction", "prohibited",
                                    "interaction", "adverse"}),
    # ---- hardware / infrastructure resources ---------------------------------
    "cores":      frozenset({"cpu", "vCPU", "processor", "threads",
                              "compute", "capacity", "workers"}),
    "disk":       frozenset({"storage", "volume", "iops", "throughput",
                              "space", "ssd", "capacity"}),
    "iops":       frozenset({"disk", "throughput", "storage", "performance",
                              "read", "write", "latency"}),
    "bandwidth":  frozenset({"network", "throughput", "rate", "speed",
                              "mbps", "gbps", "egress", "ingress"}),
    "replica":    frozenset({"instance", "node", "shard", "copy",
                              "failover", "availability", "cluster"}),
    "instance":   frozenset({"node", "server", "vm", "container", "pod",
                              "host", "replica", "worker"}),
    # ---- legal / compliance -------------------------------------------------
    "retention":  frozenset({"storage", "period", "duration", "policy",
                              "gdpr", "deletion", "purge", "archive"}),
    "sla":        frozenset({"uptime", "availability", "agreement", "guarantee",
                              "reliability", "nines", "commitment"}),
    "uptime":     frozenset({"availability", "sla", "reliability",
                              "nines", "downtime", "guarantee"}),
    "compliance": frozenset({"regulation", "policy", "requirement", "standard",
                              "audit", "gdpr", "hipaa", "soc", "pci"}),
}

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Separators inside identifiers: RATE_LIMIT, stripe_client.py, max-retries
_IDENT_SEP_RE = re.compile(r"[_.\-/\\]+")
# camelCase / PascalCase boundaries: rateLimit -> rate|Limit
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Non-word, non-space runs (punctuation) -> separator
_PUNCT_RE = re.compile(r"[^\w\s]")
# Numeric literals, matching context_manager._GTS_NUM_PATTERN
NUM_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def split_identifier(token: str) -> list[str]:
    """Split one token into its lower-cased identifier parts.

    ``rate_limit`` -> ``["rate", "limit"]``
    ``RATE_LIMIT`` -> ``["rate", "limit"]``
    ``rateLimit``  -> ``["rate", "limit"]``
    ``max-retries``-> ``["max", "retries"]``
    ``stripe``     -> ``["stripe"]``

    Casing is split BEFORE lowering, so camelCase is detected. Lowering first
    would collapse ``rateLimit`` into ``ratelimit`` and lose the boundary.
    """
    parts: list[str] = []
    for chunk in _IDENT_SEP_RE.split(token):
        if not chunk:
            continue
        parts.extend(p for p in _CAMEL_RE.split(chunk) if p)
    return [p.lower() for p in parts]


def tokenize(text: str) -> set[str]:
    """Return the scored term set for a piece of text.

    Identifier-aware: ``RATE_LIMIT = 100`` yields ``{rate, limit, 100}`` so it
    can match the prose ``"the rate limit is 100"``. The whole token is kept
    alongside its parts, so a constraint that literally says ``rate_limit``
    still matches another literal ``rate_limit``.
    """
    out: set[str] = set()
    for word in _PUNCT_RE.sub(" ", text or "").split():
        whole = word.lower()
        if len(whole) >= 3 and whole not in STOPWORDS:
            out.add(whole)
        for part in split_identifier(word):
            if len(part) >= 3 and part not in STOPWORDS:
                out.add(part)
    return out


def numbers(text: str) -> set[str]:
    """Numeric literals of at least _MIN_NUM_LEN digits."""
    return {n for n in NUM_PATTERN.findall(text or "") if len(n) >= _MIN_NUM_LEN}


def expand(tokens: set[str]) -> set[str]:
    """Expand terms through DOMAIN_SYNONYMS. For explanation only — never
    used as the trigger condition (see module docstring)."""
    expanded = set(tokens)
    for t in tokens:
        cluster = DOMAIN_SYNONYMS.get(t)
        if cluster:
            expanded |= cluster
    return expanded


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def evaluate(
    action_text: str, constraint_text: str, expand_synonyms: bool = False
) -> dict:
    """Score one action against one constraint.

    Returns ``{"block": bool, "reason": str, "shared_terms": [...],
    "shared_values": [...]}`` where ``reason`` is one of ``"value"``,
    ``"terms"``, ``"synonyms"`` or ``""``.

    ``expand_synonyms`` counts synonym-cluster agreement as evidence. Leave it
    False for anything that blocks a write; see the module docstring. It is for
    recall-oriented callers that warn rather than block.
    """
    action_terms = tokenize(action_text)
    constraint_terms = tokenize(constraint_text)
    literal_terms = action_terms & constraint_terms
    shared_terms = literal_terms
    if expand_synonyms:
        shared_terms = literal_terms | (
            expand(action_terms) & expand(constraint_terms)
        )
    shared_values = numbers(action_text) & numbers(constraint_text)

    if shared_values:
        return {
            "block": True,
            "reason": "value",
            "shared_terms": sorted(shared_terms),
            "shared_values": sorted(shared_values),
        }
    if len(shared_terms) >= MIN_OVERLAP:
        return {
            "block": True,
            "reason": (
                "terms" if len(literal_terms) >= MIN_OVERLAP else "synonyms"
            ),
            "shared_terms": sorted(shared_terms),
            "shared_values": [],
        }
    return {
        "block": False,
        "reason": "",
        "shared_terms": sorted(shared_terms),
        "shared_values": [],
    }


def evaluate_constraints(action_text: str, constraints: list[dict]) -> list[dict]:
    """Score one action against many constraints. Returns the blocking subset,
    each augmented with the evidence that decided it."""
    blocking = []
    for c in constraints:
        verdict = evaluate(action_text, c.get("content", ""))
        if verdict["block"]:
            blocking.append({**c, "_verdict": verdict})
    return blocking


def explain(action_text: str, constraint_text: str) -> list[str]:
    """Human-readable reason a write was blocked, using synonym clusters to
    name the shared domain. Returns [] when nothing is shared."""
    verdict = evaluate(action_text, constraint_text)
    if not verdict["block"]:
        return []
    if verdict["shared_values"]:
        return verdict["shared_values"]
    expanded = expand(tokenize(action_text)) & expand(tokenize(constraint_text))
    return sorted(expanded) or verdict["shared_terms"]


# ---------------------------------------------------------------------------
# Session identity
# ---------------------------------------------------------------------------

def derive_session_id() -> str:
    """Stable session id derived from the working directory.

    Shared by observer.py and hooks.py so the layer that REGISTERS a
    constraint and the layer that ENFORCES it agree on which session they
    are in. They used to disagree: the observer fell back to this id while
    the gate required CREDENCE_SESSION_ID and silently passed every tool
    call through when it was unset.
    """
    return hashlib.md5(os.getcwd().encode()).hexdigest()[:8] + "_auto"


def resolve_session_id() -> str:
    """The session id both layers must use: explicit env var, else derived."""
    return os.environ.get("CREDENCE_SESSION_ID") or derive_session_id()
