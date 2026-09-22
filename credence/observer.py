"""
credence/observer.py
====================
Passive conversation-stream observer for automatic epistemic tracking.

Registers uncertain values from user messages WITHOUT requiring model
cooperation. The model never needs to call credence_register — this hook
fires on every user message and does it automatically.

This removes the fundamental fragility of instruction-dependent enforcement:
if the model ignores CLAUDE.md, the registry is still populated.

Hook configuration (add to .claude/settings.json):
----------------------------------------------------
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [{
          "type": "command",
          "command": "python3 -m credence.observer"
        }]
      }
    ]
  }
}

The PreToolUse gate (hooks.py) is still required for enforcement.
This observer is the detection layer; hooks.py is the enforcement layer.
Both resolve the session id the same way (matching.resolve_session_id), so a
constraint registered here is enforced there even when CREDENCE_SESSION_ID is
unset.

The registry is created on first real registration rather than requiring the
database to already exist. Previously a fresh install registered nothing at
all: the hook read a path that no one had created yet and returned early.

Exit codes: always 0 — observer never blocks. A registration that fails still
returns 0, because failing a user's prompt is not this hook's job; it reports
the failure on stderr and in the event log instead, so an inert registry is
visible rather than silent.
"""

from __future__ import annotations

import json
import re
import sys

from credence.matching import log_event, resolve_db_path, resolve_session_id


# ── Uncertainty markers — authoritative copy lives in context_manager.py.
# Inline here to avoid a slow import on every hook invocation.
# Strong markers: fire unconditionally — these phrases almost exclusively signal uncertainty
_STRONG_MARKERS = frozenset({
    "not certain", "not sure", "uncertain", "tentative", "unverified",
    "approximately", "roughly", "i think", "i believe", "i'm not",
    "might be", "might not", "may be", "possibly", "perhaps",
    "i'd verify", "need to check", "should verify", "to verify",
    "approx", "tbd",
    "probably", "maybe", "provisionally", "preliminary", "supposedly",
    "ambiguous", "unclear", "hasn't clarified", "not yet clarified",
    "unconfirmed", "not confirmed", "not yet confirmed", "open question",
    "needs verification", "need to verify",
    "not yet decided", "not decided", "to be determined", "to be confirmed",
    "haven't confirmed", "haven't verified", "haven't checked",
    "depending on", "depends on whether", "subject to", "contingent on",
    "once we confirm", "once we verify", "pending confirmation",
    "as far as i know", "to my knowledge", "to my understanding",
    "if i recall", "i seem to recall", "last time i checked",
    "best of my knowledge",
    "working theory", "my assumption", "i'm assuming", "in theory",
    "could be wrong", "not 100%", "not entirely sure",
    "the vendor said", "they mentioned", "reportedly",
    "i read somewhere", "heard that", "we were told",
    "give or take", "ballpark", "order of magnitude", "in the range of",
    "somewhere around", "plus or minus", "estimated at",
    "untested", "not yet tested", "haven't tested", "not benchmarked",
    "iirc", "afaik", "if i recall correctly", "from memory",
    "off the top of my head", "as best i recall", "i think i remember",
    "i'm unsure", "unsure", "not sure which", "unsure of",
    "according to the rep", "per the ticket", "vendor claims",
    "sales rep said", "they told us", "our rep mentioned",
    "according to their docs", "according to the docs", "per their docs",
    "per the vendor", "from the vendor", "according to the vendor",
    "vendor estimate", "vendor ballpark", "vendor said",
    "the demo showed", "from the demo",
    "from a quote", "per the quote", "their estimate",
    "estimate is", "my estimate", "i'd estimate",
})

# Weak markers: only fire when a numeric value is also present.
# These phrases appear in both uncertainty and non-uncertainty contexts.
# "around 100 req/min" → register. "wrap around the list" → skip.
_WEAK_MARKERS = frozenset({
    "around", "assuming", "i guess", "i suppose",
    "seems like", "seems to be", "docs say", "the docs say",
})

_STRONG_RE = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in sorted(_STRONG_MARKERS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
_WEAK_RE = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in sorted(_WEAK_MARKERS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
# Kept for backward compatibility (used in tests)
_UNCERTAINTY_MARKERS = _STRONG_MARKERS | _WEAK_MARKERS
_MARKER_RE = _STRONG_RE

# Ghost constraint heuristic — same conditions as CLAUDE.md:
# numeric value + domain keyword + not inside a URL
_DOMAIN_KW_RE = re.compile(
    r'\b(rate[\s_-]?limit|auth[\s_-]?lifetime|token[\s_-]?expir|ttl|'
    r'api[\s_-]?version|pricing|cost[\s_-]?per|price[\s_-]?per|'
    r'quota|max[\s_-]?retries|timeout|concurrency|tokens?)\b',
    re.IGNORECASE,
)
# No trailing \b — allows matching numbers glued to units: 30s, 5MB, 256KB, 50ms
_NUMERIC_RE   = re.compile(r'\b\d+(?:\.\d+)?')
_URL_CTX_RE   = re.compile(r'(?:://|[?=&/]v?)\d')


def _classify(text: str) -> tuple[bool, str]:
    """Return (should_register, source_type) for a text fragment."""
    lower = text.lower()

    # Strong marker → register unconditionally
    if _STRONG_RE.search(lower):
        return True, "user_estimate"

    # Weak marker → only register when a numeric value is also present.
    # Prevents "wrap around the list" / "seems like a clean design" from firing.
    if _WEAK_RE.search(lower) and _NUMERIC_RE.search(text):
        return True, "user_estimate"

    # Ghost heuristic: numeric + domain keyword + not a URL value
    if (_NUMERIC_RE.search(text)
            and _DOMAIN_KW_RE.search(text)
            and not _URL_CTX_RE.search(text)):
        return True, "vendor_claim"

    return False, ""


def _report_failure(exc: Exception, db_path: str) -> None:
    """Surface a failed registration instead of swallowing it.

    This hook's contract is that it exits 0 — it must never block a prompt.
    That is a statement about the exit code, not about silence, and swallowing
    the exception made the worst outcome look like the best one: an unwritable
    registry registers nothing, the gate downstream finds nothing to enforce,
    and enforcement is inert with nothing anywhere saying so. That is the same
    failure this module has already been fixed for twice — a fresh install
    that registered nothing, and layers that disagreed about the registry
    path — so the diagnostic goes where the person who can fix it will see it.
    Stderr, because Claude Code surfaces it, and the event log, because
    `credence stats` reads it.
    """
    print(
        f"credence observer: could not register a constraint "
        f"({type(exc).__name__}: {exc}) — registry {db_path}",
        file=sys.stderr,
    )
    log_event({
        "event": "observer_error",
        "error": type(exc).__name__,
        "detail": str(exc)[:200],
        "registry": db_path,
    })


def observe(text: str, session_id: str, db_path: str) -> bool:
    """
    Inspect a text fragment and register it if uncertain.
    Returns True if a registration was made.
    """
    text = text.strip()
    if len(text) < 12:
        return False

    should, source_type = _classify(text)
    if not should:
        return False

    try:
        from credence.registry import CredenceRegistry
        reg = CredenceRegistry(db_path)
        reg.register(
            content=text[:500],
            session_id=session_id,
            j_score=0.0,
            zone="LOW",
            source=source_type,
            constraint_type="observation",
        )
        return True
    except Exception as exc:
        _report_failure(exc, db_path)
        return False


def _extract_text(payload: dict) -> str:
    """Pull user message text from a UserPromptSubmit hook payload."""
    # Claude Code sends the prompt as either a string or a list of content parts
    content = payload.get("prompt", payload.get("message", ""))
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content) if content else ""


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return 0

    # The observer's contract is that it always exits 0. Valid JSON that is
    # not an object would otherwise raise on .get() and break the contract.
    if not isinstance(payload, dict):
        return 0

    text = _extract_text(payload)
    if not text:
        return 0

    # Shared resolver — must agree with hooks.py, the MCP server, and the Rust
    # gate, or this registers constraints where the gate never looks.
    db_path    = resolve_db_path()
    session_id = resolve_session_id()

    # No existence check: observe() constructs the registry lazily, and only
    # when the text actually warrants a registration. That keeps the first
    # real use working while avoiding a stray .db in every directory the
    # hook happens to run in.
    observe(text, session_id, db_path)
    return 0  # observer never blocks


if __name__ == "__main__":
    sys.exit(main())
