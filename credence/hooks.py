"""
credence/hooks.py
=================
Claude Code PreToolUse hook for automatic epistemic enforcement.

When configured in .claude/settings.json, this script intercepts every
Write/Edit/MultiEdit/Bash/NotebookEdit tool call and checks whether its
arguments overlap with any unverified constraint in the Credence registry.
Reads are never gated: only the tools in matching.ENFORCED_TOOLS are, and a
tool this hook does not recognise passes straight through.

If overlap is found (≥2 non-stopword terms), the hook exits non-zero and
Claude Code blocks the tool call — printing a warning to the user instead.

This converts Credence from advisory to enforcing: the model cannot write
code or run commands that embed unverified values without explicit user
confirmation, regardless of whether the model called credence_gate itself.

Setup (add to your project's .claude/settings.json):
------------------------------------------------------
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit|NotebookEdit|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -m credence.hooks"
          }
        ]
      }
    ]
  }
}

The hook reads the tool call from stdin as JSON (Claude Code hook protocol).

The session id comes from CREDENCE_SESSION_ID when set, and is otherwise
derived from the working directory by credence.matching.resolve_session_id()
— the same derivation observer.py uses. Previously this hook required the
env var and passed every tool call through when it was unset, which made
enforcement a silent no-op in the documented default setup.

Overlap scoring lives in credence.matching, so this hook, the observer, and
the MCP gate all reach the same verdict.

Exit codes:
  0  — proceed (no unverified constraint overlap)
  2  — block  (overlapping unverified constraint found; warning printed to stderr)
"""

from __future__ import annotations

import json
import os
import re
import sys

from credence.matching import (
    evaluate_constraints,
    is_enforced_tool,
    log_event,
    resolve_db_path,
    resolve_session_id,
)


# ---------------------------------------------------------------------------
# Event log — ~/.credence/events.jsonl
# Every gate fire is logged so false-positive rate can be measured.
# Run `credence stats` to see signal quality from real usage.
## The path and the writer live in credence/matching.py, because this module and
# the observer both append and __main__.py reads. This file carried its own
# `_EVENTS_DIR` / `_EVENTS_FILE` pair, which is a copy of a location that
# already existed twice more.
# ---------------------------------------------------------------------------




# Overlap scoring, stopwords, thresholds, and session identity all live in
# credence/matching.py. This module used to carry its own weaker copy, which
# is why the demo blocked and real hook invocations did not.


# Depth cap for _flatten. Named rather than inline because the Rust gate
# mirrors it (`MAX_FLATTEN_DEPTH` in credence_gate/src/main.rs) and the two
# have to agree for the paths to see the same bytes; tests/unit/test_matcher_parity.py
# compares them. A runaway payload must not make the gate expensive either, on
# the path whose selling point is being fast.
_MAX_FLATTEN_DEPTH = 4


def _flatten(obj, depth: int = 0) -> str:
    """Recursively flatten a JSON object to a single string for scanning."""
    if depth > _MAX_FLATTEN_DEPTH:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten(v, depth + 1) for v in obj)
    return str(obj)


def main() -> int:
    # --- Read hook payload from stdin (Claude Code protocol) ----------------
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            # Valid JSON that is not an object (e.g. "[]") carries no tool
            # call. Treat it as empty rather than raising: a hook must never
            # turn a tool call into a crash.
            payload = {}
    except (json.JSONDecodeError, ValueError):
        payload = {}

    tool_name  = payload.get("tool_name", payload.get("name", "unknown"))
    tool_input = payload.get("tool_input", payload.get("input", {}))

    # --- Only writing tools are gated ---------------------------------------
    # Defense in depth. The documented settings.json restricts this hook with
    # `"matcher": "Write|Edit|MultiEdit|NotebookEdit|Bash"`, but a matcher is
    # configuration a user can omit, broaden, or get wrong — and without this
    # check the hook scored every tool name and argument, so a registered value
    # turned every read-only call containing that number into a block. Reading
    # is not irreversible (docs/VISION.md: "gate the action, not the text").
    #
    # Exits before opening the registry: this runs on every tool call.
    if not is_enforced_tool(tool_name):
        return 0

    action_text = f"{tool_name} {_flatten(tool_input)}"

    # --- Locate registry ----------------------------------------------------
    # Shared resolver: this hook, the observer, the MCP server, and the Rust
    # gate must all open the same file. This used to read only CREDENCE_DB, so
    # setting CREDENCE_DB_PATH pointed the MCP tools at one database and this
    # hook at another (see the note in credence/matching.py).
    db_path    = resolve_db_path()
    session_id = resolve_session_id()

    if not os.path.exists(db_path):
        # No registry configured → pass through silently
        return 0

    try:
        from credence.registry import CredenceRegistry
        registry   = CredenceRegistry(db_path)
        uncertain  = registry.list_uncertain(session_id)
    except Exception:
        # Registry unavailable → pass through silently
        return 0

    if not uncertain:
        return 0

    # --- Overlap check -------------------------------------------------------
    blocking = []
    for c in evaluate_constraints(action_text, uncertain):
        verdict = c["_verdict"]
        blocking.append({
            "constraint_id": c["constraint_id"],
            "content":       c["content"][:120],
            "overlap":       verdict["shared_values"] or verdict["shared_terms"][:6],
            "reason":        verdict["reason"],
            "zone":          c.get("zone", "UNKNOWN"),
        })

    if not blocking:
        log_event({
            "event":       "allow",
            "tool_name":   tool_name,
            "session_id":  session_id,
            "constraints": len(uncertain),
        })
        return 0

    # --- Block and warn ------------------------------------------------------
    def _clean(s: str) -> str:
        s = re.sub(r'^\[stale:[^\]]+\]\s*', '', s)
        s = re.sub(r'^\[AI-generated:[^\]]+\]\s*', '', s)
        return s
    reasons = " | ".join(_clean(b["content"])[:100] for b in blocking[:2])
    lines = [
        f"credence: blocked {tool_name} — {len(blocking)} unverified value(s)",
        f"  → {reasons}",
        "  Verify first, then retry. Use credence_constraints to see all pending.",
    ]
    print("\n".join(lines), file=sys.stderr)

    log_event({
        "event":      "block",
        "tool_name":  tool_name,
        "session_id": session_id,
        "blocked_by": [
            {"id": b["constraint_id"][:12], "content": b["content"][:80], "overlap": b["overlap"]}
            for b in blocking[:3]
        ],
        "feedback":   None,   # filled in by `credence feedback 1|2|3`
    })
    return 2   # non-zero → Claude Code blocks the tool call


if __name__ == "__main__":
    sys.exit(main())
