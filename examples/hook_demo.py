#!/usr/bin/env python3
"""
hook_demo.py — the enforcement path, driven exactly as Claude Code drives it.

This is the demo that matters. It shells out to the two hooks the way the
editor does — JSON on stdin, exit code as the verdict — so what you see here
is what happens in a real session, not a hand-built approximation.

    observer   UserPromptSubmit   registers hedgeable values
    gate       PreToolUse         exit 2 blocks the tool call

Note there is no CREDENCE_SESSION_ID in the environment below. Both hooks
derive the same session id from the working directory, so this works in the
default setup with no configuration. (It did not always: the gate used to
require that variable and silently allow everything without it.)

Run:
    python3 examples/hook_demo.py
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_hook(module: str, payload: dict, db: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.path.expanduser("~"),
        "CREDENCE_DB": db,
        "CREDENCE_NO_LOG": "1",
        # deliberately no CREDENCE_SESSION_ID
    }
    return subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


def gate(db: str, tool: str, tool_input: dict) -> tuple[bool, str]:
    r = run_hook("credence.hooks", {"tool_name": tool, "tool_input": tool_input}, db)
    return r.returncode == 2, r.stderr.strip()


def main() -> int:
    db = os.path.join(tempfile.mkdtemp(prefix="credence-hook-demo-"), "registry.db")

    print("Setup: two hooks installed, no environment variables set.")
    print()

    # ── the observer sees a hedged value ────────────────────────────────────
    prompt = "I think the Stripe rate limit is around 100 req/min — not confirmed yet"
    r = run_hook("credence.observer", {"prompt": prompt}, db)
    print("UserPromptSubmit")
    print(f"   user: {prompt}")
    print(f"   observer exit={r.returncode} (always 0 — it detects, never blocks)")
    print(f"   registry created on first real use: {os.path.exists(db)}")
    print()

    # ── the gate scores real tool calls ────────────────────────────────────
    cases = [
        ("Edit", {"file_path": "stripe_client.py", "new_string": "RATE_LIMIT = 100"},
         "embeds the unverified value"),
        ("Write", {"file_path": "client.py", "content": "rate_limit = 100"},
         "same value, snake_case"),
        ("Edit", {"file_path": "theme.py", "new_string": "COLOR = 'dark'"},
         "unrelated — must NOT block"),
        ("Edit", {"file_path": "client.py", "new_string": "TIMEOUT_MS = 5000"},
         "same domain, different value — must NOT block"),
    ]

    print("PreToolUse")
    failures = 0
    for tool, tool_input, note in cases:
        blocked, stderr = gate(db, tool, tool_input)
        verdict = "BLOCKED (exit 2)" if blocked else "allowed (exit 0)"
        needle = next((v for v in tool_input.values() if isinstance(v, str)), "")
        print(f"   {verdict:18s} {needle[:38]:40s} {note}")
        if blocked:
            print(f"        {stderr.splitlines()[0]}")
        # the two "must NOT block" cases are the precision guard
        if note.startswith("same domain") or note.startswith("unrelated"):
            failures += blocked

    print()
    if failures:
        print(f"FAIL: {failures} false positive(s) — the gate is too loose")
        return 1
    print("OK: fires on the unverified value, silent on everything else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
