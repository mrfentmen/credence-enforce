#!/usr/bin/env python3
"""
quickstart.py — Credence in 30 lines, no API key, no MCP server.

Shows the whole loop:

  1. A value the user hedged gets registered as UNVERIFIED.
  2. Code that embeds that value is scored, and blocked.
  3. The user confirms the value; the block clears.

Run:
    python3 examples/quickstart.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.matching import evaluate_constraints, explain
from credence.registry import CredenceRegistry


def main() -> int:
    db = os.path.join(tempfile.mkdtemp(prefix="credence-quickstart-"), "registry.db")
    registry = CredenceRegistry(db_path=db)
    session = "quickstart"

    # 1 ── the user hedges a value
    cid = registry.register(
        content="I think the Stripe rate limit is around 100 req/min",
        session_id=session,
        j_score=0.30,
        zone="LOW",
        source="user_stated",
        constraint_type="vendor_claim",
    )
    print("1. registered as unverified")
    print(f"   id={cid[:12]}  session={session}")

    # 2 ── code that embeds it
    action = "Edit stripe_client.py RATE_LIMIT = 100"
    blocking = evaluate_constraints(action, registry.list_uncertain(session))
    print()
    print(f"2. gate on: {action}")
    if blocking:
        c = blocking[0]
        print(f"   BLOCKED by {c['_verdict']['reason']} agreement")
        print(f"   shared values: {c['_verdict']['shared_values']}")
        print(f"   why: {' '.join(explain(action, c['content']))[:60]}")
    else:
        print("   allowed")
        return 1

    # 3 ── the user confirms
    registry.verify(cid, verified_value="100 req/min per stripe.com/docs")
    remaining = evaluate_constraints(action, registry.list_uncertain(session))
    print()
    print("3. user confirmed the value")
    if remaining:
        print(f"   still blocked by {len(remaining)} constraint(s)")
        return 1
    print("   gate cleared — this write is now allowed")

    print()
    print("Notes")
    print("  • The block came from a shared VALUE (100) plus the domain terms.")
    print("  • A write of a *different* value in the same domain is not blocked:")
    other = "Edit stripe_client.py TIMEOUT_MS = 5000"
    print(f"    {other}")
    print(f"    -> {'blocked' if evaluate_constraints(other, registry.list_uncertain(session)) else 'allowed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
