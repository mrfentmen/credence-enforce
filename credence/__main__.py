"""
credence/__main__.py
=====================
Entry point for: python -m credence  /  credence demo

Run without any setup, any API key, any configuration.
Shows the core product moment in under 30 seconds.
"""

from __future__ import annotations
import os
import sys
import tempfile
import time


def _hr(char: str = "─", width: int = 62) -> None:
    print(char * width)


def _banner() -> None:
    _hr("═")
    print("  Credence — epistemic guard for AI-generated code")
    print("  Every unverified value is flagged before it ships.")
    _hr("═")


def run_demo() -> None:
    """
    30-second smoke test. No API key. No config. No existing session.

    Shows:
      1. A value stated as uncertain in conversation → registered
      2. AI generates code containing that value + stale patterns
      3. credence_self_probe extracts all domain values
      4. credence_scan annotates the output
      5. credence_gate blocks the write
      6. User confirms the value → gate clears
    """
    import re
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        demo_db = tf.name
    os.environ["CREDENCE_DB"] = demo_db

    try:
        from credence.temporal_patterns import scan_temporal, scan_domain_assignments, TEMPORAL_J_SCORES
        from credence.mcp_server import _scan_output, _get_registry
        from credence.matching import evaluate_constraints

        reg = _get_registry()
        SESSION = "credence_demo"

        _banner()
        print()

        # ── Turn 1: user states an uncertain value ──────────────────────────
        print("  💬  You:   I think Stripe's rate limit is around 100 req/min.")
        print()
        time.sleep(0.3)

        cid = reg.register(
            content         = "I think Stripe rate limit is around 100 req/min",
            session_id      = SESSION,
            j_score         = 0.30,
            zone            = "LOW",
            source          = "user_stated",
            constraint_type = "vendor_claim",
        )
        print("  ⚙  credence_register → registered as UNVERIFIED")
        print(f"     id: {cid}  |  source: user_stated  |  state: unverified")
        print()
        time.sleep(0.3)

        # ── Turn 2: AI generates code ───────────────────────────────────────
        generated = '''\
```python
class StripeClient:
    BASE_URL    = "https://api.stripe.com/v1"
    API_VERSION = "2023-10-16"
    RATE_LIMIT  = 100
    TOKEN_EXPIRY = 3600
    MAX_RETRIES  = 3
    TIMEOUT_MS   = 5000
```'''

        print("  🤖  Agent generates StripeClient code...")
        print("      (before Credence intercepts it)")
        print()
        time.sleep(0.4)

        # ── self_probe ──────────────────────────────────────────────────────
        t_hits = scan_temporal(generated)
        d_hits = scan_domain_assignments(generated)

        for h in t_hits:
            reg.register(
                content=h.constraint_content, session_id=SESSION,
                j_score=TEMPORAL_J_SCORES.get(h.pattern_name, 0.20), zone="LOW",
                source="temporal_scan", constraint_type="vendor_claim",
            )
        for h in d_hits:
            reg.register(
                content=h.constraint_content, session_id=SESSION,
                j_score=0.0, zone="LOW",
                source="self_probe", constraint_type="config",
            )

        total = len(t_hits) + len(d_hits)
        print(f"  ⚙  credence_self_probe → {total} values extracted")
        print(f"     {len(t_hits)} stale (temporal patterns)  |  "
              f"{len(d_hits)} domain-uncertain")
        print("     Unknown = unverified. No confidence scoring.")
        print()
        time.sleep(0.4)

        # ── scan → annotated output ─────────────────────────────────────────
        annotated, hits = _scan_output(generated, reg, SESSION, turn=0)

        print("  ⚙  credence_scan → annotated output:")
        print()
        for line in annotated.splitlines():
            print(f"     {line}")
        print()
        time.sleep(0.4)

        # ── gate: blocks write ──────────────────────────────────────────────
        # Scored by credence.matching so the demo shows what the gate actually
        # does. It previously carried its own copy of the overlap rule, which
        # meant the smoke test a stranger runs first could disagree with the
        # enforcement they would get.
        uncertain = reg.list_uncertain(SESSION)
        action = "write stripe_client rate limit token expiry timeout version"
        blocked = evaluate_constraints(action, uncertain)

        _hr()
        print("  ⚙  credence_gate  →  proceed: False")
        print(f"     {len(blocked)} constraint(s) block writing stripe_client.py:")
        def _clean(s: str) -> str:
            s = re.sub(r'^\[stale:[^\]]+\]\s*', '', s)
            s = re.sub(r'^\[AI-generated:[^\]]+\]\s*', '', s)
            return s
        for b in blocked[:4]:
            print(f"     ⚠  {_clean(b['content'])[:130]}")
        print()
        time.sleep(0.4)

        # ── user confirms ───────────────────────────────────────────────────
        print("  💬  You:   Confirmed — rate limit is 100 req/min per stripe.com/docs")
        print()
        time.sleep(0.3)

        confirm = "confirmed rate limit is 100 per stripe docs"
        verified = 0
        for c in evaluate_constraints(confirm, reg.list_uncertain(SESSION)):
            reg.verify(c["constraint_id"], "confirmed: stripe.com/docs")
            verified += 1

        print(f"  ⚙  credence_autoverify → {verified} constraint(s) verified")
        print()
        time.sleep(0.3)

        # ── gate re-check ───────────────────────────────────────────────────
        uncertain_after = reg.list_uncertain(SESSION)
        blocked_after = evaluate_constraints(action, uncertain_after)

        _hr()
        remaining = len(uncertain_after)
        if len(blocked_after) < len(blocked):
            print("  ⚙  credence_gate  →  proceed: True  ✓")
            print(f"     Rate limit verified. {remaining} value(s) still annotated in code.")
        else:
            print(f"  ⚙  credence_gate  →  proceed: False  ({len(blocked_after)} still blocked)")
        print()

        # ── summary ─────────────────────────────────────────────────────────
        _hr("═")
        all_c = reg._conn.execute(
            "SELECT content, verified, source FROM constraints WHERE session_id=?",
            (SESSION,)
        ).fetchall()
        v_count = sum(1 for r in all_c if r["verified"])
        u_count = sum(1 for r in all_c if not r["verified"])

        print(f"  Session summary:  {v_count} verified  |  {u_count} still unverified")
        print()
        print("  This is Credence.")
        print("  Every value that flows from conversation into code")
        print("  is tracked and must be verified before it ships.")
        print()
        print("  Install:  pip install credence-enforce")
        print("  Docs:     github.com/mrfentmen/credence-enforce")
        _hr("═")

    finally:
        try:
            os.unlink(demo_db)
        except OSError:
            pass


def run_feedback(tag: str) -> None:
    """Tag the last gate block in ~/.credence/events.jsonl as useful/noise/skip."""
    import json as _json
    events_file = os.path.expanduser("~/.credence/events.jsonl")

    if not os.path.exists(events_file):
        print("No gate events recorded yet. Run credence in a real session first.")
        return

    with open(events_file) as fh:
        lines = fh.readlines()

    # Find last block event with no feedback
    for i in range(len(lines) - 1, -1, -1):
        try:
            ev = _json.loads(lines[i])
        except Exception:
            continue
        if ev.get("event") == "block" and ev.get("feedback") is None:
            label = {"1": "true_positive", "2": "false_positive", "3": "skip"}.get(tag, "skip")
            ev["feedback"] = label
            lines[i] = _json.dumps(ev) + "\n"
            with open(events_file, "w") as fh:
                fh.writelines(lines)
            print(f"  Logged: last gate block → {label}")
            return

    print("No unfeedback'd block found. All recent blocks already tagged.")


def run_stats() -> None:
    """Print false-positive rate from ~/.credence/events.jsonl."""
    import json as _json
    events_file = os.path.expanduser("~/.credence/events.jsonl")

    if not os.path.exists(events_file):
        print("No events yet. Install the hook and use Credence in a real session.")
        return

    blocks = allows = tp = fp = skip = untagged = 0
    with open(events_file) as fh:
        for line in fh:
            try:
                ev = _json.loads(line)
            except Exception:
                continue
            if ev.get("event") == "block":
                blocks += 1
                fb = ev.get("feedback")
                if fb == "true_positive":
                    tp += 1
                elif fb == "false_positive":
                    fp += 1
                elif fb == "skip":
                    skip += 1
                else:
                    untagged += 1
            elif ev.get("event") == "allow":
                allows += 1

    tagged = tp + fp
    fpr = (fp / tagged * 100) if tagged else None

    _hr("═")
    print("  Credence — field signal report")
    _hr()
    print(f"  Gate fires (blocks) : {blocks}")
    print(f"  Pass-throughs       : {allows}")
    print(f"  Tagged by user      : {tagged}  (true_pos={tp}, false_pos={fp}, skip={skip})")
    print(f"  Untagged            : {untagged}")
    print()
    if fpr is not None:
        flag = "✓ signal is clean" if fpr <= 20 else ("⚠ needs filter tightening" if fpr <= 40 else "✗ too noisy")
        print(f"  False positive rate : {fpr:.1f}%  {flag}")
    else:
        print("  False positive rate : — (tag gate blocks with `credence feedback 1|2|3`)")
    _hr("═")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "demo":
        run_demo()
    elif args[0] in ("--version", "-V", "version"):
        import credence as _c
        print(_c.__version__)
    elif args[0] == "server":
        from credence.mcp_server import main as server_main
        server_main()
    elif args[0] == "feedback":
        tag = args[1] if len(args) > 1 else "3"
        run_feedback(tag)
    elif args[0] == "stats":
        run_stats()
    else:
        print("Usage: credence [demo|server|feedback|stats|--version]")
        print("       credence demo         — run 30-second smoke test (no API key needed)")
        print("       credence server       — start MCP server")
        print("       credence --version    — print installed version")
        print("       credence feedback 1   — last gate block was correct (true positive)")
        print("       credence feedback 2   — last gate block was noise (false positive)")
        print("       credence feedback 3   — skip / unsure")
        print("       credence stats        — show false positive rate from real usage")
        sys.exit(1)


if __name__ == "__main__":
    main()
