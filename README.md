# Credence Enforce

AI doesn't remember what it wasn't sure about. Credence does.

[![CI](https://github.com/mrfentmen/credence-enforce/actions/workflows/ci.yml/badge.svg)](https://github.com/mrfentmen/credence-enforce/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **This is a fork of [Credence](https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai)
> by Lakshmi Chakradhar Vijayarao** (Apache 2.0; upstream PyPI name `credence-guard`).
> The published distribution here is `credence-enforce`.
> The fork exists to repair the `PreToolUse` gate — upstream's did not fire on code,
> only on prose. See [`docs/FORK.md`](docs/FORK.md) for the full list of changes and
> [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the reasoning behind each.
> The import package is still `credence`, so do not install this alongside
> `credence-guard`.

```bash
pip install credence-enforce
credence demo   # 30-second smoke test, no API key required
```

`fastmcp` is a hard dependency (since upstream 1.2.5), so the MCP server works
out of the box. The core deterministic layers — probe, Truth Buffer, Consistency
Enforcer, and the `PreToolUse` gate — need no API key and no `anthropic` package;
the SDK is imported only when a model call is actually made.

---

## The problem

You say: *"The rate limit is probably around 50 — I haven't confirmed it yet."*

Fifteen turns later, Claude writes:

```python
RATE_LIMIT = 50   # no warning. no flag. shipped.
```

The API rejects every request at 2am. The real limit was 10. Claude forgot you weren't sure.

This isn't hallucination. The model reproduced exactly what it read. What it read had the qualifier stripped — by context compression, fifteen turns back.

---

## What Credence does

Tracks uncertain values the moment you state them. Blocks writes that embed those values until you confirm them.

```
you say "rate limit is probably 50"
    → observer registers it (before Claude responds)
    → Claude writes: RATE_LIMIT = 50  # ⚠ CREDENCE[unverified]
    → write blocked until you confirm
```

Every other tool warns. Credence enforces.

---

## What it looks like

```python
# Claude generates this. Credence intercepts before it ships.

class StripeClient:
    API_VERSION  = "2023-10-16"  # ⚠⚠ CREDENCE[stale]: API date versions change on release — verify before shipping
    RATE_LIMIT   = 100           # ⚠  CREDENCE[unverified]: I think Stripe rate limit is around 100 req/min
    TOKEN_EXPIRY = 3600          # ⚠⚠ CREDENCE[stale]: Token/session lifetime values are set by the vendor — verify
    MAX_RETRIES  = 3
    TIMEOUT_MS   = 5000
```

```
credence: blocked Edit — 2 unverified value(s)
  → I think Stripe rate limit is around 100 req/min | TOKEN_EXPIRY = 3600
  Verify first, then retry. Use credence_constraints to see all pending.
```

After you confirm: `"Confirmed — rate limit is 100 req/min per stripe.com/docs"` → gate clears.

![Gate demo](demo/gate_demo.gif)

---

## Setup

**1. Add to `.mcp.json`:**
```json
{ "mcpServers": { "credence": { "command": "credence-server" } } }
```

**2. Add to `.claude/settings.json`:**
```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "python3 -m credence.observer" }] }
    ],
    "PreToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit|NotebookEdit|Bash",
        "hooks": [{ "type": "command", "command": "python3 -m credence.hooks" }]
      }
    ]
  }
}
```

Done. No API key required.

> **Registry:** Credence creates `epistemic_registry.db` in your working directory. Add `*.db` to your `.gitignore`, or set `CREDENCE_DB=~/.credence/registry.db` to keep it global. `CREDENCE_DB` is what the hook, the observer, and the Rust gate resolve; `CREDENCE_DB_PATH` also works and takes precedence over it, with `CREDENCE_REGISTRY_PATH` accepted as a legacy alias. Every layer resolves them in that order.
>
> **Session tracking:** Set `CREDENCE_SESSION_ID=my-project` to keep constraints stable across directory changes and terminal restarts.
>
> **Event log:** The gate writes block/allow events to `~/.credence/events.jsonl` (local only, never sent anywhere). Set `CREDENCE_NO_LOG=1` to disable.
>
> **Constraint cap:** The registry allows up to 500 constraints per session by default. Override with `CREDENCE_MAX_CONSTRAINTS=<n>`.
>
> **Debugging:** Set `CREDENCE_DEBUG=1` to have the Rust gate log its timing and verdict to stderr for each hook invocation.

---

## How it works

Two layers, neither requires model cooperation:

| Layer | Hook | Role |
|---|---|---|
| Observer | `UserPromptSubmit` | Passive listener — registers uncertain values before Claude generates anything |
| Gate | `PreToolUse` | Blocks writes that embed unverified values |

The observer fires before the model processes your message. If you say "I think the rate limit is 50", the registry has that entry before Claude generates a single token.

---

## What gets blocked

```
credence: blocked Edit — 2 unverified value(s)
  → rate limit is probably 50 req/min | token expires in 3600s
  Verify first, then retry. Use credence_constraints to see all pending.
```

Once verified, the gate clears.

### Which tools are gated

Only tools that can persist a value: `Write`, `Edit`, `MultiEdit`,
`NotebookEdit`, `Bash`. Read-only tools (`Read`, `Grep`, `Glob`, `WebSearch`,
`Task`, `NotebookRead`, …) are never blocked — reading is not irreversible, so
an unverified number appearing in a `Read` argument is not a write that needs
verifying.

The list lives in `credence/matching.py` as `ENFORCED_TOOLS`, and every install
snippet and `matcher` regex is derived from it, so the gate and the docs cannot
disagree. `tests/unit/test_matcher_parity.py` pins that — all four hand-written
copies of this regex once omitted `MultiEdit`, a file-writing tool that then
bypassed the gate entirely in the documented setup.

---

## What Credence does NOT do

- Does not verify facts — it cannot tell you if a value is correct
- Does not catch uncertainty that was never stated
- Does not block the model from *saying* a wrong value in prose — only from *writing* it to a file or command

---

## Measured results

46% of uncertainty qualifiers are stripped by Claude Haiku during context compression. Credence blocks 100% of those writes (n=50, bootstrap CI: [0%–0%]).

Validated across 7 open-weight models (Qwen, Mistral, Llama, Phi, Gemma) from 5 organizations: same failure mode, same block rate.

```bash
credence demo                     # smoke test, no API key
credence stats                    # false-positive rate from real gate usage
credence feedback 1|2|3           # tag last gate block: correct / noise / skip
python3 -m pytest tests/ -q       # 956 tests, 4 skipped, no API key required
python3 -m evals.latency_report   # P50/P95/P99
```

Entry points that need no API key: `credence demo`, `credence stats`,
`credence feedback`, and the full test suite — the `anthropic` SDK is imported
only when a model call is actually made, so the deterministic layers (probe,
Truth Buffer, Consistency Enforcer, GTS) run without it.

Full methodology: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

## Project layout

```
credence/         pip-installable package
  observer.py     passive UserPromptSubmit hook
  hooks.py        PreToolUse enforcement gate
  matching.py     canonical constraint matcher (shared by hook + context)
  mcp_server.py   17-tool MCP server
  registry.py     SQLite constraint store
  memory.py       cross-session persistence
tests/            956 tests, 4 skipped, no API key required
evals/            validation studies + multi-model benchmarks
docs/             technical report, architecture, ETP spec
credence_gate/    Rust gate (alternative to Python hooks.py)
experimental/     Phase 2 work — not yet shipped
paper/            Research paper draft + figures
examples/         runnable quickstart + hook demo (pinned by tests)
```

---

## Research

The scientific basis for Credence is documented in `paper/` (EQL / EQLR / FCR).

The companion geometry thesis — on confabulation detection and why the detection
axis is dissociable from the causal control axis — lives in a separate repo:
→ **[Detection Without Control](https://github.com/Lakshmi-Chakradhar-Vijayarao/detection-without-control)**

---

## Built by

**Lakshmi Chakradhar Vijayarao** — [GitHub](https://github.com/Lakshmi-Chakradhar-Vijayarao) · [LinkedIn](https://www.linkedin.com/in/lakshmichakradharvijayarao/) · [X](https://x.com/LChakradharV28)

Apache 2.0 License

<!-- mcp-name: io.github.Lakshmi-Chakradhar-Vijayarao/credence -->
