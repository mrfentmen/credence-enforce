# Credence — Development Roadmap

**Vision**: Credence is the epistemic runtime for the AI age. The layer that makes
uncertainty machine-readable, transferable, and enforceable across any AI agent,
any compression, any session, any handoff.

**Current state** (2026-05-06):
- 17 MCP tools — shipped as v1.2.0
- Tests: 935 passing, 1 skipped, 0 failing
- Core: faithfulness probe (0.07ms, 0% FCR with Haiku/Opus), Rust gate (3.4ms, 98× faster than Python hook)
- Free-tier validated: EQLR 51% → 0% blocked on llama-3.1-8b-instant (n=98, Groq)
- Observer hook live — fires before model generates, zero API key

---

## Phase 0 — Complete the MCP Server

**Goal**: 17 tools. Zero config. Tested. Ready to ship.
**Principle**: Every addition must have tests. Run full suite after each task.

| # | Task | What it is | File(s) |
|---|------|-----------|---------|
| 0.1 | `credence_score` tool | Expose J-score + zone as diagnostic. Uses `confidence_proxy.py` (already in `credence/`). Helps users understand why a compression was blocked or allowed. | `mcp_server.py` |
| 0.2 | Restore `envelope.py` from archive | Pure Python, zero deps. CredenceEnvelope frozen dataclass with trust decay math. Needed for 0.3/0.4. | `credence/envelope.py` |
| 0.3 | `credence_wrap` tool | Wrap any response as a CredenceEnvelope before passing to next agent. Returns JSON-serializable envelope with trust score. | `mcp_server.py` |
| 0.4 | `credence_unwrap` tool | Inspect envelope from upstream agent. Flags trust < 0.40, resets verified=False on propagation, increments chain_depth. | `mcp_server.py` |
| 0.5 | `source_type` on `credence_register` | Add `source_type` param: `vendor_claim / user_estimate / verified_in_code / hearsay / inference`. Stored in registry. Surfaces in `list_uncertain`. Zero API. | `mcp_server.py`, `registry.py` |
| 0.6 | `credence_autoverify` tool | Scan text for confirmation phrases ("actually", "confirmed", "I checked", "turns out", "verified"). Auto-calls `registry.verify()` on matching constraints. No API. | `mcp_server.py` |
| 0.7 | `credence_session_brief` tool | Plain-English summary of inherited uncertainties for session start: "3 things unverified: rate limit (est. 50, LOW), token expiry (est. 3600, LOW). Verify before writing code." | `mcp_server.py` |
| 0.8 | Tests S27-A through S27-N | Cover all new tools: envelope math, source_type storage, autoverify pattern matching, session_brief format. | `tests/tests.py` |
| ✓ | **Verification checkpoint** | `python3 tests/tests.py` → 0 failures. `python3 -c "import credence"` → clean. Smoke: register → wrap → unwrap → verify flow. | — |

---

## Phase 1 — Production Features

**Goal**: Full audit trail. Uncertainty inheritance in generated code. Passive data collection starts.
**Principle**: Data flywheel infrastructure goes in NOW so it collects from install #1.

| # | Task | What it is | File(s) |
|---|------|-----------|---------|
| 1.1 | `credence_audit` tool | Per-session epistemic timeline. Chronological sequence: `registered → enforced → verified / contradicted`. Uses `registry.get_trajectory()` (already built). | `mcp_server.py` |
| 1.2 | Uncertainty inheritance (AST walk) | After GTS annotates a literal in code, walk the code block with `ast.parse()` to find functions/variables that reference that literal. Annotate call sites too. Pure stdlib. | `mcp_server.py` (extend `_scan_output`) |
| 1.3 | Marker weight tracking schema | New SQLite table `marker_events`: `(session_id, marker, fired_at, fcr_outcome, session_type)`. Wire passive recording into `credence_post_compress`. Collects only — no learning yet. | `registry.py`, `mcp_server.py` |
| 1.4 | Session type detection | Keyword heuristics: `debug` (error/traceback/exception), `design` (architecture/schema/trade-off), `code_review` (review/refactor), `research` (compare/evaluate/benchmark). | `mcp_server.py` |
| 1.5 | Wire session type into tools | `credence_gate` and `credence_session_info` responses include `session_type`. Groundwork for adaptive thresholds in Phase 3. | `mcp_server.py` |
| 1.6 | Tests S28-A through S28-J | Cover audit timeline ordering, AST inheritance annotation, marker_events recording, session type classification for all 4 types. | `tests/tests.py` |
| ✓ | **Verification checkpoint** | `python3 tests/tests.py` → 0 failures. Verify `marker_events` table exists and records after `credence_post_compress` call. | — |

---

## Phase 2 — Become the Standard

**Goal**: Protocol, not just a tool. Contradiction detection. TypeScript. Cross-team.
**Principle**: Publish the spec before the SDK. The spec is what makes it a standard.

| # | Task | What it is | File(s) |
|---|------|-----------|---------|
| 2.1 | `credence_diff` tool | Compare two texts/responses for epistemic divergence. Detect contradictions: "Agent A registered rate_limit=50 (unverified). Agent B says rate_limit=100." Returns: matched_claims, contradictions, divergence_score. Pure Python. | `mcp_server.py` |
| 2.2 | Epistemic Transport Protocol (ETP) spec | JSON schema for epistemic metadata any agent can attach to any response. Fields: `etp_version`, `j_score`, `zone`, `source_type`, `verified`, `chain_depth`, `session_id`. Published as `docs/ETP_SPEC.md` + `etp_schema.json`. | `docs/ETP_SPEC.md`, `etp_schema.json` |
| 2.3 | `credence_project_status` tool | Project-wide epistemic health: total constraints, verified rate, epistemic_debt, top unresolved, session_type breakdown across all sessions. | `mcp_server.py` |
| 2.4 | TypeScript SDK (`credence-ts`) | Reimplement core probe + registry client in TypeScript. Enables native Copilot/Cursor integration without Python. Separate package in `sdk/typescript/`. | `sdk/typescript/` |
| 2.5 | Team epistemic state | Multi-user shared registry via configurable `CREDENCE_DB_PATH`. Multiple developers on same project share constraint state. | `registry.py`, `mcp_server.py` |
| 2.6 | Tests S29-A through S29-H | Cover credence_diff contradiction detection, ETP schema validation, project_status aggregation. TS SDK has separate Jest suite. | `tests/tests.py` |
| ✓ | **Verification checkpoint** | `python3 tests/tests.py` → 0 failures. ETP schema validates. TypeScript SDK builds and tests pass. | — |

---

## Phase 3 — Data Flywheel

**Goal**: System improves from real usage. Adaptive, not static.
**Principle**: Adaptive systems are DORMANT until data thresholds are met. Never activate on zero data.

| # | Task | What it is | Threshold |
|---|------|-----------|-----------|
| 3.1 | Ghost constraint heuristics | Source-type-based ghost detection — zero API. If `source_type = vendor_claim` and no hedging marker present → flag as potential ghost. Catches what probe misses. | Active from day one (deterministic) |
| 3.2 | Marker weight learning (from archive) | Wire Phase 1.3 data into `marker_weights.py` update logic. Tracks precision/recall per marker. Down-weights FP markers. Boosts reliable ones. | Activates at n_sessions ≥ 200 |
| 3.3 | Thompson sampling bandit (from archive) | Per-session-type threshold adaptation. Beta(α,β) per (session_type, theta) arm. Success = qualifier survived. Failure = qualifier stripped. Informed priors from Phase 1 data. | Activates at n_sessions ≥ 100 |
| 3.4 | `credence_marker_health` tool | Diagnostic: top/bottom 10 markers by precision. Shows which markers reliably predict FCR vs. which fire on benign text. | Returns "insufficient data" below threshold |
| 3.5 | `credence_adaptive_status` tool | Shows current bandit state: learned threshold per session type, confidence intervals, convergence. | Returns "learning" below threshold |
| 3.6 | Tests S30-A through S30-L | Ghost heuristics (active), dormancy below threshold, marker_health "insufficient data" path, adaptive_status "learning" path. | `tests/tests.py` |
| ✓ | **Final verification checkpoint** | `python3 tests/tests.py` → 0 failures. Adaptive systems respect thresholds. Ghost heuristics fire correctly. | — |

---

## Model Interaction Architecture

**When Credence genuinely needs model intelligence** (ghost detection, semantic
contradiction resolution, complex verification detection), it must not make its
own LLM API calls. Three clean tiers:

### Tier 0 — Always deterministic, always zero-API (non-negotiable)
The probe, the gate, the GTS, wrap/unwrap, all registry operations. These never
ask anything. This is the hard guarantee that works with any agent, any model,
any configuration. Never add API calls here.

### Tier 1 — MCP Sampling (the right design for hard cases)
MCP spec includes a `sampling` mechanism — the server asks the **host** to make
an LLM call on its behalf. Claude Code, Copilot, and Cursor already have model
access. Credence sends a structured request ("classify the epistemic origin of
this claim") and the host runs it through whatever model it already has.

- Zero API key on Credence's side
- No cost to Credence — absorbed by the host
- Graceful degradation: if host denies sampling, fall back to Tier 0
- Use cases: ghost constraint detection, semantic contradiction resolution
- This is architecturally honest — Credence is infrastructure, not a model.
  When it needs model intelligence, it asks the model it serves.

### Tier 2 — Opt-in API (`credence[plus]`, future)
For users who want behavioral signal or semantic entropy and have an API key.
`behavioral_signal.py`, `claim_extractor.py`, `semantic_entropy.py` live here.
Explicit opt-in. Never hidden, never default.

### Feedback / RL
The Thompson sampling bandit IS the RL layer — feedback signal is `qual_survival`
from `credence_post_compress`. Success = qualifier survived. Failure = qualifier
stripped. Beta distribution updates per session type. Human-in-the-loop
verification (`credence_verify`) is the semantic reward signal. No automated
reward for epistemic correctness — ground truth requires human confirmation.
This is the correct design: RL loop exists, uses the right signal.

---

## Invariants — Never Break These

1. **Zero API key** — no MCP tool may make an LLM API call (Tier 0 is permanent).
2. **Zero config** — `pip install credence-ai` + MCP settings = fully working.
3. **Faithfulness probe stays deterministic** — 108-marker frozenset, pure string scan, 0.017ms.
4. **Rust gate stays fast** — must remain < 10ms.
5. **Tests never regress** — every task ends with 0 failing tests.
6. **Adaptive systems are dormant below threshold** — no random behavior on zero data.
7. **MCP sampling is opt-in** — Tier 1 features degrade to Tier 0 if host denies sampling.

---

## Tool count by phase end

| After phase | MCP Tools | Resources | Tests  | Status          |
|-------------|-----------|-----------|--------|-----------------|
| Baseline    | 11        | 2         | 166    | ✓ shipped       |
| Phase 0     | 16        | 2         | 195    | ✓ shipped       |
| Phase 1     | 17        | 2         | 829    | ✓ shipped v1.2.0 |
| Phase 2     | 19 + TS SDK | 2      | ~950   | planned          |
| Phase 3     | 22        | 2         | ~1050  | planned          |

---

## Out of scope until post-Phase 3

- Federated learning (needs network of users)
- `behavioral_signal.py` (5 Haiku calls — Tier 2 opt-in only)
- `semantic_entropy.py` (3 Haiku + NLI calls — Tier 2 opt-in only)
- `dpo_proxy.py` (PyTorch + model weights — too heavy)
- Full `claim_extractor.py` (Haiku API — Tier 2 opt-in, future `credence[plus]`)
