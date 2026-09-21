# Changelog

All notable changes to credence-guard are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] — 2026-09-21

First release of the `credence-enforce` fork. Distribution name, version, and
package metadata changed; the import package is still `credence`.

### Fixed
- **The enforcement gate only fired on prose, never on code** — including the
  README's own headline example. The hook compared tokens by exact match, so
  `RATE_LIMIT = 50` after "I think the rate limit is 50" produced zero overlap
  and the write went through unblocked. The enforcing path had no tests at all;
  `tests/unit/test_gate.py` covered a different gate. Matching now goes through
  one canonical scorer (`credence/matching.py`) shared by the hook, the probe,
  and prompt building, with identifier-aware splitting (camelCase/snake_case),
  `_CE_DOMAIN_SYNONYMS` expansion, and a numeric-value rule.
- **The MCP `credence_gate` tool had its own copy of the matcher, and the copy
  was also wrong.** It lower-cased before splitting, so `RATE_LIMIT` stayed a
  single token and never matched the prose terms "rate"/"limit"; it found one
  shared term where two are required. Measured on the README's example it
  answered PROCEED while the `PreToolUse` hook blocked the identical input —
  two enforcement doors, two different answers, so which one you got depended on
  which the agent happened to use. It now calls `credence.matching`.
- **The MCP `credence_autoverify` tool carried a 20-word stopword list against
  the gate's ~190, with no synonym or identifier handling.** This one is worse
  in kind than a missing block: autoverify is what flips a constraint to
  VERIFIED, and VERIFIED is precisely what makes the gate allow a write. A
  matcher more generous than the gate's silently disarms enforcement rather than
  merely failing to apply it. It now calls `credence.matching` too.
- **`credence demo` computed its blocking set with its own copy as well** — so
  the smoke test a stranger runs first could contradict the enforcement they
  would actually get. It now calls `credence.matching`.
- **The Consistency Enforcer had the same blind spot a third time.** It
  tokenised with `text.lower().split()`, so `RATE_LIMIT = 100` became the single
  term `rate_limit`, never matched `rate`/`limit`, and the enforcer did not fire
  on any code-shaped query. It also discarded tokens of two characters or fewer,
  which made a value like `50` invisible. Measured before the fix: `RATE_LIMIT =
  100` fired, `stripeClientRateLimit = 100` did not. That path is what makes the
  model express uncertainty instead of asserting a value as fact, so on code —
  most of what a coding agent sees — it was silent. It now delegates to
  `credence.matching`.
- **The observer silently registered nothing on a fresh install** — it returned
  early instead of writing the constraint, so the gate downstream had nothing to
  enforce.
- **Observer and gate disagreed on session identity** — constraints were
  registered under one key and looked up under another, so enforcement saw an
  empty set. Session resolution is now shared (`resolve_session_id`).
- **`ContextManager` could not be constructed without the `anthropic` SDK.**
  That made the deterministic layers — probe, Truth Buffer, Consistency Enforcer,
  GTS, and every prompt-building path — unusable without it, and left
  `tests/integration/test_mock_llm.py` unable to run (5 failed, 15 errors) while
  its docstring claimed "no API key required". The client is now built on first
  use; its absence reports how to install it.
- **Perf tests measured wall-clock mean** and so failed under scheduler noise.
  They now use min-of-N with latency budgets defined once in `bench_all.py` and
  read by `test_perf.py`, so the CLI and the gate cannot disagree. The same
  defect in `tests/integration/test_session_mock.py::test_wrap_overhead_under_2ms`
  is fixed the same way — it failed under load with the code untouched. Both
  keep their original budgets; only the statistic changed. Verified under six
  concurrent CPU burners, and by mutation: making `wrap()` 3ms slower still
  fails the test (min 4.31ms against the 2.0ms budget).
- Docs: license badge said MIT while the project ships Apache 2.0; test counts
  said 829 against 898 actual.

### Added
- `credence/matching.py` — canonical constraint matcher, with
  `tests/unit/test_matching.py` pinning its synonym map to
  `context_manager._CE_DOMAIN_SYNONYMS` so the two cannot silently diverge.
- `tests/unit/test_hook_enforcement.py` — first coverage for the `PreToolUse`
  enforcing path, driven through the real hook entry point.
- Runnable `examples/quickstart.py` and `examples/hook_demo.py`, plus
  `tests/unit/test_examples.py` which executes them (the previous `examples/`
  were four scripts importing a nonexistent `esm` module).
- `tests/unit/test_matcher_parity.py` — asserts the enforcement paths agree with
  each other on a shared corpus. A test per implementation cannot catch four
  implementations disagreeing; only running them against the same input can.
  Pinned by mutation: restoring the gate's original matcher fails the tests,
  including the hook-versus-gate comparison; restoring the enforcer's original
  tokenizer fails the code-shaped corpus rows.
- `evaluate(..., expand_synonyms=True)` — the enforcer's recall-first mode, as an
  argument rather than a second matcher. Blocking keeps the literal-only rule
  (expanding both sides lets one shared cluster key satisfy the threshold by
  itself, so `TIMEOUT_MS = 5000` would be blocked merely because an unverified
  constraint mentions a different timeout). Warning wants the opposite trade: a
  spurious warning costs a sentence, a missed one costs the false certainty the
  project exists to prevent.
- `tests/conftest.py` — puts the repo root on `sys.path` once. Every test module
  previously repeated that manipulation above its imports, which is why
  `tests/` carried 39 `E402 module-import-not-at-top-of-file` violations: the
  import genuinely could not be at the top until the path existed.
- `NOTICE` — Apache 2.0 attribution for the upstream project, and the fork's
  provenance. Shipped in the wheel at `dist-info/licenses/`.
- `docs/FORK.md` — what this fork changes and why, the modified-file list
  required by Apache 2.0 §4b, and the prerequisites for publishing.
- `MANIFEST.in` — `LICENSE`, `NOTICE`, and the docs ship in the sdist.

### Removed
- `mcp_server._expand_tokens` and `mcp_server._AUTOVERIFY_STOPWORDS` — the
  private second implementations behind the two bugs above. A regression test
  asserts neither name comes back.
- `credence_runtime/` — shipped inside the wheel but unimportable (depended on a
  nonexistent `esm` module). Nothing referenced it.
- `examples/{quickstart,langchain_router,compliance_logging,dashboard}.py` — all
  four failed on `from esm import ...`. Replaced by the working examples above.

### Changed
- `Makefile` rewritten. Five of its six targets referenced files and scripts that
  do not exist in this project; `make test` and `make lint` were among them.
- Two more latency assertions converted from mean to min-of-N, with budgets
  unchanged: `test_gate_latency_under_5ms` and
  `test_wrap_overhead_under_2ms`. Both measured the machine rather than the code.
  The gate's margin narrowed for a real reason — see the note under Fixed about
  the enforcer now using the identifier-aware matcher, which took a
  20-constraint call from ~0.2ms to ~2.4ms. That is the price of catching
  `RATE_LIMIT = 100`, and it is paid once per user turn.
- `tests/` lint: `module-import-not-at-top-of-file` violations down from 39 to
  20, by removing the per-file `sys.path` juggling in every file this fork
  touched. The remaining 20 are in files left alone.
- Package metadata: `credence-guard` 1.2.5 → `credence-enforce` 1.3.0, with
  upstream authorship preserved alongside the fork's. Licence is now the PEP 639
  expression `Apache-2.0` with `license-files = ["LICENSE", "NOTICE"]`; the
  redundant `License :: OSI Approved :: Apache Software License` classifier was
  dropped (PEP 639 rejects the two together, and the build fails without this).
- `server.json` re-pointed at the fork's own registry namespace. The upstream
  `repository.id` was removed rather than carried over — that field exists to
  detect repository resurrection, so keeping upstream's ID with a new URL would
  defeat its purpose.
- Enforcement decisions are now module-level functions
  (`mcp_server.blocking_constraints`, `mcp_server.confirmable_constraints`)
  instead of logic written inside `@mcp.tool()` closures. `fastmcp` is imported
  in a `try`/`except` and the tools are only registered when it is present, so
  a decision written inside a tool body cannot be called by a test in an
  environment without it — which is exactly why the gate kept a broken matcher
  for so long. The tools now only adapt input and output.
- Install strings, the demo footer, `install_gate` guidance, `SECURITY.md`,
  `CONTRIBUTING.md`, `docs/ETP_SPEC.md`, and the issue template no longer name
  `credence-guard` or the upstream repository as this package's install source.
- `SECURITY.md` no longer publishes an inbound email address. GitHub `noreply`
  addresses do not accept mail, so the private advisory link is the real channel;
  the file says so.

---

## [1.2.5] — 2026-05-06

### Changed
- **`fastmcp` is now a hard dependency** — install command is `pip install credence-guard` (no `[mcp]` extra required). The MCP server is the primary interface; the extra was unnecessary friction.
- **MCP registry**: fixed `server.json` name casing (`io.github.Lakshmi-Chakradhar-Vijayarao/credence`) to match GitHub username.

---

## [1.2.4] — 2026-05-06

### Added
- `credence --version` / `credence -V` flag — prints installed package version.
- 6 missing tools added to CLAUDE.md quick reference: `credence_session_summary`, `credence_project_status`, `credence_scan_ghosts`, `credence_audit`, `credence_diff`, `credence_reset`.

### Fixed
- SECURITY.md: corrected "no per-session rate limit" claim — cap was already implemented at 500/session (`CREDENCE_MAX_CONSTRAINTS`).
- README: documented `CREDENCE_MAX_CONSTRAINTS` env var and `credence stats` / `credence feedback` CLI commands.

---

## [1.2.3] — 2026-05-06

### Fixed
- Added `anthropic` to `[dev]` extras so `test_mock_llm.py` integration tests can instantiate `ContextManager` without requiring a separate `pip install anthropic`.
- Added MCP registry ownership token to README for `registry.modelcontextprotocol.io` submission.

---

## [1.2.2] — 2026-05-06

### Fixed
- PyPI publish pipeline: removed broken OIDC trusted-publishing config (`environment: pypi`), switched to `PYPI_API_TOKEN` secret. v1.1.0 and v1.2.0 were published to PyPI for the first time.
- `CREDENCE_NO_LOG=1` opt-out added to `hooks.py` gate event log.
- Data storage documented in README and SECURITY.md (`epistemic_registry.db` + `~/.credence/events.jsonl`).

---

## [1.2.0] — 2026-05-06

### Changed
- **Observer two-tier marker architecture** — strong markers fire unconditionally; weak markers (`around`, `seems like`, `i guess`, `docs say`) now require a co-present numeric value before registering. Eliminates false positives like "wrap around the list" while adding `_NUMERIC_RE` fix (`\b` removed) to catch unit-glued numbers: `30s`, `5MB`, `50ms`.
- **Observer detection coverage**: 59% → 95% on a 22-phrase probe; false positive rate: 50% → 8%.
- **`TEMPORAL_J_SCORES`** extracted to `credence/temporal_patterns.py` — single source of truth, imported by both `mcp_server.py` and `__main__.py`.
- **Gate and scan display**: internal `[stale:…]` and `[AI-generated:…]` DB prefixes stripped at all user-facing output points.

### Fixed
- All 829 tests passing, 1 skipped. (22 tests removed: `test_enforce.py` and `test_manifest.py` covered `credence/enforce.py` and `credence/epistemic_manifest.py`, which were relocated to `experimental/`.)
- `CREDENCE_DB_PATH` → `CREDENCE_DB` in `credence/__main__.py` (canonical env var).
- Ruff: all 20 lint errors resolved (16 auto-fixed, 4 manual E701).

### Removed
- `evals/fcr_downstream_results.json` — v2 scorer (incorrect; contradicted canonical v3 result).
- `evals/compression_faithfulness_results_groq.json`, `_hf.json` — superseded by `compression_faithfulness_n50_results.json`.
- `evals/eqlr_compressor_results.json`, `evals/experiment_results.json` — superseded.
- `evals/data/`, `evals/training/` — DPO training pipeline (dormant; data on HuggingFace).
- `docs/LAUNCH.md`, `docs/MCP_REGISTRY_SUBMISSION.md` — internal process documents.
- `sdk/typescript/` → relocated to `experimental/typescript/` (Phase 2, not yet shipped).
- `credence/enforce.py`, `credence/epistemic_manifest.py` → relocated to `experimental/`.

### Added
- `paper/PAPER_DRAFT.md` — full arXiv-style research paper draft.
- `paper/figures/` — 6 publication-ready figures (PDF + PNG, reproducible generation scripts).
- `experimental/` — home for Phase 2 unshipped work with explicit README.
- `docs/README.md` — navigation index for the docs/ directory.

---

## [1.1.0] — 2026-05-06

### Added
- **Observer hook** (`credence/observer.py`) — passive `UserPromptSubmit` hook. Registers uncertain values before the model generates a single token. Zero API, zero config.
- **`credence_session_summary`** — plain-English digest of all unverified constraints for a session; structured for agent handoffs.
- **`credence_diff`** — detects epistemic contradictions between two texts/agent responses. Returns matched claims, contradictions, divergence score.
- **`credence_project_status`** — project-wide epistemic health across all sessions: `CLEAN / LOW_DEBT / MEDIUM_DEBT / HIGH_DEBT`.
- **`credence_scan_ghosts`** — flags constraints that match ghost heuristics (numeric + domain keyword, no documentation reference).
- **`credence_marker_health`** — diagnostic for marker precision; returns "insufficient data" until usage thresholds are met.
- **`credence_bandit_status`** — shows adaptive threshold learning state; returns "learning" below activation threshold.
- **`credence demo`** CLI entry point — 30-second smoke test, no API key required.

### Fixed
- All 851 tests passing, 0 failures.
- Stale count references corrected across all documentation.

---

## [1.0.0] — 2026-05-02

### Added
- **Faithfulness probe** — deterministic uncertainty-marker detection (0.017ms P50, 0% FCR). Blocks compression when uncertainty qualifiers are present in the segment being compressed.
- **11-tool MCP server** (`credence-server`) — zero API key, zero config. Drop-in Claude Code integration via `.mcp.json`.
- **Rust PreToolUse gate** (`credence-gate`) — 3.4ms native binary, 98× faster than Python hook. Blocks irreversible tool calls when unverified constraints overlap the planned action.
- **Generation-Time Scanner (GTS)** — annotates unverified numeric literals inline in generated code and prose before they ship.
- **Consistency Enforcer** — fires imperative enforcement when a user query keyword-overlaps a registered unverified constraint (≥2 non-stopword terms, synonym-expanded).
- **Truth Buffer** — injects all unverified constraints as epistemic context before every generation turn.
- **CredenceRegistry** — SQLite-backed constraint store with trajectory tracking, confidence decay, per-type decay rates, and cross-session memory.
- **CredenceMemory** — cross-session epistemic memory (`snapshot` → `recall_and_inject`).
- **`wrap()` API** — model-agnostic faithfulness wrapper for any `Callable[[str], str]` compression function.
- **`measure_fcr()`** — offline False Certainty Rate measurement utility.
- **`enforce()` / `CredenceViolation`** — decorator-based enforcement for functions that consume uncertain values.
- **EpistemicManifest** — session-level epistemic health summary.
- **EQL Benchmark** (n=50) — compression faithfulness study with confidence intervals. Haiku: 26% FCR → 0%. LLMLingua-sim: 70% FCR → 0%.
- **Epistemic Transport Protocol** spec (`docs/ETP_SPEC.md`, `etp_schema.json`).
- **Latency report** — all enforcement checkpoints measured. Worst-case P99 overhead: ~5.2ms (~0.10% of a typical Claude Opus API call).

### Architecture
- Zero API key required for all enforcement operations.
- All enforcement layers are deterministic string operations — no model calls at enforcement time.
- `CREDENCE_DB_PATH` / `CREDENCE_REGISTRY_PATH` env vars configure registry location (Rust gate and Python server both respect these).

---

## [Unreleased]

### Planned
- Pre-built `credence-gate` binary as PyPI platform wheel (removes `cargo build` requirement).
- GitHub Actions `credence-check` integration.
- ETP adoption in external agent frameworks.

### Already shipped (not yet tagged)
- Per-session constraint cap: 500 constraints/session default, override with `CREDENCE_MAX_CONSTRAINTS`. Implemented in `registry.py:register()`.
