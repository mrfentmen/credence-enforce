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
- **Three layers resolved the registry path three different ways, so setting
  the documented variable could disable enforcement.** `credence/hooks.py` and
  `credence/observer.py` read only `CREDENCE_DB`; `credence/mcp_server.py` read
  `CREDENCE_DB_PATH` → `CREDENCE_DB` → `CREDENCE_REGISTRY_PATH`; the Rust gate
  read `CREDENCE_DB_PATH` → `CREDENCE_REGISTRY_PATH` and not `CREDENCE_DB` at
  all. `mcp_server.py` calls `CREDENCE_DB_PATH` canonical, so a user who set it
  pointed the MCP tools at one database while the observer *registered*
  constraints into `epistemic_registry.db` in the working directory and the
  hook *looked* there — unless the working directory differed, in which case
  the two did not even agree with each other. A gate that opens the wrong file
  finds no constraints and allows every write.

  There is now one resolver, `credence.matching.resolve_db_path`, used by the
  hook, the observer, and the MCP server, with the Rust gate implementing the
  same chain. Pinned by `test_hook_honours_credence_db_path`, which blocks a
  write using only `CREDENCE_DB_PATH` set — verified by mutation.
- **The Rust gate never read `CREDENCE_DB`, so the documented setup disabled
  it.** `credence/hooks.py` and `credence/observer.py` resolve `CREDENCE_DB`,
  README.md tells users to set it, and `.github/workflows/ci.yml` builds the
  crate as a drop-in for those two layers. The binary resolved
  `CREDENCE_DB_PATH` → `CREDENCE_REGISTRY_PATH` → default instead. A user who
  followed the README had the gate open a different, empty database, find no
  constraints, and allow every write — enforcement that looks installed and is
  inert. The chain now matches `mcp_server.py`, which already read all three.

  The same binary also carries the identifier-blind tokeniser: it filters on
  `w.len() > 2` after `split_whitespace()`, so `RATE_LIMIT` is one token and
  `SYNONYM_CLUSTERS` has no `rate_limit` entry to bridge it. This is a code
  reading, not a measurement — `cargo` is not available on the machine this was
  written on, so the binary was never executed. `tests/unit/test_rust_gate_parity.py`
  asserts the agreement and skips until the crate is built.
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
- **The gate blocked read-only tools.** The Python hook carried no
  enforced-tool list at all, so it scored every tool name and every argument.
  One prompt containing a plausible value ("the timeout is 30 seconds")
  registered `30`, and from then on a `Read` with `offset=30`, a `Grep` for
  `"100"`, a `Glob` pattern, `WebSearch`, `TodoWrite` — any read-only call whose
  arguments happened to contain that number — exited 2. Enforcement now applies
  only to tools that can persist a value (`Write`, `Edit`, `MultiEdit`,
  `NotebookEdit`, `Bash`), which is what the Rust gate already did and what
  `docs/VISION.md` states: gate the action, not the text. Measured after the
  fix: 5 writing tools block, 7 read-only tools allow, and a write with no
  unverified value still allows.
- **`MultiEdit` bypassed enforcement in the documented setup.** All four
  hand-written `matcher` regexes — `README.md`, `docs/INTERNALS.md`,
  `credence/hooks.py`, and the `credence install` snippet — read
  `Write|Edit|Bash|NotebookEdit` and omitted `MultiEdit`, a tool that writes
  files. The docs are not a fallback here: `credence install` exits before
  printing the snippet when Rust is absent, so copying the docs is the only
  path for anyone without a Rust toolchain, and that path gated four of five
  writing tools. The list now lives once, as `ENFORCED_TOOLS` in
  `credence/matching.py`, and every snippet and regex is derived from it.
- **The Rust gate reached different verdicts than the Python hook on the same
  write.** `credence_gate/` is offered as a faster drop-in for
  `credence/hooks.py`, which makes it a second implementation of the blocking
  decision — and it had been written independently. Five divergences, each of
  which changes the answer for some write:

  - no identifier splitting for camelCase — the tokeniser lowered before
    splitting, so `stripeClientRateLimit` stayed one dead token and only the
    numeric rule could rescue it. This is the one the CI run caught first, on
    `Write stripeClientRateLimit = 100`;
  - no numeric-value rule at all, so `Write TIMEOUT = 100` against a constraint
    about a `100 req/min` rate limit was allowed here and blocked there;
  - synonym-cluster agreement counted as blocking evidence, where the canonical
    matcher excludes it — expanding both sides lets one shared cluster key
    satisfy the threshold on its own, so same-domain writes carrying unrelated
    values blocked here and passed there;
  - a 60-word stopword list against the canonical 78, differing in both
    directions: it stopped code words the canonical list scores ("write",
    "file", "code", "function", "method") and scored 45 the canonical list
    stops ("think", "know", "want", "is", "of", "size", "error");
  - its usage docstring advertised `Write|Edit|Bash|NotebookEdit` — the sixth
    and last copy of that regex, and the last one still omitting `MultiEdit`.
    That is the snippet users copy out of the file that installs the gate.

  All five now follow `credence/matching.py`. The tokeniser splits identifiers
  by scanning rather than with `(?<=[a-z0-9])(?=[A-Z])`, because the Rust
  `regex` crate has no lookaround. The pattern cache is hoisted into
  `OnceLock` for the same reason the rule exists at all: `tokenize` runs once
  per constraint on every tool call, and the gate's budget is single-digit
  milliseconds.
- **The Rust gate read only the top-level strings of `tool_input`, so
  `MultiEdit` was never checked.** `extract_arguments_text` mapped over the
  object's values and kept the ones that were strings. `MultiEdit` carries its
  edit text in `edits` — an array of objects — so for that tool the argument
  text was the file path and nothing else, and a write embedding an unverified
  value was allowed while `credence/hooks.py`, which flattens recursively,
  blocked the identical payload. Non-string values were dropped the same way,
  including bare numbers, which is half the blocking rule. Found by CI rather
  than by reading: the existing corpus handed both paths the same pre-flattened
  string inside a Write-shaped payload, which is exactly the shape that hides a
  field the gate never reads. `flatten` now recurses through objects and arrays
  with the same depth cap as `_flatten`, so the two cannot disagree about what
  an action contains. Verified by CI on the failing commit: `MultiEdit` was the
  only tool out of five to disagree (`assert 0 == 2`), because it is the only
  one whose text is nested.
- **Nothing ran the Rust parity test, so none of that was visible.**
  `tests/unit/test_rust_gate_parity.py` skips when the release binary is
  absent; the `rust-gate` job built the binary and stopped, and `test` ran in a
  separate job with no binary. The file skipped everywhere, CI included, and
  its own docstring recorded that wiring it up "has deliberately not been
  made" — which is how a test that cannot fail comes to look like a test. The
  job now installs Python and runs it, and the failures above are what came
  back.
- **A failed registration in the observer was swallowed, so inert enforcement
  looked like a quiet conversation.** `observe()` ended in
  `except Exception: return False`, which made the worst outcome indistinguishable
  from the best one: an unwritable registry registered nothing, the gate
  downstream found nothing to enforce, every write was allowed, and nothing
  anywhere said so. The exit code contract is unchanged — the observer still
  returns 0, because failing a user's prompt is not its job — but the failure
  now reaches stderr, which Claude Code surfaces, and the event log, which
  `credence stats` reads. This is the third time this module's silence hid the
  same class of problem (a fresh install that registered nothing; layers that
  disagreed about the registry path).
- **The event log path was written in three places and read in one.**
  `~/.credence/events.jsonl` was an inline `expanduser` string in `hooks.py`,
  `credence stats`, and `credence feedback` — so a writer and a reader could
  disagree about the file and the only symptom would be "no events yet". That
  is the same failure `resolve_db_path` already removes for the registry. The
  path is now `matching.events_file()` and the writer is `matching.log_event()`,
  resolved per call rather than at import, because the hooks run as fresh
  subprocesses where `$HOME` is whatever the caller set.
- Docs: license badge said MIT while the project ships Apache 2.0; test counts
  said 829 against 898 actual.

### Added
- `matching.events_file()` / `matching.log_event()` — one definition of where
  the event log lives and one writer for it, so the hooks that append and the
  commands that read cannot drift apart. O10 in `tests/unit/test_observer.py`
  fails if a second copy of the path comes back.
- `tests/unit/test_observer.py` O9 — a registration that fails must be reported.
  Points `CREDENCE_DB` at a directory sqlite cannot create, then asserts the
  exit code is still 0 *and* that the failure reached both stderr and the event
  log. Without it, the swallowed-exception version passes every other observer
  test in the file.
- `credence/matching.py` — canonical constraint matcher, with
  `tests/unit/test_matching.py` pinning its synonym map to
  `context_manager._CE_DOMAIN_SYNONYMS` so the two cannot silently diverge.
- `tests/unit/test_hook_enforcement.py` — first coverage for the `PreToolUse`
  enforcing path, driven through the real hook entry point.
- Runnable `examples/quickstart.py` and `examples/hook_demo.py`, plus
  `tests/unit/test_examples.py` which executes them (the previous `examples/`
  were four scripts importing a nonexistent `esm` module).
- `tests/unit/test_rust_gate_parity.py` — the fourth enforcement path is the
  Rust binary, and nothing compared it to the Python side. Asserts the gate
  blocks and allows the same corpus rows as `credence/matching.py`, and that it
  honours `CREDENCE_DB`. Skips unless `credence_gate/target/release/credence-gate`
  exists.
- `tests/unit/test_matcher_parity.py` — asserts the enforcement paths agree with
  each other on a shared corpus. A test per implementation cannot catch four
  implementations disagreeing; only running them against the same input can.
  Pinned by mutation: restoring the gate's original matcher fails the tests,
  including the hook-versus-gate comparison; restoring the enforcer's original
  tokenizer fails the code-shaped corpus rows.
- `tests/unit/test_matcher_parity.py` also pins the **tool scope**: the Rust
  `enforced_tools` literal, every documented `matcher` regex, and
  `credence.matching.ENFORCED_TOOLS` must gate the same set of tools. It
  compares what each matcher *gates* rather than its literal text, because
  alternation order carries no meaning in a regex — the first version compared
  strings and failed on a mere reorder, which only teaches people to ignore a
  test. Pinned by mutation: dropping `MultiEdit` from `README.md` fails and
  names `['MultiEdit']`; reordering the same regex passes. `credence_gate/src/main.rs`
  is in that list too — its usage docstring was the sixth copy of the regex and
  the last one still missing `MultiEdit`.
- `test_rust_gate_uses_the_canonical_stopword_list` — parses the Rust stopword
  literal and compares it to `matching.STOPWORDS`. Read as source, so it runs
  on machines without cargo; the two lists differed in both directions for as
  long as nothing compared them.
- Corpus rows that isolate the Rust matcher's divergence classes rather than
  catching them as a side effect: `Write stripeClientRateLimit = 999` blocks
  only if identifiers are split (its value is not the constraint's), and
  `Write throttle = 25` must NOT block — same domain, different value, reachable
  only through a synonym, which is exactly the verdict the Rust gate used to get
  wrong.
- `test_rust_gate_shares_the_canonical_thresholds` — `MIN_OVERLAP`, the
  minimum numeric-literal length, and the flatten depth cap are single integers
  that decide whether a write is blocked at all, and each existed as an
  independent copy in Python and Rust. The stopword list had already drifted
  60-against-78 in both directions before anything compared it, so these are
  now read out of the Rust source and compared to `matching.MIN_OVERLAP`,
  `matching._MIN_NUM_LEN`, and `hooks._MAX_FLATTEN_DEPTH` (the last promoted
  from an inline literal so there is one place to change it). Runs without
  cargo. Pinned by mutation: changing the Rust `MIN_OVERLAP` to 3 fails it.
- Real per-tool payload shapes rather than one action string: `TOOL_PAYLOADS`
  and `UNRELATED_PAYLOADS` in `tests/unit/test_matcher_parity.py`, driven
  through the `PreToolUse` hook by P9 and through the Rust binary by X5. A gate
  that scores correctly but reads the wrong field blocks nothing, which is
  invisible to a test that flattens the input for both paths. Both maps must
  cover every enforced tool, so adding one to the allowlist without a payload to
  demonstrate its extraction fails rather than passing silently.
- The `rust-gate` CI job now runs the Rust parity tests after building the
  crate, so those assertions execute against a real binary instead of skipping.
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
- Four more latency assertions converted from mean to min-of-N, with budgets
  unchanged: `test_gate_latency_under_5ms`, `test_wrap_overhead_under_2ms`, and
  both registry tests. All of them measured the machine rather than the code.
  `test_register_latency_under_5ms` failed on a Python 3.12 CI runner with
  "Register too slow: 16.76ms" while the 3.11 runner passed — the same defect,
  not a version difference. Its name said 5ms against a 15ms budget; the budget
  is unchanged and the name now matches it.
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
