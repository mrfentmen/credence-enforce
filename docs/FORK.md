# Fork notes — credence-enforce

This repository is a fork of **Credence** by
[Lakshmi Chakradhar Vijayarao](https://github.com/Lakshmi-Chakradhar-Vijayarao),
upstream at <https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai>,
distributed on PyPI as `credence-guard`.

Licence: **Apache 2.0** (see `LICENSE`). The fork is permitted and the upstream
attribution is preserved in `NOTICE`.

## Why a fork and not a pull request

The upstream repository is Apache 2.0 and accepts contributions, so these fixes
are submittable upstream. Three of them change enforcement *behaviour*, which is
the core claim of the project, so they were developed and verified here first.

## What this fork changes

Every change is listed with its reasoning in `docs/CHANGELOG.md` under
`[Unreleased]`. The four that matter:

| # | Change | Why it matters |
|---|---|---|
| 1 | Every enforcement path now fires on **code**, not only prose | The `PreToolUse` gate, the MCP `credence_gate` tool, the MCP `credence_autoverify` tool, and the Consistency Enforcer each carried their own matcher, and each missed `RATE_LIMIT = 100` after "I think the rate limit is 100" — the README's own example. The gate and the MCP tool disagreed with each other on identical input. |
| 2 | `credence/matching.py` — one canonical constraint matcher | Four scorers became one. The MCP tool had lower-cased before splitting (`RATE_LIMIT` stayed one token); the autoverifier used a 20-word stopword list against the gate's ~190, and it is the tool that marks constraints *verified*, i.e. the one that can switch enforcement off. |
| 3 | `credence_runtime/` removed | It was packaged into the wheel by `include = ["credence*"]` but could not import — it depended on a nonexistent `esm` module. |
| 4 | `ContextManager` no longer needs the `anthropic` SDK to construct | 17 tests could not run without it. The deterministic layers need no client. |

`docs/CHANGELOG.md` also records the removal of four dead `examples/` scripts,
the `Makefile` rewrite (five of six targets referenced files that do not exist),
and the documentation corrections (the licence badge said MIT; the project is
Apache 2.0).

## Modified files (Apache 2.0 §4b)

```
CONTRIBUTING.md                     test count 829 -> 898; documents credence/matching.py
Makefile                            rewritten; all targets previously broken
README.md                           licence badge, test count, entry-point notes, layout
credence/__init__.py                version; install instruction
credence/__main__.py                demo scores via the canonical matcher; install/footer strings
credence/context_manager.py         lazy API client; enforcer delegates to the canonical matcher
credence/hooks.py                   uses the canonical matcher; session-id resolution shared
credence/install_gate.py            points at the fork's repository
credence/mcp_server.py              gate + autoverifier delegate to the canonical matcher;
                                    decisions extracted to module level so tests can reach them;
                                    private matcher copies deleted
credence/observer.py                registers on a fresh install; shares session-id resolution
docs/CHANGELOG.md                   [Unreleased] section added
docs/ETP_SPEC.md                    fork install name added alongside upstream
docs/ROADMAP.md                     current-state test count corrected
examples/quickstart.py              rewritten (previous version failed on `from esm import`)
pyproject.toml                      fork name/version; PEP 639 licence expression and files
tests/integration/test_mock_llm.py  lazy-client coverage; header imports
                                    tidied
tests/integration/test_session_mock.py  wrap latency gated on min-of-N; header imports tidied
tests/perf/bench_all.py             min-of-N measurement; latency budgets defined once
tests/perf/test_perf.py             reads budgets from bench_all instead of repeating them
tests/unit/test_gate.py             enforcer latency gated on min-of-N; header imports tidied
tests/unit/test_probe.py            min-of-N to match the probe budget
server.json                         fork registry namespace; upstream repository id removed
```

Added: `credence/matching.py`, `examples/hook_demo.py`, `NOTICE`, `MANIFEST.in`,
`tests/conftest.py`, `tests/unit/test_matching.py`, `tests/unit/test_hook_enforcement.py`,
`tests/unit/test_matcher_parity.py`, `tests/unit/test_examples.py`,
`docs/FORK.md`.

Deleted: `credence_runtime/`, `examples/{compliance_logging,dashboard,langchain_router}.py`,
`mcp_server._expand_tokens`, `mcp_server._AUTOVERIFY_STOPWORDS`.

## Name collision

The import package is still `credence`. It was **not** renamed, deliberately:
renaming it would touch every one of the 936 tests and every documented code
example for no functional gain. The consequence is that `credence-enforce` and
`credence-guard` cannot both be installed in the same environment — they provide
the same module and the same console scripts. Pick one.

One upstream name is deliberately left alone: `experimental/typescript/` still
carries `"name": "credence-guard"` in its `package.json`. That tree is unshipped
("Phase 2 work — not yet shipped") and is excluded from the Python wheel, so the
name is inert. It should be renamed before anyone runs `npm publish` from it.

## Before publishing

Two steps need a human, because they cannot be done from inside the repository:

1. **Create the fork repository**, then update the `Repository`, `Bug Tracker`,
   and `"Source (fork)"` URLs in `pyproject.toml`. They currently point at
   `https://github.com/mrfentmen/credence-enforce`, which must exist before PyPI
   metadata references it.
2. **Add the `PYPI_API_TOKEN` secret** to that repository's Actions secrets.
   `.github/workflows/publish.yml` publishes on a `v*` tag and needs it.

`credence-enforce` was confirmed unclaimed on PyPI at the time of writing.
Publishing under `credence-guard` is not possible — that name belongs to the
upstream author.

## Sending changes upstream

Items 1–4 above are bug fixes with tests attached, and upstream's `CONTRIBUTING.md`
is open to them. If they are accepted upstream, this fork should be reduced to
whatever remains rejected rather than diverging indefinitely.
