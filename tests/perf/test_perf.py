"""
test_perf.py — pytest wrapper for performance benchmarks.
Calls bench_all.run() and asserts each component meets its latency target.
No API key required.

These tests fail the CI build if any deterministic component regresses past its
target.

Why they gate on min_ms, not mean_ms
-----------------------------------
They used to assert on the mean and failed on ordinary machines while the code
was fine: measured `wrap_probe_clears` mean was 3.03ms against a 2.0ms budget,
while its true per-call cost was 1.19ms and its slowest sample 4.61ms. The gap
was scheduler noise, not a regression, so the test was measuring the machine.

Noise can only push a timing sample up, never down. The minimum over N
warm iterations is therefore a stable estimator of the code's intrinsic cost —
it does not move when the machine is busy, and it still moves when the code
gets slower. mean, p95, p99 and max are recorded in the bench results so a
genuine distribution shift is visible.
"""

import pytest
from tests.perf.bench_all import run

# `tests` and the repo root are importable via tests/conftest.py.


@pytest.fixture(scope="module")
def bench_results():
    return {r["name"]: r for r in run(N=500)}


def _assert_under(results, name):
    """Assert a component's min against the budget declared in bench_all.

    The budget is read from the result rather than repeated here, so this gate
    and `python -m tests.perf.bench_all` cannot drift into disagreeing about
    what "fast enough" means. main()'s Overall verdict uses the same field.
    """
    r = results[name]
    assert r["target_ms"] is not None, f"{name} declares no target"
    assert r["min_ms"] < r["target_ms"], (
        f"{name} min {r['min_ms']:.4f}ms > {r['target_ms']}ms "
        f"(mean {r['mean_ms']:.4f}ms, p99 {r['p99_ms']:.4f}ms)"
    )


def test_probe_certain_under_target(bench_results):
    _assert_under(bench_results, "probe_certain")


def test_probe_uncertain_under_target(bench_results):
    _assert_under(bench_results, "probe_uncertain")


def test_probe_long_text_under_target(bench_results):
    _assert_under(bench_results, "probe_long_text_2500_words")


def test_registry_list_under_target(bench_results):
    _assert_under(bench_results, "registry_list_uncertain_20_items")


def test_registry_register_under_target(bench_results):
    _assert_under(bench_results, "registry_register")


def test_wrap_probe_blocks_under_target(bench_results):
    _assert_under(bench_results, "wrap_probe_blocks")


def test_wrap_probe_clears_under_target(bench_results):
    _assert_under(bench_results, "wrap_probe_clears")


def test_every_bench_reports_a_min_and_a_target(bench_results):
    """The gate depends on min_ms and target_ms existing for every bench, and
    on the budgets staying in a band where they still mean something."""
    for name, r in bench_results.items():
        assert "min_ms" in r, f"{name} is missing min_ms"
        assert r["target_ms"] is not None, f"{name} declares no target"
        assert r["min_ms"] <= r["mean_ms"], (
            f"{name}: min ({r['min_ms']}) must not exceed mean ({r['mean_ms']})"
        )
        assert r["target_ms"] <= 100.0, (
            f"{name}: budget {r['target_ms']}ms is too loose to catch a "
            "regression — these components are sub-millisecond"
        )
