"""
bench_all.py — Performance benchmarks for all deterministic components.
No API key required.

These targets are the single source of truth for the latency budgets:
tests/perf/test_perf.py reads target_ms out of these results instead of
repeating the numbers, so this CLI and the test gate cannot disagree.
They were set from measurement under load, following the "2-5x measured P99"
policy the gate used to claim but did not follow.

Targets:
  Probe (short): < 0.15ms per call
  Probe (2500 words): < 5ms per call
  Registry: < 5ms per operation
  Wrap:     < 2ms overhead (excluding compress_fn)

Run:
    python3 -m tests.perf.bench_all
    python3 -m tests.perf.bench_all --n 2000
"""

import argparse
import statistics
import sys
import tempfile
import time

from credence.context_manager import ContextManager
from credence.registry import CredenceRegistry
from credence.wrap import wrap

# Importable both as `python -m tests.perf.bench_all` (the `-m` form puts the
# repo root on sys.path) and through pytest (tests/conftest.py does it). No path
# manipulation is needed in this file.

_cm = ContextManager.__new__(ContextManager)

CERTAIN_TEXT = (
    "The rate limit is 100 req/min. The endpoint is confirmed at /api/v2. "
    "Authentication uses Bearer tokens. The timeout is 30 seconds."
)
UNCERTAIN_TEXT = (
    "I think the rate limit might be around 50 req/min, but I am not certain. "
    "The timeout is probably 30 seconds, though it may vary. "
    "Authentication might require additional configuration."
)


def bench(name: str, fn, N: int = 1000, target_ms: float = None, warmup: int = 50) -> dict:
    """Time one callable, warmup discarded.

    The pass/fail gate uses min_ms rather than mean_ms. On a shared or loaded
    machine the mean is dominated by scheduler interruptions and by the first,
    cold iteration, so the same code passes on an idle laptop and fails in CI.
    The minimum is the standard estimator of a microbenchmark's intrinsic
    cost: noise can only push a sample up, never down, so min approximates the
    real per-call cost while still moving when the code itself regresses.

    mean, p95, p99 and max are still reported — they are the interesting
    numbers for a user, they are just the wrong thing to gate on.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    mean = statistics.mean(times)
    p95 = times[int(N * 0.95)]
    p99 = times[int(N * 0.99)]
    passed = (times[0] < target_ms) if target_ms else None
    return {"name": name, "n": N,
            "min_ms":  round(times[0], 4),
            "mean_ms": round(mean, 4),
            "p95_ms": round(p95, 4), "p99_ms": round(p99, 4),
            "max_ms": round(times[-1], 4),
            "target_ms": target_ms, "passed": passed}


def run(N: int = 1000) -> list[dict]:
    results = []

    # Probe — certain text
    results.append(bench(
        "probe_certain", lambda: _cm._has_uncertainty(CERTAIN_TEXT), N, 0.15
    ))
    # Probe — uncertain text
    results.append(bench(
        "probe_uncertain", lambda: _cm._has_uncertainty(UNCERTAIN_TEXT), N, 0.15
    ))
    # Probe — long text (50× repetition)
    long_text = CERTAIN_TEXT * 50
    results.append(bench(
        "probe_long_text_2500_words", lambda: _cm._has_uncertainty(long_text), N, 5.0
    ))

    # Registry
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    reg = CredenceRegistry(db_path=db_path)
    # Pre-register some items
    for i in range(20):
        reg.register(f"constraint {i}", "s1", 0.3, "LOW")
    results.append(bench(
        "registry_list_uncertain_20_items",
        lambda: reg.list_uncertain("s1"), N, 5.0
    ))
    results.append(bench(
        "registry_register",
        lambda: reg.register("new uncertain constraint", "bench", 0.3, "LOW"), N//10, 5.0
    ))

    # Wrap — probe fires (no compress_fn call)
    results.append(bench(
        "wrap_probe_blocks",
        lambda: wrap(lambda t: t, context=UNCERTAIN_TEXT), N, 2.0
    ))
    # Wrap — probe clears (identity compress_fn)
    results.append(bench(
        "wrap_probe_clears",
        lambda: wrap(lambda t: t[:len(t)//2], context=CERTAIN_TEXT), N, 2.0
    ))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 65)
    print("CREDENCE — Performance Benchmarks")
    print("=" * 65)
    print(f"{'Component':<40} {'Min':>8} {'Mean':>8} {'P95':>8} {'P99':>8}  {'Status'}")
    print("-" * 65)

    results = run(args.n)
    all_pass = True
    for r in results:
        status = ""
        if r["passed"] is not None:
            status = "✓" if r["passed"] else f"✗ (target {r['target_ms']}ms)"
            if not r["passed"]:
                all_pass = False
        print(f"{r['name']:<40} {r['min_ms']:>7.3f}ms {r['mean_ms']:>7.3f}ms "
              f"{r['p95_ms']:>7.3f}ms {r['p99_ms']:>7.3f}ms  {status}")

    print("-" * 65)
    print(f"\nOverall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
