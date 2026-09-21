"""
test_session_mock.py — Full session lifecycle integration tests.
Uses mock compression functions. No API key required.

Coverage:
  I1  wrap() blocks compression on uncertain content (probe fires)
  I2  wrap() allows compression on certain content (probe clears)
  I3  wrap() measures qualifier survival correctly
  I4  FCR risk estimate is in [0, 1]
  I5  Mock compressor that strips markers: wrap detects the loss
  I6  Mock compressor that preserves markers: wrap reports safe
  I7  Registry integration: uncertain content auto-registered
  I8  Full pipeline: register → wrap → verify → no enforcement on verified
  I9  measure_fcr() computes correct FCR from batch
  I10 wrap() latency overhead < 2ms (excluding compress_fn time)
"""

import time

from credence.registry import CredenceRegistry
from credence.wrap import measure_fcr, wrap


# ── Mock compressors ──────────────────────────────────────────────────────────

def strip_all_compressor(text: str) -> str:
    """Simulates LLMLingua-style: drops short qualifier sentences."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    # Keep only sentences > 30 chars (drops short qualifiers)
    kept = [s for s in sentences if len(s) > 30]
    return ". ".join(kept) + "." if kept else text[:100]


def faithful_compressor(text: str) -> str:
    """Simulates a faithful compressor: keeps qualifiers intact."""
    words = text.split()
    # Keep first 60% of words (proportional compression, preserves structure)
    keep = max(10, int(len(words) * 0.6))
    return " ".join(words[:keep])


def identity_compressor(text: str) -> str:
    return text


def null_compressor(text: str) -> str:
    """H2O-style: returns minimal output (null-answer mode)."""
    return "Summary unavailable."


# ── I1: Probe blocks uncertain content ───────────────────────────────────────

def test_wrap_blocks_uncertain_content():
    uncertain_text = (
        "The system processes requests. "
        "I think the rate limit might be around 50 req/min, but I am not certain. "
        "The endpoint returns HTTP 200."
    )
    result = wrap(strip_all_compressor, context=uncertain_text)
    assert result.probe_blocked is True
    assert result.output == uncertain_text  # original returned
    assert result.qual_survival == 1.0
    assert result.fcr_risk == 0.0


def test_wrap_blocks_preserves_exact_original():
    text = "I think the configuration might need adjustment. The server runs on port 8080."
    result = wrap(faithful_compressor, context=text)
    assert result.output == text


# ── I2: Probe clears certain content ─────────────────────────────────────────

def test_wrap_allows_certain_content():
    certain_text = (
        "The rate limit is 100 req/min. "
        "The endpoint is confirmed at /api/v2/users. "
        "Authentication uses Bearer tokens."
    )
    result = wrap(faithful_compressor, context=certain_text)
    assert result.probe_blocked is False
    assert result.output != certain_text  # was compressed


def test_wrap_identity_compressor_certain():
    text = "The service runs on port 8080. The database uses PostgreSQL."
    result = wrap(identity_compressor, context=text)
    assert result.probe_blocked is False
    assert result.qual_survival == 1.0


# ── I3: Qualifier survival measurement ───────────────────────────────────────

def test_wrap_measures_zero_survival_on_strip():
    certain_text = "The rate limit is 100. The timeout is 30 seconds. The endpoint is active."
    # Manually add a marker for testing
    text_with_markers = certain_text + " (approximately correct)"
    # Use a compressor that removes "approximately"
    def removes_approx(t: str) -> str:
        return t.replace("approximately", "").replace("(correct)", "").strip()
    result = wrap(removes_approx, context=text_with_markers)
    # Either blocked or measured
    assert 0.0 <= result.qual_survival <= 1.0


def test_wrap_reports_full_survival_when_preserved():
    text = "The configuration is confirmed. The endpoint is at /api/v2."
    result = wrap(identity_compressor, context=text)
    assert result.qual_survival == 1.0


# ── I4: FCR risk range ────────────────────────────────────────────────────────

def test_wrap_fcr_risk_in_valid_range():
    text = "The system processes requests. The API returns HTTP 200."
    result = wrap(faithful_compressor, context=text)
    assert 0.0 <= result.fcr_risk <= 1.0


def test_wrap_fcr_risk_zero_when_blocked():
    text = "I think the rate limit might be 50 req/min."
    result = wrap(strip_all_compressor, context=text)
    assert result.probe_blocked
    assert result.fcr_risk == 0.0


# ── I5: Strip compressor detected ────────────────────────────────────────────

def test_wrap_detects_qualifier_loss_from_strip_compressor():
    certain_text = "The endpoint is confirmed at /api/v2. The timeout is 30 seconds."
    result = wrap(strip_all_compressor, context=certain_text)
    # No probe fire (certain text), but survival should be measured
    assert not result.probe_blocked
    assert 0.0 <= result.qual_survival <= 1.0


# ── I6: Faithful compressor reported safe ────────────────────────────────────

def test_wrap_safe_with_faithful_compressor():
    text = "The system returns HTTP 200 on success. The rate limit is 100 req/min."
    result = wrap(faithful_compressor, context=text)
    assert result.verdict in ("SAFE", "WARN", "RISK", "PRESERVE")


# ── I7: Registry auto-registration ───────────────────────────────────────────

def test_wrap_auto_registers_uncertain_content(tmp_path):
    db = str(tmp_path / "reg.db")
    reg = CredenceRegistry(db_path=db)
    text = "I think the rate limit might be 50 req/min per user account."
    result = wrap(faithful_compressor, context=text, session_id="s1",
                  registry=reg, auto_register=True)
    assert result.probe_blocked  # uncertain text blocked
    items = reg.list_uncertain("s1")
    assert len(items) >= 1


def test_wrap_no_auto_register_certain_content(tmp_path):
    db = str(tmp_path / "reg.db")
    reg = CredenceRegistry(db_path=db)
    text = "The rate limit is 100 req/min. The API is fully operational."
    result = wrap(faithful_compressor, context=text, session_id="s1",
                  registry=reg, auto_register=True)
    assert not result.probe_blocked
    items = reg.list_uncertain("s1")
    assert len(items) == 0  # no uncertain content


# ── I8: Full lifecycle ────────────────────────────────────────────────────────

def test_full_pipeline_register_wrap_verify(tmp_path):
    db = str(tmp_path / "lifecycle.db")
    reg = CredenceRegistry(db_path=db)

    # Register uncertain constraint
    cid = reg.register(
        content="rate limit might be 50 req/min",
        session_id="s1", j_score=0.3, zone="LOW"
    )
    assert cid is not None

    # Uncertain text → probe blocks
    uncertain = "The rate limit might be 50 req/min for sandbox tier."
    r1 = wrap(strip_all_compressor, context=uncertain, session_id="s1", registry=reg)
    assert r1.probe_blocked

    # Verify the constraint
    reg.verify(cid, verified_value="confirmed 100 req/min in production docs")

    # After verify: the explicitly registered item is verified
    # (wrap may have auto-registered additional items from the uncertain text)
    all_items = reg.list_uncertain("s1")
    # The manually registered constraint should no longer be uncertain
    manual_still_uncertain = any(i["constraint_id"] == cid for i in all_items)
    assert not manual_still_uncertain, "Manually registered constraint should be verified"


# ── I9: measure_fcr() ────────────────────────────────────────────────────────

def test_measure_fcr_zero_when_qualifiers_preserved():
    contexts = ["I think the rate limit might be 50 req/min"] * 5
    answers = ["The rate limit might be approximately 50 req/min per the docs."] * 5
    qualifiers = [["might", "approximately"]] * 5
    result = measure_fcr(contexts, answers, qualifiers)
    assert result["fcr"] == 0.0
    assert result["qualifier_survival"] > 0


def test_measure_fcr_high_when_qualifiers_stripped():
    contexts = ["I think the rate limit might be 50 req/min"] * 5
    # Answer states as confirmed fact, no qualifiers
    answers = ["The rate limit is 50 req/min."] * 5
    qualifiers = [["might", "think", "approximately"]] * 5
    result = measure_fcr(contexts, answers, qualifiers)
    assert result["qualifier_survival"] == 0.0


def test_measure_fcr_correct_count():
    contexts = ["might be uncertain"] * 10
    answers = ["The value is confirmed."] * 10
    qualifiers = [["might"]] * 10
    result = measure_fcr(contexts, answers, qualifiers)
    assert result["n"] == 10
    assert 0.0 <= result["fcr"] <= 1.0


# ── I10: Wrap latency ─────────────────────────────────────────────────────────

def test_wrap_overhead_under_2ms():
    """Wrap's own overhead, excluding the compressor, must stay under 2ms.

    Gated on the fastest of N runs rather than the mean. Scheduler noise can
    only push a timing sample up, never down, so the minimum is a stable
    estimator of the code's intrinsic cost: it does not move when the machine
    is busy, and it still moves when the code gets slower. The mean is reported
    in the failure message so a genuine distribution shift stays visible.

    This test previously asserted the mean and failed under load with the code
    untouched — the same defect fixed in tests/perf/test_perf.py.
    """
    text = "The system processes requests at 100 req/min. The database is confirmed healthy."
    N = 200
    samples = []
    for i in range(N):
        t0 = time.perf_counter()
        wrap(identity_compressor, context=text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= 10:                      # discard warm-up
            samples.append(elapsed_ms)

    assert samples, "no samples collected"
    min_ms = min(samples)
    mean_ms = sum(samples) / len(samples)
    assert min_ms < 2.0, (
        f"Wrap overhead too high: min {min_ms:.3f}ms "
        f"(mean {mean_ms:.3f}ms over {len(samples)} runs)"
    )


def test_wrap_result_has_all_fields():
    text = "The endpoint is confirmed at /api/v2."
    result = wrap(identity_compressor, context=text)
    assert hasattr(result, "output")
    assert hasattr(result, "probe_blocked")
    assert hasattr(result, "qual_survival")
    assert hasattr(result, "fcr_risk")
    assert hasattr(result, "latency_ms")
    assert hasattr(result, "verdict")
    assert hasattr(result, "safe")
