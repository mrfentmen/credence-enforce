"""
test_scan_annotation.py — the annotation path, which had no pytest coverage.

`_scan_output` is what puts `⚠ CREDENCE[unverified]: …` on a generated line and
what `credence_scan` reports hits from. Two defects lived in it, both found by a
suite whose results were being discarded rather than by any pytest run:

  1. Every fenced literal produced TWO hits, one `source="code"` and one
     `source="prose"`. `_GTS_CODE_BLOCK` has three capturing groups, so
     `re.split` returns the code body as a segment of its own, and the pass-2
     guard only skipped segments that started or ended with a fence — so the
     body was scanned again as if it were prose.

  2. Re-annotating already-annotated text appended a second marker instead of
     skipping it. The guard tested for the literal `"CREDENCE:"`, which no
     emitted marker contains: the markers are `CREDENCE[unverified]:`,
     `CREDENCE[stale]:`, and `CREDENCE[inherited from …]`. It is now
     `has_credence_marker()` in context_manager.py, on the real prefix.

Coverage:
  A1 One fenced literal yields exactly one hit and one marker   [regression]
  A2 An already-annotated line is not annotated again           [regression]
  A3 Prose outside a code block is still annotated and reported
"""

from credence.mcp_server import _scan_output
from credence.registry import CredenceRegistry

CONSTRAINT = "[AI-generated:rate_limit] RATE_LIMIT = 100 — Rate limits vary by plan"
VALUE_LINE = "RATE_LIMIT = 100"


def _registry(tmp_path, session_id="annot"):
    reg = CredenceRegistry(db_path=str(tmp_path / "reg.db"))
    reg.register(CONSTRAINT, session_id, j_score=0.3, zone="LOW",
                 source="self_probe", constraint_type="config")
    return reg


# ── A1 ───────────────────────────────────────────────────────────────────────

def test_fenced_literal_yields_exactly_one_hit(tmp_path):
    """Regression: the code body was re-scanned as prose.

    The count matters beyond tidiness: `credence_scan` returns these hits to the
    agent, so a duplicate per literal doubles every number the agent reads off
    it, and the duplicate's `line` carried the marker twice.
    """
    reg = _registry(tmp_path)
    annotated, hits = _scan_output(f"```python\n{VALUE_LINE}\n```", reg, "annot", turn=0)

    assert len(hits) == 1, (
        f"one literal produced {len(hits)} hits — target={VALUE_LINE}\n"
        f"{[{'source': h['source'], 'line': h['line']} for h in hits]}"
    )
    assert hits[0]["source"] == "code"
    assert annotated.count("CREDENCE[") == 1, (
        f"the line carries {annotated.count('CREDENCE[')} markers: {annotated!r}"
    )


# ── A2 ───────────────────────────────────────────────────────────────────────

def test_already_annotated_line_is_not_annotated_again(tmp_path):
    """Regression: the suppression guard tested for a string nobody emits.

    Annotating the same source twice is a real path — an agent generates code,
    scans it, then scans again after an edit — and markers must not accumulate.
    """
    reg = _registry(tmp_path)
    once, _ = _scan_output(f"```python\n{VALUE_LINE}\n```", reg, "annot", turn=0)
    twice, hits = _scan_output(once, reg, "annot", turn=0)

    assert twice.count("CREDENCE[") == 1, (
        f"re-scanning compounded the marker: {twice!r}"
    )
    assert not hits, (
        f"re-scanning an annotated block reported hits for a value that is "
        f"already annotated: {[h['line'] for h in hits]}"
    )


# ── A3 ───────────────────────────────────────────────────────────────────────

def test_prose_outside_a_code_block_is_still_annotated(tmp_path):
    """The fix must not silence pass 2 — only stop it seeing code.

    Narrowing pass 2 to the text outside code blocks is the point; dropping it
    would leave prose values unannotated.
    """
    reg = _registry(tmp_path)
    annotated, hits = _scan_output(f"The {VALUE_LINE} setting is current.", reg,
                                   "annot", turn=0)

    assert len(hits) == 1, f"prose value not reported: {hits}"
    assert hits[0]["source"] == "prose"
    assert "CREDENCE[" in annotated
