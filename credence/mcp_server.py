"""
credence/mcp_server.py
======================
Credence MCP Server — zero API key required.

Works with any coding agent: Claude Code, Codex, Cursor, Copilot.
The agent uses its own model for intelligence; Credence provides the
extraction, registration, and annotation layer deterministically.

Tools (17 total, zero LLM calls):
    credence_register        — register an uncertain constraint explicitly
    credence_verify          — mark a constraint as verified
    credence_autoverify      — auto-verify constraints on confirmation phrases
    credence_constraints     — query all unverified constraints for a session
    credence_gate            — pre-tool gate (block if unverified constraints apply)
    credence_scan            — scan model output for unverified numeric literals
    credence_self_probe      — extract domain values from generated code;
                               auto-register stale + domain-risky values as unverified
    credence_memory_snapshot — persist unverified constraints as project memory
    credence_memory_recall   — load project memory into a new session
    credence_audit           — per-session epistemic timeline
    credence_reset           — clear all constraints for a session
    credence_session_summary — plain-English digest of session epistemic state
    credence_diff            — detect epistemic contradictions between two texts
    credence_project_status  — project-wide epistemic health dashboard
    credence_scan_ghosts     — flag ghost constraints (numeric + domain, no docs)
    credence_marker_health   — marker precision diagnostic (returns "insufficient data" below threshold)
    credence_bandit_status   — adaptive threshold state (returns "learning" below threshold)

Resources (passive, epistemic:// URI scheme):
    epistemic://session/{session_id}/ledger             — all constraints
    epistemic://session/{session_id}/constraint/{id}    — single constraint + trajectory

Install:
    pip install credence-enforce
    credence-server

No ANTHROPIC_API_KEY or any other key needed.
Design principle: Unknown = unverified. Every value extracted is unverified
until the user explicitly calls credence_verify with evidence.
"""

import os
import re
import threading
from typing import Optional

try:
    from fastmcp import FastMCP
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FASTMCP_AVAILABLE = False

from .context_manager import (
    _GTS_NUM_PATTERN,
    _GTS_CODE_BLOCK,
    _GTS_SKIP_PREFIXES,
    _GTS_SENTENCE_SPLIT,
)
from .matching import evaluate_constraints
from .registry import CredenceRegistry
from .temporal_patterns import scan_temporal, scan_domain_assignments, TEMPORAL_J_SCORES

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Credence",
    instructions=(
        "Credence tracks uncertain values from conversation and enforces verification\n"
        "before they reach code. Works with any coding agent. Zero API key required.\n\n"
        "Principle: unknown = unverified. Binary state only — verified or not.\n\n"
        "Lifecycle:\n"
        "  1. credence_memory_recall  — session START (existing project)\n"
        "  2. credence_autoverify     — EVERY user message\n"
        "  3. credence_register       — user states uncertain value\n"
        "  4. credence_self_probe     — AFTER generating code (auto-extract values)\n"
        "  5. credence_scan           — AFTER generating code (annotate output)\n"
        "  6. credence_gate           — BEFORE Write / Edit / Bash\n"
        "  7. credence_verify         — user confirms a value with evidence\n"
        "  8. credence_memory_snapshot — session END\n\n"
        "Annotations in generated code:\n"
        "  ⚠⚠ CREDENCE[stale]      — structurally stale (versions, auth lifetimes, pricing)\n"
        "  ⚠  CREDENCE[unverified] — must verify before shipping"
    ),
) if _FASTMCP_AVAILABLE else None

# Process-level registry singleton
_registry: Optional[CredenceRegistry] = None
_registry_lock = threading.Lock()


def _get_registry() -> CredenceRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:  # double-checked locking
                db_path = (
                    os.environ.get("CREDENCE_DB_PATH")
                    or os.environ.get("CREDENCE_DB")           # alias used by hooks.py / observer.py
                    or os.environ.get("CREDENCE_REGISTRY_PATH")
                    or "epistemic_registry.db"
                )
                _registry = CredenceRegistry(db_path=db_path)
    return _registry


# ---------------------------------------------------------------------------
# Enforcement decisions — module level, so tests can reach them
# ---------------------------------------------------------------------------
#
# These live outside the @mcp.tool() closures deliberately. `fastmcp` is
# imported in a try/except and the tools are only registered when it is
# present, so logic written inside a tool body cannot be called by a test in an
# environment without it. That is precisely how the gate kept a private, broken
# matcher for so long: nothing could reach it to check.
#
# The tool now only adapts input and output. The decision is a plain function.


def blocking_constraints(
    tool_name: str, arguments_summary: str, constraints: list[dict]
) -> list[dict]:
    """The subset of `constraints` that should block this tool call.

    Delegates to credence.matching.evaluate_constraints — the same scorer the
    PreToolUse hook and the autoverifier use. A second implementation here
    could only drift, and it did: see the note in credence_gate.

    Each returned constraint carries `overlap_terms` (the evidence) and
    `match_reason` ("value" or "terms").
    """
    blocking = []
    for hit in evaluate_constraints(f"{tool_name} {arguments_summary}", constraints):
        verdict = hit.pop("_verdict")
        hit["overlap_terms"] = (
            verdict["shared_values"] + verdict["shared_terms"]
        )[:6]
        hit["match_reason"] = verdict["reason"]
        blocking.append(hit)
    return blocking


def confirmable_constraints(text: str, constraints: list[dict]) -> list[str]:
    """Ids of constraints that `text` confirms, by the canonical matcher.

    This is the one decision in the system that can turn enforcement OFF — a
    confirmed constraint is removed from `list_uncertain`, and the gate only
    blocks on uncertain ones. It must therefore be scored exactly as the gate
    scores blocking, or a confirmation that the gate would not have honoured
    can disarm it. One matcher, both directions.
    """
    return [
        c["constraint_id"]
        for c in evaluate_constraints(text, constraints)
    ]


# ---------------------------------------------------------------------------
# Standalone GTS scan (no ContextManager needed)
# ---------------------------------------------------------------------------

def _scan_output(output_text: str, registry: CredenceRegistry, session_id: str, turn: int = 0) -> tuple[str, list]:
    """Scan output for unverified numeric literals. Returns (annotated_text, hits)."""
    constraints = registry.list_uncertain(session_id)
    if not constraints:
        return output_text, []

    # Build value map: numeric_value_str → constraint dict with eff_conf
    value_map: dict[str, dict] = {}
    for c in constraints:
        nums = _GTS_NUM_PATTERN.findall(c.get("content", ""))
        eff_conf = registry.get_effective_confidence(c["constraint_id"], turn)
        c_with_conf = dict(c, eff_conf=eff_conf)
        for n in nums:
            if len(n) >= 2:
                value_map[n] = c_with_conf

    if not value_map:
        return output_text, []

    def _annotation(c: dict, snippet: str, for_code: bool) -> str:
        source = c.get("source", "")
        text   = (c.get("content") or "")[:110]
        # Strip internal prefixes added by temporal_patterns.py before displaying
        text = re.sub(r'^\[stale:[^\]]+\]\s*', '', text)
        text = re.sub(r'^\[AI-generated:[^\]]+\]\s*', '', text)
        if source == "temporal_scan":
            tier = "⚠⚠ CREDENCE[stale]"
        else:
            # Everything else: unverified — binary, no confidence tiers
            tier = "⚠ CREDENCE[unverified]"
        return (f"  # {tier}: {text}" if for_code else f" ⚠ {tier}: {text}")

    hits: list[dict] = []
    result = output_text

    # Pass 1 — code blocks (annotate literals + inherit uncertainty downstream)
    _ASSIGN_RE = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=')

    def annotate_code_block(m: re.Match) -> str:
        fence, body, close = m.group(1), m.group(2), m.group(3)
        lines = body.split("\n")
        out   = []

        # First pass: annotate direct literal hits, collect annotated var names
        annotated_vars: dict[str, dict] = {}  # var_name → constraint
        for line in lines:
            if any(line.lstrip().startswith(p) for p in _GTS_SKIP_PREFIXES):
                out.append(line)
                continue
            if "CREDENCE:" in line:
                out.append(line)
                continue
            for val, c in value_map.items():
                if re.search(r'\b' + re.escape(val) + r'\b', line):
                    m_assign = _ASSIGN_RE.match(line)
                    if m_assign:
                        annotated_vars[m_assign.group(1)] = c
                    line = line.rstrip() + _annotation(c, val, for_code=True)
                    hits.append({"value": val, "constraint_id": c["constraint_id"],
                                 "constraint_text": c.get("content","")[:80],
                                 "line": line.strip(), "eff_conf": c.get("eff_conf",0.5),
                                 "source": "code"})
                    break
            out.append(line)

        # Inheritance pass: annotate lines that reference annotated variable names
        if annotated_vars:
            out2 = []
            for line in out:
                if "CREDENCE:" in line:
                    out2.append(line)
                    continue
                if any(line.lstrip().startswith(p) for p in _GTS_SKIP_PREFIXES):
                    out2.append(line)
                    continue
                # Skip the assignment line itself (already annotated)
                if _ASSIGN_RE.match(line):
                    out2.append(line)
                    continue
                for var_name, c in annotated_vars.items():
                    if re.search(r'\b' + re.escape(var_name) + r'\b', line):
                        line = line.rstrip() + f"  # CREDENCE[inherited from {var_name}, unverified]"
                        hits.append({"value": var_name,
                                     "constraint_id": c["constraint_id"],
                                     "constraint_text": c.get("content","")[:80],
                                     "line": line.strip(),
                                     "source": "code_inherited"})
                        break
                out2.append(line)
            out = out2

        return fence + "\n".join(out) + close

    result = _GTS_CODE_BLOCK.sub(annotate_code_block, result)

    # Pass 2 — prose (non-code segments)
    segments = _GTS_CODE_BLOCK.split(result)
    prose_out = []
    for seg in segments:
        if seg.startswith("```") or seg.endswith("```"):
            prose_out.append(seg)
            continue
        sentences = _GTS_SENTENCE_SPLIT.split(seg)
        new_sents = []
        for sent in sentences:
            if "CREDENCE:" in sent:
                new_sents.append(sent)
                continue
            for val, c in value_map.items():
                if re.search(r'\b' + re.escape(val) + r'\b', sent):
                    sent = sent.rstrip() + _annotation(c, val, for_code=False)
                    hits.append({"value": val, "constraint_id": c["constraint_id"],
                                 "constraint_text": c.get("content","")[:80],
                                 "line": sent.strip(), "eff_conf": c.get("eff_conf",0.5),
                                 "source": "prose"})
                    break
            new_sents.append(sent)
        prose_out.append(" ".join(new_sents))

    # prose_out will have interleaved code+prose; simplest: just return result from code pass
    # (prose pass on the already-annotated result is correct)
    return result, hits


# ---------------------------------------------------------------------------
# Session type detection (keyword heuristics — zero API)
# ---------------------------------------------------------------------------

_SESSION_TYPE_KEYWORDS: dict[str, frozenset] = {
    "debug": frozenset({
        "error", "exception", "traceback", "stack trace", "stacktrace",
        "bug", "fix", "crash", "fails", "failing", "broken", "not working",
        "undefined", "null pointer", "segfault", "timeout", "deadlock",
        "infinite loop", "memory leak", "500", "404", "503",
    }),
    "design": frozenset({
        "architecture", "schema", "design", "trade-off", "tradeoff",
        "pattern", "approach", "system", "service", "component",
        "microservice", "monolith", "event-driven", "message queue",
        "database", "scalability", "availability", "consistency",
        "sharding", "replication", "caching", "load balancer",
    }),
    "code_review": frozenset({
        "review", "refactor", "clean up", "cleanup", "improve",
        "simplify", "readability", "maintainability", "best practice",
        "naming", "duplication", "dry", "solid", "smell",
    }),
    "research": frozenset({
        "compare", "evaluate", "benchmark", "analysis", "survey",
        "pros and cons", "trade offs", "alternatives", "options",
        "which is better", "difference between", "versus", "vs",
        "recommend", "suggestion",
    }),
}


def _detect_session_type(text: str) -> str:
    """
    Classify session type from text using keyword heuristics.
    Returns: "debug" | "design" | "code_review" | "research" | "general"
    """
    text_lower = text.lower()
    scores = {stype: 0 for stype in _SESSION_TYPE_KEYWORDS}
    for stype, keywords in _SESSION_TYPE_KEYWORDS.items():
        scores[stype] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.tool()
    def credence_register(
        content:     str,
        session_id:  str,
        source_type: str = "observation",
    ) -> dict:
        """
        Register an uncertain constraint in the epistemic registry.

        Use whenever the user states something uncertain: an unconfirmed vendor
        claim, an assumption, a 'I think' statement, a number from a quick search.
        All registered values are UNVERIFIED until explicitly confirmed.

        source_type classifies what kind of uncertain claim this is:
          "vendor_claim"  — value stated by a vendor / docs / external party
          "user_estimate" — user's rough guess or approximation
          "observation"   — user's direct measurement (unconfirmed)
          "assumption"    — working hypothesis
          "config"        — configuration value that may change
          "inference"     — model-derived, not directly stated

        Args:
            content:     The uncertain claim text (exact quote preferred).
            session_id:  Session identifier.
            source_type: Epistemic origin (see above). Default "observation".

        Returns:
            constraint_id, status, content, source_type
        """
        _TYPE_MAP = {"user_estimate": "estimate", "inference": "assumption"}
        registry_type = _TYPE_MAP.get(source_type, source_type)
        if registry_type not in {"vendor_claim", "estimate", "observation",
                                  "assumption", "compliance", "config", "performance"}:
            registry_type = "observation"

        registry = _get_registry()
        cid = registry.register(
            content         = content,
            session_id      = session_id,
            j_score         = 0.0,   # unknown = unverified, no confidence scoring
            zone            = "LOW",
            constraint_type = registry_type,
        )
        return {
            "constraint_id": cid,
            "status":        "registered",
            "content":       content,
            "session_id":    session_id,
            "source_type":   source_type,
            "message": (
                f"Registered as UNVERIFIED [{source_type}] (id={cid}). "
                f"Call credence_verify('{cid}', <confirmed_value>, '{session_id}') "
                "when the user confirms this value."
            ),
        }

    @mcp.tool()
    def credence_verify(
        constraint_id:  str,
        verified_value: str,
        session_id:     str,
        evidence:       str = "",
        source:         str = "user",
    ) -> dict:
        """
        Mark a registered constraint as verified with its confirmed value.

        After verification the constraint is excluded from Truth Buffer injection
        and Consistency Enforcer enforcement. An audit trail is recorded — who
        verified, on what basis, and what the confirmed value is.

        Args:
            constraint_id:  ID from credence_register.
            verified_value: The confirmed value (e.g. "100 req/min per Stripe docs §4.2").
            session_id:     Session identifier.
            evidence:       What was checked to confirm this. Strongly recommended.
                            Examples: "checked Stripe dashboard 2026-05-02",
                            "confirmed in production logs", "vendor email attached".
                            An empty evidence string is accepted but leaves no audit basis.
            source:         Who verified this. Use "user" for human confirmation,
                            "api_response" for automated checks, "agent:<name>" for
                            downstream agents, "external_doc" for documentation.

        Returns:
            Updated constraint dict with verified=True and audit fields.
        """
        registry = _get_registry()
        result   = registry.verify(constraint_id, verified_value, evidence=evidence, source=source)
        if "error" in result:
            return result
        result["status"]   = "verified"
        result["audit"]    = {"source": source, "evidence": evidence or "(none provided)"}
        result["message"]  = (
            f"Verified by '{source}'. Confirmed value: '{verified_value}'. "
            + (f"Evidence: {evidence}. " if evidence else "Warning: no evidence recorded. ")
            + "Safe to implement code that depends on this value."
        )
        return result

    @mcp.tool()
    def credence_constraints(session_id: str) -> dict:
        """
        List all unverified constraints for a session.

        Use before writing code that may embed user-stated values, or at
        session end to audit what still needs confirmation.

        Args:
            session_id: Session identifier.

        Returns:
            count, constraints list (id, content, j_score, zone), message.
        """
        registry    = _get_registry()
        constraints = registry.list_uncertain(session_id)
        count       = len(constraints)
        message = (
            "All constraints verified."
            if count == 0
            else f"{count} unverified constraint(s) — confirm before shipping code."
        )
        return {"count": count, "constraints": constraints, "message": message}

    @mcp.tool()
    def credence_gate(
        tool_name:         str,
        arguments_summary: str,
        session_id:        str,
    ) -> dict:
        """
        Pre-execution epistemic gate (CP4): block irreversible tool calls that
        embed unverified constraint values.

        Call BEFORE write_file, execute_code, send_request, deploy, or any
        tool that would embed a user-stated value into code or infrastructure.
        Scored by `credence.matching` — the same decision the `PreToolUse` hook
        makes, so this tool and the hook cannot return different verdicts for
        the same input. The match is on a shared numeric value or on literal
        shared terms after identifier splitting, not on synonym clusters: see
        the module docstring in `credence/matching.py` for why clusters are
        excluded from a decision that blocks a write.

        Args:
            tool_name:           Name of the tool about to be called.
            arguments_summary:   Brief summary of arguments (omit secrets).
            session_id:          Session identifier.

        Returns:
            proceed: bool, blocked_by: list, recommendation: str.
        """
        registry  = _get_registry()
        uncertain = registry.list_uncertain(session_id)

        if not uncertain:
            return {
                "proceed":          True,
                "blocked_by":       [],
                "unverified_count": 0,
                "recommendation":   "PROCEED — no unverified constraints in this session.",
            }

        # Scored by credence.matching — the same implementation the PreToolUse
        # hook uses. This tool previously carried its own copy of the overlap
        # rule, and the copy was wrong: it lower-cased before splitting, so
        # `RATE_LIMIT` stayed one token and never matched the prose terms
        # "rate"/"limit". With the README's own example it found one shared term
        # ("100"), needed two, and answered PROCEED — while the hook blocked the
        # same input. Two gates giving two answers to one question is worse than
        # either answer, because the answer you get depends on which door the
        # agent happened to use.
        blocking = blocking_constraints(tool_name, arguments_summary, uncertain)

        if blocking:
            cids = ", ".join(c["constraint_id"] for c in blocking)
            recommendation = (
                f"BLOCK — {len(blocking)} unverified constraint(s) may affect this action. "
                f"Verify first (IDs: {cids})."
            )
        else:
            recommendation = (
                f"PROCEED — {len(uncertain)} unverified constraint(s) exist "
                "but none are topically related to this action."
            )

        session_type = _detect_session_type(f"{tool_name} {arguments_summary}")

        return {
            "proceed":          len(blocking) == 0,
            "blocked_by":       blocking,
            "unverified_count": len(uncertain),
            "session_type":     session_type,
            "recommendation":   recommendation,
        }

    @mcp.tool()
    def credence_scan(output_text: str, session_id: str, current_turn: int = 0) -> dict:
        """
        Generation-Time Constraint Scanner (CP3): scan model output for numeric
        literals that match registered unverified constraints.

        Two annotation tiers (no confidence scores — unknown = unverified):
            ⚠⚠ CREDENCE[stale]      — source="temporal_scan" (structurally stale values)
            ⚠  CREDENCE[unverified]  — all other registered constraints

        Scans both code blocks and prose.

        Args:
            output_text:  Raw model output to scan.
            session_id:   Session whose constraints to check against.
            current_turn: Turn number for confidence decay (default 0).

        Returns:
            annotated_output, scan_hits, hit_count, high_risk_count, recommendation.
        """
        registry   = _get_registry()
        annotated, hits = _scan_output(output_text, registry, session_id, current_turn)

        if hits:
            recommendation = (
                f"REVIEW — {len(hits)} unverified literal(s) annotated. "
                "Confirm values before shipping."
            )
        else:
            recommendation = "CLEAN — no output literals matched unverified constraints."

        return {
            "annotated_output": annotated,
            "scan_hits":        hits,
            "hit_count":        len(hits),
            "recommendation":   recommendation,
        }

    @mcp.tool()
    def credence_self_probe(code: str, session_id: str) -> dict:
        """
        Extract domain-relevant values from generated code and register them
        as unverified by default — zero API calls, zero model judgment.

        Works with any coding agent (Claude Code, Codex, Cursor, Copilot).
        The agent's own model is NOT asked to rate its confidence. Instead,
        every extracted value is treated as unverified until the user
        explicitly calls credence_verify with evidence.

        Two extraction passes:
          1. Temporal patterns — structurally stale values (API versions,
             semver strings, auth magic numbers, rate literals, pricing).
             Auto-registered as source="temporal_scan", j_score≤0.25.
          2. Domain assignment patterns — variable names that signal
             uncertain config values (rate_limit, timeout, token_expiry…).
             Auto-registered as source="self_probe", j_score≤0.40.

        After calling this, call credence_scan to annotate the code output
        with the registered constraints before showing it to the user.

        Args:
            code:       The generated code block (raw string, fenced or plain).
            session_id: Session identifier.

        Returns:
            stale_count, domain_count, total_registered,
            stale_hits, domain_hits, annotation_hint.
        """
        registry = _get_registry()

        # --- Pass 1: temporal patterns (structurally stale) ---
        # j_score from the pattern definition; stale values are HIGH RISK by default.
        temporal_hits = scan_temporal(code)
        stale_registered: list[dict] = []
        for h in temporal_hits:
            j = TEMPORAL_J_SCORES.get(h.pattern_name, 0.20)
            cid = registry.register(
                content         = h.constraint_content,
                session_id      = session_id,
                j_score         = j,
                zone            = "LOW",
                source          = "temporal_scan",
                constraint_type = "vendor_claim",
            )
            stale_registered.append({
                "constraint_id": cid,
                "value":         h.value,
                "category":      h.category,
                "stale_reason":  h.stale_reason,
                "verify_hint":   h.verify_hint,
                "j_score":       j,
            })

        # --- Pass 2: domain assignment patterns ---
        # j_score=0.0 for all domain hits: unknown = unverified, no confidence guessing.
        # Three states only: unverified | stale | verified.
        domain_hits = scan_domain_assignments(code)
        domain_registered: list[dict] = []
        for h in domain_hits:
            cid = registry.register(
                content         = h.constraint_content,
                session_id      = session_id,
                j_score         = 0.0,
                zone            = "LOW",
                source          = "self_probe",
                constraint_type = "config",
            )
            domain_registered.append({
                "constraint_id": cid,
                "var_name":      h.var_name,
                "value":         h.value,
                "domain":        h.domain,
                "stale_reason":  h.stale_reason,
                "j_score":       0.0,
            })

        total = len(stale_registered) + len(domain_registered)
        annotation_hint = (
            f"Registered {total} value(s) as unverified. "
            "Call credence_scan(code, session_id) to annotate the output "
            "with CREDENCE tiers before showing it to the user."
        ) if total > 0 else (
            "No domain-relevant or temporally stale values detected in this code block."
        )

        return {
            "stale_count":      len(stale_registered),
            "domain_count":     len(domain_registered),
            "total_registered": total,
            "stale_hits":       stale_registered,
            "domain_hits":      domain_registered,
            "annotation_hint":  annotation_hint,
            "message": (
                f"Extracted {len(stale_registered)} stale + "
                f"{len(domain_registered)} domain-uncertain value(s). "
                "All registered as unverified. "
                "Unknown = unverified — no confidence scoring applied."
            ),
        }

    @mcp.tool()
    def credence_memory_snapshot(session_id: str, project_id: str) -> dict:
        """
        Persist all unverified constraints from a session as project memory.

        Call at the END of a session. Next session on the same project calls
        credence_memory_recall to inherit what was still uncertain — the new
        session starts knowing what it doesn't know.

        Args:
            session_id: Current session ID.
            project_id: Stable project identifier (e.g. "my-api-project").

        Returns:
            saved_count, items, message.
        """
        from .memory import CredenceMemory
        mem  = CredenceMemory(_get_registry())
        snap = mem.snapshot(session_id=session_id, project=project_id)
        return {
            "project_id":  snap.project_id,
            "session_id":  snap.session_id,
            "saved_count": snap.saved_count,
            "items": [
                {"constraint_id": i.get("constraint_id"),
                 "content":       i.get("content"),
                 "zone":          i.get("zone"),
                 "j_score":       i.get("j_score")}
                for i in snap.items
            ],
            "message": snap.summary(),
        }

    @mcp.tool()
    def credence_memory_recall(
        project_id:     str,
        new_session_id: str,
        context_hint:   str = "",
    ) -> dict:
        """
        Load project memories into a new session at session start.

        Call at the START of a new session. Injects all previously unverified
        constraints into the new session so enforcement works from turn 1.

        Args:
            project_id:     Project identifier matching credence_memory_snapshot.
            new_session_id: ID for the new session.
            context_hint:   Optional keyword filter.

        Returns:
            injected_count, system_block (prepend to system prompt), items, message.
        """
        from .memory import CredenceMemory
        mem    = CredenceMemory(_get_registry())
        recall = mem.recall_and_inject(
            project        = project_id,
            new_session_id = new_session_id,
            context_hint   = context_hint,
        )
        return {
            "project_id":     recall.project_id,
            "new_session_id": recall.new_session_id,
            "injected_count": recall.injected_count,
            "system_block":   recall.system_block,
            "items": [
                {"constraint_id": i.get("constraint_id"),
                 "content":       i.get("content"),
                 "zone":          i.get("zone"),
                 "j_score":       i.get("j_score")}
                for i in recall.items
            ],
            "is_empty": recall.is_empty(),
            "message": (
                f"Loaded {recall.injected_count} unverified constraint(s) from "
                f"project '{project_id}' into session '{new_session_id}'."
                if not recall.is_empty()
                else f"No unverified constraints found for project '{project_id}'."
            ),
        }

    @mcp.tool()
    def credence_autoverify(text: str, session_id: str) -> dict:
        """
        Scan text for natural-language verification signals and auto-verify
        matching unverified constraints — zero API calls.

        When a user says "actually it's 3600", "confirmed: rate limit is 100",
        or "I checked, the port is 5432", this tool detects those confirmation
        phrases and automatically marks matching constraints as verified.

        Matching: a constraint is a candidate if ≥ 2 non-stopword tokens from
        the constraint text appear in the confirmation sentence.

        Args:
            text:       The user or assistant message to scan for confirmations.
            session_id: Session whose constraints to check against.

        Returns:
            verified_count, verified_ids, candidates_checked, message.
        """
        _CONFIRM_PHRASES = frozenset({
            "actually", "confirmed", "i checked", "turns out", "verified",
            "it is", "it's", "the answer is", "we confirmed", "just checked",
            "double checked", "double-checked", "in fact", "correction",
            "to clarify", "clarification", "the actual", "the real",
            "as per", "according to", "per the docs", "per docs",
            "i verified", "we verified", "found out", "it turns out",
        })
        text_lower = text.lower()
        has_signal = any(p in text_lower for p in _CONFIRM_PHRASES)
        if not has_signal:
            return {
                "verified_count":    0,
                "verified_ids":      [],
                "candidates_checked": 0,
                "message":           "No confirmation signal detected in text.",
            }

        registry  = _get_registry()
        uncertain = registry.list_uncertain(session_id)
        if not uncertain:
            return {
                "verified_count":    0,
                "verified_ids":      [],
                "candidates_checked": 0,
                "message":           "Confirmation detected but no unverified constraints in session.",
            }

        # Scored by credence.matching, for the same reason as the gate above —
        # and with more at stake. This tool is what flips a constraint to
        # VERIFIED, and VERIFIED is what makes the gate allow a write. A
        # matcher here that is more generous than the gate's silently disarms
        # enforcement, so "do these two texts concern the same thing?" must have
        # exactly one answer in the codebase. The stopword set this used to
        # carry had 20 entries against the gate's ~190, and no synonym or
        # identifier handling at all.
        verified_ids = confirmable_constraints(text, uncertain)
        for cid in verified_ids:
            registry.verify(cid, f"auto-verified from: {text[:80]}")

        n = len(verified_ids)
        return {
            "verified_count":    n,
            "verified_ids":      verified_ids,
            "candidates_checked": len(uncertain),
            "message": (
                f"Auto-verified {n} constraint(s) matching confirmation signal."
                if n > 0
                else "Confirmation signal present but no constraints matched."
            ),
        }

    @mcp.tool()
    def credence_audit(session_id: str) -> dict:
        """
        Per-session epistemic timeline — all constraints (verified and unverified)
        in chronological order with full certainty trajectory.

        Use to answer "what have we tracked this session?" or "what's still open?"
        before starting an implementation phase.

        Args:
            session_id: The session to audit.

        Returns:
            constraint_count, unverified_count, verified_count, constraints list,
            timeline (each constraint with its trajectory events).
        """
        registry = _get_registry()
        all_rows = registry._conn.execute(
            "SELECT * FROM constraints WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        all_constraints = [registry._row_to_dict(r) for r in all_rows]
        uncertain = [c for c in all_constraints if not c.get("verified")]
        verified  = [c for c in all_constraints if c.get("verified")]
        timeline = [
            {**c, "trajectory": registry.get_trajectory(c["constraint_id"])}
            for c in all_constraints
        ]
        return {
            "session_id":       session_id,
            "constraint_count": len(all_constraints),
            "unverified_count": len(uncertain),
            "verified_count":   len(verified),
            "constraints":      all_constraints,
            "timeline":         timeline,
        }

    @mcp.tool()
    def credence_reset(session_id: str) -> dict:
        """
        Clear all constraints for a session.

        Use after completing a verification pass, or when starting a fresh
        implementation phase where all prior uncertain values have been resolved.

        Args:
            session_id: The session to clear.

        Returns:
            cleared (bool), session_id, cleared_count.
        """
        registry = _get_registry()
        count = registry._conn.execute(
            "SELECT COUNT(*) FROM constraints WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        registry._conn.execute(
            "DELETE FROM constraints WHERE session_id=?", (session_id,)
        )
        registry._conn.commit()
        return {"cleared": True, "session_id": session_id, "cleared_count": count}

    @mcp.tool()
    def credence_session_summary(session_id: str, project_id: str = None) -> dict:
        """
        Brief of unverified constraints for a session.

        Returns action_required, unverified_count, and a summary list so the
        model can decide whether to snapshot or prompt the user to verify.

        Args:
            session_id:  Session to summarize.
            project_id:  Optional project to associate with the snapshot.
        """
        registry = _get_registry()
        uncertain = registry.list_uncertain(session_id)
        summaries = [
            {"content": c["content"][:100], "constraint_id": c["constraint_id"],
             "source": c.get("source", ""), "verified": c.get("verified", False)}
            for c in uncertain
        ]
        return {
            "session_id":          session_id,
            "project_id":          project_id,
            "action_required":     len(uncertain) > 0,
            "unverified_count":    len(uncertain),
            "constraint_summaries": summaries,
        }

    @mcp.tool()
    def credence_diff(text_a: str, text_b: str, session_id: str = None) -> dict:
        """
        Compare two texts for numeric contradictions.

        Extracts numeric claims from both texts and detects when the same
        topic context has different values. Optionally checks against
        verified registry constraints when session_id is provided.

        Args:
            text_a:     First text (e.g. prior agent response).
            text_b:     Second text (e.g. new agent response).
            session_id: Optional — check text_b against verified constraints.

        Returns:
            matched_claims, contradictions, registry_conflicts, divergence_score,
            contradiction_count, recommendation, etp_version.
        """
        import re as _re
        _num_re = _re.compile(r'\b\d+(?:\.\d+)?\b')
        nums_a = set(_num_re.findall(text_a))
        nums_b = set(_num_re.findall(text_b))
        matched  = list(nums_a & nums_b)
        only_a   = nums_a - nums_b
        only_b   = nums_b - nums_a
        contradictions = []
        for n in only_a:
            contradictions.append({"value_a": n, "value_b": None, "note": f"{n!r} in text_a only"})
        for n in only_b:
            contradictions.append({"value_a": None, "value_b": n, "note": f"{n!r} in text_b only"})

        registry_conflicts: list = []
        if session_id:
            registry = _get_registry()
            verified = [c for c in registry.get_all(session_id) if c.get("verified")]
            for c in verified:
                v_nums = set(_num_re.findall(c.get("verified_value") or c["content"]))
                for n in v_nums:
                    if n not in nums_b and v_nums:
                        registry_conflicts.append({
                            "constraint_id": c["constraint_id"],
                            "verified_value": c.get("verified_value"),
                            "conflict_note":  f"verified {n!r} not present in text_b",
                        })

        total = max(1, len(nums_a | nums_b))
        divergence_score = round(len(contradictions) / total, 4)

        if contradictions:
            recommendation = "DIVERGE — numeric values differ between the two texts"
        elif not nums_a and not nums_b:
            recommendation = "AGREE — no numeric claims to compare"
        else:
            recommendation = "AGREE — numeric values consistent"

        return {
            "matched_claims":    matched,
            "contradictions":    contradictions,
            "registry_conflicts": registry_conflicts,
            "divergence_score":  divergence_score,
            "contradiction_count": len(contradictions),
            "recommendation":    recommendation,
            "etp_version":       "1.0",
        }

    @mcp.tool()
    def credence_project_status(project_id: str) -> dict:
        """
        Project-wide epistemic health dashboard.

        Shows all unverified constraints across sessions that have been
        snapshotted to this project via credence_memory_snapshot.

        Args:
            project_id: Project identifier.

        Returns:
            total_constraints, unverified_count, verified_count, epistemic_debt,
            health (CLEAN/LOW_DEBT/MEDIUM_DEBT/HIGH_DEBT), etp_version.
        """
        registry = _get_registry()
        constraints = registry.get_all_project_constraints(project_id)
        total      = len(constraints)
        unverified = [c for c in constraints if not c.get("verified")]
        debt       = len(unverified)
        if debt == 0:
            health = "CLEAN"
        elif debt <= 3:
            health = "LOW_DEBT"
        elif debt <= 10:
            health = "MEDIUM_DEBT"
        else:
            health = "HIGH_DEBT"
        return {
            "project_id":       project_id,
            "total_constraints": total,
            "unverified_count": len(unverified),
            "verified_count":   total - len(unverified),
            "epistemic_debt":   debt,
            "health":           health,
            "etp_version":      "1.0",
        }

    @mcp.tool()
    def credence_scan_ghosts(session_id: str) -> dict:
        """
        Scan a session for ghost constraints.

        Ghost constraints are vendor-supplied facts registered without hedging
        language — they look certain but are actually unverified. The ghost
        detector flags unverified vendor_claim constraints with no hedging.

        Args:
            session_id: Session to scan.

        Returns:
            ghost_count, ghost_candidates (with ghost_reason), recommendation.
        """
        registry = _get_registry()
        flagged  = registry.flag_ghost_constraints(session_id)
        if flagged:
            recommendation = (
                f"⚠ {len(flagged)} ghost constraint(s) detected. These vendor claims "
                "lack hedging language — verify before treating as fact."
            )
        else:
            recommendation = (
                "No ghost constraints detected. All vendor claims include appropriate "
                "hedging language or are already verified."
            )
        return {
            "session_id":       session_id,
            "ghost_count":      len(flagged),
            "ghost_candidates": flagged,
            "recommendation":   recommendation,
        }

    @mcp.tool()
    def credence_marker_health() -> dict:
        """
        Marker statistics from accumulated session data.

        Shows which uncertainty markers are most reliable (high precision)
        vs. noisy (low precision) based on observed FCR outcomes.
        Requires 10+ sessions before returning data.

        Returns:
            status (insufficient_data|available), threshold, markers list.
        """
        registry = _get_registry()
        stats    = registry.get_marker_stats()
        if not stats:
            return {
                "status":    "insufficient_data",
                "threshold": 10,
                "markers":   [],
                "message":   "Need 10+ sessions with marker event data.",
            }
        return {
            "status":    "available",
            "threshold": 10,
            "markers":   stats,
            "message":   f"{len(stats)} markers with precision/recall data.",
        }

    @mcp.tool()
    def credence_bandit_status() -> dict:
        """
        Adaptive compression threshold status (Thompson sampling bandit).

        Returns current learned thresholds per session type, or the static
        defaults if insufficient data has been collected (< 100 sessions).

        Returns:
            status (learning|active), threshold, current_thresholds, message.
        """
        registry = _get_registry()
        return registry.get_bandit_state()

# ---------------------------------------------------------------------------
# MCP Resources — epistemic:// URI scheme
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.resource("epistemic://session/{session_id}/ledger")
    def epistemic_ledger(session_id: str) -> str:
        """Epistemic ledger — all constraints for a session."""
        import json as _json
        registry = _get_registry()
        uncertain = registry.list_uncertain(session_id)
        all_rows  = registry._conn.execute(
            "SELECT * FROM constraints WHERE session_id=? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        all_constraints = [registry._row_to_dict(r) for r in all_rows]
        return _json.dumps({
            "session_id":        session_id,
            "total_constraints": len(all_constraints),
            "unverified_count":  len(uncertain),
            "verified_count":    len(all_constraints) - len(uncertain),
            "constraints":       all_constraints,
            "etp_version":       "1.0",
        }, indent=2)

    @mcp.resource("epistemic://session/{session_id}/constraint/{constraint_id}")
    def epistemic_constraint(session_id: str, constraint_id: str) -> str:
        """Single constraint with full certainty trajectory."""
        import json as _json
        registry = _get_registry()
        rows = registry._conn.execute(
            "SELECT * FROM constraints WHERE constraint_id=? AND session_id=?",
            (constraint_id, session_id),
        ).fetchone()
        if rows is None:
            return _json.dumps({"error": f"constraint '{constraint_id}' not found"})
        constraint = registry._row_to_dict(rows)
        trajectory = registry.get_trajectory(constraint_id)
        return _json.dumps({
            "session_id":  session_id,
            "constraint":  constraint,
            "trajectory":  trajectory,
            "event_count": len(trajectory),
            "etp_version": "1.0",
        }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _FASTMCP_AVAILABLE:
        print("fastmcp not installed. Run: pip install 'credence-ai[mcp]'")
        return
    mcp.run()
