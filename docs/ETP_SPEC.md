# Epistemic Transport Protocol (ETP) v1.0

**A model-agnostic standard for attaching epistemic metadata to AI-generated claims.**

---

## Problem

Every AI system today passes information between agents by value. Nobody passes it by
epistemic weight. When Agent A says "the rate limit is probably 50 req/min" and hands
that off to Agent B, Agent B receives "the rate limit is 50 req/min" — the uncertainty
qualifier is gone. When Agent B compresses it into a summary, or Agent C executes code
based on it, the original hedge has been laundered into apparent fact.

ETP fixes this by defining how epistemic metadata should travel with every piece of
AI-generated information through every hop of every pipeline.

---

## Core Objects

### EpistemicConstraint

A single uncertain claim tracked across sessions and agent hops.

```json
{
  "constraint_id":   "a3f1b2c4d5e6",
  "session_id":      "session-2026-05-01",
  "content":         "I think the rate limit is around 50 req/min",
  "j_score":         0.32,
  "zone":            "LOW",
  "verified":        false,
  "verified_value":  null,
  "created_at":      "2026-05-01T14:22:00Z",
  "updated_at":      "2026-05-01T14:22:00Z"
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `constraint_id` | string | Globally unique identifier (12-char content hash) |
| `session_id` | string | Conversation or session the claim originated in |
| `content` | string | The uncertain claim verbatim, as stated or extracted |
| `j_score` | number [0–1] | Epistemic confidence at registration time |
| `zone` | "LOW" \| "MEDIUM" \| "HIGH" | Confidence zone — controls compression policy |
| `verified` | boolean | True once confirmed against an authoritative source |
| `verified_value` | string \| null | Confirmed factual value; null until verified |
| `created_at` | ISO 8601 | Timestamp of first registration |
| `updated_at` | ISO 8601 | Timestamp of last update |

---

### EpistemicZone

| Zone | j_score range | Policy |
|------|---------------|--------|
| `HIGH` | ≥ 0.70 | Epistemically resolved. Safe to compress or summarize. |
| `MEDIUM` | 0.45–0.70 | Borderline. Trim but do not compress. |
| `LOW` | < 0.45 | Explicit uncertainty. Preserve verbatim. |

---

### EpistemicEnvelope

Provenance wrapper for AI-generated content in multi-agent pipelines. Travels with
content as it moves between agents, compression layers, and memory systems.

```json
{
  "content":               "The endpoint accepts 50 requests per minute",
  "j_score":               0.32,
  "zone":                  "LOW",
  "source":                "credence",
  "verified":              false,
  "chain_depth":           1,
  "trust_score":           0.27,
  "should_verify":         true,
  "safe_to_compress":      false,
  "uncertainty_preserved": true,
  "content_type":          "text",
  "session_id":            "session-2026-05-01"
}
```

**Trust decay formula:**

```
trust_score = max(0, j_score − (chain_depth × 0.05) − source_penalty)
```

Where `source_penalty = 0.10` for unknown sources; `0.0` for trusted sources
(`"credence"`, `"user"`, `"system"`).

**Key derived fields:**

| Field | Meaning |
|-------|---------|
| `should_verify` | `trust_score < 0.40 AND NOT verified` — must verify before acting |
| `safe_to_compress` | `zone == "HIGH" AND trust_score ≥ 0.40 AND NOT uncertainty_preserved` |

---

### EpistemicEvent

A single event in a constraint's **certainty trajectory** — the event log showing
how confidence in a claim evolved over time.

```json
{
  "event_id":      1,
  "constraint_id": "a3f1b2c4d5e6",
  "timestamp":     "2026-05-01T14:22:00Z",
  "event_type":    "register",
  "j_score":       0.32,
  "zone":          "LOW",
  "notes":         "session=session-2026-05-01 turn=3"
}
```

**Event types:**

| Type | Meaning |
|------|---------|
| `register` | Constraint first observed |
| `scout` | Auto-extracted by Scout classifier |
| `chat_update` | Confidence revised mid-session |
| `verify` | Confirmed with external evidence |
| `contradict` | New claim conflicts with prior verification |

---

### EpistemicLedger

Full epistemic state for a session. The canonical machine-readable record.

```json
{
  "session_id":        "session-2026-05-01",
  "total_constraints": 3,
  "unverified_count":  2,
  "verified_count":    1,
  "constraints":       [...],
  "etp_version":       "1.0"
}
```

Accessible via MCP Resource: `epistemic://session/{session_id}/ledger`

---

## MCP Interface

Credence is the reference implementation of ETP, deployed as an MCP server.

### Tools

| Tool | Phase | Purpose |
|------|-------|---------|
| `credence_register` | Any time | Register an uncertain constraint |
| `credence_autoverify` | On every user message | Auto-match confirmations to open constraints |
| `credence_verify` | On confirmation | Mark a constraint as verified |
| `credence_constraints` | Any time | Query unverified constraints for a session |
| `credence_gate` | Before irreversible action | Block if unverified constraints apply |
| `credence_scan` | After generation | Annotate output with ⚠⚠/⚠ markers |
| `credence_self_probe` | After code generation | Extract and register values from code |
| `credence_scan` | Before shipping code | Annotate unverified literals inline |
| `credence_wrap` | Before agent handoff | Attach ETP envelope to content |
| `credence_unwrap` | On receiving from agent | Inspect provenance, increment chain_depth |
| `credence_diff` | After receiving agent response | Detect contradictions vs. prior response |
| `credence_project_status` | Any time | Project-wide epistemic health dashboard |
| `credence_memory_snapshot` | Session end | Persist unverified constraints as project memory |
| `credence_memory_recall` | Session start | Load prior session's constraints |

### Resources

| URI | Returns |
|-----|---------|
| `epistemic://session/{session_id}/ledger` | EpistemicLedger |
| `epistemic://session/{session_id}/constraint/{id}` | EpistemicConstraint + trajectory |

---

## Team Sharing (Multi-Developer)

Set `CREDENCE_DB_PATH` to a shared path so multiple developers on the same project
share the constraint registry:

```json
{
  "mcpServers": {
    "credence": {
      "command": "credence-server",
      "env": {
        "CREDENCE_DB_PATH": "/shared/projects/myapp/epistemic_registry.db"
      }
    }
  }
}
```

All developers working on the project then see the same unverified constraints,
verified facts, and trajectory history.

---

## Design Principles

1. **Zero API key** — all ETP tooling must be deterministic. No LLM calls in the
   enforcement path.

2. **Epistemic weight travels with content** — EpistemicEnvelope ensures that
   j_score, zone, and chain_depth are available to every downstream consumer.

3. **Trust degrades with distance** — each agent hop reduces trust_score by 0.05.
   Unknown sources receive an additional 0.10 penalty.

4. **Uncertainty is preserved, not compressed** — content with `uncertainty_preserved=true`
   must never be compressed regardless of zone.

5. **Verification stops decay** — once a constraint is verified, its j_score is
   frozen and does not decay further.

---

## JSON Schema

The machine-readable schema is at `docs/etp_schema.json`.

```
$schema: https://json-schema.org/draft/2020-12/schema
$id:     https://credence-ai.io/etp/v1
```

---

## Versioning

This is ETP v1.0. The `etp_version` field in all objects is `"1.0"`.

Future versions will be backwards-compatible at the field level. New fields will
be additive. Removed fields will be deprecated for one major version before removal.

---

## Reference Implementation

[credence-ai](https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai) — Python MCP server.
Install (upstream): `pip install credence-guard`
Install (this fork): `pip install credence-enforce`
