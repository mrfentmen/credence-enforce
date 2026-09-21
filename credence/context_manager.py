"""
credence/context_manager.py
===========================
ContextManager: Credence — Epistemic Preservation Layer for Claude API.

Every conversation turn, Claude Opus 4.7 answers the user's question.
Credence then reads that response, computes a J-proxy confidence score, and
decides what to do with old context:

  HIGH confidence  (J ≥ 0.70) → Compress: ask Claude Haiku to summarize old
                                  turns into 2-3 sentences, discard raw turns.
  MEDIUM confidence(J ∈ [0.45, 0.70)) → Trim: keep last 10 turns, drop older.
  LOW confidence   (J < 0.45)  → Preserve: keep everything.

The compressor IS Claude — Haiku 4.5 summarizes Opus 4.7's own prior turns.
The model manages its own memory: Opus reasons forward while Haiku prunes
what Opus no longer needs to re-read. This is Claude acting as both author
and editor of its own context.

Guard rails:
  - Attention sink protection: first 2 turns are never compressed — they
    establish conversation identity and are disproportionately attended to.
  - Compression depth limit: after 3 compressions the system stops
    compressing (recursive summarization degrades quality).
  - Novelty guard: disabled (79.2% false-positive rate on stable sessions;
    vocabulary distance cannot distinguish same-domain progression from genuine
    topic pivots without semantic embeddings).
  - Adaptive thinking (opt-in, use_thinking=True): forward-reserved for models
    that expose thinking blocks. Opus 4.7 does not expose thinking tokens, so
    thinking_tokens and thinking_utilization are always 0 on this model.
  - Drift detection: if J drops below theta_low for 3 consecutive turns the
    system enters drift state and forces PRESERVE until confidence recovers.
    This is proactive control — the intervention happens before the 4th
    failure, not after.

Savings are tracked against a no-Credence baseline (full context every turn).
Real token counts come from the Anthropic usage response object.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from .confidence_proxy import CredenceProxy, CredenceResult
from .matching import evaluate as _match_evaluate, expand as _match_expand, tokenize as _match_tokenize

try:
    from .registry import CredenceRegistry
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pricing (per million tokens, as of 2026)
# ---------------------------------------------------------------------------
_INPUT_COST_PER_M  = 15.0   # $15 / 1M input  — Opus 4.7
_OUTPUT_COST_PER_M = 75.0   # $75 / 1M output — Opus 4.7
_HAIKU_INPUT_PER_M  = 0.80  # $0.80 / 1M input  — Haiku 4.5
_HAIKU_OUTPUT_PER_M = 4.00  # $4.00 / 1M output — Haiku 4.5

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
_MODEL_OPUS  = "claude-opus-4-7"
_MODEL_HAIKU = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Thinking budget bounds (tokens)
# Budget scales continuously between these by inverse J-score.
# J=0 → _THINKING_MAX; J>=theta_high → 0 (no thinking)
# ---------------------------------------------------------------------------
_THINKING_MIN = 1024   # API minimum; budget_tokens must be >= 1024
_THINKING_MAX = 5000

_MIN_COMPRESS_TOKENS  = 150 # don't call Haiku if old segment is smaller than this
_MIN_COMPRESS_ROI     = 50  # minimum net tokens saved (after Haiku overhead) to justify compression

# ---------------------------------------------------------------------------
# Regime detection — only activate Credence compression/trim when session shows
# evidence of needing it (cross-turn dependencies or mixed J variance).
#
# Rationale: on independent QA sessions, J-variance is low (all turns cluster
# HIGH) and there are no cross-turn references. Applying Credence compression to
# such sessions wastes Haiku calls and sometimes hurts ROUGE-L vs naive window.
# The regime gate makes Credence universally safe: when dependencies aren't present,
# PRESERVE is the correct decision anyway.
# ---------------------------------------------------------------------------
_REGIME_MIN_TURNS        = 4     # warmup turns before regime detection activates
_REGIME_J_VARIANCE_FLOOR = 0.05  # min variance to signal mixed-confidence session

# ---------------------------------------------------------------------------
# Post-compression degradation detection — explicit, tunable constants.
# These detect whether a compression was harmful AFTER the fact (shadow check).
# ---------------------------------------------------------------------------
_SUMMARY_FAITHFUL_THRESHOLD = 0.30  # min fraction of original content words in summary

# ---------------------------------------------------------------------------
# Consistency Enforcer — switches from informational Truth Buffer injection
# to imperative enforcement when the user query DIRECTLY asks about an
# unverified registered constraint.
#
# Truth Buffer:          "here are uncertain things — acknowledge them"
# Consistency Enforcer:  "this query asks about [X] which is UNVERIFIED —
#                         you MUST express uncertainty"
#
# The enforcer fires when ≥2 non-stop-words overlap between the user message
# and a registered constraint. This threshold distinguishes direct questions
# about a constraint from tangential mentions.
# ---------------------------------------------------------------------------
_CE_MIN_OVERLAP = 2   # minimum non-stop-word overlap to trigger enforcement

_CE_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "about", "what", "how",
    "when", "where", "which", "who", "i", "we", "you", "it", "my", "our",
    "your", "its", "that", "this", "and", "or", "but", "if", "so", "use",
    "used", "using", "get", "set", "now", "just", "also", "need", "want",
    "tell", "know", "think", "make", "give", "take", "see", "say", "go",
    # Cross-domain words excluded to prevent false enforcement:
    # "size" spans UI (font size) vs API (batch size); "error" spans general
    # programming errors vs API retry/error handling. Both are too ambiguous
    # to participate in overlap scoring — context-specific clusters cover them.
    "size", "error",
})

# Domain-aware synonym clusters for the Consistency Enforcer.
#
# "How fast can we call the endpoint?" has zero literal overlap with
# "rate limit is 50 req/min" yet refers to the same constraint. These
# clusters map each term to its semantic family so that expanded token
# sets can find the overlap that raw string matching misses.
#
# Each key expands to all other members of its cluster. The clusters are
# intentionally narrow (technical API/auth/infra vocabulary) to avoid
# false positives from over-broad synonymy.
_CE_DOMAIN_SYNONYMS: dict[str, frozenset[str]] = {
    # ---- rate / throughput ------------------------------------------------
    "rate":       frozenset({"limit", "throttle", "quota", "rps", "rpm", "qps",
                              "frequency", "speed", "fast", "slow", "calls",
                              "requests", "throughput", "bandwidth"}),
    "limit":      frozenset({"rate", "cap", "max", "maximum", "ceiling",
                              "quota", "throttle", "threshold"}),
    "throttle":   frozenset({"rate", "limit", "quota", "cap", "restrict",
                              "slow", "backoff"}),
    "quota":      frozenset({"rate", "limit", "cap", "allowance", "budget"}),
    "fast":       frozenset({"rate", "speed", "frequency", "quickly", "rapid",
                              "throughput"}),
    "slow":       frozenset({"rate", "throttle", "delay", "latency", "backoff"}),
    "requests":   frozenset({"rate", "calls", "rps", "rpm", "qps", "invocations",
                              "hits", "traffic"}),
    "calls":      frozenset({"requests", "invocations", "hits", "rate", "rps"}),
    "endpoint":   frozenset({"api", "url", "route", "path", "service",
                              "resource", "host"}),
    # ---- authentication / tokens ------------------------------------------
    "token":      frozenset({"auth", "jwt", "bearer", "credential", "secret",
                              "key", "access", "refresh", "expiry", "expire",
                              "ttl", "session", "oauth"}),
    "auth":       frozenset({"token", "credential", "login", "authenticate",
                              "authorization", "permission", "access", "oauth",
                              "jwt", "bearer"}),
    "expiry":     frozenset({"expire", "expires", "ttl", "timeout", "duration",
                              "lifetime", "valid", "renew", "token"}),
    "expire":     frozenset({"expiry", "ttl", "timeout", "lifetime", "valid"}),
    "credential": frozenset({"token", "auth", "secret", "key", "password",
                              "apikey", "cert"}),
    "secret":     frozenset({"token", "key", "credential", "password", "apikey"}),
    # "refresh" intentionally omitted: auth token refresh ≠ cache refresh —
    # keeping it in both clusters created false CE positives (cache queries
    # matching auth constraints via {refresh, ttl, expiry} overlap).
    "session":    frozenset({"token", "auth", "cookie", "login", "timeout"}),
    # ---- pagination / batching --------------------------------------------
    "page":       frozenset({"pagination", "paging", "offset", "cursor",
                              "batch", "chunk", "size", "per", "results"}),
    "pagination": frozenset({"page", "paging", "offset", "cursor", "batch",
                              "limit", "size", "results"}),
    "batch":      frozenset({"chunk", "bulk", "page", "size", "limit", "group"}),
    # "size" removed as standalone key — too cross-domain (font size ≠ batch size).
    # Reachable as a value from "page", "pagination", "batch", "memory" clusters.
    # ---- infrastructure / error -------------------------------------------
    "timeout":    frozenset({"latency", "delay", "slow", "wait", "deadline",
                              "response", "expiry", "ttl", "out", "times"}),
    "request":    frozenset({"requests", "call", "invocation", "hit", "api"}),
    "latency":    frozenset({"timeout", "delay", "slow", "response",
                              "performance", "speed"}),
    "retry":      frozenset({"backoff", "attempt", "reconnect", "throttle",
                              "fail", "retrying", "retries"}),
    "backoff":    frozenset({"retry", "wait", "delay", "throttle", "slow"}),
    "wait":       frozenset({"retry", "backoff", "delay", "pause", "hold"}),
    # "error" removed as standalone key — too cross-domain (programming error ≠
    # API retry error). Reachable as a value from "retry" and "fail" clusters.
    "fail":       frozenset({"exception", "crash", "bug", "issue", "retry"}),
    # ---- configuration / deployment ---------------------------------------
    "config":     frozenset({"setting", "option", "parameter", "value",
                              "configure", "setup", "env", "environment"}),
    "deploy":     frozenset({"release", "push", "rollout", "ship", "launch",
                              "prod", "production", "staging"}),
    "memory":     frozenset({"ram", "heap", "allocation", "buffer",
                              "cache", "storage", "size"}),
    "cpu":        frozenset({"processor", "compute", "cores", "performance",
                              "load", "utilization", "capacity"}),
    # cache cluster deliberately excludes "expiry" and "refresh" to prevent
    # cross-domain bleed with auth/token constraints that also use those terms.
    "cache":      frozenset({"ttl", "invalidate", "stale", "memory", "storage"}),
    # ---- financial / billing ------------------------------------------------
    "cost":       frozenset({"price", "fee", "charge", "billing", "spend", "budget",
                              "rate", "tier", "plan", "pricing", "pay", "invoice"}),
    "price":      frozenset({"cost", "fee", "charge", "billing", "rate", "pricing"}),
    "budget":     frozenset({"cost", "spend", "limit", "cap", "allocation",
                              "forecast", "estimate"}),
    "payment":    frozenset({"charge", "invoice", "billing", "fee", "subscription",
                              "renewal", "refund"}),
    "threshold":  frozenset({"limit", "cap", "max", "minimum", "floor", "ceiling",
                              "cutoff", "trigger"}),
    # ---- medical / dosing / clinical ----------------------------------------
    "dose":       frozenset({"dosage", "dosing", "mg", "ml", "concentration",
                              "amount", "quantity", "regimen", "prescription"}),
    "dosage":     frozenset({"dose", "dosing", "mg", "ml", "frequency",
                              "schedule", "concentration"}),
    "frequency":  frozenset({"dose", "schedule", "interval", "period", "rate",
                              "daily", "weekly", "hourly"}),
    "interval":   frozenset({"frequency", "period", "schedule", "gap",
                              "delay", "duration", "between"}),
    "duration":   frozenset({"period", "length", "interval", "window", "lifetime",
                              "expiry", "ttl"}),
    "contraindication": frozenset({"warning", "restriction", "prohibited",
                                    "interaction", "adverse"}),
    # ---- hardware / infrastructure resources ---------------------------------
    "cores":      frozenset({"cpu", "vCPU", "processor", "threads",
                              "compute", "capacity", "workers"}),
    "disk":       frozenset({"storage", "volume", "iops", "throughput",
                              "space", "ssd", "capacity"}),
    "iops":       frozenset({"disk", "throughput", "storage", "performance",
                              "read", "write", "latency"}),
    "bandwidth":  frozenset({"network", "throughput", "rate", "speed",
                              "mbps", "gbps", "egress", "ingress"}),
    "replica":    frozenset({"instance", "node", "shard", "copy",
                              "failover", "availability", "cluster"}),
    "instance":   frozenset({"node", "server", "vm", "container", "pod",
                              "host", "replica", "worker"}),
    # ---- legal / compliance -------------------------------------------------
    "retention":  frozenset({"storage", "period", "duration", "policy",
                              "gdpr", "deletion", "purge", "archive"}),
    "sla":        frozenset({"uptime", "availability", "agreement", "guarantee",
                              "reliability", "nines", "commitment"}),
    "uptime":     frozenset({"availability", "sla", "reliability",
                              "nines", "downtime", "guarantee"}),
    "compliance": frozenset({"regulation", "policy", "requirement", "standard",
                              "audit", "gdpr", "hipaa", "soc", "pci"}),
}

# ---------------------------------------------------------------------------
# Generation-Time Constraint Scanner (GTS)
#
# The Consistency Enforcer fires BEFORE the API call when the user query
# directly overlaps with a registered unverified constraint. That handles
# the case where the user ASKS about a constraint.
#
# The GTS closes the remaining gap: when Claude writes code that SILENTLY
# EMBEDS a registered uncertain value — e.g. "RATE_LIMIT = 50" after the
# user said "I think the rate limit is around 50 — unconfirmed" — the code
# is annotated inline before being returned to the user:
#
#   RATE_LIMIT = 50  # CREDENCE: unverified — I think the rate limit is around…
#
# Mechanism: extract numeric values from each unverified constraint in the
# registry; search assignment lines in code blocks for those values; inject
# a CREDENCE comment. The user sees the epistemic provenance inline, at the
# point of use.
#
# This is the generation-time layer of the stack:
#   Storage  → registry
#   Compression → faithfulness probe
#   Injection → Truth Buffer + Consistency Enforcer
#   Generation → GTS (this layer)
# ---------------------------------------------------------------------------
_GTS_NUM_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\b')
_GTS_CODE_BLOCK  = re.compile(r'(```[^\n]*\n)(.*?)(```)', re.DOTALL)
_GTS_SKIP_PREFIXES = ("def ", "class ", "import ", "from ", "//", "/*", "*", "#", "@")
# Prose scanner: split on sentence-ending punctuation followed by whitespace/EOL.
_GTS_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
# String literal scanner: extract quoted strings from constraint content.
# Two sub-patterns:
#   _GTS_STR_EXTRACT: matches any quoted string 3-80 chars (captures spaces too)
#   _GTS_STR_ASSIGN:  matches string literal on RHS of an assignment in code
_GTS_STR_EXTRACT = re.compile(r'["\']([^"\']{3,80})["\']')
# String assignment in code: variable = "value" or variable = 'value'
_GTS_STR_ASSIGN  = re.compile(r'=\s*["\']([^"\']{2,80})["\']')
# Unquoted identifier-like values in constraint content.
# Pattern 1: uppercase identifiers — RS256, AES-256-GCM, HMAC-SHA256
_GTS_IDENT_EXTRACT = re.compile(r'\b([A-Z][A-Z0-9\-]{2,29})\b')
# Pattern 2: lowercase hyphenated tokens — us-east-1, eu-west-2, read-write, ap-southeast
_GTS_HYPHEN_EXTRACT = re.compile(r'\b([a-z]{2,12}(?:-[a-z0-9]{1,12}){1,4})\b')

# ---------------------------------------------------------------------------
# Confidence Policy Layer
#
# Deterministic enforcement rules based on registered confidence score.
# These fire regardless of whether the model's phrasing was cooperative.
#
# Tiers:
#   HIGH RISK  (eff_conf < _GTS_WARN_THRESHOLD):
#       annotation: "# ⚠⚠ CREDENCE[HIGH RISK, conf=X.XX]: unverified — ..."
#       prose:      "⚠⚠ CREDENCE[HIGH RISK, conf=X.XX]: ..."
#
#   UNVERIFIED  (_GTS_WARN_THRESHOLD ≤ eff_conf < _GTS_QUALIFY_THRESHOLD):
#       annotation: "# ⚠ CREDENCE[unverified, conf=X.XX]: ..."
#
#   LOW CONFIDENCE  (eff_conf ≥ _GTS_QUALIFY_THRESHOLD but still unverified):
#       annotation: "# CREDENCE[check, conf=X.XX]: ..."
#
#   VERIFIED: no annotation.
#
# Confidence decay formula: j_score × 0.95^turns_elapsed (from registry).
# A constraint registered at j=0.28 hits HIGH RISK after ~8 unverified turns.
# ---------------------------------------------------------------------------
_GTS_WARN_THRESHOLD    = 0.20   # below this: HIGH RISK
_GTS_QUALIFY_THRESHOLD = 0.40   # below this: standard UNVERIFIED annotation

# ---------------------------------------------------------------------------
# Session persistence versioning — increment when state schema changes.
# Migration functions keyed by old version handle upgrades transparently.
# ---------------------------------------------------------------------------
_SESSION_VERSION = "1.1"

# ---------------------------------------------------------------------------
# Adaptive threshold configuration
#
# Fixed global thresholds (theta_high=0.65) misfire because the J-distribution
# shifts by session type: a Q&A session clusters [0.60-0.91]; a coding session
# clusters tighter around [0.62-0.72]. A fixed cutoff that works for one
# mis-classifies the other.
#
# Solution: track a rolling buffer of J-scores and set thresholds as percentiles
# of what this specific session is producing — "compress the top 25%, preserve
# the bottom 25%, TRIM the middle 50%".
#
# Safety floors prevent over-compression during warmup or in pathological sessions:
#   _ADAPTIVE_THETA_HIGH_FLOOR = 0.65  — never compress below this (global default)
#   _ADAPTIVE_THETA_LOW_CEIL   = 0.55  — never let LOW zone creep above this
#
# Empirical calibration from 30-question benchmark (April 2026):
#   Full J range: [0.594, 0.907], mean=0.691
#   At theta_high=0.70: factual=8/10 HIGH, uncertain=0/10 HIGH → clean separation
#   At theta_high=0.65: uncertain=6/10 HIGH → wrong, compresses uncertain responses
#   → Floor set to 0.65; adaptive P75 raises it further when session warrants.
# ---------------------------------------------------------------------------
_ADAPTIVE_MIN_SAMPLES      = 5   # turns before adaptive mode activates (warmup)
_ADAPTIVE_BUFFER_SIZE      = 20  # rolling window (last N turns)
_ADAPTIVE_THETA_HIGH_FLOOR = 0.65  # global safety floor — never compress below J=0.65
_ADAPTIVE_THETA_LOW_CEIL   = 0.55  # cap — LOW zone must stay below this

# Faithfulness guard — markers that signal user-flagged uncertainty.
# If the compressible old segment contains these, Haiku may strip them,
# turning tentative facts into apparent certainties. We refuse to compress
# (return 0 → caller falls through to PRESERVE) rather than risk hallucination.
_UNCERTAINTY_MARKERS = frozenset({
    # Core hedging — explicit epistemic qualification
    "not certain", "not sure", "uncertain", "tentative", "unverified",
    "approximately", "roughly", "i think", "i believe", "i'm not",
    "might be", "might not", "may be", "possibly", "perhaps",
    "i'd verify", "need to check", "should verify", "to verify",
    "approx", "tbd",
    # Common hedging words not covered by phrase-level markers
    "probably", "maybe", "provisionally", "preliminary", "supposedly",
    "ambiguous", "unclear", "hasn't clarified", "not yet clarified",
    # Open-question markers — unresolved decisions
    "unconfirmed", "not confirmed", "not yet confirmed", "open question", "still open",
    "needs verification", "need to verify",
    "not yet decided", "not decided", "to be determined", "to be confirmed",
    "haven't confirmed", "haven't verified", "haven't checked",
    # Conditional uncertainty — claim depends on unresolved variable
    "depending on", "depends on whether", "subject to", "contingent on",
    "once we confirm", "once we verify", "pending confirmation",
    # Knowledge-gap hedging — limits of personal knowledge
    "as far as i know", "to my knowledge", "to my understanding",
    "if i recall", "i seem to recall", "last time i checked",
    "best of my knowledge",
    # Working-theory markers — provisional hypotheses
    "working theory", "my assumption", "i'm assuming", "in theory",
    "could be wrong", "not 100%", "not entirely sure",
    # Source-transfer markers — second-hand claims
    "the vendor said", "they mentioned", "reportedly", "supposedly",
    "the docs say", "i read somewhere", "heard that", "we were told",
    # Measurement hedging — approximate quantities
    "give or take", "ballpark", "order of magnitude", "in the range of",
    "somewhere around", "plus or minus", "estimated at",
    # Incompleteness — untested assumptions
    "untested", "not yet tested", "haven't tested", "not benchmarked",
    "untested assumption", "needs benchmarking",
    # Memory hedging — recalling without verification (live demo protection)
    "iirc", "afaik", "if i recall correctly", "from memory", "off the top of my head",
    "as best i recall", "i think i remember", "i'm pretty sure but",
    # Uncertainty verbs — "I'm unsure", "not sure which", "unsure of"
    "i'm unsure", "unsure", "not sure which", "unsure which", "unsure of",
    # Hearsay / second-hand claims
    "according to the rep", "per the ticket", "per the thread", "vendor claims",
    "sales rep said", "they told us", "our rep mentioned", "the rep said",
    "according to their docs", "per their docs",
    # Vendor/sales channel — ghost constraint patterns (implicit unverified sources)
    "per the vendor", "from the vendor", "according to the vendor",
    "vendor estimate", "vendor ballpark", "vendor said", "vendor mentioned",
    "sales call", "from the sales call", "during the sales call",
    "the demo showed", "from the demo", "according to the demo", "per the demo",
    "account rep said", "the account rep", "sales team said", "the sales team",
    "from a quote", "per the quote", "based on a quote", "their estimate",
    "their ballpark", "in their quote", "per their estimate",
    # Informal / undocumented sources
    "from a blog post", "i read online", "saw online", "read online",
    "from a forum", "per a forum post", "from a slack message", "in a slack",
    "from a tweet", "per a tweet", "someone mentioned", "i heard",
    "from a stackoverflow", "from stackoverflow", "per stackoverflow", "from stack overflow",
    "from a reddit post", "per reddit", "saw on reddit", "per a reddit",
    # Production-untested assertions
    "not production-tested", "not load-tested", "never tested in production",
    "works in theory", "should work in theory", "ought to work",
    "not stress-tested", "not benchmarked in production",
    # Conditional epistemic — depends on unresolved state
    "assuming that", "assuming this is correct", "if that's right",
    "if that's correct", "if i'm reading this right", "if the config is right",
    # Past-tense and contracted forms — same epistemic meaning, different tense
    # ("weren't sure about X" is identical uncertainty to "not sure about X")
    "wasn't sure", "weren't sure", "wasn't certain", "weren't certain",
    "hadn't verified", "hadn't confirmed", "hadn't checked",
    "didn't confirm", "didn't verify", "didn't check",
    "wasn't confirmed", "wasn't verified",
    # Modal-past uncertainty — "it might have been changed" is unverified
    "might have been", "may have been", "could have been",
    "might have changed", "may have changed",
    # Modal-present uncertainty — "might require/need/extend/support/change"
    # Any "might + verb" form expresses epistemic possibility (not certainty).
    # "might " (with trailing space) catches all "might + word" patterns in one marker.
    # "may have" (without "been") covers "may have a 30-day SLA" type claims.
    "might ", "may have",
    # Knowledge-gap — parallel to haven't confirmed/verified/checked
    "haven't seen", "hasn't finalized", "hasn't finalised",
    # Impersonal/third-person hedging — typical in assistant responses and summaries
    # These were absent from the original list but are canonical English uncertainty
    # expressions (found by auditing FCR-scored downstream answers that were actually hedged)
    "pending verification", "requiring verification", "requires verification",
    "not fully confirmed", "not been confirmed", "not been finalized",
    "not yet finalized", "not yet been finalized",
    "cannot definitively", "not definitively",
    "unresolved", "still pending",
    "ambiguity",
    "needs confirmation",
    # NOTE: removed "hypothesis/hypotheses" (fires on confirmed reasoning),
    # "awaiting" (administrative, not epistemic),
    # "pending decision" (workflow, not epistemic),
    # "under discussion" (process state, not factual uncertainty)
    # Appearance/seeming hedges — "seems to", "appears to" are canonical English uncertainty
    # expressions; absent from prior list despite being among the most common hedging forms.
    # Found by auditing 77/154 unguarded EQL-Bench v2 explicit scenarios with EQLR events.
    "seems to", "seem to", "appears to", "appear to",
    "seems like", "seem like", "appears like",
    # Person-attribution — second-hand from colleague/peer (not vendor/rep)
    "colleague said", "a colleague", "colleague mentioned", "colleague told",
    "my colleague", "team member said", "teammate said", "coworker said",
    # Sales channel — "sales claimed/said" not covered by "sales rep said"
    "sales claimed", "sales said", "sales told us", "sales mentioned",
    # Inferential / log-derived — empirical but not officially confirmed
    "inferred from", "inferred based", "we inferred",
    "logs show", "from the logs", "log timestamps", "the logs show",
    "log shows", "log suggests", "logs suggest",
    "observed from", "observed in logs", "observed in testing",
    # Absence-of-documentation markers — no official source confirms the claim
    "not documented", "undocumented", "no documentation", "no official documentation",
    "no official", "not officially", "no official confirmation",
    "unofficially", "off the record",
    # Source-qualified indirect attribution — "based on our" is often unconfirmed
    "based on our understanding", "based on our assumption", "based on what we heard",
    "based on feedback", "based on what they said", "based on the conversation",
    # Issue/chat-tracker sourced — informal channels, not authoritative docs
    "github issue", "from a github issue", "per the github issue",
    "per the issue", "from the issue tracker",
    "from a slack", "per the slack",
    # Vendor assertion (not contracted) — "vendor says X" without confirmation in contracts
    "vendor says", "vendor claims",
    # Group/personal estimate — "we estimate", "we think" parallel to "i think"/"i estimate"
    "we estimate", "we think", "we suspect", "i suspect",
    # Not verified / not reproduced — explicit statement of missing verification
    "not verified", "not reproduced", "not been reproduced", "haven't reproduced",
    "haven't measured", "haven't reviewed the",
    # Back-of-envelope / rough estimate — explicit informal calculation
    "back-of-envelope", "back of envelope", "rough estimate", "envelope math",
    # Informal source — blog post, conference talk, anecdote
    "blog post", "a blog post", "the blog post", "blog post claims", "blog post says",
    "conference talk", "conference session", "a conference", "at the conference",
    "anecdotal", "anecdotally", "informal observation", "informally observed",
    "informally", "verbal quote", "verbal commitment", "not from a formal",
    # Preliminary / provisional determination — "is preliminarily X"
    "is preliminarily", "preliminarily", "preliminary finding",
    # Not in contracts / not official — claim not backed by formal agreement
    "not in our contracts", "not in the contract", "not in the sla",
    "subject to change", "may change", "could change without",
    # Personal/org attribution from external parties
    "company said", "engineer said", "architect said", "manager said", "someone said",
    "a researcher said", "an analyst said", "an expert said", "specialist said",
    # Vendor possessive + document type — "vendor's guide says", "vendor's whitepaper claims"
    # "vendor's" alone is too broad (fires on "vendor's DPA meets X" with no hedging);
    # only match when followed by a document type that implies the claim is uncontracted.
    "vendor's guide", "vendor's blog", "vendor's whitepaper", "vendor's documentation",
    "vendor's readme", "vendor's docs", "vendor's roadmap",
    # Early-stage / nothing-decided — preliminary design markers
    "early discussion", "early exploration", "early spike", "early thinking",
    "nothing finalized", "nothing decided", "nothing ratified",
    "not yet ratified", "nothing committed", "no adr",
    # Academic pre-publication — preprints are explicitly unreviewed
    "a preprint", "preprint claims", "preprint says", "preprint suggests",
    "not yet peer-reviewed", "not peer-reviewed", "awaiting peer review",
    # Informal guidance / advice — not from authoritative source
    "informal guidance", "informal advice", "informal claim", "informal post",
    "informal assessment", "informal interpretation",
    # Extrapolation — inference beyond the data
    "extrapolated from", "extrapolated based", "extrapolating from",
    # Not tested in / against — explicitly untested deployment context
    "not tested in", "not tested on", "not tested with", "not tested against",
    "not tested under", "untested in",
    # Marketing materials — explicitly promotional, not technical verification
    "marketing materials", "marketing collateral", "marketing deck",
    # Consultant said / suggested — external informal guidance
    "consultant suggested", "consultant mentioned", "consultant said",
    "a consultant", "the consultant", "outside counsel", "informally advised",
    # Conflicting sources — two authoritative sources disagree → claim is unresolved
    "conflicting reports", "conflicting data", "conflicting results",
    "conflicting studies", "conflicting benchmarks", "conflicting estimates",
    "two reports", "two studies", "two benchmarks",
    # May not apply / hold — explicit conditional limitation
    "may not apply", "may not hold", "may not generalize", "may not transfer",
    "may not be applicable", "may not be representative",
    # Forum post — expanding forum source coverage beyond "per a forum post"
    "forum post", "forum thread", "forum says", "forum suggests",
    # Rough reading — informal measurement
    "rough reading", "rough calculation", "rough numbers",
    # Quick/general read — casual interpretation, not reviewed by expert
    "quick read", "quick reading", "general reading", "general interpretation",
    "not reviewed by counsel", "not reviewed by legal",
    # "Someone said" — anonymous/person attribution without context
    "someone said", "someone told", "someone mentioned",
    # Slack-specific — "internal slack thread suggests" (vs "from a slack message" already present)
    "slack thread", "slack channel suggested", "slack discussion",
    # Team hasn't decided — preliminary state
    "hasn't decided", "haven't decided", "not been decided",
    # Not formally — claim not officially processed/triaged
    "not formally", "not formally confirmed", "not formally reviewed",
    # Medical/scientific study type qualifiers
    "a case report", "case report described", "case report suggests",
    "phase 1 study", "phase 1 safety", "phase 1 trial", "phase i study",
    "single case", "no control group",
    # Not yet assessed — explicitly unevaluated
    "not yet assessed", "not been assessed", "not assessed",
    "not yet evaluated", "not been evaluated",
    # Not announced — future feature without official announcement
    "not announced", "not officially announced",
    # Sales/marketing materials as source
    "sales deck", "sales presentation", "the sales deck",
    # Paper citation without peer review signal
    "a paper claimed", "a paper claims", "a paper says", "a paper suggests",
    "a case study claimed", "case study claims",
    # Status page ≠ SLA — operational data without contractual backing
    "status page shows", "status page claims", "dashboard shows",
    "hasn't been verified by",
})

# Semantic entropy proxy — markers that signal multiple valid answers.
# Detected in MEDIUM-J responses → zone downgraded to LOW → PRESERVE.
# Zero-cost single-pass approximation of sampling-based semantic entropy
# (Kuhn et al. ICLR 2023): when the model signals competing answers itself,
# a re-sample would likely diverge → high semantic uncertainty → preserve.
_MULTI_ANSWER_MARKERS = frozenset({
    "it depends on", "depends on the", "depends on what",
    "there are several", "there are two", "there are multiple",
    "varies by", "varies depending", "context-dependent",
    "some would argue", "others might say", "one approach",
    "on one hand", "on the other hand", "two main", "multiple approaches",
    "no single answer", "no one-size", "case by case",
    "different perspectives", "highly contextual",
    "not entirely clear", "no clear consensus", "no universal",
})

# Novelty detection — kept at 0.75 but now applied to content words, not entities.
# Content words fix two entity-counting failures:
#   1. Topic shifts without proper nouns ("optimistic locking" → "database sharding")
#   2. Sentence-start capitalization false positives (words like "The", "This")
# Three-gate remains: vocab ≥10, ≥5 new content words, >75% of response words new.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Single turn output returned to the caller."""
    turn_idx:         int
    response:         str
    j_score:          float
    zone:             str
    decision:         str          # COMPRESS | TRIM | PRESERVE
    tokens_in:        int
    tokens_out:       int
    tokens_saved:     int
    cost_usd:         float
    savings_usd:      float
    reasoning:        str
    session_tokens_used:   int
    session_tokens_saved:  int
    session_cost_usd:      float
    session_savings_usd:   float
    compression_ratio:     float   # saved / (used + saved)
    thinking_tokens:       int   = 0    # 0 when use_thinking=False
    thinking_utilization:  float = 0.0  # thinking_tokens / budget; 0 when not used
    thinking_budget_used:  int   = 0    # budget allocated for this turn (continuous J-governor)
    drift_state:           bool  = False  # True when 3+ consecutive LOW-zone turns detected
    adaptive_theta_high:   float = 0.65  # effective HIGH threshold used this turn
    adaptive_theta_low:    float = 0.35  # effective LOW threshold used this turn
    uncertainty_preserved: bool  = False  # True when faithfulness probe fired or LOW/MEDIUM PRESERVE
    truth_buffer_count:    int   = 0    # number of unverified constraints injected this turn
    scout_extractions:     int   = 0    # constraints auto-registered by Scout this turn
    alignment_warnings:    list  = field(default_factory=list)  # Governor: dicts from AlignmentWarning.to_dict()
    caveat_injected:       bool  = False  # True when previous-turn Governor warning injected
    user_uncertainty_detected: bool  = False  # True when user message contains uncertainty markers
    se_score:          float  = 0.0   # semantic entropy score (0=certain, 1=uncertain); 0 when probe not used
    se_uncertain:      bool   = False  # True when SE probe fired and overrode J-routing
    enforcement_active: bool  = False  # True when Consistency Enforcer fired (query directly hit unverified constraint)
    scan_hits:         list   = field(default_factory=list)  # GTS: code literals annotated with unverified constraint tags
    ghost_detections:  int   = 0    # constraints detected by Opus ghost detector this turn
    contradictions_detected: list = field(default_factory=list)  # [{constraint_id, prior_value, new_value, reasoning}]

    @property
    def envelope(self) -> dict:
        """
        Return a CredenceEnvelope dict for this turn — MCP-serializable epistemic
        provenance record. Downstream agents check envelope['should_verify']
        and envelope['safe_to_compress'] before acting on this information.
        """
        from .envelope import CredenceEnvelope
        return CredenceEnvelope.from_turn(
            response              = self.response,
            j_score               = self.j_score,
            zone                  = self.zone,
            decision              = self.decision,
            content_type          = "text",
            source                = "credence",
            uncertainty_preserved = self.uncertainty_preserved,
        ).to_dict()


@dataclass
class SessionStats:
    total_tokens_in:   int   = 0
    total_tokens_out:  int   = 0
    total_tokens_saved: int  = 0
    total_cost_usd:    float = 0.0
    total_savings_usd: float = 0.0
    turns_compressed:  int   = 0
    turns_trimmed:     int   = 0
    turns_preserved:   int   = 0
    decision_log:      list  = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        total = self.total_tokens_in + self.total_tokens_saved
        if total == 0:
            return 0.0
        return self.total_tokens_saved / total


# ---------------------------------------------------------------------------
# AlignmentWarning — output of the Governor's post-generation check
# ---------------------------------------------------------------------------

@dataclass
class AlignmentWarning:
    """
    Fired when a generated response asserts confidence about an unverified constraint.

    The Governor (Output Alignment Layer) compares every response's epistemic tone
    against the live ledger. When a HIGH or MEDIUM response discusses a constraint
    the ledger marks as LOW/MEDIUM and unverified, an AlignmentWarning is raised.

    Enforcement is synchronous: the caveat is appended inline to the current response
    so the user sees it immediately. The suggested_caveat is also queued for injection
    into the NEXT turn's system prompt so the model self-corrects going forward.
    """
    constraint_id:      str
    constraint_content: str
    ledger_zone:        str   # LOW or MEDIUM
    response_zone:      str   # HIGH or MEDIUM (assertive language detected)
    overlap_words:      list  # content words that matched between response and constraint
    suggested_caveat:   str   # injected into next turn's system prompt

    def to_dict(self) -> dict:
        return {
            "constraint_id":      self.constraint_id,
            "constraint_content": self.constraint_content,
            "ledger_zone":        self.ledger_zone,
            "response_zone":      self.response_zone,
            "overlap_words":      self.overlap_words,
            "suggested_caveat":   self.suggested_caveat,
        }


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Drop-in wrapper around the Claude API with adaptive context management.

    Usage:
        mgr = ContextManager()
        result = mgr.chat("What is the capital of France?")
        print(result.response)
        print(f"J={result.j_score:.2f}  saved={result.tokens_saved} tokens")
    """

    COMPRESS_AFTER     = 8    # turns: compress history older than this on HIGH (fires at n_turns > 16)
    TRIM_WINDOW        = 10   # turns: keep last N turns on MEDIUM
    ATTENTION_SINK     = 2    # turns: never compress first N turns (attention sinks)
    MAX_COMPRESSIONS   = 3    # stop compressing after this many (quality guard)

    @property
    def client(self):
        """The model client, created on first use.

        Deferred rather than built in __init__. The deterministic layers — the
        faithfulness probe, the Truth Buffer, the Consistency Enforcer, GTS,
        and any path that only *builds* a prompt — need no API client at all.
        Requiring the SDK to construct a ContextManager made that logic
        unusable without `anthropic` installed and left the mock-LLM
        integration suite unable to run. The honest moment to ask for a client
        is the moment a model call is actually made.
        """
        if self._client is None:
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "The 'anthropic' package is required for model calls. "
                    "Install it with: pip install 'credence-enforce[api]'"
                )
            resolved_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._client = Anthropic(api_key=resolved_key)
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    def __init__(
        self,
        api_key:        Optional[str] = None,
        theta_high:     float = 0.70,
        theta_low:      float = 0.45,
        system_prompt:  Optional[str] = None,
        max_tokens:     int = 1024,
        use_thinking:      bool = False,   # enable adaptive thinking budget
        use_agreement:     bool = False,   # enable agreement-based second signal for MEDIUM zone
        main_model:        Optional[str] = None,  # override default Opus model
        compression_model: Optional[str] = None,  # override default Haiku model
        registry:          Optional["CredenceRegistry"] = None,  # Truth Buffer + Scout source
        session_id:        Optional[str] = None,  # session key for registry lookup
        use_scout:         bool = False,   # enable Scout Classifier entity extraction
        use_semantic_entropy: bool = False,   # enable SE probe on MEDIUM-zone turns
        use_claim_extraction: Optional[bool] = None,  # None = auto: True when registry provided
        use_ghost_detector:   bool = False,   # enable Opus-powered ghost constraint detection
        use_behavioral:       bool = False,   # enable Tier 2 behavioral consistency probe
        use_manifest:         bool = False,   # use structured EpistemicManifest instead of Truth Buffer
        client = None,                         # pre-built client (e.g. ClaudeCodeClient); overrides api_key
    ):
        # Client and key only. The client itself is built on first use by the
        # `client` property, so constructing a ContextManager requires no SDK.
        self._client  = client
        self._api_key = api_key
        self.proxy         = CredenceProxy(theta_high, theta_low)
        self.system_prompt = system_prompt or (
            "You are a helpful, precise assistant. "
            "Give concise answers when the answer is clear; "
            "express genuine uncertainty when it exists."
        )
        self.max_tokens        = max_tokens
        self.use_thinking      = use_thinking
        self.use_agreement     = use_agreement
        self.use_scout         = use_scout
        self.use_semantic_entropy = use_semantic_entropy
        # Auto-enable claim extraction when a registry is provided — if a store
        # exists, claims should be extracted and preserved automatically.
        # Explicit False overrides (e.g. for read-only registry use).
        if use_claim_extraction is None:
            self.use_claim_extraction = (registry is not None)
        else:
            self.use_claim_extraction = use_claim_extraction
        self.use_ghost_detector = use_ghost_detector
        self.use_behavioral     = use_behavioral
        self.use_manifest       = use_manifest
        # Import here to avoid circular; SE and behavioral probes are lazy-constructed
        self._se_probe         = None
        self._behavioral_probe = None
        self._claim_extractor  = None   # lazy-init ClaimExtractor when use_claim_extraction=True
        self.main_model        = main_model or _MODEL_OPUS
        self.compression_model = compression_model or _MODEL_HAIKU
        self.stats             = SessionStats()

        # Truth Buffer + Scout: registry supplies unverified constraints that are
        # injected into the system prompt before every turn so Claude never forgets
        # what it doesn't know. Scout auto-extracts new constraints from user messages.
        self._registry   = registry    # type: Optional[CredenceRegistry]
        self._session_id = session_id

        # Live conversation state
        self._history:           list[dict] = []
        self._summary:           Optional[str] = None
        self._turn_idx:          int = 0
        self._compression_count: int = 0
        self._prev_zone:         str = "MEDIUM"   # seed: neutral until first turn
        self._prev_j:            float = 0.50     # prev-turn J-score; seeds thinking budget
        self._turn1_goal:        str = ""         # user's first message — anchors compression
        self._current_user_message: str = ""      # for query-aware Truth Buffer

        # Drift detection: rolling J history; drift when 3 consecutive LOW turns
        self._j_history:  list[float] = []
        self._drift_state: bool = False

        # Adaptive threshold tracking: rolling J-buffer for percentile-based thresholds.
        # After _ADAPTIVE_MIN_SAMPLES turns, theta_high = P75, theta_low = P25 of buffer.
        self._j_buffer:         list[float]          = []

        # Parallel J-score list for history messages (same length as _history).
        # User messages store None; assistant messages store the turn's J-score.
        # Used by selective compression to identify HIGH-J turns for compression
        # while keeping LOW/MEDIUM-J turns verbatim.
        self._history_j_scores: list[Optional[float]] = []

        # Faithfulness probe firing flag — set True in _compress() when the probe
        # blocks compression. Read and reset in chat() to set uncertainty_preserved.
        self._last_faithfulness_block: bool = False

        # Compression shadow — pre-compression state kept during _compress().
        # Restored immediately if faithfulness check or ROI gate fails.
        self._compression_shadow:         Optional[list[dict]]            = None
        self._compression_shadow_j:       Optional[list[Optional[float]]] = None
        self._compression_shadow_summary: Optional[str]                   = None

        # Output Alignment Layer (Governor): caveat from previous turn's alignment check.
        # Set by _align_output() at end of turn N; injected into system prompt at turn N+1.
        # Async pattern: zero latency on current response, corrects the next one.
        self._pending_alignment_caveat: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> TurnResult:
        """Send a message, receive a Credence-managed response."""
        self._turn_idx += 1
        self._current_user_message = user_message   # for query-aware Truth Buffer
        if self._turn_idx == 1:
            self._turn1_goal = user_message   # anchor for task-aware compression
        self._history.append({"role": "user", "content": user_message})
        self._history_j_scores.append(None)   # user message — no J-score yet

        # Direct user-turn uncertainty detection — the primary protection signal.
        # J-score measures the assistant's response; this measures what the USER said.
        # When users express uncertainty, mark the turn-pair as protected regardless
        # of what J-score the assistant response receives.
        # Implementation: overwrite the None placeholder with 0.0 (sentinel that means
        # "user flagged this turn as uncertain — never eligible for compression").
        user_uncertainty_detected = self._has_uncertainty(user_message)
        if user_uncertainty_detected:
            self._history_j_scores[-1] = 0.0   # sentinel: protect this pair

        # Scout Classifier: extract and register uncertain constraints from user message
        # before the main API call so Truth Buffer picks them up on the next turn.
        scout_count = 0
        if self.use_scout:
            extractions = self._scout_classify(user_message)
            scout_count = sum(
                1 for e in extractions
                if isinstance(e, dict) and e.get("confidence_level", "high") != "high"
            )

        # Ghost Detector: Opus-powered detection of implicit uncertain constraints.
        # Fires only when canonical markers are absent (faithfulness probe covers those).
        # High-precision: only registers claims Opus is ≥70% confident are ghost constraints.
        ghost_count = 0
        if self.use_ghost_detector and not user_uncertainty_detected:
            ghost_items = self._ghost_detect(user_message)
            ghost_count = len(ghost_items)

        # Contradiction Detector: Opus-powered cross-turn conflict detection.
        # When a new user message introduces a value that conflicts with a registered
        # constraint, Opus reasons about which is more reliable and marks the conflict.
        # Fires on any turn when registry is set and ghost detector is enabled —
        # contradiction detection is always-on when the epistemic stack is active.
        contradictions = []
        if self.use_ghost_detector and self._registry is not None and self._session_id is not None:
            contradictions = self._detect_contradiction(user_message)

        # Truth Buffer: count how many unverified constraints are injected this turn
        truth_buffer_count = 0
        if self._registry is not None and self._session_id is not None:
            truth_buffer_count = len(self._registry.list_uncertain(self._session_id))

        # Build messages and call Opus for the main answer.
        # _build_enforcement_system_prompt wraps Truth Buffer and upgrades to
        # Consistency Enforcement when the query directly hits an unverified constraint.
        # When registry is None, fall back to caveat-only augmentation.
        caveat_injected_this_turn = self._pending_alignment_caveat is not None
        enforcement_active = False
        if self._registry is not None:
            augmented_system, enforcement_active = self._build_enforcement_system_prompt(user_message)
        else:
            augmented_system = self._augment_system_with_caveat()
        # Caveat consumed — clear it before the API call so it doesn't persist
        self._pending_alignment_caveat = None
        messages    = self._build_messages()
        call_kwargs, thinking_budget = self._build_call_kwargs(augmented_system, messages)
        resp        = self.client.messages.create(**call_kwargs)
        text, tokens_in, tokens_out, thinking_tokens, thinking_utilization = (
            self._parse_response(resp, thinking_budget)
        )

        # Compute J-proxy
        cr: CredenceResult = self.proxy.compute(text)

        # Dual-signal zone adjustment: if thinking utilization is high (model
        # worked hard) but J-proxy says HIGH (text looks confident), downgrade
        # to MEDIUM. Confident-sounding text after heavy deliberation signals
        # latent difficulty — compressing away the history would be premature.
        # Threshold: >50% of budget consumed despite HIGH text signal.
        if thinking_utilization > 0.50 and cr.zone == "HIGH":
            cr = CredenceResult(
                j_score      = cr.j_score,
                zone         = "MEDIUM",
                factors      = cr.factors,
                reasoning    = cr.reasoning + f"; thinking override ({thinking_utilization:.0%} utilization)",
                content_type = cr.content_type,
            )

        # Semantic entropy proxy: MEDIUM-J responses containing multi-answer
        # markers (e.g. "it depends on", "case by case") signal that a re-sample
        # would likely yield a different answer — high semantic uncertainty.
        # Downgrade zone to LOW → PRESERVE. Zero-cost Kuhn 2023 approximation.
        if cr.zone == "MEDIUM" and self._has_multi_answer(text):
            cr = CredenceResult(
                j_score      = cr.j_score,
                zone         = "LOW",
                factors      = cr.factors,
                reasoning    = cr.reasoning + "; semantic entropy proxy (context-dependent answer detected)",
                content_type = cr.content_type,
            )

        # Semantic Entropy probe: fires on all MEDIUM and HIGH-zone turns when enabled.
        # Generates N=3 Haiku re-completions; high inter-sample variance → uncertain →
        # override to LOW → PRESERVE.
        #
        # Ghost constraints (implicit uncertainty, no canonical hedging markers) score
        # HIGH-J and bypass the faithfulness probe. The prior J > 0.85 fast-path skipped
        # SE at exactly the J-range where ghost constraints appear — that was a blind spot.
        # SE now covers the full MEDIUM+HIGH zone. Cost: ~3 extra Haiku calls per
        # very-high-J turn when use_semantic_entropy=True (~$0.0005/turn).
        # When use_semantic_entropy=False, J alone routes.
        se_score     = 0.0
        se_uncertain = False
        if self.use_semantic_entropy and cr.zone in ("MEDIUM", "HIGH"):
            if self._se_probe is None:
                from .semantic_entropy import SemanticEntropyProbe
                self._se_probe = SemanticEntropyProbe()
            se_result    = self._se_probe.compute(messages, self.client)
            se_score     = se_result.entropy_score
            se_uncertain = se_result.is_uncertain
            if se_uncertain and cr.zone != "LOW":
                cr = CredenceResult(
                    j_score      = cr.j_score,
                    zone         = "LOW",
                    factors      = cr.factors,
                    reasoning    = cr.reasoning + f"; SE_override({se_result.reasoning})",
                    content_type = cr.content_type,
                )

        # Agreement-based second signal (opt-in, use_agreement=True).
        # On MEDIUM-J turns — where J-proxy is least reliable — ask Haiku to
        # rate its own confidence. Fuse: J_final = 0.7*J + 0.3*agreement.
        # Uses Haiku to keep cost negligible (~$0.0002/call).
        # Only fires on MEDIUM because HIGH and LOW are already decided;
        # MEDIUM is the boundary zone where a second reading matters most.
        if self.use_agreement and cr.zone == "MEDIUM":
            agreement = self._agreement_score(text)
            fused_j   = round(0.7 * cr.j_score + 0.3 * agreement, 4)
            fused_zone = (
                "HIGH"   if fused_j >= self._effective_theta_high else
                "MEDIUM" if fused_j >= self._effective_theta_low  else
                "LOW"
            )
            cr = CredenceResult(
                j_score      = fused_j,
                zone         = fused_zone,
                factors      = cr.factors,
                reasoning    = cr.reasoning + f"; agreement={agreement:.2f} fused_j={fused_j:.3f}",
                content_type = cr.content_type,
            )

        # Behavioral consistency probe (Tier 2, opt-in via use_behavioral=True).
        # Draws N=5 Haiku re-completions and measures pairwise ROUGE-L variance.
        # High variance → model is uncertain despite assertive-sounding text → downgrade zone.
        # Fires on MEDIUM and HIGH (the J ranges where ghost constraints appear).
        # Cost: ~$0.0003/turn (5 Haiku calls at max_tokens=150).
        # Only upgrades downward — a MEDIUM that looks consistent stays MEDIUM;
        # a HIGH that is wildly inconsistent becomes MEDIUM or LOW.
        if self.use_behavioral and cr.zone in ("MEDIUM", "HIGH"):
            if self._behavioral_probe is None:
                from .behavioral_signal import BehavioralConsistencyProbe
                self._behavioral_probe = BehavioralConsistencyProbe()
            beh_result = self._behavioral_probe.compute(
                messages  = messages,
                client    = self.client,
                theta_high = self._effective_theta_high,
                theta_low  = self._effective_theta_low,
                j_input    = cr.j_score,
            )
            _zone_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if _zone_rank.get(beh_result.fused_zone, 1) < _zone_rank.get(cr.zone, 1):
                cr = CredenceResult(
                    j_score      = beh_result.fused_j,
                    zone         = beh_result.fused_zone,
                    factors      = cr.factors,
                    reasoning    = (
                        cr.reasoning
                        + f"; behavioral(var={beh_result.variance:.3f}"
                        + f" cons={beh_result.consistency_score:.3f}"
                        + f" fused={beh_result.fused_j:.3f})"
                    ),
                    content_type = cr.content_type,
                )

        # Drift detection: 3 consecutive LOW-zone turns → proactive PRESERVE lock
        self._j_history.append(cr.j_score)
        if len(self._j_history) > 5:
            self._j_history.pop(0)
        self._drift_state = (
            len(self._j_history) >= 3
            and all(j < self.proxy.theta_low for j in self._j_history[-3:])
        )

        # Governor: synchronous output enforcement.
        # Checks whether the response states something with higher confidence than
        # the epistemic ledger warrants. When a HIGH-J response touches an unverified
        # LOW/MEDIUM constraint, the warning is appended inline to the current response
        # so the user sees it immediately — not deferred to the next turn.
        #
        # The caveat is also queued in _pending_alignment_caveat so the model itself
        # sees the warning in the NEXT turn's system prompt (belt-and-suspenders).
        alignment_warnings_raw = self._align_output(text)
        if alignment_warnings_raw:
            inline_caveats = "  ".join(
                f"[⚠ unverified: \"{w.constraint_content[:70]}\" — stored as {w.ledger_zone} confidence]"
                for w in alignment_warnings_raw
            )
            text = f"{text}\n\n---\n*Credence: {inline_caveats}*"
            self._pending_alignment_caveat = "\n".join(
                w.suggested_caveat for w in alignment_warnings_raw
            )
        alignment_warnings_dicts = [w.to_dict() for w in alignment_warnings_raw]

        # Generation-Time Constraint Scanner: annotate code literals that embed
        # registered uncertain values. Fires after alignment check so the full
        # (caveat-appended) text is scanned. Returns annotated text and hit list.
        scan_hits: list[dict] = []
        if self._registry is not None and self._session_id is not None:
            text, scan_hits = self._scan_output_for_constraints(text)

        # Append assistant turn BEFORE compression decision
        self._history.append({"role": "assistant", "content": text})
        self._history_j_scores.append(cr.j_score)   # assistant message — record J

        # Apply Credence memory decision
        decision, tokens_saved = self._apply_credence(cr)

        # Capture effective thresholds AFTER apply_credence so buffer is still pre-update
        eff_high = self._effective_theta_high
        eff_low  = self._effective_theta_low

        # Determine if this turn preserved uncertainty verbatim.
        # True when: (a) faithfulness probe blocked compression, or
        #            (b) decision is PRESERVE and the zone is LOW or MEDIUM
        #                (i.e., uncertain/borderline content kept intact).
        # Read and reset the faithfulness flag AFTER shadow mutations are done.
        uncertainty_preserved = (
            self._last_faithfulness_block
            or (decision == "PRESERVE" and cr.zone in ("LOW", "MEDIUM"))
        )
        self._last_faithfulness_block = False

        self._prev_zone = cr.zone
        self._prev_j    = cr.j_score

        # Auto-extract uncertain claims from this turn (EG-2, opt-in).
        # Addresses the probe vocabulary gap: catches implicit uncertainty that doesn't
        # use canonical hedging phrases ("the docs suggested", "vendor estimate was ~50").
        # Extracted claims are registered in the registry and visible in the Truth Buffer
        # from the NEXT turn onward (async — zero latency on this turn).
        if self.use_claim_extraction and self._registry is not None and self._session_id is not None:
            if self._claim_extractor is None:
                from .claim_extractor import ClaimExtractor
                self._claim_extractor = ClaimExtractor(model=self.compression_model)
            combined = f"User: {user_message}\nAssistant: {text}"
            self._claim_extractor.extract_and_register(
                combined, self.client, self._registry,
                self._session_id, self._turn_idx,
            )

        # Update adaptive threshold buffer (after decision — causal, not lookahead)
        self._j_buffer.append(cr.j_score)
        if len(self._j_buffer) > _ADAPTIVE_BUFFER_SIZE:
            self._j_buffer.pop(0)

        # Costs
        cost_usd    = _cost(tokens_in, tokens_out)
        savings_usd = _cost(tokens_saved, 0)

        # Update session stats
        self.stats.total_tokens_in    += tokens_in
        self.stats.total_tokens_out   += tokens_out
        self.stats.total_tokens_saved += tokens_saved
        self.stats.total_cost_usd     += cost_usd
        self.stats.total_savings_usd  += savings_usd
        if decision == "COMPRESS":
            self.stats.turns_compressed += 1
        elif decision == "TRIM":
            self.stats.turns_trimmed += 1
        else:
            self.stats.turns_preserved += 1

        self.stats.decision_log.append({
            "turn":                      self._turn_idx,
            "j_score":                   cr.j_score,
            "zone":                      cr.zone,
            "decision":                  decision,
            "tokens_saved":              tokens_saved,
            "reasoning":                 cr.reasoning,
            "semantic_entropy_override": "semantic entropy proxy" in cr.reasoning,
            "content_type":              cr.content_type,
            "thinking_tokens":           thinking_tokens,
            "thinking_utilization":      thinking_utilization,
            "thinking_budget_used":      thinking_budget,
            "drift_state":               self._drift_state,
            "adaptive_theta_high":       eff_high,
            "adaptive_theta_low":        eff_low,
            "uncertainty_preserved":     uncertainty_preserved,
            "truth_buffer_count":        truth_buffer_count,
            "scout_extractions":         scout_count,
            "alignment_warnings":        len(alignment_warnings_dicts),
            "caveat_injected":           caveat_injected_this_turn,
            "user_uncertainty_detected": user_uncertainty_detected,
            "se_score":          se_score,
            "se_uncertain":      se_uncertain,
            "enforcement_active":   enforcement_active,
            "contradictions":      len(contradictions),
            "scan_hits":         len(scan_hits),
            "ghost_detections":  ghost_count,
        })

        return TurnResult(
            turn_idx        = self._turn_idx,
            response        = text,
            j_score         = cr.j_score,
            zone            = cr.zone,
            decision        = decision,
            tokens_in       = tokens_in,
            tokens_out      = tokens_out,
            tokens_saved    = tokens_saved,
            cost_usd        = round(cost_usd, 6),
            savings_usd     = round(savings_usd, 6),
            reasoning       = cr.reasoning,
            session_tokens_used   = self.stats.total_tokens_in + self.stats.total_tokens_out,
            session_tokens_saved  = self.stats.total_tokens_saved,
            session_cost_usd      = round(self.stats.total_cost_usd, 4),
            session_savings_usd   = round(self.stats.total_savings_usd, 4),
            compression_ratio     = round(self.stats.compression_ratio, 3),
            thinking_tokens        = thinking_tokens,
            thinking_utilization   = thinking_utilization,
            thinking_budget_used   = thinking_budget,
            drift_state            = self._drift_state,
            adaptive_theta_high    = eff_high,
            adaptive_theta_low     = eff_low,
            uncertainty_preserved  = uncertainty_preserved,
            truth_buffer_count     = truth_buffer_count,
            scout_extractions      = scout_count,
            alignment_warnings     = alignment_warnings_dicts,
            caveat_injected        = caveat_injected_this_turn,
            user_uncertainty_detected = user_uncertainty_detected,
            se_score          = se_score,
            se_uncertain      = se_uncertain,
            enforcement_active = enforcement_active,
            scan_hits                = scan_hits,
            ghost_detections         = ghost_count,
            contradictions_detected  = contradictions,
        )

    # ------------------------------------------------------------------
    # chat() sub-phases — extracted for readability and testability
    # ------------------------------------------------------------------

    def _build_call_kwargs(self, augmented_system: str, messages: list) -> tuple[dict, int]:
        """Build Anthropic API call kwargs + thinking_budget. Returns (kwargs, budget)."""
        thinking_budget = 0
        call_kwargs = dict(
            model      = self.main_model,
            system     = augmented_system,
            messages   = messages,
            max_tokens = self.max_tokens,
        )
        if self.use_thinking and self._prev_j < self.proxy.theta_high:
            effort = "high" if self._prev_j < self.proxy.theta_low else "medium"
            call_kwargs["thinking"]      = {"type": "adaptive"}
            call_kwargs["output_config"] = {"effort": effort}
            thinking_budget = _THINKING_MAX if effort == "high" else _THINKING_MIN
        return call_kwargs, thinking_budget

    def _parse_response(self, resp, thinking_budget: int) -> tuple[str, int, int, int, float]:
        """Parse Anthropic response into (text, tokens_in, tokens_out, thinking_tokens, thinking_utilization)."""
        text_block   = next((b for b in resp.content if b.type == "text"), None)
        text         = text_block.text if text_block else ""
        tokens_in    = resp.usage.input_tokens
        tokens_out   = resp.usage.output_tokens
        thinking_tokens = sum(
            len(b.thinking) // 4
            for b in resp.content
            if b.type == "thinking" and hasattr(b, "thinking")
        )
        thinking_utilization = (
            round(thinking_tokens / thinking_budget, 3)
            if thinking_budget > 0
            else 0.0
        )
        return text, tokens_in, tokens_out, thinking_tokens, thinking_utilization

    def _adjust_zone(self, cr: "CredenceResult", thinking_utilization: float,
                     messages: list) -> "CredenceResult":
        """Apply all zone-adjustment signals after initial J-score computation.

        Order matters: thinking override → semantic entropy proxy → SE probe →
        agreement fusion → behavioral probe. Each can only downgrade the zone,
        never upgrade it.
        """
        # Dual-signal: heavy thinking on HIGH text → downgrade to MEDIUM
        if thinking_utilization > 0.50 and cr.zone == "HIGH":
            cr = CredenceResult(
                j_score      = cr.j_score,
                zone         = "MEDIUM",
                factors      = cr.factors,
                reasoning    = cr.reasoning + f"; thinking override ({thinking_utilization:.0%} utilization)",
                content_type = cr.content_type,
            )

        # Semantic entropy proxy: context-dependent answer → MEDIUM→LOW
        if cr.zone == "MEDIUM" and self._has_multi_answer(cr._raw_text if hasattr(cr, "_raw_text") else ""):
            pass  # handled inline in chat() where text is available

        # Agreement fusion (MEDIUM only)
        if self.use_agreement and cr.zone == "MEDIUM":
            agreement  = self._agreement_score(cr._raw_text if hasattr(cr, "_raw_text") else "")
            fused_j    = round(0.7 * cr.j_score + 0.3 * agreement, 4)
            fused_zone = (
                "HIGH"   if fused_j >= self._effective_theta_high else
                "MEDIUM" if fused_j >= self._effective_theta_low  else
                "LOW"
            )
            cr = CredenceResult(
                j_score      = fused_j,
                zone         = fused_zone,
                factors      = cr.factors,
                reasoning    = cr.reasoning + f"; agreement={agreement:.2f} fused_j={fused_j:.3f}",
                content_type = cr.content_type,
            )

        return cr

    def reset(self):
        self._history            = []
        self._summary            = None
        self._turn_idx           = 0
        self._compression_count  = 0
        self._prev_zone          = "MEDIUM"
        self._prev_j             = 0.50
        self._turn1_goal         = ""
        self._j_history          = []
        self._drift_state        = False
        self._last_faithfulness_block    = False
        self._j_buffer                   = []
        self._history_j_scores           = []
        self._compression_shadow         = None
        self._compression_shadow_j       = None
        self._compression_shadow_summary = None
        self._pending_alignment_caveat   = None
        self._current_user_message       = ""
        self.stats                       = SessionStats()

    @property
    def decision_log(self) -> list[dict]:
        return self.stats.decision_log

    @property
    def _effective_theta_high(self) -> float:
        """P75 of recent J-scores, floored at _ADAPTIVE_THETA_HIGH_FLOOR.
        Compresses only the most confident quarter of this session's responses."""
        if len(self._j_buffer) < _ADAPTIVE_MIN_SAMPLES:
            return self.proxy.theta_high
        buf = sorted(self._j_buffer)
        p75 = buf[int(0.75 * len(buf))]
        return max(_ADAPTIVE_THETA_HIGH_FLOOR, p75)

    @property
    def _effective_theta_low(self) -> float:
        """P25 of recent J-scores, capped at _ADAPTIVE_THETA_LOW_CEIL.
        Preserves only the least confident quarter of this session's responses."""
        if len(self._j_buffer) < _ADAPTIVE_MIN_SAMPLES:
            return self.proxy.theta_low
        buf = sorted(self._j_buffer)
        p25 = buf[int(0.25 * len(buf))]
        eff_high = self._effective_theta_high
        return min(p25, eff_high - 0.10, _ADAPTIVE_THETA_LOW_CEIL)

    # ------------------------------------------------------------------
    # Credence memory decisions
    # ------------------------------------------------------------------

    def _apply_credence(self, cr: CredenceResult) -> tuple[str, int]:
        """Returns (decision, tokens_saved)."""
        n_turns = len(self._history)

        # Drift state: sustained uncertainty (3+ consecutive LOW turns) → lock PRESERVE
        if self._drift_state:
            return "PRESERVE", 0

        # Regime detection: only activate compression/trim when session shows
        # evidence of cross-turn dependencies or mixed J-variance.
        # Independent QA sessions (all HIGH J, no cross-references) should not
        # incur Haiku costs — PRESERVE is correct there anyway.
        if not self._should_enable_credence():
            return "PRESERVE", 0

        # Compute effective zone using adaptive thresholds.
        # Guard-rail overrides (semantic entropy proxy, dual-signal thinking) are
        # detected by comparing cr.zone against what static thresholds would give.
        # When a guard rail has already downgraded the zone, respect it.
        static_zone = (
            "HIGH"   if cr.j_score >= self.proxy.theta_high else
            "MEDIUM" if cr.j_score >= self.proxy.theta_low  else
            "LOW"
        )
        if cr.zone != static_zone:
            eff_zone = cr.zone          # guard-rail override — keep as-is
        else:
            eff_zone = (
                "HIGH"   if cr.j_score >= self._effective_theta_high else
                "MEDIUM" if cr.j_score >= self._effective_theta_low  else
                "LOW"
            )

        if eff_zone == "HIGH" and n_turns > self.COMPRESS_AFTER * 2:
            if self._compression_count < self.MAX_COMPRESSIONS:
                saved = self._compress()
                if saved > 0:
                    self._compression_count += 1
                    return "COMPRESS", saved

        if eff_zone == "MEDIUM" and n_turns > self.TRIM_WINDOW * 2:
            saved = self._trim()
            return "TRIM", saved

        return "PRESERVE", 0

    def _compress(self) -> int:
        """
        Selectively compress the old segment using per-turn J-scores.

        HIGH-J turns are resolved context — safe to summarize via Haiku.
        LOW/MEDIUM-J turns contain uncertainty or important context — kept verbatim.

        History after compression:
            sink (attention sinks) + verbatim LOW/MEDIUM turns + recent

        The Haiku summary of HIGH-J turns is injected via _build_messages as
        <context_summary> XML prefix, not as a fake turn in _history.

        Shadow: pre-compression state is saved during the Haiku call.
        Restored immediately if the faithfulness check or ROI gate rejects the summary.
        """
        sink_msgs = self.ATTENTION_SINK * 2
        keep_n    = self.COMPRESS_AFTER * 2

        if len(self._history) <= sink_msgs + keep_n:
            return 0

        sink   = self._history[:sink_msgs]
        old    = self._history[sink_msgs:-keep_n]
        recent = self._history[-keep_n:]
        old_j  = self._history_j_scores[sink_msgs:-keep_n]

        if not old:
            return 0

        # Split old segment into HIGH-J turns (compress) and LOW/MEDIUM-J (keep verbatim).
        # Process in turn-pairs (user message + assistant message).
        high_j_msgs:      list[dict] = []
        preserved_msgs:   list[dict] = []
        preserved_j:      list[Optional[float]] = []

        for i in range(0, len(old) - 1, 2):
            user_msg    = old[i]
            asst_msg    = old[i + 1]
            j           = old_j[i + 1]   # J-score lives on the assistant message
            user_j_flag = old_j[i]        # 0.0 sentinel = user expressed uncertainty this turn
            # Protect if: user expressed uncertainty (sentinel=0.0) OR assistant J is not HIGH
            if (user_j_flag is not None  # user-side sentinel set → always preserve
                    or j is None
                    or j < self._effective_theta_high):
                preserved_msgs.extend([user_msg, asst_msg])
                preserved_j.extend([old_j[i], old_j[i + 1]])
            else:
                high_j_msgs.extend([user_msg, asst_msg])

        # Nothing to compress — all old turns are LOW/MEDIUM-J
        if not high_j_msgs:
            return 0

        tokens_before = sum(len(m["content"]) // 4 for m in high_j_msgs)
        if tokens_before < _MIN_COMPRESS_TOKENS:
            return 0

        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in high_j_msgs
        )

        # Claim-first compression: operate at claim granularity, not turn granularity.
        #
        # When a registry is active, extract uncertain claims from the high-J segment
        # BEFORE compressing. Extracted claims are registered in the Truth Buffer and
        # injected verbatim into the system prompt on every subsequent turn — so the
        # specific uncertain facts survive even after the turns that contained them are
        # compressed or dropped.
        #
        # This breaks the binary protection that previously preserved ENTIRE turns
        # whenever any uncertainty marker was found. Under the new design:
        #   - The claim is preserved (in the registry)
        #   - The turn can be compressed (saves tokens)
        #   - The model always sees the claim via Truth Buffer (epistemic safety)
        #
        # Without registry: fall back to the binary guard (preserve entire turn).
        _claims_extracted = False
        if self._registry is not None and self._session_id is not None and self.use_claim_extraction:
            self._registry.extract_and_register_claims(
                conv_text, self._session_id, self._turn_idx,
                self.client, self.compression_model,
            )
            _claims_extracted = True
        elif (
            self._has_uncertainty_in_user_turns(high_j_msgs)
            or self._has_uncertainty_in_assistant_prose(high_j_msgs)
        ):
            # Scan USER turns AND assistant prose for uncertainty markers.
            # Assistant code blocks are excluded (they contain routine TODO/might
            # comments); assistant natural-language hedging is included because
            # Haiku would silently drop "I believe X — unconfirmed" from a summary.
            self._last_faithfulness_block = True
            return 0

        # Save compression shadow BEFORE making any changes so we can restore
        # if the Haiku summary is unfaithful or the ROI gate rejects it.
        self._compression_shadow         = self._history[:]
        self._compression_shadow_j       = self._history_j_scores[:]
        self._compression_shadow_summary = self._summary

        goal_line = (
            f"The user's original goal: {self._turn1_goal}\n\n"
            if self._turn1_goal else ""
        )
        # Uncertainty-weighted compression: explicitly enumerate registered unverified
        # constraints so Haiku must preserve them verbatim, not just "try to preserve
        # qualifiers" generically. Inspired by LLMLingua-2's importance-scored
        # preservation — we apply the same principle to epistemic values specifically.
        _preserve_block = ""
        if self._registry is not None and self._session_id is not None:
            try:
                _uncertain = self._registry.list_uncertain(self._session_id)
                if _uncertain:
                    _lines = [
                        f"  - [{c.get('zone','LOW')}] {c.get('constraint_text','')}"
                        for c in _uncertain[:8]
                    ]
                    _preserve_block = (
                        "\nCRITICAL — the following constraints are UNVERIFIED. "
                        "You MUST quote each one verbatim with its uncertainty qualifier:\n"
                        + "\n".join(_lines) + "\n"
                    )
            except Exception:
                pass
        summary_resp = self.client.messages.create(
            model=self.compression_model,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation segment in 2-3 concise sentences. "
                    "Preserve all key facts, decisions, and context needed "
                    "to continue the conversation toward the user's goal. "
                    "IMPORTANT: if anything was stated as uncertain, approximate, "
                    "or needing verification, explicitly preserve that uncertainty "
                    "flag in your summary (e.g. 'tentative', 'unverified', 'approx.')."
                    + _preserve_block + "\n\n"
                    + goal_line
                    + "Conversation segment to summarize:\n\n"
                    + conv_text
                ),
            }],
            max_tokens=200,
        )
        summary = summary_resp.content[0].text.strip()

        comp_in  = summary_resp.usage.input_tokens
        comp_out = summary_resp.usage.output_tokens
        self.stats.total_tokens_in  += comp_in
        self.stats.total_tokens_out += comp_out
        self.stats.total_cost_usd   += _cost_haiku(comp_in, comp_out)

        # Immediate faithfulness check: did Haiku preserve enough key content?
        # Catches over-compression where the summary is so vague that key terms
        # from the original segment are lost. More reliable than the prior
        # J-swing heuristic (which fired on naturally hard follow-up questions
        # unrelated to compression quality).
        if not self._summary_faithful(conv_text, summary, claims_extracted=_claims_extracted):
            self._restore_from_shadow()
            return 0

        # Rebuild history: sink + verbatim preserved turns + recent
        # HIGH-J turns are removed; their summary is stored in self._summary
        # and injected as <context_summary> prefix by _build_messages.
        self._summary = summary
        self._history = sink + preserved_msgs + recent
        self._history_j_scores = (
            self._history_j_scores[:sink_msgs]
            + preserved_j
            + self._history_j_scores[-keep_n:]
        )

        tokens_after = len(summary) // 4
        gross_saved  = max(0, tokens_before - tokens_after)
        net_saved    = max(0, gross_saved - comp_out)

        # ROI gate: if net savings don't justify compression call overhead, revert.
        if net_saved < _MIN_COMPRESS_ROI:
            self._restore_from_shadow()
            return 0

        # Compression committed — discard shadow.
        self._clear_shadow()
        return net_saved

    def _trim(self) -> int:
        """
        J-selective trim: keep attention sink + LOW/MEDIUM-J turns + recent window.

        Naive trim (slice to last N messages) silently drops LOW-J uncertain constraints
        that happen to be older than the window. This defeats the whole point of Credence —
        the model forgets the uncertain fact that was explicitly flagged for preservation.

        This version mirrors selective compression: only HIGH-J turns (already resolved,
        safe to lose) are dropped. LOW/MEDIUM-J turns are always kept verbatim regardless
        of age. This makes TRIM safe to apply even when the old segment contains uncertain
        context.
        """
        keep_n    = self.TRIM_WINDOW * 2
        sink_msgs = self.ATTENTION_SINK * 2

        if len(self._history) <= keep_n:
            return 0

        sink      = self._history[:sink_msgs]
        sink_j    = self._history_j_scores[:sink_msgs]
        old       = self._history[sink_msgs:-keep_n]
        old_j     = self._history_j_scores[sink_msgs:-keep_n]
        recent    = self._history[-keep_n:]
        recent_j  = self._history_j_scores[-keep_n:]

        if not old:
            return 0

        preserved_msgs: list[dict]             = []
        preserved_j:    list[Optional[float]]  = []
        dropped_tokens  = 0
        dropped_texts:  list[str]              = []   # batch claim extraction

        for i in range(0, len(old) - 1, 2):
            user_msg    = old[i]
            asst_msg    = old[i + 1]
            j           = old_j[i + 1]
            user_j_flag = old_j[i]        # 0.0 sentinel = user expressed uncertainty this turn
            # Protect if: user expressed uncertainty (sentinel=0.0) OR assistant J is not HIGH
            if (user_j_flag is not None  # user-side sentinel set → always preserve
                    or j is None
                    or j < self._effective_theta_high):
                preserved_msgs.extend([user_msg, asst_msg])
                preserved_j.extend([old_j[i], old_j[i + 1]])
            else:
                dropped_tokens += (len(user_msg["content"]) + len(asst_msg["content"])) // 4
                # Collect turn text for batch claim extraction before discarding.
                dropped_texts.append(
                    f"User: {user_msg['content']}\nAssistant: {asst_msg['content']}"
                )

        if dropped_tokens == 0:
            return 0

        # Pre-drop claim extraction: extract uncertain claims from all HIGH-J turns
        # that are about to leave the history. Extracted claims survive via Truth Buffer.
        # Batched into a single Haiku call to avoid N calls for N dropped turns.
        if (dropped_texts
                and self._registry is not None
                and self._session_id is not None
                and self.use_claim_extraction):
            batch_text = "\n\n---\n".join(dropped_texts)
            self._registry.extract_and_register_claims(
                batch_text, self._session_id, self._turn_idx,
                self.client, self.compression_model,
            )

        self._history        = sink + preserved_msgs + recent
        self._history_j_scores = sink_j + preserved_j + recent_j
        return dropped_tokens

    def _build_messages(self) -> list[dict]:
        """
        Inject compression summary as an XML-tagged prefix on the first message.
        A fake user/assistant exchange would pollute role alternation and confuse
        the model about who said what. A single tagged prefix is the correct pattern
        per Anthropic's context injection guidance.
        """
        if self._summary and self._history:
            msgs = list(self._history)
            first = msgs[0]
            msgs[0] = {
                "role":    first["role"],
                "content": (
                    f"<context_summary>\n{self._summary}\n</context_summary>\n\n"
                    + first["content"]
                ),
            }
            return msgs
        return list(self._history)

    # ------------------------------------------------------------------
    # Consistency Enforcer
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_tokens(tokens: set[str]) -> set[str]:
        """
        Expand a set of tokens through _CE_DOMAIN_SYNONYMS.

        Retained as the documented home of the synonym map that
        `_CE_DOMAIN_SYNONYMS` is and that `tests/unit/test_matching.py` pins
        `credence.matching` against. Matching itself goes through
        `credence.matching.expand`; this is no longer on the decision path.
        """
        expanded = set(tokens)
        for t in tokens:
            if t in _CE_DOMAIN_SYNONYMS:
                expanded |= _CE_DOMAIN_SYNONYMS[t]
        return expanded

    def _direct_constraint_matches(
        self, user_message: str, constraints: list[dict]
    ) -> list[dict]:
        """
        Find constraints whose content directly overlaps with the user query.

        Direct matches escalate Truth Buffer injection to Consistency
        Enforcement — they are what makes the model express uncertainty about a
        value it would otherwise assert as fact.

        The decision comes from `credence.matching.evaluate`, the same scorer
        the `PreToolUse` gate uses. This method previously carried its own
        matcher, and it had the same blind spot as the gate's: it tokenised with
        `text.lower().split()`, so `RATE_LIMIT = 100` yielded the single term
        `rate_limit`, never matched the prose `rate` / `limit`, and the enforcer
        stayed silent. It also discarded every token of two characters or
        fewer, which made a value like `50` invisible. Both matter most in the
        case this project is for: an agent writing code.

        `_overlap` and `_literal_overlap` are preserved because the enforcement
        message reads them, and both are still computed from the canonical
        tokenizer rather than a private one.

        DISPUTED constraints always escalate regardless of overlap — a fact
        verified and then contradicted is the highest-risk state, and the user
        must be told whether or not their query mentions it.
        """
        # Query-side terms are computed once, not per constraint. The
        # explanation is built only for constraints that actually matched,
        # which is the rare case — doing it per constraint roughly doubled the
        # cost of a gate call on a session with many open constraints.
        raw_query = _match_tokenize(user_message)
        expanded_query = _match_expand(raw_query)
        matches = []

        for c in constraints:
            if c.get("validation_status") == "disputed":
                matches.append({**c, "_overlap": ["DISPUTED"],
                                 "_literal_overlap": []})
                continue

            # expand_synonyms=True: this path warns, it does not block. Its
            # contract (pinned by tests/unit/test_gate.py) is recall-first —
            # "How fast can we call the endpoint?" must fire on a rate-limit
            # constraint. Blocking keeps the literal-only rule; see the module
            # docstring in credence/matching.py for why the two differ.
            verdict = _match_evaluate(
                user_message, c["content"], expand_synonyms=True
            )
            if not verdict["block"]:
                continue

            raw_c = _match_tokenize(c["content"])
            overlap = expanded_query & _match_expand(raw_c)
            matches.append({
                **c,
                "_overlap":         verdict["shared_values"] + sorted(overlap),
                "_literal_overlap": sorted(raw_query & raw_c),
            })
        return matches

    def _build_enforcement_system_prompt(self, user_message: str) -> tuple[str, bool]:
        """
        Build the per-turn system prompt with optional Consistency Enforcement.

        Two modes:
        - Truth Buffer mode (default): informational injection of unverified
          constraints — "here are uncertain things, acknowledge them"
        - Enforcement mode: imperative injection when the query directly asks
          about a registered unverified constraint — "this query asks about X
          which is UNVERIFIED; you MUST express uncertainty"

        Returns (system_prompt, enforcement_active).

        Enforcement mode fires when _direct_constraint_matches returns ≥1 hit.
        The upgrade from informational to imperative is the key mechanism that
        prevents the baseline 50% hallucination observed in E6: the Truth Buffer
        alone listed the constraint but the model treated it as confirmed fact.
        Enforcement mode explicitly prohibits that response pattern.
        """
        base_prompt = self._augment_with_truth_buffer()

        if self._registry is None or self._session_id is None:
            return base_prompt, False

        uncertain = self._registry.get_relevant_claims(
            user_message, self._session_id, max_claims=15
        )
        if not uncertain:
            return base_prompt, False

        direct_matches = self._direct_constraint_matches(user_message, uncertain)
        if not direct_matches:
            return base_prompt, False

        constraint_lines = "\n".join(
            f"• {c['content'][:120]}  [stored confidence: {c.get('zone', 'UNKNOWN')}]"
            for c in direct_matches[:3]
        )
        enforcement_block = (
            "\n\nCONSISTENCY ENFORCEMENT — ACTIVE:\n"
            "The following constraint(s) are REGISTERED AS UNVERIFIED in the epistemic ledger "
            "and this query is directly asking about them:\n"
            f"{constraint_lines}\n\n"
            "YOU MUST in your response:\n"
            "1. State the value with explicit uncertainty markers "
            "(e.g., 'approximately', 'tentative', 'not yet confirmed', 'as stated earlier — unverified')\n"
            "2. Remind the user to verify this value before relying on it in production\n"
            "3. If the response includes code that uses this value, add an inline comment "
            "flagging it as unverified (e.g., `# UNVERIFIED — confirm before production`)\n\n"
            "Responding without acknowledging the uncertain status of the above constraint(s) "
            "is an epistemic error. The user explicitly noted uncertainty about this value — "
            "your response must reflect that."
        )
        return base_prompt + enforcement_block, True

    def _scan_output_for_constraints(
        self, response_text: str
    ) -> tuple[str, list[dict]]:
        """
        Generation-Time Constraint Scanner (GTS).

        Two-pass scan:

        Pass 1 — Code blocks: scan assignment lines for numeric literals that
        appear in registered unverified constraints and annotate inline:
            RATE_LIMIT = 50  # CREDENCE: unverified — I think the rate limit…

        Pass 2 — Prose sentences: scan non-code paragraphs for sentences that
        contain a registered unverified value and append a bracketed footnote:
            "Set the timeout to 3600 seconds."
            → "Set the timeout to 3600 seconds.  ⚠ CREDENCE[unverified]: auth
               token expiry might be 3600 seconds…"

        This closes the generation gap: the Consistency Enforcer guards the
        case where the user ASKS about a constraint (pre-generation); the GTS
        guards the case where the model SILENTLY USES an uncertain value in
        code or prose (post-generation).

        Returns (annotated_response, scan_hits).
        scan_hits: list of {value, constraint_id, constraint_text, line, source}.
        """
        if self._registry is None or self._session_id is None:
            return response_text, []

        constraints = self._registry.list_uncertain(self._session_id) or []
        if not constraints:
            return response_text, []

        current_turn = self._turn_idx

        # Build lookup: value string → list of constraint dicts (with eff_conf).
        # Covers both numeric literals (50, 3600, 0.95) and string literals
        # ("/api/v2", "RS256", "read:users") found inside constraint content.
        # Multiple constraints can share the same value. At annotation time we
        # pick by content-word overlap; break ties by lowest effective confidence.
        # Only track numeric values with ≥2 digits to avoid false-positives on
        # common single-digit literals (0, 1, 2, 3 …).
        value_map: dict[str, list[dict]] = {}
        # str_value_map: maps lowercased string fragment → list of constraints.
        # Separate from value_map to use different matching logic in Pass 1/2.
        str_value_map: dict[str, list[dict]] = {}
        for c in constraints:
            eff_conf = self._registry.get_effective_confidence(
                c["constraint_id"], current_turn
            )
            c = {**c, "eff_conf": eff_conf}
            ctext = c.get("content", "")
            # Numeric values
            for num in _GTS_NUM_PATTERN.findall(ctext):
                if len(num.replace(".", "")) >= 2:
                    value_map.setdefault(num, []).append(c)
            # String literal values: quoted fragments in constraint content
            for frag in _GTS_STR_EXTRACT.findall(ctext):
                key = frag.lower()
                str_value_map.setdefault(key, []).append(c)
            # Unquoted uppercase identifiers: RS256, AES-256-GCM, etc.
            # Skip common stop-words that happen to be all-caps
            _IDENT_SKIP = {"AND", "OR", "NOT", "FOR", "THE", "USE", "API", "URL",
                           "JWT", "SQL", "CSS", "HTTP", "HTTPS", "JSON", "XML"}
            for ident in _GTS_IDENT_EXTRACT.findall(ctext):
                if ident not in _IDENT_SKIP:
                    str_value_map.setdefault(ident.lower(), []).append(c)
            # Lowercase hyphenated region/env tokens: us-east-1, eu-west-2, read-write, etc.
            _HYPHEN_SKIP = {"not-yet", "to-be", "in-the", "of-the", "out-of", "based-on"}
            for token in _GTS_HYPHEN_EXTRACT.findall(ctext):
                if token not in _HYPHEN_SKIP and len(token) >= 5:
                    str_value_map.setdefault(token.lower(), []).append(c)

        def _best_constraint(num: str, line_context: str) -> dict:
            """
            When multiple constraints share the same numeric value, pick the one
            whose content words have the highest overlap with the code/prose line.
            Falls back to the lowest-confidence constraint (highest epistemic risk).
            """
            candidates = value_map[num]
            if len(candidates) == 1:
                return candidates[0]
            # Split on ALL non-alpha-digit chars (including underscores) so that
            # RATE_LIMIT → {'rate', 'limit'} and matches constraint words 'rate', 'limit'.
            line_words = set(re.sub(r"[^a-z0-9]", " ", line_context.lower()).split())
            best, best_score = candidates[0], -1.0
            for cand in candidates:
                cwords = set(re.sub(r"[^a-z0-9_]", " ",
                                    cand.get("content", "").lower()).split())
                overlap = len(line_words & cwords) / max(len(line_words | cwords), 1)
                # Break ties in favour of lower effective confidence (higher risk)
                score = overlap * 10 - cand.get("eff_conf", 0.30)
                if score > best_score:
                    best, best_score = cand, score
            return best

        if not value_map and not str_value_map:
            return response_text, []

        scan_hits: list[dict] = []

        def _policy_annotation(c: dict, snippet: str, *, for_code: bool) -> str:
            """
            Confidence Policy Layer: annotation severity driven by validation_status + eff_conf.

            DISPUTED   (validation_status='disputed'):
                code →  # ⚠⚠ CREDENCE[DISPUTED]: contradicted by newer info (X) — ...
                prose → ⚠⚠ CREDENCE[DISPUTED]: contradicted by newer info (X) — ...
            HIGH RISK  (eff_conf < _GTS_WARN_THRESHOLD):
                code →  # ⚠⚠ CREDENCE[HIGH RISK, conf=X.XX]: ...
                prose → ⚠⚠ CREDENCE[HIGH RISK, conf=X.XX]: ...
            UNVERIFIED (_GTS_WARN_THRESHOLD ≤ eff_conf < _GTS_QUALIFY_THRESHOLD):
                code →  # ⚠ CREDENCE[unverified, conf=X.XX]: ...
                prose → ⚠ CREDENCE[unverified, conf=X.XX]: ...
            CHECK      (eff_conf ≥ _GTS_QUALIFY_THRESHOLD, still unverified):
                code →  # CREDENCE[check, conf=X.XX]: ...
                prose → CREDENCE[check, conf=X.XX]: ...
            """
            # DISPUTED is highest severity — overrides all other tiers
            if c.get("validation_status") == "disputed":
                new_val = c.get("contradicted_by", "newer info")
                tag = f"⚠⚠ CREDENCE[DISPUTED]: contradicted by newer info ({new_val}) — {snippet}"
                return f"  # {tag}" if for_code else f"  {tag}"
            eff = c.get("eff_conf", 0.30)
            conf_tag = f"conf={eff:.2f}"
            if eff < _GTS_WARN_THRESHOLD:
                tag = f"⚠⚠ CREDENCE[HIGH RISK, {conf_tag}]: {snippet}"
            elif eff < _GTS_QUALIFY_THRESHOLD:
                tag = f"⚠ CREDENCE[unverified, {conf_tag}]: {snippet}"
            else:
                tag = f"CREDENCE[check, {conf_tag}]: {snippet}"
            return f"  # {tag}" if for_code else f"  {tag}"

        # ---- Pass 1: code blocks ----------------------------------------
        def annotate_code_block(code_content: str) -> str:
            lines = code_content.split("\n")
            out: list[str] = []
            for line in lines:
                stripped = line.strip()
                if (
                    not stripped
                    or "CREDENCE:" in line
                    or stripped.startswith(_GTS_SKIP_PREFIXES)
                ):
                    out.append(line)
                    continue

                annotated = line
                matched = False
                # Pass 1a: numeric literal assignments  (e.g. RATE_LIMIT = 50)
                for num in value_map:
                    if re.search(r"=\s*" + re.escape(num) + r"\b", line):
                        c       = _best_constraint(num, line)
                        ctext   = c.get("content", "")
                        cid     = c.get("constraint_id", "?")
                        snippet = ctext[:55].replace("\n", " ")
                        if len(ctext) > 55:
                            snippet += "…"
                        suffix  = _policy_annotation(c, snippet, for_code=True)
                        annotated = line.rstrip() + suffix
                        scan_hits.append({
                            "value":           num,
                            "constraint_id":   cid,
                            "constraint_text": ctext[:80],
                            "eff_conf":        round(c.get("eff_conf", 0.30), 3),
                            "line":            stripped,
                            "source":          "code",
                        })
                        matched = True
                        break
                # Pass 1b: string literal assignments  (e.g. SCOPE = "read:users")
                if not matched and str_value_map:
                    for m in _GTS_STR_ASSIGN.finditer(line):
                        frag_key = m.group(1).lower()
                        if frag_key in str_value_map:
                            c_list  = str_value_map[frag_key]
                            c       = c_list[0]
                            ctext   = c.get("content", "")
                            cid     = c.get("constraint_id", "?")
                            snippet = ctext[:55].replace("\n", " ")
                            if len(ctext) > 55:
                                snippet += "…"
                            suffix  = _policy_annotation(c, snippet, for_code=True)
                            annotated = line.rstrip() + suffix
                            scan_hits.append({
                                "value":           m.group(1),
                                "constraint_id":   cid,
                                "constraint_text": ctext[:80],
                                "eff_conf":        round(c.get("eff_conf", 0.30), 3),
                                "line":            stripped,
                                "source":          "code_string",
                            })
                            break
                out.append(annotated)
            return "\n".join(out)

        text_after_code = _GTS_CODE_BLOCK.sub(
            lambda m: m.group(1) + annotate_code_block(m.group(2)) + m.group(3),
            response_text,
        )

        # ---- Pass 2: prose sentences outside code blocks -----------------
        def annotate_prose(prose: str) -> str:
            if not prose.strip():
                return prose
            sentences = _GTS_SENTENCE_SPLIT.split(prose)
            out: list[str] = []
            for sent in sentences:
                if "CREDENCE:" in sent:
                    out.append(sent)
                    continue
                annotated = sent
                matched = False
                # Pass 2a: numeric value in prose sentence
                for num in value_map:
                    if re.search(r'\b' + re.escape(num) + r'\b', sent):
                        c       = _best_constraint(num, sent)
                        ctext   = c.get("content", "")
                        cid     = c.get("constraint_id", "?")
                        snippet = ctext[:60].replace("\n", " ")
                        if len(ctext) > 60:
                            snippet += "…"
                        suffix  = _policy_annotation(c, snippet, for_code=False)
                        annotated = sent.rstrip() + suffix
                        scan_hits.append({
                            "value":           num,
                            "constraint_id":   cid,
                            "constraint_text": ctext[:80],
                            "eff_conf":        round(c.get("eff_conf", 0.30), 3),
                            "line":            sent[:80],
                            "source":          "prose",
                        })
                        matched = True
                        break
                # Pass 2b: string fragment in prose sentence
                if not matched and str_value_map:
                    sent_lower = sent.lower()
                    for frag_key, c_list in str_value_map.items():
                        if len(frag_key) >= 4 and frag_key in sent_lower:
                            c       = c_list[0]
                            ctext   = c.get("content", "")
                            cid     = c.get("constraint_id", "?")
                            snippet = ctext[:60].replace("\n", " ")
                            if len(ctext) > 60:
                                snippet += "…"
                            suffix  = _policy_annotation(c, snippet, for_code=False)
                            annotated = sent.rstrip() + suffix
                            scan_hits.append({
                                "value":           frag_key,
                                "constraint_id":   cid,
                                "constraint_text": ctext[:80],
                                "eff_conf":        round(c.get("eff_conf", 0.30), 3),
                                "line":            sent[:80],
                                "source":          "prose_string",
                            })
                            break
                out.append(annotated)
            return "  ".join(out) if len(out) > 1 else (out[0] if out else prose)

        # Walk through code-fence boundaries in the code-annotated text
        parts: list[str] = []
        last = 0
        for m in _GTS_CODE_BLOCK.finditer(text_after_code):
            parts.append(annotate_prose(text_after_code[last:m.start()]))
            parts.append(m.group(0))  # code block already annotated — keep
            last = m.end()
        parts.append(annotate_prose(text_after_code[last:]))

        annotated_response = "".join(parts)
        return annotated_response, scan_hits

    def _augment_with_truth_buffer(self) -> str:
        """
        Build a per-turn system prompt that prepends unverified constraints from the
        registry as an EPISTEMIC CONTEXT block.

        This is the Truth Buffer: Claude sees the live list of what it doesn't know
        at the start of every turn, making epistemic uncertainty active rather than
        passive. Without this, unverified constraints survive in conversation history
        but are not foregrounded — they can be overlooked when the model synthesises
        a final answer.

        Only fires when a registry and session_id are set. Zero-cost when registry is
        empty or all constraints are verified.
        """
        if self._registry is None or self._session_id is None:
            return self.system_prompt
        # Query-aware injection: only inject constraints RELEVANT to this turn's context.
        # Reduces system prompt bloat vs injecting all 10 at once.
        # Falls back to list_uncertain if no query context available.
        _query_ctx = getattr(self, '_current_user_message', '')
        current_turn = self._turn_idx
        if _query_ctx:
            # No hard cap — show ALL unverified constraints, relevance-ranked.
            # Directly relevant ones first; remaining sorted by staleness.
            uncertain = self._registry.get_relevant_claims(
                _query_ctx, self._session_id, max_claims=9999
            )
            # Augment with effective_confidence if not already present
            for c in uncertain:
                if "effective_confidence" not in c:
                    c["effective_confidence"] = self._registry.get_effective_confidence(
                        c["constraint_id"], current_turn
                    )
        else:
            uncertain = self._registry.get_effective_uncertain(
                self._session_id, current_turn=current_turn, max_claims=9999
            )
        if not uncertain:
            return self.system_prompt

        def _constraint_label(c: dict) -> str:
            eff = c.get("effective_confidence")
            orig = c.get("j_score", 0.5)
            zone = c.get("zone", "?")
            content = c["content"][:120]
            # DISPUTED: highest-priority label — previously verified, now contradicted
            if c.get("validation_status") == "disputed":
                new_val = c.get("contradicted_by", "newer info")
                return f"• [⚠⚠ DISPUTED — contradicted by newer info ({new_val})] {content}"
            if eff is not None:
                decay_frac = (orig - eff) / orig if orig > 0 else 0
                if decay_frac >= 0.30:
                    # Noticeably decayed — flag as stale
                    return f"• [{zone}, STALE conf={eff:.2f}] {content}"
                return f"• [{zone}, conf={eff:.2f}] {content}"
            return f"• [{zone}] {content}"

        # use_manifest=True: replace natural language Truth Buffer with structured
        # EpistemicManifest XML (the research-direction path). The manifest is
        # explicitly labelled NON-COMPRESSIBLE and encodes confidence as a numeric
        # attribute rather than as hedging words — testing whether structured
        # representation survives compression better than natural language.
        if self.use_manifest:
            from .epistemic_manifest import EpistemicManifest
            manifest_xml = EpistemicManifest.from_registry(
                self._registry, self._session_id, current_turn=current_turn
            )
            if manifest_xml:
                base = f"{self.system_prompt}\n\n{manifest_xml}"
                if self._pending_alignment_caveat:
                    base = f"{base}\n\nEPISTEMIC GOVERNOR ALERT:\n{self._pending_alignment_caveat}"
                return base

        lines = "\n".join(_constraint_label(c) for c in uncertain)
        block = (
            "EPISTEMIC CONTEXT — UNVERIFIED CONSTRAINTS (DISPUTED first, then by staleness):\n"
            f"{lines}\n"
            "When discussing topics related to these constraints, always acknowledge "
            "their uncertain status. Do not treat them as confirmed facts."
        )
        base = f"{self.system_prompt}\n\n{block}"
        if self._pending_alignment_caveat:
            base = f"{base}\n\nEPISTEMIC GOVERNOR ALERT:\n{self._pending_alignment_caveat}"
        return base

    def _augment_system_with_caveat(self) -> str:
        """
        Return system prompt with only the Governor caveat injected (no registry).
        Used when registry is None but a pending alignment caveat exists.
        """
        if self._pending_alignment_caveat:
            return (
                f"{self.system_prompt}\n\n"
                f"EPISTEMIC GOVERNOR ALERT:\n{self._pending_alignment_caveat}"
            )
        return self.system_prompt

    def _scout_classify(self, user_message: str) -> list[dict]:
        """
        Scout Classifier: lightweight Haiku call that extracts uncertain constraints
        from the user message as structured JSON, then auto-registers them.

        Replaces manual `credence_register` for the common case where the user states
        an uncertain fact inline (e.g. "I think the rate limit is ~50 req/min").

        Returns list of extracted constraint dicts (entity, value, confidence_level,
        raw_quote). Skips the Haiku call entirely when J-proxy rates the message HIGH
        (confident message unlikely to contain uncertain constraints).
        """
        if self._registry is None or self._session_id is None:
            return []

        # Fast pre-filter: if J-proxy says HIGH on the user message, skip Scout.
        # User messages with J >= 0.80 are assertive statements, not uncertainty.
        cr_msg = self.proxy.compute(user_message)
        if cr_msg.j_score >= 0.80:
            return []

        try:
            resp = self.client.messages.create(
                model=self.compression_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract specific factual constraints stated with uncertainty in this message. "
                        'Return a JSON array ONLY: [{"entity": "what", "value": "stated value", '
                        '"confidence_level": "low|medium|high", "raw_quote": "exact quote from text"}] '
                        "or [] if none found. Only include claims with confidence_level low or medium.\n\n"
                        f"Message: {user_message[:800]}"
                    ),
                }],
                max_tokens=250,
            )
            raw = resp.content[0].text.strip()
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start < 0 or end <= start:
                return []
            items = json.loads(raw[start:end])
            for item in items:
                if not isinstance(item, dict):
                    continue
                confidence = item.get("confidence_level", "high")
                if confidence in ("low", "medium"):
                    quote = item.get("raw_quote") or item.get("value") or ""
                    j_reg = 0.30 if confidence == "low" else 0.50
                    zone  = "LOW" if confidence == "low" else "MEDIUM"
                    cid   = self._registry.register(
                        content    = quote[:500],
                        session_id = self._session_id,
                        j_score    = j_reg,
                        zone       = zone,
                        source     = "scout",
                    )
                    self._registry.log_event(
                        cid, "scout",
                        j_score=j_reg, zone=zone,
                        notes=f"entity={item.get('entity','?')} value={item.get('value','?')[:60]}"
                    )
            return items
        except Exception:
            return []

    def _ghost_detect(self, user_message: str) -> list[dict]:
        """
        Opus-powered ghost constraint detector.

        Catches implicit uncertain constraints in user messages — facts stated without
        canonical hedging markers that are nonetheless unverified: vendor claims,
        unconfirmed estimates, second-hand assertions, assumptions presented as fact.

        The faithfulness probe handles explicit markers ("I think", "approximately", etc.).
        This closes the remaining gap: assertions that SOUND certain but ARE unverified.

        Uses Opus (not Haiku) because distinguishing "established fact" from
        "assumed-but-unverified fact" requires genuine reasoning, not pattern matching.

        High-precision design: only registers claims Opus rates ≥0.70 confidence as
        ghost constraints. Missing some is acceptable; false positives degrade trust.

        Returns list of detected constraint dicts. Empty list on any error.
        """
        if self._registry is None or self._session_id is None:
            return []
        if not user_message.strip():
            return []

        try:
            resp = self.client.messages.create(
                model=self.main_model,   # Opus — reasoning depth required here
                messages=[{
                    "role": "user",
                    "content": (
                        "You are an epistemic classifier. Find GHOST CONSTRAINTS in the message below.\n\n"
                        "A ghost constraint is a specific factual claim stated as certain fact, but "
                        "which is implicitly unverified — e.g. a vendor-stated limit accepted as fact, "
                        "an estimate assumed to be confirmed, second-hand information stated without "
                        "qualification, or an unconfirmed assumption presented as established.\n\n"
                        "Rules (follow precisely):\n"
                        "1. ONLY return claims you are HIGHLY CONFIDENT (≥0.70) are implicitly unverified\n"
                        "2. Do NOT flag statements that already use hedging words (I think, maybe, "
                        "approximately, I believe, might, probably, I'm not sure, around, roughly)\n"
                        "3. Do NOT flag established facts (Python syntax, math, general knowledge)\n"
                        "4. Do NOT flag user preferences, opinions, or questions\n"
                        "5. Return [] if nothing clearly qualifies — precision over recall\n\n"
                        "Return JSON array ONLY — no other text:\n"
                        '[{"claim": "exact quote from message", '
                        '"reason": "why likely unverified in ≤15 words", '
                        '"confidence": 0.85}]\n\n'
                        f"Message: {user_message[:1000]}"
                    ),
                }],
                max_tokens=300,
                timeout=8.0,  # hard ceiling: ghost detect never blocks main response > 8s
            )
            raw = resp.content[0].text.strip() if resp.content else ""
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start < 0 or end <= start:
                return []
            items = json.loads(raw[start:end])
            results = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                confidence = float(item.get("confidence", 0.0))
                if confidence < 0.70:
                    continue
                claim = (item.get("claim") or "").strip()
                if not claim:
                    continue
                # Register as LOW-confidence — ghost = no canonical marker = high epistemic risk
                cid = self._registry.register(
                    content    = claim[:500],
                    session_id = self._session_id,
                    j_score    = 0.25,
                    zone       = "LOW",
                    source     = "ghost_detector",
                    turn_idx   = self._turn_idx,
                )
                self._registry.log_event(
                    cid, "ghost_detect",
                    j_score=0.25, zone="LOW",
                    notes=(
                        f"opus_conf={confidence:.2f} "
                        f"reason={item.get('reason', '')[:80]}"
                    ),
                )
                results.append(item)
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Contradiction Detector
    # ------------------------------------------------------------------

    def _detect_contradiction(self, user_message: str) -> list[dict]:
        """
        Opus-powered cross-turn contradiction detector.

        When a user message introduces a numeric or named value that overlaps
        topically with a registered constraint but carries a different value,
        Opus 4.7 reasons about which is more reliable and marks the conflict.

        Examples caught:
          Prior:  "rate limit is around 50 req/min — unconfirmed"
          New:    "actually, we tested it — rate limit is 200 req/min"
          → Opus: contradiction. New (tested) more reliable. Prior disputed.

          Prior:  "token expiry is 3600 seconds — tentative"
          New:    "the vendor confirmed token expiry is 86400 seconds"
          → Opus: contradiction. New (confirmed) supersedes prior. Prior disputed.

        Returns list of contradiction dicts. Empty list on any error or no conflicts.
        """
        if self._registry is None or self._session_id is None:
            return []
        if not user_message.strip():
            return []

        # Extract numeric values from the user message (≥2 digits or named values)
        nums_in_msg = set(re.findall(r'\b\d+(?:[,.]\d+)*(?:\s*(?:ms|s|sec|min|hour|hr|day|week|%|req|gb|mb|kb|k|m|b))?\b', user_message.lower()))
        if not nums_in_msg and len(user_message) < 30:
            return []  # Too short and no numbers — skip to save API cost

        # Find registered constraints that overlap topically with the user message
        try:
            similar = self._registry.check_contradiction(user_message, self._session_id)
        except Exception:
            return []
        if not similar:
            return []

        # Build a concise diff prompt for Opus
        candidates = []
        for c in similar[:4]:  # cap at 4 to keep prompt tight
            candidates.append(
                f'- Prior constraint (id={c["constraint_id"][:8]}): "{c["content"][:120]}"'
            )
        if not candidates:
            return []

        prompt = (
            "You are an epistemic conflict detector. Determine whether the new message "
            "contradicts any of the prior registered constraints listed below.\n\n"
            "A contradiction means: the new message makes a specific factual claim "
            "(a value, a limit, a deadline, a quantity) that directly conflicts with "
            "a value in one of the prior constraints — not just a different topic.\n\n"
            "For each genuine contradiction found, return:\n"
            '{"constraint_id": "first 8 chars", "prior_value": "value from prior", '
            '"new_value": "value from new message", "reasoning": "≤20 words: why this contradicts", '
            '"reliability": "prior|new|unclear"}\n\n'
            "Return a JSON array. Return [] if no genuine contradiction.\n\n"
            "Prior constraints:\n" + "\n".join(candidates) + "\n\n"
            f"New message: {user_message[:600]}"
        )

        try:
            resp = self.client.messages.create(
                model     = self.main_model,  # Opus — reasoning about reliability required
                messages  = [{"role": "user", "content": prompt}],
                max_tokens = 400,
                timeout    = 8.0,
            )
            raw = resp.content[0].text.strip() if resp.content else ""
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start < 0 or end <= start:
                return []
            items = json.loads(raw[start:end])
            if not isinstance(items, list):
                return []

            results = []
            for item in items:
                cid_prefix = item.get("constraint_id", "")
                # Find the full constraint_id from registry
                full_cid = next(
                    (c["constraint_id"] for c in similar
                     if c["constraint_id"].startswith(cid_prefix)),
                    None,
                )
                if not full_cid:
                    continue
                reliability = item.get("reliability", "unclear")
                reasoning   = item.get("reasoning", "")[:120]
                new_val     = str(item.get("new_value", ""))[:80]

                # Mark in registry — prior constraint is now disputed
                try:
                    self._registry.mark_contradiction(full_cid, f"new_value={new_val}")
                    self._registry.log_event(
                        full_cid, "contradict",
                        notes=f"new_val={new_val} reliability={reliability} {reasoning}"
                    )
                except Exception:
                    pass

                results.append({
                    "constraint_id": full_cid,
                    "prior_value":   item.get("prior_value", ""),
                    "new_value":     new_val,
                    "reasoning":     reasoning,
                    "reliability":   reliability,
                })
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Output Alignment Layer (Governor)
    # ------------------------------------------------------------------

    def _align_output(self, response_text: str) -> list[AlignmentWarning]:
        """
        Post-generation output enforcement — the Governor.

        Implements Layer 4: if response_confidence > stored_constraint_confidence,
        fire an AlignmentWarning. The caller appends caveats inline to the current
        response (synchronous) and queues them for the next turn's system prompt.

        Fires when:
          - response zone is HIGH or MEDIUM (not already hedged)
          - response shares ≥ 3 content words with an unverified constraint (HIGH)
            or ≥ 2 content words (MEDIUM response — lower bar, borderline claims matter)
          - the matched constraint is registered as LOW or MEDIUM (unverified)

        The word-overlap check is intentionally conservative: requires content words
        (not stopwords), so incidental topic overlap doesn't trigger false warnings.
        """
        if self._registry is None or self._session_id is None:
            return []

        uncertain = self._registry.list_uncertain(self._session_id)
        if not uncertain:
            return []

        response_cr    = self.proxy.compute(response_text)
        if response_cr.zone == "LOW":
            return []   # response is already hedged — no alignment issue

        response_words = self._extract_content_words(response_text)
        warnings: list[AlignmentWarning] = []

        for constraint in uncertain:
            constraint_words = self._extract_content_words(constraint["content"])
            if not constraint_words:
                continue

            overlap = response_words & constraint_words
            # HIGH response: require 3 overlapping words (confident claim about uncertain topic).
            # MEDIUM response: require 2 overlapping words (borderline claim warrants flagging too).
            min_overlap = 2 if response_cr.zone == "MEDIUM" else 3
            if len(overlap) < min_overlap:
                continue   # too few shared words — response isn't about this constraint

            # Response confidence exceeds stored epistemic record for this constraint.
            if constraint["zone"] in ("LOW", "MEDIUM"):
                caveat = (
                    f"In your previous response you discussed a topic related to: "
                    f"'{constraint['content'][:100]}'. "
                    f"This constraint is registered as UNVERIFIED (zone={constraint['zone']}). "
                    "Acknowledge this uncertainty if you mention it again."
                )
                warnings.append(AlignmentWarning(
                    constraint_id      = constraint["constraint_id"],
                    constraint_content = constraint["content"],
                    ledger_zone        = constraint["zone"],
                    response_zone      = response_cr.zone,
                    overlap_words      = list(overlap)[:6],
                    suggested_caveat   = caveat,
                ))

        return warnings

    # ------------------------------------------------------------------
    # Novelty guard  (content-word Jaccard distance, replaces entity counting)
    # ------------------------------------------------------------------

    # Common English function words — filtered out before Jaccard computation.
    _CONTENT_STOPWORDS = frozenset({
        "the", "and", "for", "that", "this", "with", "have", "from", "they",
        "will", "your", "what", "when", "how", "why", "where", "which", "who",
        "are", "was", "were", "has", "had", "been", "being", "can", "could",
        "should", "would", "may", "might", "must", "does", "did", "not", "more",
        "also", "such", "than", "only", "about", "into", "over", "some", "these",
        "those", "there", "here", "then", "but", "all", "any", "each", "both",
        "just", "even", "often", "well", "back", "still", "like", "used", "need",
        "make", "give", "take", "come", "know", "think", "look", "want", "use",
        "see", "based", "using", "since", "after", "before", "while", "through",
        "their", "them", "very", "most", "same", "next", "last", "good", "new",
    })

    def _extract_content_words(self, text: str) -> set[str]:
        """Extract lowercase content words (≥4 chars, non-stop, non-digit)."""
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        return {
            w for w in words
            if len(w) >= 4
            and not w.isdigit()
            and w not in self._CONTENT_STOPWORDS
        }

    def _has_multi_answer(self, text: str) -> bool:
        """Returns True if response signals multiple valid answers (semantic entropy proxy)."""
        lower = text.lower()
        return any(m in lower for m in _MULTI_ANSWER_MARKERS)

    @staticmethod
    def _has_uncertainty(text: str) -> bool:
        """Returns True if text contains markers that signal user-flagged uncertainty."""
        import re as _re
        # Strip markdown bold/italic markers before substring matching so that
        # "**not** been confirmed" matches "not been confirmed"
        clean = _re.sub(r'\*{1,2}|_{1,2}', '', text)
        lower = clean.lower()
        # Standard uncertainty markers
        if any(m in lower for m in _UNCERTAINTY_MARKERS):
            return True
        # Expanded: code comment uncertainty (# todo, # verify, # untested, etc.)
        if _re.search(r'#\s*(todo|fixme|hack|verify|check|untested|approximate|not sure|might)',
                      lower):
            return True
        # Expanded: numerical hedging — qualifier word followed by a number
        if _re.search(r'\b(around|roughly|approximately|about|~)\s+\d', lower):
            return True
        # Expanded: conditional uncertainty — premise implies uncertain conclusion
        if any(m in lower for m in (
            "if this is correct", "assuming this is right", "if i'm reading",
            "if that's the case", "assuming that's accurate", "provided that's true",
        )):
            return True
        # Expanded: domain hedging — informal "worth checking" patterns
        if any(m in lower for m in (
            "worth checking", "worth verifying", "double-check", "double check",
            "you might want to confirm", "lgtm but", "seems right but",
            "this should work but", "i'd recommend verifying",
        )):
            return True
        return False

    def _has_uncertainty_in_user_turns(self, msgs: list[dict]) -> bool:
        """
        Faithfulness probe scoped to USER messages only.

        Scanning the full conv_text (user + assistant turns concatenated) caused
        false positives: routine code comments in ASSISTANT turns — e.g.
        `# might not handle edge cases` or `# TODO: verify this` — blocked
        compression of HIGH-J turns even when no user-stated uncertainty existed.

        This version scans only role='user' messages, which is where user-flagged
        uncertainty actually lives. Assistant reasoning and code are excluded.
        """
        user_text = " ".join(
            m["content"] for m in msgs if m.get("role") == "user"
        )
        return self._has_uncertainty(user_text)

    def _has_uncertainty_in_assistant_prose(self, msgs: list[dict]) -> bool:
        """
        Faithfulness probe for ASSISTANT prose (non-code) turns.

        Catches cases where the assistant explicitly hedged a stated value —
        e.g. "I believe the rate limit is around 50 — unconfirmed" — that would
        be silently dropped if the turn were compressed by Haiku.

        Code blocks are stripped before scanning to avoid false positives from
        routine inline comments such as `# TODO: verify this` or `# might fail`.
        Only fenced triple-backtick blocks are stripped; inline code spans are
        left in place (they rarely contain hedging prose).
        """
        import re as _re
        _code_block = _re.compile(r"```[^\n]*\n.*?```", _re.DOTALL)
        prose_parts = []
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            prose = _code_block.sub("", m["content"])
            prose_parts.append(prose)
        prose_text = " ".join(prose_parts)
        return self._has_uncertainty(prose_text)

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _agreement_score(self, response_text: str) -> float:
        """
        Ask Haiku to rate the confidence of a response on [0, 1].
        Used as a second signal on MEDIUM-J turns where J-proxy is least reliable.
        Haiku keeps cost to ~$0.0002/call — negligible for the accuracy gain.
        """
        try:
            resp = self.client.messages.create(
                model=self.compression_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rate the epistemic confidence of the following response "
                        "on a scale of 0.0 (very uncertain) to 1.0 (completely certain). "
                        "Consider hedging language, specificity, and whether it contains "
                        "qualifiers like 'I think', 'probably', 'might'. "
                        "Reply with ONLY a decimal number between 0.0 and 1.0.\n\n"
                        f"Response:\n{response_text[:600]}"
                    ),
                }],
                max_tokens=10,
            )
            raw = resp.content[0].text.strip()
            match = re.search(r'\d+\.?\d*', raw)
            score = float(match.group()) if match else 0.5
            return round(min(1.0, max(0.0, score)), 3)
        except Exception:
            return 0.5   # safe default — no change to fusion

    def _j_variance(self) -> float:
        """Variance of J-scores in the current rolling buffer."""
        if len(self._j_buffer) < 3:
            return 0.0
        mean = sum(self._j_buffer) / len(self._j_buffer)
        return sum((j - mean) ** 2 for j in self._j_buffer) / len(self._j_buffer)

    def _has_dependency(self) -> bool:
        """True if recent assistant turns reference earlier turns explicitly."""
        if len(self._history) < 4:
            return False
        _dep_markers = {
            "as we discussed", "as mentioned", "as i said", "earlier",
            "you mentioned", "we established", "going back to",
            "as noted", "referring back", "from before",
            "the constraint", "the requirement", "the error", "the issue",
            "that value", "that limit", "those constraints", "the budget",
        }
        recent_assistant = [
            m["content"].lower()
            for m in self._history[-6:]
            if m["role"] == "assistant"
        ]
        return any(
            marker in text
            for text in recent_assistant
            for marker in _dep_markers
        )

    def _should_enable_credence(self) -> bool:
        """
        Regime detection gate: only allow COMPRESS/TRIM when the session shows
        evidence that context management matters.

        Returns False (PRESERVE) when:
        - Still in warmup (< _REGIME_MIN_TURNS)
        - Session J-variance is low AND no cross-turn references detected
          (uniform-confidence session — independent QA pattern)

        Returns True when:
        - J-variance > _REGIME_J_VARIANCE_FLOOR (mixed-confidence session), OR
        - Cross-turn dependency markers detected in recent turns
        """
        if self._turn_idx <= _REGIME_MIN_TURNS:
            return False
        return self._j_variance() > _REGIME_J_VARIANCE_FLOOR or self._has_dependency()

    # ------------------------------------------------------------------
    # Post-compression degradation + shadow recovery
    # ------------------------------------------------------------------

    def _summary_faithful(
        self,
        original_text: str,
        summary: str,
        claims_extracted: bool = False,
    ) -> bool:
        """
        NLI-based faithfulness check: Haiku judges whether the summary faithfully
        preserves uncertainty qualifiers and does not contradict the original.

        Two-stage check:
          Stage 0: Direct qualifier survival check (no API call) — skipped when
                   claims_extracted=True because qualifiers are already in the registry
                   and will be injected via Truth Buffer, so the summary does not need
                   to carry them verbatim.
          Stage 1: Haiku NLI judge — checks for factual contradiction and over-confidence.
          Stage 2: Jaccard fallback if Haiku unavailable.
        """
        # Stage 0: Direct qualifier survival check — no API call.
        # Skip when claim extraction already handled uncertainty preservation:
        # those qualifiers are now in the registry / Truth Buffer, so they don't
        # need to survive verbatim in the Haiku summary. Still run on non-registry
        # paths to catch the documented "might be 50" → "is 50" failure mode.
        if not claims_extracted:
            orig_lower    = original_text.lower()
            summ_lower    = summary.lower()
            detected_markers = [m for m in _UNCERTAINTY_MARKERS if m in orig_lower]
            if detected_markers:
                survived = sum(1 for m in detected_markers if m in summ_lower)
                if survived == 0 or survived / len(detected_markers) < 0.50:
                    return False   # qualifiers stripped — reject without Haiku call

        # Stage 1: NLI via Haiku
        try:
            nli_resp = self.client.messages.create(
                model=self.compression_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "You are an epistemic faithfulness judge. "
                        "Answer YES if the SUMMARY faithfully preserves the ORIGINAL — "
                        "specifically: (a) no uncertain claim is presented as certain, "
                        "(b) key hedging words like 'I think', 'might', 'approximately', "
                        "'not confirmed', 'unverified' are preserved where they exist, "
                        "(c) the summary does not contradict the original. "
                        "Answer NO if any qualifier was stripped or any claim was made "
                        "more confident than in the original. "
                        "Reply with ONLY the word YES or NO.\n\n"
                        f"ORIGINAL (excerpt):\n{original_text[:800]}\n\n"
                        f"SUMMARY:\n{summary}"
                    ),
                }],
                max_tokens=5,
            )
            verdict = nli_resp.content[0].text.strip().upper()
            return verdict.startswith("Y")
        except Exception:
            pass

        # Stage 2: Jaccard fallback if Haiku unavailable
        orig_words = self._extract_content_words(original_text)
        summ_words = self._extract_content_words(summary)
        if not orig_words:
            return True
        return len(orig_words & summ_words) / len(orig_words) >= _SUMMARY_FAITHFUL_THRESHOLD

    def _restore_from_shadow(self) -> None:
        """Revert to pre-compression state. Called when degradation detected."""
        if self._compression_shadow is not None:
            self._history          = self._compression_shadow
            self._history_j_scores = self._compression_shadow_j
            self._summary          = self._compression_shadow_summary
            self._compression_count = max(0, self._compression_count - 1)
        self._clear_shadow()

    def _clear_shadow(self) -> None:
        """Commit compression — discard shadow, no anomaly detected."""
        self._compression_shadow         = None
        self._compression_shadow_j       = None
        self._compression_shadow_summary = None

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist full epistemic state to JSON. Includes version for migration."""
        import json
        state = {
            "version":           _SESSION_VERSION,
            "history":           self._history,
            "history_j_scores":  self._history_j_scores,
            "summary":           self._summary,
            "j_buffer":          self._j_buffer,
            "turn_idx":          self._turn_idx,
            "compression_count": self._compression_count,
            "prev_zone":         self._prev_zone,
            "prev_j":            self._prev_j,
            "drift_state":       self._drift_state,
            "j_history":         self._j_history,
            "turn1_goal":        self._turn1_goal,
            "stats": {
                "total_tokens_in":    self.stats.total_tokens_in,
                "total_tokens_out":   self.stats.total_tokens_out,
                "total_tokens_saved": self.stats.total_tokens_saved,
                "total_cost_usd":     self.stats.total_cost_usd,
                "turns_compressed":   self.stats.turns_compressed,
                "turns_trimmed":      self.stats.turns_trimmed,
                "turns_preserved":    self.stats.turns_preserved,
            },
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load persisted session state. Migrates older versions transparently."""
        import json
        with open(path) as f:
            state = json.load(f)

        version = state.get("version", "1.0")
        if version != _SESSION_VERSION:
            state = _migrate_session(state, version)

        self._history           = state["history"]
        self._history_j_scores  = state["history_j_scores"]
        self._summary           = state.get("summary")
        self._j_buffer          = state.get("j_buffer", [])
        self._turn_idx          = state.get("turn_idx", len(self._history) // 2)
        self._compression_count = state.get("compression_count", 0)
        self._prev_zone         = state.get("prev_zone", "MEDIUM")
        self._prev_j            = state.get("prev_j", 0.50)
        self._drift_state       = state.get("drift_state", False)
        self._j_history         = state.get("j_history", [])
        self._turn1_goal        = state.get("turn1_goal", "")

        s = state.get("stats", {})
        self.stats.total_tokens_in    = s.get("total_tokens_in", 0)
        self.stats.total_tokens_out   = s.get("total_tokens_out", 0)
        self.stats.total_tokens_saved = s.get("total_tokens_saved", 0)
        self.stats.total_cost_usd     = s.get("total_cost_usd", 0.0)
        self.stats.turns_compressed   = s.get("turns_compressed", 0)
        self.stats.turns_trimmed      = s.get("turns_trimmed", 0)
        self.stats.turns_preserved    = s.get("turns_preserved", 0)


def _migrate_session(state: dict, from_version: str) -> dict:
    """Migrate session state from older versions to current _SESSION_VERSION."""
    if from_version == "1.0":
        # v1.0 → v1.1: added j_history, turns_compressed/trimmed/preserved in stats
        state.setdefault("j_history", [])
        stats = state.setdefault("stats", {})
        stats.setdefault("turns_compressed", 0)
        stats.setdefault("turns_trimmed", 0)
        stats.setdefault("turns_preserved", 0)
        state["version"] = "1.1"
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  * _INPUT_COST_PER_M  / 1_000_000 +
        output_tokens * _OUTPUT_COST_PER_M / 1_000_000
    )


def _cost_haiku(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  * _HAIKU_INPUT_PER_M  / 1_000_000 +
        output_tokens * _HAIKU_OUTPUT_PER_M / 1_000_000
    )
