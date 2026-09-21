/*!
credence-gate — Native Claude Code PreToolUse enforcement hook.

Reads the Claude Code hook payload from stdin (JSON), checks it against the
Credence epistemic registry, and outputs a decision:

  BLOCK  → tool call stopped, user sees warning
  ALLOW  → tool call proceeds

Zero Python startup overhead: binary starts in <1ms vs Python's ~300ms.

This matters because PreToolUse fires on EVERY Write/Edit/Bash call. In a
100-tool-call session, Python hook overhead is 100 × 300ms = 30 seconds.
credence-gate: 100 × <1ms = 0.1 seconds.

Usage in .claude/settings.json:
  {
    "hooks": {
      "PreToolUse": [{
        "matcher": "Write|Edit|MultiEdit|NotebookEdit|Bash",
        "hooks": [{"type": "command", "command": "credence-gate"}]
      }]
    }
  }

Protocol (Claude Code hook protocol):
  - Reads JSON from stdin: { "tool_name": "...", "tool_input": {...} }
  - Exit code 0 = ALLOW
  - Exit code 2 = BLOCK (with stderr message shown to user)
  - Writes blocking message to stderr

Registry: reads epistemic_registry.db from the current working directory.
*/

use std::collections::HashSet;
use std::io::{self, Read};
use std::time::Instant;

use serde::Deserialize;
use rusqlite::{Connection, params};
use regex::Regex;

// ---------------------------------------------------------------------------
// Constants — must match credence/matching.py
// ---------------------------------------------------------------------------

const MIN_OVERLAP: usize = 2;
// Numeric literals shorter than this do not take part in the value rule.
const MIN_NUM_LEN: usize = 2;
const DB_PATH: &str = "epistemic_registry.db";

// Canonical copy: credence/matching.py (`STOPWORDS`). This list was 60 words
// here and 78 there, differing in both directions — it carried code words the
// canonical list scores ("write", "edit", "file", "code", "function",
// "method") and omitted 45 the canonical list stops ("think", "know",
// "want", "is", "of", "size", "error"). Either direction changes which
// writes block, so a constraint phrased "I think the rate limit is ..." scored
// differently in the two gates.
// tests/unit/test_matcher_parity.py parses this literal and fails if the two
// lists diverge.
static STOPWORDS: &[&str] = &[
    "a", "about", "also", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "error", "for", "from",
    "get", "give", "go", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "know", "make", "may", "might", "my", "need",
    "now", "of", "on", "or", "our", "say", "see", "set", "should", "size",
    "so", "take", "tell", "that", "the", "think", "this", "through", "to",
    "use", "used", "using", "want", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "would", "you", "your",
];

// No synonym clusters here, deliberately. credence/matching.py excludes
// synonym-cluster agreement from the *blocking* decision: expanding both sides
// lets one shared cluster key satisfy the overlap threshold by itself, so
// `TIMEOUT_MS = 5000` would be blocked merely because an unverified constraint
// mentions some other timeout. This binary only ever blocks, so it has no
// recall-oriented caller that would want the expansion — carrying it here is
// what made the two gates return different verdicts for the same write.

// ---------------------------------------------------------------------------
// Input schema
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct HookInput {
    tool_name: Option<String>,
    tool_input: Option<serde_json::Value>,
    // Claude Code may also send session info
    session_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Constraint from registry
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Constraint {
    constraint_id: String,
    content: String,
    zone: String,
    j_score: f64,
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

fn stopword_set() -> HashSet<&'static str> {
    STOPWORDS.iter().copied().collect()
}

/// Split one token into its lower-cased identifier parts.
///
/// `rate_limit` -> `[rate, limit]`, `rateLimit` -> `[rate, limit]`,
/// `stripeClientRateLimit` -> `[stripe, client, rate, limit]`.
///
/// Casing is split BEFORE lowering: lowering first would collapse
/// `rateLimit` into `ratelimit` and lose the boundary that lets it match the
/// prose "rate limit". The canonical `regex` crate has no lookaround, so the
/// camel boundary is found by scanning rather than with the Python side's
/// `(?<=[a-z0-9])(?=[A-Z])`.
fn split_identifier(token: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for chunk in token.split(|c| matches!(c, '_' | '.' | '-' | '/' | '\\')) {
        if chunk.is_empty() {
            continue;
        }
        let chars: Vec<char> = chunk.chars().collect();
        let mut cur = String::new();
        for (i, &c) in chars.iter().enumerate() {
            if i > 0 && c.is_ascii_uppercase() {
                let prev = chars[i - 1];
                if (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && !cur.is_empty() {
                    out.push(cur.to_lowercase());
                    cur.clear();
                }
            }
            cur.push(c);
        }
        if !cur.is_empty() {
            out.push(cur.to_lowercase());
        }
    }
    out
}

fn tokenize(text: &str, stopwords: &HashSet<&str>) -> HashSet<String> {
    // Punctuation becomes a separator. `_` is a word character, so it survives
    // this pass and is handled by split_identifier — which is what makes
    // RATE_LIMIT score as `rate` and `limit` rather than one dead token.
    let re = Regex::new(r"[^\w\s]").unwrap();
    let cleaned = re.replace_all(text, " ");
    let mut out: HashSet<String> = HashSet::new();
    for word in cleaned.split_whitespace() {
        // Keep the whole token alongside its parts, so a constraint that
        // literally says `rate_limit` still matches another literal
        // `rate_limit`.
        let whole = word.to_lowercase();
        if whole.chars().count() >= 3 && !stopwords.contains(whole.as_str()) {
            out.insert(whole);
        }
        for part in split_identifier(word) {
            if part.chars().count() >= 3 && !stopwords.contains(part.as_str()) {
                out.insert(part);
            }
        }
    }
    out
}


/// Numeric literals of at least `MIN_NUM_LEN` digits, matching `NUM_PATTERN` in
/// credence/matching.py and `_GTS_NUM_PATTERN` in context_manager.py.
///
/// The Python side blocks on a shared *value* even when no term overlaps; this
/// binary had no such rule, so `Write TIMEOUT = 100` against a constraint about
/// a `100 req/min` rate limit was allowed here and blocked there.
fn numbers(text: &str) -> HashSet<String> {
    // The whole match is the digits, so no capture group is needed; converting
    // to an owned String inside the loop keeps this free of borrow subtleties.
    let re = Regex::new(r"\b\d+(?:\.\d+)?\b").unwrap();
    let mut out: HashSet<String> = HashSet::new();
    for m in re.find_iter(text) {
        let n = m.as_str();
        if n.chars().count() >= MIN_NUM_LEN {
            out.insert(n.to_string());
        }
    }
    out
}

fn resolve_db_path() -> String {
    // Match the Python side exactly. hooks.py and observer.py — the two layers
    // this binary replaces — resolve CREDENCE_DB, and SECURITY.md tells users to
    // set it. This binary previously read only CREDENCE_DB_PATH and
    // CREDENCE_REGISTRY_PATH, so a user who followed the documented setup had
    // this gate open a different, empty database, find no constraints, and allow
    // every write. Enforcement looked installed and was inert.
    //
    // Chain matches mcp_server.py: CREDENCE_DB_PATH (canonical) → CREDENCE_DB
    // (what hooks.py / observer.py use) → CREDENCE_REGISTRY_PATH (legacy) →
    // default.
    std::env::var("CREDENCE_DB_PATH")
        .or_else(|_| std::env::var("CREDENCE_DB"))
        .or_else(|_| std::env::var("CREDENCE_REGISTRY_PATH"))
        .unwrap_or_else(|_| DB_PATH.to_string())
}

fn load_constraints(session_id: &Option<String>) -> Vec<Constraint> {
    let db_path = resolve_db_path();
    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(_) => return vec![],  // no registry = no constraints = ALLOW
    };

    // Use parameterized queries throughout — never interpolate session_id into SQL.
    let rows: Vec<Constraint> = if let Some(sid) = session_id {
        // Two separate queries ORed in Rust to keep parameterization clean.
        // Query A: constraints for this specific session.
        let mut a_results: Vec<Constraint> = {
            let sql = "SELECT constraint_id, content, zone, j_score FROM constraints \
                       WHERE verified=0 \
                       AND (validation_status='unverified' OR validation_status IS NULL) \
                       AND session_id=?1";
            let mut stmt = match conn.prepare(sql) {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            let iter = match stmt.query_map(params![sid], |row| {
                Ok(Constraint {
                    constraint_id: row.get(0)?,
                    content:       row.get(1)?,
                    zone:          row.get(2)?,
                    j_score:       row.get(3)?,
                })
            }) {
                Ok(i) => i,
                Err(_) => return vec![],
            };
            iter.filter_map(|r| r.ok()).collect()
        };

        // Query B: cross-session memories (is_memory=1) — project-scoped, no sid filter.
        let b_results: Vec<Constraint> = {
            let sql = "SELECT constraint_id, content, zone, j_score FROM constraints \
                       WHERE verified=0 \
                       AND (validation_status='unverified' OR validation_status IS NULL) \
                       AND is_memory=1 AND project_id IS NOT NULL";
            let mut stmt = match conn.prepare(sql) {
                Ok(s) => s,
                Err(_) => return a_results,  // best-effort: return session constraints at minimum
            };
            let iter = match stmt.query_map([], |row| {
                Ok(Constraint {
                    constraint_id: row.get(0)?,
                    content:       row.get(1)?,
                    zone:          row.get(2)?,
                    j_score:       row.get(3)?,
                })
            }) {
                Ok(i) => i,
                Err(_) => return a_results,
            };
            iter.filter_map(|r| r.ok()).collect()
        };

        // Deduplicate by constraint_id before returning.
        let mut seen = std::collections::HashSet::new();
        a_results.retain(|c| seen.insert(c.constraint_id.clone()));
        let mut combined = a_results;
        for c in b_results {
            if seen.insert(c.constraint_id.clone()) {
                combined.push(c);
            }
        }
        combined
    } else {
        let sql = "SELECT constraint_id, content, zone, j_score FROM constraints \
                   WHERE verified=0 \
                   AND (validation_status='unverified' OR validation_status IS NULL)";
        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let iter = match stmt.query_map([], |row| {
            Ok(Constraint {
                constraint_id: row.get(0)?,
                content:       row.get(1)?,
                zone:          row.get(2)?,
                j_score:       row.get(3)?,
            })
        }) {
            Ok(i) => i,
            Err(_) => return vec![],
        };
        iter.filter_map(|r| r.ok()).collect()
    };

    rows
}

fn extract_arguments_text(tool_input: &Option<serde_json::Value>) -> String {
    match tool_input {
        None => String::new(),
        Some(v) => match v {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(map) => {
                // Concatenate all string values from tool input
                map.values()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            _ => v.to_string(),
        }
    }
}

fn format_block_message(
    tool_name: &str,
    matched: &[(String, String, String, f64)], // (cid, content, zone, j)
) -> String {
    let mut msg = String::new();
    msg.push_str("\n╔══════════════════════════════════════════════════════════════╗\n");
    msg.push_str("║  CREDENCE GATE — TOOL BLOCKED                                ║\n");
    msg.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");
    msg.push_str(&format!("  Tool:    {}\n\n", tool_name));
    for (cid, content, zone, j) in matched {
        let content_short = if content.len() > 60 {
            format!("{}…", &content[..60])
        } else {
            content.clone()
        };
        msg.push_str(&format!(
            "  ⚠ [{}, conf={:.2}] {}\n    id: {}\n\n",
            zone, j, content_short, cid
        ));
    }
    msg.push_str("  Use credence_verify(<id>, <confirmed_value>) to resolve.\n");
    msg.push_str("  Or: credence_verify_all to confirm all pending constraints.\n");
    msg
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let t_start = Instant::now();

    // Read stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap_or(0);

    if input.trim().is_empty() {
        // No input — ALLOW (safety: don't block if misconfigured)
        std::process::exit(0);
    }

    // Parse hook input
    let hook: HookInput = match serde_json::from_str(&input) {
        Ok(h) => h,
        Err(_) => {
            // Malformed JSON — ALLOW (don't block on parse error)
            std::process::exit(0);
        }
    };

    let tool_name = hook.tool_name.as_deref().unwrap_or("unknown");

    // Only enforce on write-side tools
    // Must match ENFORCED_TOOLS in credence/matching.py — the Python hook
    // carries the same list, and it is shared policy for the same reason the
    // registry path is: the two gates disagreeing means one of them enforces
    // something the other does not. tests/unit/test_matcher_parity.py parses
    // this literal and fails if the two lists drift apart.
    let enforced_tools = ["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"];
    if !enforced_tools.contains(&tool_name) {
        std::process::exit(0);
    }

    // Load constraints from registry
    let constraints = load_constraints(&hook.session_id);
    if constraints.is_empty() {
        // No unverified constraints — ALLOW
        std::process::exit(0);
    }

    // Build argument text to check
    let args_text = extract_arguments_text(&hook.tool_input);
    if args_text.is_empty() {
        std::process::exit(0);
    }

    // Tokenise. The query side is built once; the constraint side is rebuilt
    // per constraint, which is the dominant cost of this loop.
    let stopwords = stopword_set();
    let query_tokens = tokenize(&args_text, &stopwords);
    let query_values = numbers(&args_text);

    // Check each constraint for overlap
    let mut matched: Vec<(String, String, String, f64)> = vec![];
    for constraint in &constraints {
        let constraint_tokens = tokenize(&constraint.content, &stopwords);

        // Two independent rules, mirroring credence/matching.py's blocking
        // policy: a shared numeric value blocks on its own, otherwise two
        // shared literal terms are needed. Synonym agreement is deliberately
        // not evidence here — see the note where the clusters used to live.
        let shared_terms = query_tokens.intersection(&constraint_tokens).count();
        let shared_value = !query_values.is_disjoint(&numbers(&constraint.content));

        if shared_value || shared_terms >= MIN_OVERLAP {
            matched.push((
                constraint.constraint_id.clone(),
                constraint.content.clone(),
                constraint.zone.clone(),
                constraint.j_score,
            ));
        }
    }

    let elapsed_us = t_start.elapsed().as_micros();

    if matched.is_empty() {
        // No overlap — ALLOW
        // Optionally log timing for benchmarks
        if std::env::var("CREDENCE_DEBUG").is_ok() {
            eprintln!("[credence-gate] ALLOW  tool={} constraints_checked={}  elapsed={}µs",
                tool_name, constraints.len(), elapsed_us);
        }
        std::process::exit(0);
    }

    // BLOCK — print message to stderr (Claude Code shows this to user)
    let msg = format_block_message(tool_name, &matched);
    eprintln!("{}", msg);

    if std::env::var("CREDENCE_DEBUG").is_ok() {
        eprintln!("[credence-gate] BLOCK  tool={}  matched={}  elapsed={}µs",
            tool_name, matched.len(), elapsed_us);
    }

    // Exit code 2 = BLOCK in Claude Code hook protocol
    std::process::exit(2);
}
