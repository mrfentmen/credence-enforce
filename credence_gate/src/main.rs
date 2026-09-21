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
        "matcher": "Write|Edit|Bash|NotebookEdit",
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

use std::collections::{HashMap, HashSet};
use std::io::{self, Read};
use std::time::Instant;

use serde::Deserialize;
use rusqlite::{Connection, params};
use regex::Regex;

// ---------------------------------------------------------------------------
// Constants — must match credence/context_manager.py
// ---------------------------------------------------------------------------

const MIN_OVERLAP: usize = 2;
const DB_PATH: &str = "epistemic_registry.db";

static STOPWORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "have", "from",
    "are", "was", "were", "has", "had", "been", "can", "will",
    "not", "but", "all", "any", "its", "into", "over", "also",
    "than", "only", "such", "very", "more", "just", "you", "may",
    "might", "should", "would", "could", "about", "what", "write",
    "edit", "run", "set", "get", "use", "make", "call", "add",
    "value", "values", "update", "configure", "config", "let",
    "new", "old", "file", "path", "code", "function", "method",
];

// Domain synonym clusters (subset of the 32 in context_manager.py)
// Key insight: expand tokens so "endpoint" matches "rate" matches "limit"
const SYNONYM_CLUSTERS: &[&[&str]] = &[
    &["rate", "limit", "throttle", "quota", "ratelimit", "freq", "frequency", "speed", "throughput", "fast", "slow", "calls", "requests", "req"],
    &["token", "expiry", "expire", "expires", "ttl", "timeout", "session", "auth", "jwt", "oauth", "credential", "credentials"],
    &["retry", "backoff", "attempt", "attempts", "max_retries"],
    &["endpoint", "url", "host", "port", "address", "api", "service"],
    &["cost", "price", "pricing", "billing", "charge", "fee"],
    &["memory", "ram", "heap", "buffer", "cache", "size"],
    &["latency", "delay", "response", "time", "ms", "millisecond", "seconds"],
    &["concurrent", "parallel", "workers", "threads", "connections"],
];

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

fn build_synonym_map() -> HashMap<&'static str, Vec<&'static str>> {
    let mut map: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    for cluster in SYNONYM_CLUSTERS {
        for &word in *cluster {
            let mut others: Vec<&'static str> = cluster.iter().copied()
                .filter(|&w| w != word)
                .collect();
            map.entry(word).or_default().append(&mut others);
        }
    }
    map
}

fn tokenize(text: &str, stopwords: &HashSet<&str>) -> HashSet<String> {
    // Replace underscores and non-word chars with spaces so RATE_LIMIT → rate limit
    let re = Regex::new(r"[^\w\s]|_").unwrap();
    let cleaned = re.replace_all(text, " ");
    cleaned.split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 2 && !stopwords.contains(w.as_str()))
        .collect()
}

fn expand_tokens<'a>(
    tokens: &HashSet<String>,
    syn_map: &'a HashMap<&'static str, Vec<&'static str>>,
) -> HashSet<String> {
    let mut expanded = tokens.clone();
    for token in tokens.iter() {
        if let Some(synonyms) = syn_map.get(token.as_str()) {
            for &syn in synonyms {
                expanded.insert(syn.to_string());
            }
        }
    }
    expanded
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
    let enforced_tools = ["Write", "Edit", "Bash", "NotebookEdit", "MultiEdit"];
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

    // Tokenize and expand query
    let stopwords = stopword_set();
    let syn_map = build_synonym_map();
    let query_tokens = tokenize(&args_text, &stopwords);
    let query_expanded = expand_tokens(&query_tokens, &syn_map);

    // Check each constraint for overlap
    let mut matched: Vec<(String, String, String, f64)> = vec![];
    for constraint in &constraints {
        let constraint_tokens = tokenize(&constraint.content, &stopwords);
        let constraint_expanded = expand_tokens(&constraint_tokens, &syn_map);

        // Count literal + expanded overlap (use expanded for counting)
        let overlap: HashSet<&String> = query_expanded.iter()
            .filter(|t| constraint_expanded.contains(*t))
            .collect();

        if overlap.len() >= MIN_OVERLAP {
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
