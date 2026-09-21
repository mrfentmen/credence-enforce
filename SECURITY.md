# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.2.x   | ✅ Active  |
| 1.1.x   | ✅ Active  |
| 1.0.x   | ⚠️ Patches only |
| < 1.0   | ❌ No support |

## Threat Model

Credence is a **local development tool**. It runs as:
- A Python library imported into your code
- An MCP server over stdio (not a network service)
- A native binary (`credence-gate`) invoked by Claude Code hooks

The registry (`epistemic_registry.db`) is a local SQLite file. It is not a network service and has no authentication layer by design — it is scoped to your local machine or shared team filesystem.

**What Credence does not protect against:**
- Malicious code running on the same machine (same-process trust boundary)
- A compromised MCP host (Claude Code itself)
- Network-level attacks (Credence has no network listener)

## Data Storage

Credence stores data in two local locations. **No data is ever transmitted off the machine.**

| Store | Default path | Override | Contains |
|---|---|---|---|
| Registry | `./epistemic_registry.db` | `CREDENCE_DB` env var | Uncertain statements (≤500 chars), session IDs, verification status |
| Event log | `~/.credence/events.jsonl` | `CREDENCE_NO_LOG=1` to disable | Gate block/allow events, short constraint excerpts (≤80 chars), timestamps |

**Recommendations:**
- Add `*.db` to your project's `.gitignore`
- Set `CREDENCE_DB=~/.credence/registry.db` if working in cloud-synced directories (Dropbox, iCloud, etc.)
- Set `CREDENCE_NO_LOG=1` if you do not want gate events written to disk at all
- Do not store secrets as constraint content — the registry is not encrypted at rest

## Reporting a Vulnerability

If you discover a security vulnerability in `credence-enforce`, please report it **privately** before public disclosure.

**Do not open a public GitHub issue for security vulnerabilities.**

**Report via:**
- GitHub private security advisory: [Security Advisories](https://github.com/mrfentmen/credence-enforce/security/advisories/new)

This fork does not publish a security email address. Please use the private
advisory link above — it notifies the maintainer without disclosing the issue
publicly. Note that GitHub `noreply` addresses (including the one on this
package's metadata) do not accept inbound mail, so email is not a working
channel here.

If the vulnerability is in code inherited from upstream — that is, it reproduces
against `credence-guard` 1.2.5 as well — please also report it upstream to the
original author, Lakshmi Chakradhar Vijayarao, via
<https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai/security/advisories/new>.
Fixes for upstream defects belong upstream.

**Please include:**
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

**Response timeline:**
- Acknowledgement within 48 hours
- Fix or mitigation plan within 7 days for critical issues
- Coordinated public disclosure after fix is released

## Known Security Boundaries

- `credence_register()` accepts arbitrary content strings. The registry enforces a per-session cap of 500 constraints by default (override: `CREDENCE_MAX_CONSTRAINTS=<n>`). A malicious agent filling the registry is bounded by this cap; blast radius is your local disk.
- The registry is not encrypted at rest. Do not store secrets as constraint content.
- `CREDENCE_DB` env var accepts arbitrary file paths. Validate in shared/multi-user environments.
