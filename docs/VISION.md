# Credence: Vision and Research Arc

## How We Got Here

Credence did not start as a project about uncertainty. It started much further upstream.

The original research direction was mechanistic interpretability — understanding not just what models output, but what happens inside them during inference. That led us to Fisher information as a signal: the idea that the Fisher score, computed from the gradient of the log-likelihood, could serve as a per-token measure of informational importance. If a token carries high Fisher weight, the model's predictions are sensitive to it. If it carries low Fisher weight, it can be dropped with minimal consequence. That was the hypothesis behind what we called the Fisher signal — a principled, geometry-aware alternative to attention-based importance scoring.

From there, the natural extension was adaptive resource management. If you know which tokens are informationally important at the model level, you can make better decisions about memory — specifically about the KV cache. The KV cache stores key-value pairs for every token in the context window. Evicting entries aggressively saves memory and speeds up inference, but naive eviction loses information. Fisher-weighted eviction was the idea: evict low-Fisher entries first, retain high-Fisher entries, and tie the eviction policy directly to the model's own sensitivity signal rather than heuristics like recency or attention scores.

That work was progressing. Then we hit the roadblock.

The roadblock was this: Fisher-weighted eviction preserves informationally important tokens. But informational importance and epistemic importance are not the same thing. A token like "approximately" or "I think" carries low Fisher weight — it is a hedging word, not a content word, and the model's predictions are relatively insensitive to it. By the Fisher criterion, it should be evicted. But epistemically, it is the most important token in the sentence. It is the token that tells you whether you can trust what follows.

That gap — between what compression systems consider important and what *epistemically* matters — was the insight. We had been thinking about compression as a resource management problem. The roadblock forced us to see it as an epistemic event. Every compression decision is simultaneously a decision about certainty. When a compression system drops "I think the rate limit is around 50 — unconfirmed" to "rate limit: 50," it has not just saved tokens. It has made a claim: *this is confirmed*.

From that realization, the project crystallized. We stopped asking "how do we compress efficiently?" and started asking "what does compression do to the epistemic state of the conversation?" That question had never been measured. So we measured it. We designed experiments, built a scorer, ran n=50 studies, and found a 26% qualifier strip rate under Haiku and 68% under aggressive importance scoring. Then we built the enforcement layer — not as a compression improvement, but as a guard that sits *before* compression and asks a different question: does this segment contain uncertainty that should not be lost?

The research arc is: mechanistic interpretability → Fisher signal → adaptive KV cache management → compression as epistemic event → Credence.

Every step was necessary. Without the Fisher work, we would not have understood how compression systems score token importance. Without the KV cache work, we would not have understood why epistemic tokens are systematically undervalued by importance scorers. Without the roadblock, we would not have asked the question that turned into this project.

---

## The Thesis

We define **Epistemic Qualifier Loss (EQL)**: the loss of user-stated uncertainty markers during context window summarization, causing downstream models to treat explicitly uncertain claims as confirmed facts. The **EQL Rate (EQLR)** measures how often this happens: 46% under Haiku compression, 68% under LLMLingua-style scoring (n=50). The downstream False Certainty Rate (FCR) — how often the answering model states the uncertain value without any qualifier — is 6% and 74% respectively (corrected scorer v2, 198 markers). LLMLingua: 3 in 4 after compression. Both drop to 0% with Credence.

This failure has a name now. It didn't before. Practitioners at Factory.ai, SwirlAI, and others have described it for years as "semantic drift" or "false certainty from compression" — without a number, without a causal mechanism, without a fix. The academic literature on compression (LLMLingua-2, SnapKV, StreamingLLM) defines faithfulness as lexical or task-level fidelity and never measures whether uncertainty qualifiers survive. The epistemic uncertainty literature (Semantic Entropy, UProp, R-Tuning) addresses model confidence calibration and never touches the compression pipeline. EQL lives at the intersection nobody studied.

Context compression is routinely treated as a token-density problem: given a budget, maximize information retained. The pivot is recognizing that **compression is also an epistemic event.** The decision to summarize a sentence is simultaneously a decision about its certainty status. When Haiku compresses ten turns of conversation, it implicitly decides which qualifiers matter. They don't — from Haiku's perspective. "I think the rate limit is ~50" and "the rate limit is 50" have identical informational cores. The qualifier is collateral loss. EQLR measures how much collateral loss. Credence prevents it.

Memory systems (MemGPT/Letta, Mem0, Zep) persist facts across sessions but strip epistemic qualifiers at write time by design. Uncertainty quantification tools detect uncertainty but never intervene at the compression boundary. Credence is the missing enforcement layer: deterministic, sub-millisecond, wired to every operation that can corrupt epistemic state.

---

## Architecture Value

Five checkpoints, each encoding a distinct architectural principle:

- **CP1 — Faithfulness Probe (0.011ms, zero API calls):** The probe is deterministic, not probabilistic. A 198-marker frozenset scan on user turns only. EQLR 46% → 0%; FCR 74% → 0% (LLMLingua sim, n=50, 95% CI [0%, 7.1%]). *Principle: enforcement that requires a model call is not enforcement — it is a suggestion.*

- **CP2 — Truth Buffer + Consistency Enforcer:** Injects all unverified constraints into the system prompt every turn, with imperative prohibition when the user's query keyword-overlaps a registered constraint (32 domain synonym clusters, 0% FP rate). *Principle: the model should never have to remember an uncertain constraint; inject it explicitly at generation time.*

- **CP3 — Generation-Time Scanner:** Annotates numeric and string literals in generated code and prose with confidence tiers (HIGH RISK / UNVERIFIED / CHECK) derived from the live registry. Catches `RATE_LIMIT = 50`, `ALGORITHM = "RS256"`, `BASE_URL = "/api/v2"`. *Principle: enforcement must extend to the artifact, not just the conversation.*

- **CP4 — Rust Gate (3.4ms, 98× faster than Python, 0% FP rate):** Native PreToolUse hook. Blocks Write/Edit/MultiEdit/NotebookEdit/Bash when tool arguments overlap unverified constraints. *Principle: irreversible actions are where epistemic errors become real costs — gate the action, not the text.*

- **CP5 — Epistemic Memory:** Cross-session constraint registry with certainty trajectories and confidence decay. CS-FCR 40% (no memory) → 0% (Credence Memory), n=20 callbacks. *Principle: epistemic state is session-persistent by nature, not by accident of context window size.*

---

## The Research Arc

**Now (deployed):** Deterministic enforcement at five checkpoints. 22-tool MCP server. 178 passing tests (S1–S26, 11 skipped offline-only). Ghost Gauntlet: BothRate 0.200 → 1.000 (n=10 sessions). E6: 19.6% → 100% correction recall (n=23). Precision eval: 0% FP on CE, GTS, and probe. ETP schema defined. The system prevents the measured failure with no false positives on the precision eval set.

**DPO Phase 3 (in progress, 2026-05-01):** Base Phi-2 FCR established at 31.2% pre-training. DPO fine-tuning running on Kaggle T4 (3 epochs, 5,000 triples). Expected result: FCR drops to ~15% post-DPO — the soft learning layer reduces the baseline before the deterministic probe takes over. The three-point comparison (31.2% → ~15% → 0%) will be the headline Layer 2 validation.

**6 months — calibrated epistemic compression:** Replace binary block/proceed with a continuous epistemic importance weight per sentence. Analogous to LLMLingua-2's token importance scoring, but the weight is derived from the constraint registry: sentences containing registered uncertain values receive 10× importance in compression scheduling. This gives a principled hybrid — compress everything else aggressively; treat uncertain sentences as near-incompressible. Technical path: replace the binary Haiku gate with a weighted retention policy that annotates compression-safe vs. compression-risky sentences before the compressor runs.

**6 months — domain-learned uncertainty profiles:** Run 50 production sessions per project, identify which markers co-occurred with actual FCR events, output `epistemic_profile.json` with project-specific marker weights. Medical, legal, and financial domains each have distinct hedging vocabularies that the generic 108-term list misses. Thompson Sampling on marker weights (Layer 1 bandit) provides the mechanism; the product artifact is a committable, auditable profile alongside the codebase.

**6 months — certainty trajectory as compliance artifact:** Surface the `credence_trajectory` audit trail as a structured report at PR merge time: which constraints were unverified, which lines they appeared in, whether they were ever resolved. This is epistemic debt reporting — the way test coverage reports code quality. For regulated industries (HIPAA, SOC 2, PCI-DSS), this becomes a mandatory audit artifact rather than a developer convenience.

**2 years — ETP as open standard:** Make the Epistemic Transport Protocol a community standard adopted by AutoGPT, LangChain, CrewAI, and native model provider APIs. The model: HTTP headers carry request metadata; ETP headers carry epistemic metadata. Every agent handoff in every pipeline passes `{j_score, zone, verified, chain_depth}` alongside the content. The Ghost Detector and SE probe become standard middleware callable by any framework. Credence becomes the reference implementation of a protocol.

**Concurrent work framing:** arXiv:2509.11208 (ICML 2025) independently reached the same framing — compression decisions are epistemic decisions, and epistemic failures compound through pipelines — from the evidence adjudication direction rather than context compression. Two independent convergences on the same thesis from different domains is the strongest possible signal that the problem class is real and the timing is right.

---

## Limitations as Research Directions

**FCR measures hedging absence, not factual incorrectness.** A response stating "the rate limit is 50 req/min" fails FCR if the source said "I think around 50." That is the right harm to measure. But FCR conflates two harms — stripped qualifier and wrong fact — that have different remedies. Disentangling them requires ground-truth verification, which Credence does not currently perform. *Research direction: integrate the registry with external verification sources (API docs, database schemas) to enable factual FCR alongside hedging FCR.*

**Truth Buffer and CE are probabilistic.** CP1 and CP4 are deterministic. CP2 depends on the model following injected instructions. Prompt-only instruction achieves 90.0% qualifier survival (not 100%), versus 100% with the probe. The gap is the model's compliance probability. *Research direction: apply conformal prediction to give PAC-style bounds — characterize per-constraint-type compliance rates and surface the gap to users.*

**GTS over-annotation on common literals.** A constraint registering value `50` annotates every `= 50` assignment in generated code, including unrelated loop indices. Over-annotation is safer than under-annotation; but it is noise. *Research direction: sentence-level embedding similarity between the constraint text and the code context around the matched literal, with a configurable annotation threshold.*

**Confident-wrong ceiling.** J-score measures linguistic assertiveness (ρ = −0.034 with factual correctness). A confidently wrong statement scores HIGH-J, bypasses the faithfulness probe, and is neither flagged nor blocked. Ghost Detector catches many of these via Opus reasoning. *Research direction: distill a lightweight confident-wrong classifier from Ghost Detector training data, runnable at probe speed without an API call.*

**Short sessions.** Compression fires at turn 16 (COMPRESS) or 20 (TRIM). Sessions shorter than this do not trigger CP1. *Research direction: lower compression threshold adaptively for sessions with early high-stakes constraint registration.*

---

## Why This Becomes Mandatory

Type checking became a production requirement not because developers wanted more tooling, but because silent type errors at scale were more expensive than the overhead of enforcement. The same cost structure applies here. At 10 agent hops with a 10% per-hop FCR, the probability that at least one agent in the chain encounters a false certainty is 65%. Credence's deterministic probe brings that to near zero. The compounding math is unambiguous: the per-hop cost of enforcement is constant; the per-chain cost of not enforcing grows exponentially with chain length.

The Rust gate is the key infrastructure signal. The PreToolUse hook pattern — running before every irreversible action, not just occasionally — is the same model as security scanners before every git commit. Developers accept that 3.4ms overhead because the cost of a security failure dominates the cost of the scan. Epistemic failures in deployed AI systems have the same asymmetric cost structure: a wrong rate limit baked into a deployed service, a wrong auth algorithm committed to a codebase, a wrong expiry value shipped to a client. Once a team has experienced zero FCR in production, the gate becomes load-bearing infrastructure that nobody removes.

---

## Honest Assessment of the Contribution

The problem is real. Context compression does silently strip uncertainty qualifiers — not because models are broken, but because to a compression model, "I think the rate limit is ~50 req/min — unconfirmed" and "the rate limit is 50 req/min" have identical informational cores. The qualifier is collateral loss. Measuring it with methodology — naming it EQL, defining FCR, running n=50 studies — is a genuine contribution. Nobody had a number for this before.

The engineering is serious. A Rust gate at 3.4ms, a faithfulness probe at 0.07ms, a working MCP server, 178 tests, and a Ghost Detector that uses Opus 4.7's reasoning to classify the *origin and reliability* of claims — not just their surface text — is a real system, not a demo with a fake backend. The Ghost Detector insight specifically is non-obvious: Haiku sees the same characters as Opus; only Opus reasons about epistemic provenance.

**What kind of contribution this is.** Credence is a proof-of-concept that a real problem exists and is solvable, with a working reference implementation. The lasting value is the concept and the measurement. EQL, EQLR, and FCR as defined metrics will age well. The framing that "compression is also an epistemic event" is the insight. Whether *this* implementation is the one that survives in production, or whether it proves the concept for someone to build it natively into the platform — both are good outcomes. Both mean the idea mattered.

**The honest limitation.** The generation-side layers — Truth Buffer and Consistency Enforcer — inject constraints and instruct the model to treat them as unverified. That is a strong suggestion, not a mechanical guarantee. The probe and the Rust gate are fully deterministic. The generation side is deterministic injection with probabilistic compliance. This distinction matters for anyone evaluating production guarantees. CP1 and CP4 make a class of errors structurally impossible. CP2 and CP3 make a class of errors much less likely.

**The absorption scenario is the best outcome.** The biggest risk to Credence as middleware is that Anthropic or a similar platform builds epistemic state tracking natively into the context management layer — making third-party enforcement unnecessary. If that happens, it is not a failure. It is evidence that the problem was real enough to be worth solving at the infrastructure level. Credence would then be remembered as the proof of concept that made the case. The type-checking analogy is exact: many early type-checkers were third-party tools before type systems became native to compilers and languages. The concept survived; the middleware didn't need to.

**Why the concept survives regardless.** As AI agents take longer autonomous sessions and make more real-world decisions — committing code, deploying infrastructure, executing financial transactions — the question "was this value actually confirmed?" becomes existential. The compounding math is unambiguous: at 10 agent hops with even a 10% per-hop false certainty rate, the probability that at least one agent in the chain acts on an unverified value exceeds 65%. Epistemic metadata is not optional in agentic systems. Someone will build this. This project names the problem, measures it, and shows it is solvable. That is the contribution that persists.

---

## The Standard

`etp_v1.json` is a model-agnostic JSON Schema for epistemic metadata transport. It defines four primitives: `EpistemicConstraint` (a tracked uncertain claim), `EpistemicEnvelope` (a provenance wrapper for AI-generated content with trust decay per hop), `EpistemicLedger` (full session state), and `AlignmentWarning` (fired when a response is more confident than the ledger warrants).

The design principle: *every AI system today passes information between agents by value. Nobody passes it by epistemic weight.* ETP proposes to fix this by making epistemic metadata first-class in agent protocols — the same way HTTP headers made request metadata first-class in web protocols.

Credence is the reference implementation. The standard is the destination.
