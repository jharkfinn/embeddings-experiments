# KV-Prepend Probing for Multivector Embedding Design (v13)

## What this is

An experiment to decide what to extract from a frozen decoder-only MoE LLM for multivector retrieval. "Multivector" includes ColBERT-style tokenwise late interaction, facet vectors, summary representations, hybrids, or something unanticipated. KV-prepend works on dense models; we're probing whether and how to use it on MoE, where routing boundaries may create richer structure. We have not committed to any embedding construction. The analysis should tell us what to build. The goal is a general method across MoE models, not an optimized pipeline for any single task.

Two framing principles. First, "best signal" is the wrong question. The right questions are: which channels should be **compressed**, which should remain **structured**, and where should their evidence be combined — inside one embedding, at retrieval scoring, or after candidate union. Second, each signal should be evaluated on three frontiers: **abstraction** (semantics or token identity?), **heterogeneity/transport** (per-token diversity that survives downstream?), and **coverage/cost** (unique relevant documents per stored bit?). These assign roles to signals rather than ranking them. Add a fourth operational frontier in reporting: **leakage/privacy**. A channel that improves recall but is easily decodable back to text may still be unusable at scale.

Central caution: information-rich is not retrieval-useful. Router traces reconstruct 91% of tokens (Nuriyev & Kulp) but may be too lexical for semantic retrieval. A zero treatment effect doesn't mean no missing context; it means the bottlenecked single-slot summary didn't contain what that token needed.

## Prior work and gaps

- **KV-Embedding** (Tang & Yang, 2026): prepending final token's KV improves training-free embeddings ~10% on MTEB. Broad midrange robustness to prepend bias with meaningful layer effects; early layers are clearly worse and ID-based selection matters. *Gap: all dense models.*
- **MoEE** (Li et al., ICLR 2025 Oral): router logits complementary to hidden states; weighted similarity sums best. *Gap: never perturbed via context injection.*
- **VA/AlignedWVA** (Zhang et al., 2026): value vectors + last-token attention + W_O beat hidden states. *Gap: not studied on MoE or under intervention.*
- **Nuriyev & Kulp** (ICLR 2026 Workshop): sequence decoder recovers 91% of tokens from expert selections; per-token only 63%. Cross-position structure critical. Middle layers most independent. *Gap: proves capacity, not semantic utility.*

No prior work combines all three signals, tests KV-prepend on MoE, or compares causal vs. context-corrected routing traces.

## Why MoE changes expectations

Dense models showed broad midrange robustness to prepend bias/layers. In MoE, small perturbations can cross routing boundaries, producing discontinuously different expert activations. Whether this creates sharper structure is a primary question. SD-MoE found experts share aligned spectral components, so distinctness is empirical. Not every routing flip matters.

## Model and setup

Qwen3-30B-A3B at FP8. 48 layers, all softmax GQA (32 Q heads, 4 KV heads), MoE every layer (128 experts, top-8). ~30GB, single 80GB GPU. Softmax-then-top-k; we hook pre-softmax. The 4 KV heads mean the prepended summary is 4 memories through 32 query patterns; analyze at both levels. Router logits and β accumulated in bf16/fp32. Generative on both query and index sides.

RoPE: test three modes on subset (end-position-coded, re-rotated to position 0, de-rotated canonical).

**Quantization: trinary {-1, 0, +1}** throughout. Trinary preserves sparsity, which is critical for router logits (128 dims, ~8 strongly active): it zeros out inactive dimensions rather than forcing all 128 to vote equally. Float baselines on calibration subsets for rate-distortion analysis. Prompt scaffold tokens from the compression template should be masked out of all retrieval representations, token-collapse metrics, and semantic-abstraction evaluations; otherwise identical prompt tokens can dominate both pooling and tokenwise similarity.

## Two-pass design (~2.4x cost)

**Pass 1 (causal + local probes):** standard forward pass; at each layer, also run attention with final token's KV prepended, not propagated.

**Pass 2 (propagating + local probes):** prepend applied and propagated at layers 16–47; layers 0–15 pure causal (avoids early-layer damage; matches Pass 1 as sanity check). At each propagated layer, also run attention without prepend.

Three measurements per layer:

1. **Causal-base local marginal** (all 48): prepend effect, all else untouched.
2. **Propagated-base local marginal** (16–47): prepend effect with upstream already corrected.
3. **Full propagated state** (16–47): cumulative treated state.

Interaction Γ_l = δ_l^prop − δ_l^causal decomposed into gain, angle, aligned shift. Reduced subset: receiver × key-source × value-source factorial.

## What we collect (per layer, all conditions)

1. Summary uptake β per Q head per token
2. Pre-attention residual stream h_in (for offline routing-spectrum reconstruction)
3. Pre-MoE hidden states (post-attention, post-residual, post-layernorm)
4. Pre-softmax router logits (ℝ^{n×128})
5. Top-k expert selections (integer indices)
6. Raw value vectors (4 × d_head per token)
7. Attention output before residual add (post-W_O)
8. Final-token K and V per KV group (from both passes)
9. Causal attention weights (32 Q heads)
10. Query vectors on calibration subset

**Multi-slot extension (calibration subset only):** after the compression prompt prefill, generate 3 additional continuation tokens. Prepend all 4 KV pairs (original final + 3 generated) as separate slots. Nearly free: 3 decode steps on existing cache, attention goes from n+1 to n+4. Compare single-slot vs. multi-slot treatment effects. Key question: do tokens that showed zero d under single-slot show nonzero d under multi-slot? If yes, the bottleneck was the summary, not the mechanism. Also measure: does the model distribute attention across slots or concentrate on one? Do different tokens prefer different slots?

Calibration subset additionally: expert outputs near flip points; next-token entropy under compression prompt.

## Evaluation design

**Controls:** first-token KV, random-token KV, shuffled-sentence KV, prompt-elicited continuation KV, cross-example KV swaps, K-only and V-only with matched-norm nulls.

**Sentence batches:** natural diversity (short ~32, medium ~256, long ~1024–2048); semantic minimal pairs (negation, scope, coreference, entity disambiguation, temporal, clause attachment); semantic-abstraction battery (paraphrases, adversarial near-duplicates, controlled lexical overlap).

**Benchmark:** NanoBEIR (13 BEIR subsets). Per-task breakdown critical: token-alignment QA (FiQA, NQ, HotpotQA), holistic topical match (SciFact, ClimateFEVER, DBPedia), paraphrase/duplicate (Quora). **Long-context stress test:** add at least one long-document retrieval slice where decisive evidence appears late in the document. NanoBEIR documents are mostly short-to-medium; this directly tests the core KV-prepend thesis. Semantic abstraction should be judged per task family, not globally: a "too lexical" channel may be exactly right for QA-style retrieval.

**Dense comparator:** same probe (minus router analyses) on a dense Qwen sibling, matched setup.

**Bridge experiment:** compare echo and KV-prepend on the same backbone and benchmark slice to disentangle extraction method from model.

## Early experimental findings (SciFact, trinary, echo on Qwen 3.5)

Preliminary, directional observations from one task on one model using echo extraction (not KV-prepend) on Qwen 3.5-35B-A3B. These are hypotheses for the broader experiment, not conclusions:

- Attention single-vector (0.65) slightly beat attention multivector (0.63). Router multivector (0.60) beat router single-vector (0.18) by 3x. Hypothesis: attention prefers compression, router requires structure. Needs cross-task/model validation.
- Top-3 router layers >> all-48 (0.60 vs. 0.53). Layer selection matters for routers.
- Ordered top-k helps single-layer (+6 pts) but not multi-layer (best beta=0). Cross-layer patterns may subsume within-layer ordering.
- Union oracle: 0.86. Best fusion (interleave): 0.68. Score fusion / RRF: 0.64. Complementarity is at the document level (different systems find different docs), not query level. 18-point gap to oracle remains the main open problem.
- Interleaving beat RRF, suggesting value is in surfacing unique finds, not confirming agreement.

## Analysis (five clusters)

### 1. Response tracing

Mediation chain: uptake (β, entropy, sink displacement) → assimilation (Δz_attn, Δh_pre-MoE) → specialization (Δr, expert flips, propagated divergence). Radial vs. tangential decomposition before RMSNorm. Rank-growth through the chain. (m, d, u) decomposition under both regimes. **Token-collapse panel:** within-sequence diversity separately for m and d. If d is diverse but m is collapsed, extraction is "pool the core, index the response" (delta multivectors).

### 2. Interaction structure

Γ_l gain/angle/alignment across layers. Sender/receiver × key × value factorial. Sparse continuation runs (~10 layers) for transport/survival coefficients. Future-information normalization (suffix truncation). Propagation-onset sweep over a few start layers.

### 3. Signal quality

Shared vs. private variance across channels (CCA/regression). Router representation ladder: raw logits, softmax, ordered top-k, unordered top-k, margins, position-binned pooling, route motifs; plot quality vs. storage. Semantic-abstraction frontier: do distances track semantics or token identity? Specificity indices. Query-conditioned hard negatives: can each channel discriminate against the other channel's near-miss errors?

### 4. MoE-specific dynamics

Bias-spectrum for ~50 values per layer; derivatives, phase transitions, margins, KL. Routing-trace entropy and cross-layer MI under both conditions. Expert-output distinctness at flip points. Intrinsic dimensionality on causal m, causal d, propagated d, router distributions, value aggregates separately.

### 5. Retrieval evaluation

NanoBEIR per-task breakdown. Rate-distortion frontier across all signal types and compressions. Fusion methods: score fusion, RRF, interleaving, deep retrieval + reranking, conditional retrieval, learned gating. **Stage-asymmetric pipelines:** each signal as retriever with other as reranker; default assumption is compact document core with expanded query views (query-time generation cheap, index-time expensive). **Candidate-overlap decomposition:** split oracle gap into unique-to-A, unique-to-B, shared-but-ranked-differently, missed-by-both; inspect unique finds as document clusters, not just counts. **Router-specific operators:** if route motifs survive stronger compressions, test motif-kernel or sequence-alignment scoring, not only cosine/MaxSim.

## Decision tree

**What to extract:**
- m collapsed but d diverse → delta multivectors (pool core, index response)
- β and routing-spectrum derivatives identify high-susceptibility tokens → **susceptibility-weighted pooling or selection** for which tokens get indexed
- Sender (value) dominates + paraphrase-stable → summary-level representations
- Final-token K/V pairs compact and semantically useful → **summary-memory embeddings** (index K/V directly, potentially score via simulated cross-attention rather than cosine)
- Receiver dominates + sparse + survives transport + localizes on minimal pairs → tokenwise late interaction
- Router carries private variance + experts distinct + paraphrase-stable → expert-conditioned facets
- Router needs cross-position structure after stronger compressions → ordered trace representation
- Experts redundant or router paraphrase-unstable → routing as diagnostics/weights only

**How to combine:**
- Complementarity at document level → surface unique finds (interleave, deep-retrieve-rerank), not reward agreement
- Stage-asymmetric: router recall + attention reranking, or vice versa; delta for tie-breaking
- Oracle gap varies by task → task-adaptive or query-adaptive fusion

**Which regime:**
- Propagated-vs-causal differ in gain → accumulation; regime less critical
- Differ in angle → rewriting; path-aware extraction
- Delta mostly radial → absorbed by normalization; mostly tangential → genuine rotation

## What to expect from MoE sharpening

**If it holds:** rank jumps after routing, modest bias triggers clustered top-k transitions, propagated-vs-causal shows angular rewriting, decisive tokens in minimal pairs have small routing margins and surviving transport. Strongest form: "kinks that survive transport and improve retrieval." Extraction: value/attention core + structured router channel + sparse delta corrections. Pipeline: router recall, value reranking, delta tie-breaking.

**If it doesn't:** dense and MoE look similar; differences mostly gain not angle; router deltas predictable from hidden/value; flips between redundant experts. Best system: value/attention core + small router side-channel. Probe's value: layer/token selection and weighting.

**Note:** if single-slot probe looks promising, follow-up with 2–4 generated facet summaries could be disproportionately informative.

## Caveats

- **Signal spaces are geometrically different.** Hidden states, values, and router logits live in different spaces (residual stream, pre-W_O, simplex/logit). Naively concatenating or comparing them over/under-counts correlated directions. Whiten per-channel or use weighted similarity sums.
- **Expert identities are layer-local.** Expert 17 at layer 20 is a different function from expert 17 at layer 35. Keep layer identity explicit in any facet analysis or cluster into meta-facets from corpus statistics.
- **Near-tie instability.** Even with bf16/fp32 accumulation, very small 8th-vs-9th margins near routing boundaries may be unreliable. Report confidence bands at flip points; exclude tiny margins from strong conclusions.
- **Final-token contamination.** The prepended K/V is shaped by end-of-sequence position, prompt template, and next-token prediction bias. Controls (shuffled-sentence KV, cross-example swaps, first-token KV) exist to disentangle this, but interpret the summary memory as a position-and-prompt-conditioned object, not a clean semantic summary.

## Privacy note

Router traces reconstruct 91% of tokens. Treat as sensitive at scale.

## What we want from you

1. What can the collected data tell us about optimal embedding extraction, given three signal types, three conditions, two regimes, MoE routing, generative capability, and the understanding that routing traces are informationally rich but possibly too lexical?
2. Are there analyses or patterns we're missing?
3. What should we expect if MoE-sharpening holds vs. doesn't?
4. What are we overlooking?

**Scope note:** this experiment discovers a MoE role-assignment methodology on one Qwen-style architecture. Cross-family validation (e.g., Mixtral-style, DeepSeek-style MoE) is future work.


## Review amendments (v13.1)

These clarifications are implementation-critical and should be treated as part of the spec:

1. **Prompt-token masking is mandatory.** Constant scaffold tokens from the compression prompt should not be indexed or pooled together with content tokens. Keep a content-span mask and apply it consistently to pooling, MaxSim, token-collapse, abstraction tests, and candidate analysis.

2. **Leakage/privacy is a design frontier, not just a note.** Router traces are information-rich enough that storage format and retention policy should be treated as first-class evaluation criteria alongside abstraction, transport, and cost.

3. **Single-slot null effects are bottlenecked-null effects.** A near-zero treatment effect means the chosen summary slot did not carry the needed information; it does not imply that future-aware context would be useless. Multi-slot follow-up remains a core disambiguation step.

4. **Bridge fairness matters.** Echo vs. KV-prepend comparisons should match evaluation format as closely as possible, including prompt scaffolding, token masking, quantization protocol, and retrieval operator. Otherwise differences may reflect prompt format rather than extraction mechanics.
