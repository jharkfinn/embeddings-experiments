# Implementation Instructions: KV-Prepend Probing Experiment

Reference: `kv_prepend_probing_v13.md` contains the full experiment design. These instructions translate it into code.

## Overview

We are instrumenting a frozen Qwen3-30B-A3B (MoE, 48 layers) to collect internal signals under two forward-pass conditions (causal and propagating-prepend), then evaluating extracted embeddings on NanoBEIR. The model runs at FP8 for weights/activations. All router logits and attention uptake scalars are accumulated in bf16. Extracted embeddings use trinary quantization {-1, 0, +1}.

---

## Phase 0: Environment and Model Setup

**Task:** Load Qwen3-30B-A3B at FP8 on a single 80GB GPU.

- Use HuggingFace transformers. The model type is `qwen3_moe`.
- Verify the model config: 48 layers, hidden_size=2048, 32 Q heads, 4 KV heads, 128 experts, top-8 routing.
- Verify that in the HF implementation:
  - KV heads are repeated across query-head groups (4 KV → 32 Q via repeat_kv).
  - Attention output is added to the residual BEFORE post-attention RMSNorm.
  - The router applies softmax BEFORE top-k selection.
- Do NOT use the stock `output_router_logits` — it returns post-softmax values. We need pre-softmax.

---

## Phase 1: Hook Infrastructure

**Task:** Build a hook system that captures tensors at each of the 48 decoder layers during forward passes.

For each layer, we need hooks at these exact points in the Qwen3MoE decoder layer:

### 1.1 Tensors to capture

| # | Tensor | Hook location | Shape per token | Precision |
|---|--------|--------------|-----------------|-----------|
| 1 | `h_in` (pre-attention residual) | Input to the attention module | (n, 2048) | fp16 |
| 2 | `Q` (query projections) | After Q projection, before RoPE | (n, 32, d_head) | calibration subset only |
| 3 | `V_raw` (value projections) | After V projection, before attention | (n, 4, d_head) | fp16 |
| 4 | `A_causal` (attention weights) | Post-softmax attention matrix | (32, n, n) | fp16 |
| 5 | `z_attn` (attention output) | After W_O projection, BEFORE residual add | (n, 2048) | fp16 |
| 6 | `h_pre_moe` (pre-MoE state) | After attention residual add + post-attention RMSNorm | (n, 2048) | fp16 |
| 7 | `router_logits` (pre-softmax) | Inside the router, BEFORE softmax | (n, 128) | bf16 |
| 8 | `top_k_indices` (expert selections) | After top-k selection | (n, 8) as int16 | int16 |
| 9 | `final_token_K` | K projection of last token, per KV group | (4, d_head) | fp16 |
| 10 | `final_token_V` | V projection of last token, per KV group | (4, d_head) | fp16 |

### 1.2 Hook implementation notes

- Hook the router by intercepting the linear layer inside `Qwen3MoeSparseMoeBlock` BEFORE the softmax call. The HF code does `router_logits = self.gate(hidden_states)` then immediately softmaxes. Hook after `self.gate()` but before softmax.
- For `h_pre_moe`: this is the output of `self.post_attention_layernorm(residual + attn_output)`. Hook after this layernorm, before the MoE block.
- For `h_in`: this is the hidden state entering the layer, before attention. Hook at layer input.
- For `z_attn`: hook after the attention module's output projection (o_proj) but before it's added to the residual.
- For attention weights: you'll need to modify the attention forward to return weights. Set `output_attentions=True` or hook inside the attention computation after softmax.
- For `final_token_K/V`: extract K[-1] and V[-1] from the KV projections (the last token's key and value vectors, at the 4 KV-head granularity).

### 1.3 Storage

Create a `LayerCapture` dataclass per layer that holds all tensors for one condition. Use a `PassCapture` that holds 48 `LayerCapture` objects. Tensors should be moved to CPU immediately after capture to free GPU memory. Consider saving to disk per-batch if memory is tight.

---

## Phase 2: KV-Prepend Mechanism

**Task:** Implement the KV-prepend intervention that can be applied at attention time.

### 2.1 Single-slot prepend

Given a layer's Q, K, V projections for all n tokens, and the final token's K_n, V_n:

```
# K_n shape: (4, d_head) — one per KV group
# V_n shape: (4, d_head)
# Repeat K_n, V_n to match the 32 Q heads via repeat_kv (same as model does for GQA)
# Prepend to K, V along the sequence dimension:
#   K_prepend = cat([K_n.unsqueeze(seq_dim), K], dim=seq_dim)  # (n+1, ...)
#   V_prepend = cat([V_n.unsqueeze(seq_dim), V], dim=seq_dim)
# Adjust causal mask to allow all tokens to attend to position 0
# Compute attention with the augmented K, V
# The attention output for the prepend condition is z_attn_prepend
```

The summary uptake scalar β for each Q head h and token t is:
```
beta[h, t] = attn_weights_prepend[h, t, 0]  # attention weight on the prepended slot
```

We do NOT need to store the full prepend attention matrix. β plus the causal weights determine it:
```
attn_prepend[h, t, j] = (1 - beta[h, t]) * attn_causal[h, t, j]  for j > 0
```

### 2.2 RoPE handling

The final token's K already has RoPE applied at its original position (end of sequence). We test three modes:
- **Mode A (default):** reuse the rotated K as-is. The prepended key is end-position-coded.
- **Mode B:** undo the original RoPE rotation, then apply RoPE at position 0.
- **Mode C (calibration only):** use de-rotated key with no position encoding.

Implement all three; default to Mode A for the main run, test B and C on a subset.

### 2.3 Multi-slot prepend (calibration subset only)

After the compression prompt prefill, generate 3 additional tokens autoregressively (3 decode steps on existing KV cache). Now you have 4 summary tokens' KV pairs. Prepend all 4 as separate slots:
```
K_prepend = cat([K_summary_0, K_summary_1, K_summary_2, K_summary_3, K], dim=seq_dim)
# attention sequence length becomes n+4
```

Capture β per slot per head per token (shape: 4 × 32 × n).

---

## Phase 3: Two-Pass Collection

**Task:** Implement the two forward passes with local marginal probes.

### 3.1 Pass 1: Causal baseline + local marginal probes

```python
for layer_idx in range(48):
    # Normal causal attention → capture all tensors (condition: "causal")
    # Run attention AGAIN with prepend but do NOT propagate →
    #   capture z_attn_prepend, beta, router_logits_prepend, h_pre_moe_prepend
    #   (condition: "local_prepend_causal_base")
    # Continue forward pass with the CAUSAL (unprepended) outputs
```

The local prepend probe at each layer:
1. Take the same Q, K, V projections already computed
2. Prepend the final token's K_n, V_n to K, V
3. Run attention → get z_attn_prepend
4. Compute h_pre_moe_prepend = RMSNorm(h_in + z_attn_prepend)
5. Compute router_logits_prepend = gate(h_pre_moe_prepend)  [pre-softmax]
6. Capture beta, z_attn_prepend, h_pre_moe_prepend, router_logits_prepend, top_k_prepend
7. DISCARD — do not feed into the MoE block or propagate

### 3.2 Pass 2: Propagating prepend + local marginal probes

```python
for layer_idx in range(48):
    if layer_idx < 16:
        # Pure causal, identical to Pass 1 (sanity check)
        # Still run local prepend probe (should match Pass 1 exactly)
    else:
        # Run attention WITH prepend → this is the propagated path
        # Capture all tensors (condition: "propagated")
        # Also run attention WITHOUT prepend on the same (propagated) hidden states
        #   → capture (condition: "local_noprepend_propagated_base")
        # Continue forward pass with the PREPENDED outputs (propagated path)
```

At layers 16+, the local marginal on the propagated base is:
- Propagated hidden states arrive (modified by upstream prepend interventions)
- Run attention WITHOUT prepend → "what would this layer produce without its local prepend, given the propagated upstream?"
- Run attention WITH prepend → "what does this layer produce with its local prepend?"
- The difference is the propagated-base local marginal

### 3.3 Three conditions per layer (16+)

For layers 16-47 you end up with four captured states:
1. `causal` (from Pass 1)
2. `local_prepend_causal_base` (from Pass 1 — prepend at this layer, causal upstream)
3. `propagated` (from Pass 2 — full propagated state)
4. `local_noprepend_propagated_base` (from Pass 2 — no prepend at this layer, propagated upstream)

The three marginal effects are:
- Causal-base local marginal: (2) - (1)
- Propagated-base local marginal: (3) - (4)
- Full propagated vs causal: (3) - (1)

### 3.4 Summary K/V from both passes

Store final-token K/V from Pass 1 (causal-path summary) AND from Pass 2 (propagated-path summary) at each layer. These differ because in Pass 2, the final token's hidden states have been modified by upstream prepend interventions, so its K/V projections produce different summaries. Both are needed for the sender/receiver factorial.

---

## Phase 4: Compression Prompt and Sentence Batching

**Task:** Prepare inputs and manage batching.

### 4.1 Compression prompt

Wrap each input text:
```
"Context: {text} Compress the Context in one word:"
```
For queries:
```
"Query: {text} Compress the Query in one word:"
```

### 4.2 Sentence batches

Prepare three batches:

1. **Natural diversity:** ~200 sentences balanced across short (~32 tokens), medium (~256 tokens), long (~1024-2048 tokens).

2. **Semantic minimal pairs:** ~50 pairs covering negation, scope ambiguity, coreference, entity disambiguation, temporal modification, clause attachment. Each pair should differ in one semantic dimension.

3. **Semantic-abstraction battery:** ~50 triplets of (original, paraphrase, adversarial near-duplicate with similar wording but different meaning).

Mark a **calibration subset** (~50-100 sentences) for the expensive extras: Q vectors, expert outputs at flip points, multi-slot prepend, RoPE mode B/C.

---

## Phase 5: Trinary Quantization

**Task:** Implement trinary quantization for all extracted embedding vectors.

```python
def trinarize(x: Tensor, threshold_percentile: float = 33.0) -> Tensor:
    """
    Map to {-1, 0, +1}.
    Values above the upper threshold → +1
    Values below the lower threshold → -1
    Values in between → 0
    """
    pos_thresh = torch.quantile(x.abs(), threshold_percentile / 100.0)
    result = torch.zeros_like(x)
    result[x > pos_thresh] = 1
    result[x < -pos_thresh] = -1
    return result
```

The threshold percentile is a hyperparameter. For router logits (128 dims, ~8 active), something around the 90th percentile of absolute values may be appropriate (keeping ~top-8 and bottom-8 active). For attention outputs (2048 dims, denser), a lower threshold. Tune per channel.

Apply trinarization to: hidden states, pre-MoE states, router logits, attention outputs, value vectors, and any derived vectors (m, d). Keep float versions on the calibration subset.

---

## Phase 6: NanoBEIR Evaluation

**Task:** Evaluate extracted embeddings on NanoBEIR.

### 6.1 Setup

```python
from datasets import load_dataset
# Load NanoBEIR subsets
dataset_names = [
    "SciFact", "FiQA2018", "NQ", "HotpotQA", "MSMARCO",
    "ClimateFEVER", "FEVER", "DBPedia", "NFCorpus",
    "ArguAna", "QuoraRetrieval", "SciDocs", "Touche2020"
]
```

### 6.2 For each signal type and regime, evaluate:

**Single-vector (mean-pooled trinary):**
- Cosine similarity after trinarization and mean pooling over tokens
- NDCG@10 per task

**Multivector (tokenwise trinary, MaxSim):**
- Each token is a trinary vector
- Score = max over document tokens of cosine(query_token, doc_token), summed over query tokens
- NDCG@10 per task

**Signal types to evaluate independently:**
- Attention output (z_attn) at selected layers
- Pre-MoE hidden states at selected layers
- Router logits (full 128-dim) at selected layers
- Top-k expert selections (as 128-dim binary indicator) at selected layers
- Value vectors at selected layers
- Delta vectors (d = prepend - causal) at selected layers

**Evaluate all conditions:** causal, local-prepend, propagated

### 6.3 Layer selection

Don't assume which layers are best. Evaluate:
- Each individual layer
- Top-3 by individual performance
- All layers concatenated
- ID-selected layers (after running TwoNN in Phase 7)

### 6.4 Fusion evaluation

For each pair of signal types, evaluate:
- Score-level linear fusion (sweep weights)
- RRF (k=60)
- Interleaving (both orderings)
- Deep retrieval + rerank: top-100 from each, rerank union with the other signal
- Per-task breakdown of all fusion methods

Compute **oracle analysis** for each pair:
- Weak oracle (pick better system per query)
- Strong union oracle (best docs from either, optimally ranked)
- Candidate-overlap decomposition: unique-to-A, unique-to-B, shared, missed-by-both

### 6.5 Long-context stress test

Add at least one long-document retrieval task where relevant evidence appears late in the document. If no suitable NanoBEIR subset qualifies, use a slice of LoCoV1 or construct a synthetic version.

---

## Phase 7: Post-hoc Analysis (CPU)

**Task:** Implement offline analyses on collected tensors.

### 7.1 (m, d, u) decomposition

For each signal, layer, and regime:
```python
m = (x_base + x_prepend) / 2      # stable content
d = x_prepend - x_base             # context-deficit response
# u = uptake statistics: β, router entropy change, top-k flip count
```

### 7.2 Γ_l interaction quantity

For layers 16-47 where both marginals exist:
```python
delta_causal = local_prepend_causal_base - causal          # causal-base local marginal
delta_prop = propagated - local_noprepend_propagated_base   # propagated-base local marginal
# Decompose their difference:
gain = norm(delta_prop) / (norm(delta_causal) + eps)
angle = cosine_angle(delta_prop, delta_causal)
gamma_parallel = dot(delta_prop - delta_causal, normalize(delta_causal))
```

### 7.3 Radial vs. tangential decomposition

Before RMSNorm, decompose the attention-output delta:
```python
h_base = h_in + z_attn_causal                    # pre-norm state, causal
delta_z = z_attn_prepend - z_attn_causal          # attention output delta
radial = dot(delta_z, normalize(h_base)) * normalize(h_base)
tangential = delta_z - radial
# radial component mostly absorbed by RMSNorm; tangential survives to router
```

### 7.4 Token-collapse panel

For each channel and regime, measure within-sequence diversity:
```python
# Centered covariance rank of token vectors within each sequence
# Pairwise cosine similarity spread
# Compare diversity of m vs d: if d is diverse but m is collapsed → "pool core, index response"
```

### 7.5 Bias-spectrum (routing-spectrum)

For each layer, sweep bias b over ~50 values from -5 to +10:
```python
for b in bias_values:
    beta_b = sigmoid(summary_logit + b)     # summary_logit = q · k_summary / sqrt(d)
    z_b = (1 - beta_b) * z_attn_causal + beta_b * v_summary
    h_pre_moe_b = RMSNorm(h_in + z_b)       # need h_in for exact reconstruction
    router_logits_b = router_weight @ h_pre_moe_b   # (128,)
    top_k_b = topk(router_logits_b, 8)
```

Extract per-token, per-layer:
- dr/db at b=0 (sensitivity)
- d²r/db² at b=0 (curvature)
- First b where top-k changes (phase transition)
- Number of top-k changes across sweep
- 8th-vs-9th expert margin at b=0
- KL(router(b) || router(0))

### 7.6 Routing-trace structure

- Per-layer entropy of top-k selections under causal and prepend conditions
- Cross-layer mutual information matrix of top-k selections
- Per-token routing divergence: fraction of top-8 that changed between conditions
- Positional structure: is routing divergence position-dependent or uniform?

### 7.7 Intrinsic dimensionality (TwoNN)

Run TwoNN on point clouds across all 48 layers for:
- causal m, causal d, propagated d
- router logit distributions
- value-space aggregates

### 7.8 Shared-vs-private variance

```python
# For each pair of channels, fit linear regression:
#   router_d ~ hidden_d → R² tells you overlap
# Or use CCA between channel pairs
# Residual variance = private information in that channel
```

### 7.9 Specificity indices

For each alternative summary source (first-token, random-token, shuffled, cross-example, K-only, V-only):
```python
specificity = effect_own_summary - effect_alternative_summary
# Semantic specificity: own - cross-example
# Positional specificity: own - first-token
# Address/payload specificity: full - K-only, full - V-only
```

### 7.10 Sparse continuation runs

On calibration subset, for ~10 selected layers:
```python
# At layer l, branch:
#   Branch A: causal at layer l, then continue causal to layer 47
#   Branch B: prepend at layer l, then continue causal to layer 47
# Compare final-layer representations
# Fit: how much of local delta_l survived to layer 47?
transport_coeff = dot(final_delta, transported_direction) / norm(local_delta)
```

---

## Phase 8: Controls

**Task:** Run the main collection pipeline with alternative summary sources.

For each control, replace the prepended K/V with:
1. **First-token KV:** K[0], V[0] instead of K[-1], V[-1]
2. **Random-token KV:** K[rand_idx], V[rand_idx]
3. **Shuffled-sentence KV:** run a shuffled version of the input, take its final-token K/V
4. **Cross-example KV:** take final-token K/V from a different sentence in the batch
5. **K-only:** prepend real K, use matched-norm random V (not zero)
6. **V-only:** prepend matched-norm random K, use real V

"Matched-norm" means the null vector has the same L2 norm as the real vector it replaces, but a random direction.

Controls can run on a reduced subset, not the full batch.

---

## Phase 9: Bridge Experiment

**Task:** Compare echo extraction vs. KV-prepend on the same model and benchmark slice.

On a SciFact (or NanoBEIR subset) slice:
1. Run echo extraction (text + separator + text, take second copy signals)
2. Run KV-prepend extraction (compression prompt, prepend at selected layers)
3. Same trinarization, same layers, same evaluation
4. Compare: do the same channels play the same roles under both extraction methods?

---

## Implementation Order

1. **Phase 0:** Model loading, config verification
2. **Phase 1:** Hook infrastructure (most complex, get this right first)
3. **Phase 2:** KV-prepend mechanism (Mode A only initially)
4. **Phase 4:** Sentence batching and prompt formatting
5. **Phase 3:** Two-pass collection (Pass 1 only first, verify tensors, then add Pass 2)
6. **Phase 5:** Trinary quantization
7. **Phase 6:** NanoBEIR evaluation (start with single-signal, single-layer, one task to validate pipeline)
8. **Phase 7:** Post-hoc analyses (incremental, start with m/d/u and Γ_l)
9. **Phase 8:** Controls (after main pipeline validated)
10. **Phase 9:** Bridge experiment (after main results)

---

## Key Implementation Risks

- **Memory:** 48 layers × 10 tensors × multiple conditions. Move to CPU aggressively. Consider streaming to disk per batch. The biggest tensors are attention weights (32 × n × n per layer) — consider storing only for selected layers or only β + causal weights.
- **Correctness of hook points:** The difference between "post-attention pre-residual" and "post-attention post-residual" is one addition. Getting these wrong invalidates everything. Write explicit tests comparing hooked values against manually computed values on tiny inputs.
- **Router softmax ordering:** Qwen3 does softmax-then-top-k. Verify by comparing `router_logits` captured pre-softmax against the stock model's `router_logits` output (which is post-softmax). They should differ by exactly a softmax.
- **RoPE in prepend:** Make sure the prepended key either keeps its original rotation (Mode A) or gets correctly re-rotated. Incorrect RoPE handling silently produces garbage.
- **GQA repeat:** The 4 KV heads must be repeated to match 32 Q heads the same way the model does it internally. Use the same `repeat_kv` function.
- **Pass 2 propagation:** The prepend at layer l must use the PROPAGATED final-token K/V (computed from the modified hidden states), not the Pass-1 causal K/V. The summary itself changes under propagation.
