# Implementation Instructions: KV-Prepend Probing Experiment (reviewed)

Reference: `kv_prepend_probing_v13.md` contains the full experiment design. These instructions translate it into code, with a few corrections to make the collection mathematically exact and the evaluation fair.

## Overview

We are instrumenting a frozen Qwen3-30B-A3B (MoE, 48 layers) to collect internal signals under two forward-pass conditions (causal and propagating-prepend), then evaluating extracted embeddings on NanoBEIR. The model runs with FP8 weights if your stack supports it, but all collected diagnostic tensors that drive analysis should be accumulated in at least bf16.

Extracted embeddings use trinary quantization {-1, 0, +1}. Float baselines are retained on the calibration subset for rate-distortion analysis.

---

## Phase 0: Environment and Model Setup

**Task:** Load Qwen3-30B-A3B on a single 80GB GPU and pin down the exact runtime stack.

- Use HuggingFace transformers with an explicitly specified attention backend and quantization backend.
- Record the exact software stack used for FP8 execution. "HF + FP8" is not specific enough for reproducibility.
- Verify the model config: 48 layers, hidden_size=2048, 32 Q heads, 4 KV heads, 128 experts, top-8 routing.
- Verify that in the HF implementation:
  - KV heads are repeated across query-head groups via `repeat_kv`.
  - Attention output is added to the residual **before** post-attention RMSNorm.
  - The router applies softmax **before** top-k selection.
- Do **not** use the stock `output_router_logits` path for analysis. We need pre-softmax router logits.

---

## Phase 1: Hook Infrastructure

**Task:** Build a hook system that captures tensors at each of the 48 decoder layers during forward passes.

### 1.1 Tensors to capture

| # | Tensor | Hook location | Shape per token | Storage |
|---|--------|--------------|-----------------|---------|
| 1 | `resid_pre_attn` (decoder-layer input residual) | Decoder-layer input, **before** input_layernorm | (n, 2048) | fp8; calibration bf16 |
| 2 | `Q_pre_rope` | After Q projection and q_norm, before RoPE | (n, 32, d_head) | calibration subset only, bf16 |
| 3 | `V_raw` | After V projection, before attention | (n, 4, d_head) | fp8; calibration bf16 |
| 4 | `A_causal_slice` | Post-softmax causal attention weights, **only needed slices** | see note | bf16, calibration/selected layers only |
| 5 | `z_attn` | After `o_proj`, before residual add | (n, 2048) | fp8; calibration bf16 |
| 6 | `h_pre_moe` | After attention residual add + post-attention RMSNorm | (n, 2048) | fp8; calibration bf16 |
| 7 | `router_logits_pre_softmax` | Inside router, before softmax | (n, 128) | fp8; calibration bf16 |
| 8 | `top_k_indices` | After top-k selection | (n, 8) | **int8** (values 0-127) |
| 9 | `final_token_K_raw` | Last-token K before RoPE, per KV group | (4, d_head) | fp8; calibration bf16 |
| 10 | `final_token_K_rot` | Last-token K after RoPE, per KV group | (4, d_head) | fp8; calibration bf16 |
| 11 | `final_token_V` | Last-token V per KV group | (4, d_head) | fp8; calibration bf16 |
| 12 | `position_ids` or RoPE cos/sin slice | Needed to reconstruct alternative RoPE modes | small | calibration subset or all if cheap |

### 1.2 Hook implementation notes

- `resid_pre_attn` is **not** the input to the attention submodule after input-layernorm. It is the decoder-layer input residual **before** the input-layernorm. This is the state you need for exact residual-add and RMSNorm reconstruction.
- If you also want to inspect attention-internal states, treat them as separate tensors. Do not overload `h_in` to mean both things.
- Hook the router by intercepting the linear map before softmax inside `Qwen3MoeTopKRouter`.
- For `h_pre_moe`, hook after `post_attention_layernorm`, before the MoE block.
- For `z_attn`, hook after `o_proj`, before the residual add.
- For RoPE experiments, store enough to reconstruct all three modes. The easiest option is to store both raw and rotated last-token keys plus position information.
- **Do not store the full causal attention matrix for all layers/lengths.** For most runs, store only:
  - the summary uptake `beta`
  - any query-token slices required for AlignedWVA-style weighting
  - any additional slices needed for sink-displacement analysis
  Full `A_causal` should be limited to a calibration subset or selected layers, otherwise memory blows up.

### 1.3 Storage

Create a `LayerCapture` dataclass per layer and condition, wrapped by a `PassCapture` with 48 layers.

Move tensors to CPU quickly, but do not assume that "move to CPU" solves the storage problem for attention weights. For long contexts, even fp16 attention matrices are too large to store indiscriminately.

Persist per-batch to disk in a chunked format (e.g. safetensors, zarr, parquet for small metadata) with a schema version.

---

## Phase 2: KV-Prepend Mechanism

**Task:** Implement the KV-prepend intervention at attention time.

### 2.1 Single-slot prepend

Given a layer's Q, K, V projections for all n tokens, and the final token's summary K/V:

- Prepend the summary K/V along the sequence dimension.
- Adjust the causal mask so every token may attend to the prepended slot.
- Reuse the same GQA expansion path as the model (`repeat_kv`).
- The summary uptake scalar is:
  `beta[h, t] = attn_weights_prepend[h, t, 0]`

You do **not** need the full prepend attention matrix. Once the original logits are held fixed and only one new slot is added, the old probabilities are just rescaled by `1 - beta`.

### 2.2 RoPE handling

The prepend intervention should support three modes:

- **Mode A:** reuse the already-rotated end-position key as-is.
- **Mode B:** de-rotate the end-position key, then re-rotate it to position 0.
- **Mode C:** de-rotate the key and leave it unrotated.

To make these modes reconstructible, store both the last-token raw key and the last-token rotated key, plus position info or RoPE cos/sin.

### 2.3 Multi-slot prepend (calibration subset only)

Generate 3 continuation tokens after the compression prompt and prepend all 4 summary slots.

This is cheap on the decode side, but it is **not** "nearly free" end-to-end, because attention and analysis now operate over more slots at every probed layer. Keep this confined to a calibration subset.

Capture per-slot uptake `beta[slot, head, token]`.

---

## Phase 3: Two-Pass Collection

**Task:** Implement the two forward passes with local marginal probes.

### 3.1 Pass 1: Causal baseline + local marginal probes

For each layer:

1. Run the normal causal attention path and capture the causal tensors.
2. Re-run local attention with the prepended summary **without propagation**.
3. Compute:
   - `z_attn_prepend`
   - `h_pre_moe_prepend = RMSNorm(resid_pre_attn + z_attn_prepend)`
   - `router_logits_prepend`
   - `top_k_prepend`
   - `beta`
4. Discard the prepend branch after capture; continue the model on the causal path.

### 3.2 Pass 2: Propagating prepend + local marginal probes

For layers 0-15:
- Pure causal forward, plus the same local prepend probe used in Pass 1 for sanity checks.

For layers 16-47:
- Run the propagated path **with** prepend and continue forward on that propagated state.
- Also run the local **without-prepend** branch on the same propagated incoming state.
- Capture both so that the propagated-base local marginal is well-defined.

### 3.3 Conditions

At layers 16-47 you will have:

1. `causal`
2. `local_prepend_causal_base`
3. `propagated`
4. `local_noprepend_propagated_base`

The derived differences are:

- causal-base local marginal = (2) - (1)
- propagated-base local marginal = (3) - (4)
- full propagated effect = (3) - (1)

### 3.4 Sender-side summaries from both passes

Store summary K/V from both passes. In Pass 2 the summary memory itself changes, so sender/receiver analyses need both the causal-path and propagated-path summaries.

---

## Phase 4: Prompting, Content Masks, and Sentence Batching

### 4.1 Compression prompt

Wrap inputs as planned, but define a **content-span mask** for every sequence.

This mask should exclude scaffold tokens such as:
- `"Context:"`
- `"Query:"`
- `"Compress the ... in one word:"`

Prompt scaffold tokens should be excluded from:
- pooled embeddings
- tokenwise retrieval vectors
- token-collapse / diversity statistics
- semantic-abstraction analyses

Generated summary tokens should be tracked separately.

### 4.2 Sentence batches

Prepare:

1. natural diversity batch
2. semantic minimal-pair battery
3. semantic-abstraction battery

Mark a calibration subset for:
- `Q_pre_rope`
- full or partial attention slices
- expert outputs near flip points
- multi-slot prepend
- alternative RoPE modes
- higher-precision retention

---

## Phase 5: Storage Format and Evaluation-Time Encoding

**All vectors are stored at fp8 (1 byte/dim).** Top-k expert indices are stored as int8 (values 0-127). β is stored in bf16. Calibration subset retains bf16 for router logits and other diagnostic tensors.

**Quantization/encoding is applied at evaluation time**, not at collection time. This enables testing different schemes from the same stored data without rerunning the forward pass.

Encoding schemes to compare at evaluation time:

```python
def trinarize_by_fraction(x: torch.Tensor, nonzero_fraction: float) -> torch.Tensor:
    thresh = torch.quantile(x.abs(), 1.0 - nonzero_fraction)
    y = torch.zeros_like(x)
    y[x > thresh] = 1
    y[x < -thresh] = -1
    return y
```

Critical details:

- Calibrate thresholds per **channel × layer family**, not independently per sentence, or comparability will suffer.
- Evaluate both **pool-then-quantize** and **quantize-then-pool** for single-vector baselines.
- For router features, do not assume the bottom-k negative logits are as useful as the positive top-k. Compare:
  - full signed trinary {-1, 0, +1}
  - positive-only trinary {0, +1} (zeros out negative logits; may be better for routers where negative = irrelevant, not informative)
  - top-k indicator {0, 1}
  - ordered top-k + margins
  - pooled route motifs
- For attention outputs, full signed trinary is likely correct since both signs are meaningful in residual stream space.

---

## Phase 6: NanoBEIR Evaluation

### 6.1 Dataset plumbing

Use the canonical NanoBEIR dataset identifiers expected by the evaluator stack, e.g. lowercase names such as:
- `scifact`
- `fiqa2018`
- `nq`
- `hotpotqa`
- `msmarco`
- `climatefever`
- `fever`
- `dbpedia`
- `nfcorpus`
- `arguana`
- `quoraretrieval`
- `scidocs`
- `touche2020`

Do not assume that mixed-case names passed to `load_dataset` will map correctly.

### 6.2 Representation families to evaluate

For each signal type and regime, evaluate:

**Single-vector**
- mean or hybrid pooled, trinary
- both pool-then-quantize and quantize-then-pool where relevant
- float baselines on calibration subset for rate-distortion comparison

**Multivector**
- tokenwise representations with MaxSim
- prompt tokens masked out
- optional susceptibility-based token selection

**Signal types**
- `z_attn`
- `h_pre_moe`
- router logits
- ordered top-k / unordered top-k / margins
- value-derived vectors
- delta vectors

**Regimes**
- causal
- local prepend
- propagated

### 6.3 Layer selection

Evaluate:
- each layer
- top-k layers by held-out performance
- concatenated multi-layer variants
- ID-selected layers

### 6.4 Fusion evaluation

Evaluate:
- score-level linear fusion
- RRF
- interleaving
- candidate union + rerank
- stage-asymmetric pipelines (A retrieves, B reranks; B retrieves, A reranks)

Do oracle analysis per pair:
- weak oracle
- strong union oracle
- candidate-overlap decomposition

### 6.5 Long-context stress test

Add at least one long-document retrieval slice where decisive evidence appears late.

### 6.6 Query/document asymmetry

Run a dedicated short-query / long-document slice. Query-time generation is cheap and index-time generation is not; the winning representation may be asymmetric.

---

## Phase 7: Post-hoc Analysis

### 7.1 (m, d, u) decomposition

For each signal:
- `m = (x_base + x_treated) / 2`
- `d = x_treated - x_base`
- `u = uptake / routing-change / entropy statistics`

### 7.2 Interaction quantity

Keep the gain / angle / aligned-shift decomposition, but whiten or normalize per channel before cross-layer comparisons when possible.

### 7.3 Radial vs tangential decomposition

Use the pre-norm state:
- `h_base = resid_pre_attn + z_attn_causal`
- `delta_z = z_attn_treated - z_attn_causal`

Project `delta_z` into radial and tangential components relative to `h_base`.

### 7.4 Token-collapse panel

Measure within-sequence diversity separately for `m` and `d`.

### 7.5 Bias-spectrum (routing-spectrum) — computed ONLINE, calibration subset only

**This is not an offline analysis.** During the forward pass, W_O, the gate, and the layernorm are all live on GPU. Run the ~50-point sweep inline at each layer and store only compact signatures (~8 floats per token per layer: sensitivity, curvature, first phase transition, flip count, margin, KL). No weight extraction or offline reconstruction needed. ~17% extra compute on ~100 docs.

This part needs a more exact formula than a naive post-o_proj interpolation.

**Do not use**
`beta_b = sigmoid(q·k_summary/sqrt(d) + b)`
unless that symbol already means the **summary-vs-original-logsumexp margin**.

Instead, for the original summary slot:
- either use the observed `beta_0` and set `margin_0 = logit(beta_0)`
- then sweep `beta_b = sigmoid(margin_0 + b)`

or explicitly compute:
- `margin_0 = l_summary - logsumexp(l_original_tokens)`

Then reconstruct the treated attention output in **headwise pre-`o_proj` space**:

1. compute causal headwise weighted values from `A_causal_slice` and `V_raw`
2. mix those with the summary value using `beta_b`
3. concatenate heads
4. apply `o_proj`
5. add `resid_pre_attn`
6. apply RMSNorm
7. apply router linear map

The previous shortcut
`z_b = (1 - beta_b) * z_attn_causal + beta_b * v_summary`
is not exact unless `v_summary` has already been mapped into the same post-`o_proj` space with the correct per-head mixing.

Store per-token, per-layer:
- Compact signatures (~8 floats): sensitivity, curvature, first phase transition, flip count, margin, KL
- **Sparse transition records**: for each top-k change during the sweep, store `(b_critical, expert_out, expert_in, rank_position)`. Typically 2-5 transitions per token, ~8 bytes each. Negligible storage, directly identifies which routing boundaries the prepend crosses and which experts are competing.

### 7.6 Routing-trace structure

Keep:
- entropy
- cross-layer MI
- per-token divergence
- positional dependence

### 7.7 Intrinsic dimensionality

Run TwoNN separately on:
- causal `m`
- causal `d`
- propagated `d`
- router objects
- value aggregates

### 7.8 Shared vs private variance

Use regression / CCA / SVCCA as planned, but report **utility-conditioned** private variance when possible: private variance that actually improves retrieval or abstraction metrics is more interesting than purely geometric residuals.

### 7.9 Specificity indices

Keep as planned.

### 7.10 Sparse continuation runs

The original note left `transported_direction` undefined.

For each selected layer, compute at least:
- final-delta norm
- cosine between local delta and final delta in a common comparison space
- variance explained by the best linear transport map on a calibration subset

Also add a propagated-base continuation variant for a small subset, not just causal-base continuation, so you can test whether transport differs by regime.

---

## Phase 8: Controls

Keep the control family, but report both:
- matched-norm random nulls
- zero nulls

This matters for K-only / V-only interpretation.

---

## Phase 9: Bridge Experiment

Compare echo extraction vs KV-prepend on the same benchmark slice, but also equalize:
- prompt scaffolding
- token masking
- quantization protocol
- retrieval operator

Otherwise the bridge may mostly measure prompt-format differences.

---

## Key Implementation Risks

- **Hook ambiguity:** `resid_pre_attn` versus attention-submodule input must not be conflated.
- **RoPE reconstructibility:** store enough state to reproduce all modes.
- **Attention-matrix blowup:** full `A_causal` is too large for general use.
- **Bias-spectrum exactness:** reconstruct in headwise pre-`o_proj` space, not by mixing post-`o_proj` tensors with raw values.
- **Prompt contamination:** unmasked scaffold tokens will distort both pooling and MaxSim.
- **Dataset naming / loader drift:** use the canonical NanoBEIR identifiers supported by the evaluator stack.
- **Quantization confounds:** float baselines on the calibration subset help distinguish model effects from trinary artifacts in diagnostic analyses.
- **Propagation semantics:** propagated summaries must come from the propagated path, not reused from the causal pass.
