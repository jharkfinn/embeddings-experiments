# KV-Prepend Probing Experiment

This folder contains a self-contained implementation of the reviewed
`kv_prepend_probing_v13_reviewed 2.md` plan and
`implementation_instructions_reviewed 2.md`.

Everything created for this experiment lives in this folder:

- `default_experiment.json`: default runtime spec
- `spec_main_hf_teacher_forcing_3tasks.json`: lean 3-task main-run contract
- `spec_calibration_hf_3tasks.json`: exact calibration run
- `spec_controls_hf_3tasks.json`: control-only calibration run
- `spec_bridge_hf_3tasks.json`: bridge echo run
- `run_kv_prepend_experiment.py`: CLI entrypoint
- `kv_prepend_experiment/`: collection, evaluation, analysis, and runtime modules

## Target runtime

This is intended to run on the Thunder instance, not on the local laptop.
The local machine does not currently have the runtime stack installed.

## Expected Python dependencies

Install from `requirements_thunder.txt`, then add any GPU-specific wheels needed by
your CUDA / PyTorch build.

Minimum expected packages:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `sentencepiece`
- `safetensors`

For FP8 loading, use a Transformers build that exposes an FP8 quantization config.
The loader checks this at runtime and fails fast if the build does not support it.

## CLI

### Verify model contract

```bash
python run_kv_prepend_experiment.py verify-model
```

### Write split Thunder specs

```bash
python run_kv_prepend_experiment.py write-split-specs
```

### Describe what a run collects

```bash
python run_kv_prepend_experiment.py describe-run --run-name main_hf_teacher_forcing_3tasks
python run_kv_prepend_experiment.py describe-run --run-name calibration_hf_3tasks
python run_kv_prepend_experiment.py describe-run --run-name controls_hf_3tasks
python run_kv_prepend_experiment.py describe-run --run-name bridge_hf_3tasks
```

### Collect captures

```bash
python run_kv_prepend_experiment.py \
  --spec default_experiment.json \
  collect \
  --records-json records.json \
  --dataset-name scifact_slice \
  --run-controls \
  --run-bridge
```

For the lean Thunder main run, use the generated teacher-forced HF spec:

```bash
python run_kv_prepend_experiment.py \
  --spec spec_main_hf_teacher_forcing_3tasks.json \
  collect \
  --records-json records.json \
  --dataset-name scifact
```

`records.json` is a list of objects with:

```json
[
  {"text_id": "doc-1", "kind": "doc", "text": "Document text..."},
  {"text_id": "query-1", "kind": "query", "text": "Query text..."}
]
```

### Evaluate one signal

```bash
python run_kv_prepend_experiment.py \
  --spec default_experiment.json \
  evaluate \
  --task-name scifact \
  --signal attention_output \
  --condition causal \
  --pass-name pass1 \
  --layers 35
```

### Evaluate the task suite

```bash
python run_kv_prepend_experiment.py \
  --spec default_experiment.json \
  evaluate-suite \
  --task-name scifact
```

### Run post-hoc capture analysis

```bash
python run_kv_prepend_experiment.py \
  --spec default_experiment.json \
  analyze
```

## Notes

- The default main path is now a batched teacher-forced HF collector.
- The older vLLM backend has been moved under `legacy/vllm_backend/`.
- The main HF path is optimized around dynamic token-budget batching, manual
  padding of pretokenized prompts, pass-2 reuse of the pass-1 prefix states,
  and no attention-weight materialization on lean branches.
- The active main fast path uses a strict `flex_packed` attention backend for
  no-weight collection, with packed per-document attention segments and no
  eager/CPU fallback path.
- `torch.compile` is applied to the packed no-weight attention kernel, and the
  main run also uses fixed sequence-length buckets for compile stability.
- Calibration, controls, and bridge runs stay on the exact `sdpa` path because
  they need dense attention weights and richer exact captures.
- The collector runs decoder layers directly so it can compute causal and prepend
  variants from the same Q/K/V projections and store fp8-first capture tensors.
- Encoding stays an evaluation-time decision: signed trinary, positive-only router
  trinary, and top-k indicators are derived from stored fp8 tensors.
- Pass 1 and Pass 2 are both implemented.
- Multi-slot prepend is implemented through greedy continuation re-forwards on the
  calibration subset.
- Controls include first-token, random-token, shuffled-sentence, cross-example,
  K-only, and V-only, with matched-norm and zero nulls.
- Calibration captures include online bias-spectrum signatures and sparse routing
  transition records.
- The bridge experiment reuses the same collector with echo-formatted prompts.
- Controls and bridge default to summary-only storage, so they keep comparative
  analysis outputs without writing full capture bundles.
- Calibration subset selection is now deterministic and order-invariant: a
  stable hash keyed by `seed + dataset_name + kind + text_id + text`, stratified
  by dataset and doc/query kind. The selected IDs are written under
  `artifacts*/calibration_manifests/`.
- Non-calibration captures now honor `main_dense_layers`, `main_router_layers`,
  and `main_capture_signals`, so lean corpus runs only retain the selected retrieval
  tensors instead of the full calibration tensor set.
- In the current lean setup, router signals are kept for all 48 layers while
  dense `attention_output` and `pre_moe` captures stay on a selected shortlist.

## Outputs

By default the CLI writes under the folder given by `--root`:

- `captures/`
- `evaluation/`
- `analysis/`
- `artifacts/`
- `controls/`
- `bridge/`
