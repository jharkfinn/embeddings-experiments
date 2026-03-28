# KV-Prepend Probing Experiment

This folder contains a self-contained implementation of the reviewed
`kv_prepend_probing_v13_reviewed 2.md` plan and
`implementation_instructions_reviewed 2.md`.

Everything created for this experiment lives in this folder:

- `default_experiment.json`: default runtime spec
- `spec_main_hf_teacher_forcing_3tasks.json`: lean 3-task main-run contract
- `spec_main_hf_teacher_forcing_l40s_3tasks.json`: reduced-memory main-run contract for 48GB-class GPUs
- `spec_calibration_hf_3tasks.json`: exact calibration run
- `spec_controls_hf_3tasks.json`: control-only calibration run
- `spec_bridge_hf_3tasks.json`: bridge echo run
- `run_kv_prepend_experiment.py`: CLI entrypoint
- `kv_prepend_experiment/`: collection, evaluation, analysis, and runtime modules

## Target runtime

This is intended to run on the Thunder instance, not on the local laptop.
The local machine does not currently have the runtime stack installed.

## Expected Python dependencies

Use the pinned Thunder installer:

```bash
./install_thunder_env.sh
```

That script creates `.venv` and installs the exact working stack we validated on
Thunder:

- `torch==2.10.0+cu126`
- `fbgemm-gpu==1.5.0+cu126`
- `fbgemm-gpu-genai==1.5.0+cu126`
- `transformers==5.4.0`
- `datasets==4.8.4`
- `accelerate==1.13.0`

The pinned requirements live in `requirements_thunder.txt`. This is the supported
FP8 + FlexAttention stack for the active teacher-forcing path; do not substitute
older Transformers builds or unpinned CUDA wheels unless you are prepared to
revalidate the runtime contract.

`default_experiment.json` is now aligned with the active teacher-forcing main path.
Use the explicit split specs when you want calibration, controls, or bridge runs.

Every CLI run now writes a timestamped log under `artifacts/logs/` by default.
Use `--log-path` or `--log-level` to override that behavior.

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
python run_kv_prepend_experiment.py describe-run --run-name main_hf_teacher_forcing_l40s_3tasks
python run_kv_prepend_experiment.py describe-run --run-name calibration_hf_3tasks
python run_kv_prepend_experiment.py describe-run --run-name controls_hf_3tasks
python run_kv_prepend_experiment.py describe-run --run-name bridge_hf_3tasks
```

### Collect captures

Build a reproducible 3-task records file first:

```bash
python build_nanobeir_records.py --output records_nanobeir_3tasks.json
```

For the lean Thunder main run, use the default active spec or the generated
teacher-forced HF spec:

```bash
python run_kv_prepend_experiment.py \
  --spec spec_main_hf_teacher_forcing_3tasks.json \
  collect \
  --records-json records_nanobeir_3tasks.json \
  --dataset-name nanobeir_3tasks
```

For a reduced-memory 48GB-class GPU target such as an L40S, use the dedicated
L40S profile instead. It keeps the same signal families but lowers the main
batch envelope and loads with `torch_dtype="auto"`:

```bash
python run_kv_prepend_experiment.py \
  --spec spec_main_hf_teacher_forcing_l40s_3tasks.json \
  collect \
  --records-json records_nanobeir_3tasks.json \
  --dataset-name nanobeir_3tasks
```

For calibration, controls, and bridge, use the generated split specs instead of
the default main spec.

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
- `torch.compile` is applied to the packed no-weight attention kernel in the
  standard default mode, and the main run also uses fixed sequence-length
  buckets for compile stability.
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
- Evaluation now includes summary-memory signal families:
  `summary_value`, `summary_key_rot`, `summary_key_raw`, and `summary_memory`.
- Grouped layer reporting defaults to `all_selected_layers`; in-task top-N layer
  selection is kept as diagnostic output and is labeled explicitly in the saved
  evaluation results.
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
