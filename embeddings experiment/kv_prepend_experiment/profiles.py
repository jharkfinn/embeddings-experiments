from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import ExperimentSpec

THREE_TASKS = ["scifact", "fiqa2018", "quoraretrieval"]
MAIN_DENSE_LAYERS = [15, 23, 31, 39, 47]
MAIN_ROUTER_LAYERS = list(range(48))
MAIN_SIGNALS = ["attention_output", "pre_moe", "router_logits", "top_k_binary"]


def _clone_spec(spec: ExperimentSpec) -> ExperimentSpec:
    return copy.deepcopy(spec)


def build_main_hf_teacher_forcing_spec(base: ExperimentSpec) -> ExperimentSpec:
    spec = _clone_spec(base)
    spec.collection.runtime_backend = "hf_teacher_forcing"
    spec.collection.calibration_subset_size = 0
    spec.collection.capture_q_vectors = False
    spec.collection.capture_attention_weights_for_all_layers = False
    spec.collection.run_controls = False
    spec.collection.run_bridge = False
    spec.collection.streaming_batch_size = max(spec.collection.streaming_batch_size, 16)
    spec.collection.max_batch_tokens = max(spec.collection.max_batch_tokens, 8192)
    spec.collection.attention_backend = "flex_packed"
    spec.collection.sequence_length_buckets = [128, 256, 384, 512, 768, 1024, 1536, 2048]
    spec.collection.pad_main_batches_to_streaming_size = True
    spec.collection.enable_attention_compile = True
    spec.collection.attention_compile_mode = "reduce-overhead"
    spec.collection.attention_compile_fullgraph = False
    spec.collection.main_dense_layers = list(MAIN_DENSE_LAYERS)
    spec.collection.main_router_layers = list(MAIN_ROUTER_LAYERS)
    spec.collection.main_capture_signals = list(MAIN_SIGNALS)
    spec.evaluation.dataset_names = list(THREE_TASKS)
    spec.evaluation.selected_layers = list(MAIN_ROUTER_LAYERS)
    spec.output.artifacts_dir = "artifacts_main"
    spec.output.captures_dir = "captures_main"
    spec.output.evaluation_dir = "evaluation_main"
    spec.output.analysis_dir = "analysis_main"
    spec.output.controls_dir = "controls_main"
    spec.output.bridge_dir = "bridge_main"
    return spec


def build_calibration_hf_spec(base: ExperimentSpec) -> ExperimentSpec:
    spec = _clone_spec(base)
    spec.collection.runtime_backend = "hf"
    spec.collection.calibration_subset_size = max(spec.collection.calibration_subset_size, 100)
    spec.collection.attention_backend = "sdpa"
    spec.collection.enable_attention_compile = False
    spec.collection.capture_q_vectors = True
    spec.collection.capture_attention_weights_for_all_layers = False
    spec.collection.run_controls = False
    spec.collection.run_bridge = False
    spec.collection.streaming_batch_size = 2
    spec.collection.max_batch_tokens = min(spec.collection.max_batch_tokens, 4096)
    spec.evaluation.dataset_names = list(THREE_TASKS)
    spec.evaluation.selected_layers = list(range(48))
    spec.output.artifacts_dir = "artifacts_calibration"
    spec.output.captures_dir = "captures_calibration"
    spec.output.evaluation_dir = "evaluation_calibration"
    spec.output.analysis_dir = "analysis_calibration"
    spec.output.controls_dir = "controls_calibration"
    spec.output.bridge_dir = "bridge_calibration"
    return spec


def build_controls_hf_spec(base: ExperimentSpec) -> ExperimentSpec:
    spec = build_calibration_hf_spec(base)
    spec.collection.run_controls = True
    spec.collection.run_bridge = False
    spec.collection.controls_storage_mode = "summary_only"
    spec.output.artifacts_dir = "artifacts_controls"
    spec.output.captures_dir = "captures_controls"
    spec.output.evaluation_dir = "evaluation_controls"
    spec.output.analysis_dir = "analysis_controls"
    spec.output.controls_dir = "controls"
    spec.output.bridge_dir = "bridge_controls"
    return spec


def build_bridge_hf_spec(base: ExperimentSpec) -> ExperimentSpec:
    spec = build_calibration_hf_spec(base)
    spec.collection.run_controls = False
    spec.collection.run_bridge = True
    spec.collection.bridge_storage_mode = "summary_only"
    spec.output.artifacts_dir = "artifacts_bridge"
    spec.output.captures_dir = "captures_bridge"
    spec.output.evaluation_dir = "evaluation_bridge"
    spec.output.analysis_dir = "analysis_bridge"
    spec.output.controls_dir = "controls_bridge"
    spec.output.bridge_dir = "bridge"
    return spec


def named_run_specs(base: ExperimentSpec) -> dict[str, ExperimentSpec]:
    main_spec = build_main_hf_teacher_forcing_spec(base)
    return {
        "main_hf_teacher_forcing_3tasks": main_spec,
        "calibration_hf_3tasks": build_calibration_hf_spec(base),
        "controls_hf_3tasks": build_controls_hf_spec(base),
        "bridge_hf_3tasks": build_bridge_hf_spec(base),
    }


def _common_collection_contract() -> dict[str, Any]:
    return {
        "passes": ["pass1", "pass2"],
        "conditions": [
            "causal",
            "local_prepend_causal_base",
            "propagated",
            "local_noprepend_propagated_base",
        ],
        "prompt_masking": "content_token_mask is stored and scaffold tokens are excluded during evaluation/analysis",
    }


def describe_run(spec: ExperimentSpec, run_name: str) -> dict[str, Any]:
    common = _common_collection_contract()
    if run_name == "main_hf_teacher_forcing_3tasks":
        return {
            "run_name": run_name,
            "runtime_backend": spec.collection.runtime_backend,
            "datasets": list(spec.evaluation.dataset_names),
            "captures": {
                **common,
                "dense_layers": list(spec.collection.main_dense_layers),
                "router_layers": list(spec.collection.main_router_layers),
                "signals": list(spec.collection.main_capture_signals),
                "execution_mode": "batched teacher-forced HF forward passes; no generation backend",
                "storage_policy": "fp8-first lean corpus cache; no calibration-only tensors",
                "performance": {
                    "attention_backend": spec.collection.attention_backend,
                    "attention_compile": bool(spec.collection.enable_attention_compile),
                    "attention_compile_mode": spec.collection.attention_compile_mode,
                    "attention_compile_fullgraph": bool(spec.collection.attention_compile_fullgraph),
                    "sequence_length_buckets": list(spec.collection.sequence_length_buckets),
                    "pad_main_batches_to_streaming_size": bool(spec.collection.pad_main_batches_to_streaming_size),
                },
                "omits": [
                    "q_pre_rope",
                    "attention_weights",
                    "beta",
                    "final_token_k_raw",
                    "final_token_k_rot",
                    "final_token_v",
                    "position_ids",
                    "multi_slot_summaries",
                    "bias_spectrum_signature",
                    "bias_spectrum_transitions",
                ],
            },
            "outputs": {
                "captures_dir": spec.output.captures_dir,
                "evaluation_dir": spec.output.evaluation_dir,
                "analysis_dir": spec.output.analysis_dir,
            },
        }
    if run_name == "calibration_hf_3tasks":
        return {
            "run_name": run_name,
            "runtime_backend": spec.collection.runtime_backend,
            "datasets": list(spec.evaluation.dataset_names),
            "captures": {
                **common,
                "layers": "all 48 layers",
                "signals": [
                    "resid_pre_attn",
                    "q_pre_rope",
                    "v_raw",
                    "attention_weights (selected calibration layers)",
                    "beta",
                    "attention_output",
                    "pre_moe",
                    "router_logits",
                    "top_k_binary",
                    "final_token_k_raw",
                    "final_token_k_rot",
                    "final_token_v",
                    "position_ids",
                ],
                "extra_calibration_only": [
                    "multi_slot_summaries",
                    "bias_spectrum_signature",
                    "bias_spectrum_transitions",
                    "float baselines / rate-distortion inputs",
                ],
                "storage_policy": "fp8 for stored vectors, bf16 for calibration-sensitive diagnostics",
            },
            "outputs": {
                "captures_dir": spec.output.captures_dir,
                "evaluation_dir": spec.output.evaluation_dir,
                "analysis_dir": spec.output.analysis_dir,
            },
        }
    if run_name == "controls_hf_3tasks":
        return {
            "run_name": run_name,
            "runtime_backend": spec.collection.runtime_backend,
            "datasets": list(spec.evaluation.dataset_names),
            "captures": {
                "uses_calibration_capture_contract": True,
                "control_modes": [
                    "first_token",
                    "random_token",
                    "k_only",
                    "v_only",
                    "k_only_zero",
                    "v_only_zero",
                    "shuffled_sentence",
                    "cross_example",
                    "prompt_continuation",
                ],
                "storage_mode": spec.collection.controls_storage_mode,
            },
            "outputs": {
                "controls_dir": spec.output.controls_dir,
                "analysis_dir": spec.output.analysis_dir,
            },
        }
    if run_name == "bridge_hf_3tasks":
        return {
            "run_name": run_name,
            "runtime_backend": spec.collection.runtime_backend,
            "datasets": list(spec.evaluation.dataset_names),
            "captures": {
                "uses_calibration_capture_contract": True,
                "intervention": "echo-formatted bridge prompts using the bridge separator",
                "storage_mode": spec.collection.bridge_storage_mode,
            },
            "outputs": {
                "bridge_dir": spec.output.bridge_dir,
                "analysis_dir": spec.output.analysis_dir,
            },
        }
    raise ValueError(f"Unknown run name: {run_name}")


def write_named_specs(base: ExperimentSpec, root: str | Path) -> list[Path]:
    root = Path(root)
    paths: list[Path] = []
    for name, spec in named_run_specs(base).items():
        path = root / f"spec_{name}.json"
        spec.save(path)
        paths.append(path)
    return paths
