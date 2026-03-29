from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    model_name: str = "Qwen/Qwen3-30B-A3B-FP8"
    tokenizer_name: str | None = None
    torch_dtype: str = "bfloat16"
    quantization: str = "fp8"
    device: str = "cuda"
    device_map: str = "cuda:0"
    attn_implementation: str = "eager"
    max_length: int = 2048
    trust_remote_code: bool = True
    output_attentions: bool = True
    preflight_max_used_memory_gib: float | None = 8.0


@dataclass
class PromptSpec:
    context_template: str = "Context: {text} Compress the Context in one word:"
    query_template: str = "Query: {text} Compress the Query in one word:"
    bridge_separator: str = "\n<kv-prepend-separator>\n"


@dataclass
class CollectionSpec:
    propagate_from_layer: int = 16
    calibration_subset_size: int = 0
    runtime_backend: str = "hf_teacher_forcing"
    capture_q_vectors: bool = False
    capture_attention_weights_for_all_layers: bool = False
    attention_weight_layers: list[int] = field(default_factory=lambda: [15, 23, 31, 39, 47])
    rope_modes: list[str] = field(default_factory=lambda: ["reuse_rotated", "rerotate_zero", "derotate"])
    default_rope_mode: str = "reuse_rotated"
    multi_slot_decode_steps: int = 3
    streaming_batch_size: int = 16
    max_batch_tokens: int = 8192
    adaptive_max_batch_tokens: bool = True
    adaptive_target_gpu_utilization: float = 0.9
    adaptive_gpu_reserve_gib: float = 2.0
    sequence_length_buckets: list[int] = field(default_factory=lambda: [128, 256, 384, 512, 768, 1024, 1536, 2048])
    pad_main_batches_to_streaming_size: bool = True
    trim_stored_sequence_length: bool = True
    store_prompt_text_in_payload: bool = False
    store_token_ids_in_payload: bool = False
    attention_backend: str = "hybrid"
    enable_attention_compile: bool = True
    attention_compile_mode: str = "default"
    attention_compile_fullgraph: bool = False
    sort_by_length: bool = True
    writer_queue_size: int = 4
    random_seed: int = 0
    save_every_batch: bool = True
    run_controls: bool = False
    run_bridge: bool = False
    controls_storage_mode: str = "summary_only"
    bridge_storage_mode: str = "summary_only"
    bias_sweep_points: int = 51
    bias_sweep_min: float = -6.0
    bias_sweep_max: float = 6.0
    main_capture_signals: list[str] = field(
        default_factory=lambda: ["attention_output", "v_raw", "router_logits", "top_k_binary"]
    )
    main_dense_layers: list[int] = field(default_factory=lambda: [15, 23, 31, 39, 47])
    main_router_layers: list[int] = field(default_factory=lambda: list(range(48)))
    main_value_layers: list[int] = field(default_factory=lambda: list(range(48)))


@dataclass
class QuantizationSpec:
    default_percentile: float = 33.0
    router_percentile: float = 90.0
    attention_percentile: float = 33.0
    hidden_percentile: float = 33.0
    value_percentile: float = 33.0
    channelwise: bool = True
    router_positive_only: bool = False


@dataclass
class EvaluationSpec:
    nanobeir_repo: str = "zeta-alpha-ai/NanoBEIR"
    dataset_names: list[str] = field(
        default_factory=lambda: [
            "scifact",
            "fiqa2018",
            "nq",
            "hotpotqa",
            "msmarco",
            "climatefever",
            "fever",
            "dbpedia",
            "nfcorpus",
            "arguana",
            "quoraretrieval",
            "scidocs",
            "touche2020",
        ]
    )
    long_context_dataset: str = "scifact"
    top_k: int = 10
    fusion_weights: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    rrf_k: int = 60
    candidate_pool_k: int = 100
    selected_layers: list[int] = field(default_factory=lambda: list(range(48)))
    topn_layers_for_grouping: int = 3
    layer_grouping_policy: str = "all_selected_layers"
    fixed_group_layers: list[int] = field(default_factory=list)


@dataclass
class OutputSpec:
    base_dir: str = "."
    artifacts_dir: str = "artifacts"
    captures_dir: str = "captures"
    evaluation_dir: str = "evaluation"
    analysis_dir: str = "analysis"
    controls_dir: str = "controls"
    bridge_dir: str = "bridge"


@dataclass
class ExperimentSpec:
    model: ModelSpec = field(default_factory=ModelSpec)
    prompts: PromptSpec = field(default_factory=PromptSpec)
    collection: CollectionSpec = field(default_factory=CollectionSpec)
    quantization: QuantizationSpec = field(default_factory=QuantizationSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def resolve_path(self, root: str | Path, *parts: str) -> Path:
        return Path(root).joinpath(*parts)


def _coerce_dataclass(dataclass_type, raw: dict[str, Any]):
    return dataclass_type(**raw) if raw else dataclass_type()


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentSpec(
        model=_coerce_dataclass(ModelSpec, raw.get("model", {})),
        prompts=_coerce_dataclass(PromptSpec, raw.get("prompts", {})),
        collection=_coerce_dataclass(CollectionSpec, raw.get("collection", {})),
        quantization=_coerce_dataclass(QuantizationSpec, raw.get("quantization", {})),
        evaluation=_coerce_dataclass(EvaluationSpec, raw.get("evaluation", {})),
        output=_coerce_dataclass(OutputSpec, raw.get("output", {})),
    )
