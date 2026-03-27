from __future__ import annotations

import json
import os
from pathlib import Path

from kv_prepend_experiment.config import ExperimentSpec
from kv_prepend_experiment.prompts import build_prompt_examples


def import_vllm():
    import vllm  # type: ignore

    return vllm


def _annotate_examples(examples, tokenizer, spec: ExperimentSpec):
    for example in examples:
        prefix = "Query: " if example.kind == "query" else "Context: "
        suffix = " Compress the Query in one word:" if example.kind == "query" else " Compress the Context in one word:"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        text_ids = tokenizer.encode(example.text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        full_ids = (prefix_ids + text_ids + suffix_ids)[: spec.model.max_length]
        content_mask = ([0] * len(prefix_ids) + [1] * len(text_ids) + [0] * len(suffix_ids))[: len(full_ids)]
        example.prompt_token_ids = full_ids
        example.content_token_mask = content_mask
        example.token_count = len(full_ids)
    return examples


def _prompt_key(token_ids: list[int]) -> str:
    import hashlib
    import numpy as np

    arr = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _iter_batches(examples, spec: ExperimentSpec):
    ordered = list(examples)
    if spec.collection.sort_by_length:
        ordered.sort(key=lambda ex: int(ex.token_count or 0), reverse=True)
    current = []
    current_max = 0
    for example in ordered:
        proposed_max = max(current_max, int(example.token_count or 0))
        proposed_size = len(current) + 1
        exceeds_size = proposed_size > spec.collection.streaming_batch_size
        exceeds_tokens = proposed_size * proposed_max > spec.collection.max_batch_tokens
        if current and (exceeds_size or exceeds_tokens):
            yield current
            current = [example]
            current_max = int(example.token_count or 0)
        else:
            current.append(example)
            current_max = proposed_max
    if current:
        yield current


def collect_main_vllm(spec: ExperimentSpec, root: str | Path, records: list[dict], dataset_name: str):
    if spec.collection.runtime_backend != "vllm":
        raise ValueError("collect_main_vllm requires a vLLM-backed spec")

    vllm = import_vllm()
    LLM = vllm.LLM
    SamplingParams = vllm.SamplingParams
    TokensPrompt = vllm.TokensPrompt

    root = Path(root)
    output_dir = root / spec.output.captures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    model_kwargs = {
        "model": spec.model.model_name,
        "trust_remote_code": spec.model.trust_remote_code,
        "enforce_eager": True,
        "gpu_memory_utilization": getattr(spec.collection, "vllm_gpu_memory_utilization", 0.9),
        "max_model_len": spec.model.max_length,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": getattr(spec.collection, "vllm_max_num_batched_tokens", 16384),
        "worker_extension_cls": "kv_prepend_experiment.worker_extension_main.VLLMMainCaptureExtension",
    }
    if spec.model.quantization:
        model_kwargs["quantization"] = spec.model.quantization

    llm = LLM(**model_kwargs)
    tokenizer = llm.get_tokenizer()

    examples = build_prompt_examples(records, spec.prompts, calibration_ids=set())
    examples = _annotate_examples(examples, tokenizer, spec)

    hook_config = {
        "default_rope_mode": spec.collection.default_rope_mode,
        "propagate_from_layer": spec.collection.propagate_from_layer,
        "main_capture_signals": list(spec.collection.main_capture_signals),
        "main_dense_layers": list(spec.collection.main_dense_layers),
        "main_router_layers": list(spec.collection.main_router_layers),
    }
    llm.collective_rpc("setup_hooks", args=(json.dumps(hook_config),))

    sampling = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    output_paths = []
    for batch_index, batch_examples in enumerate(_iter_batches(examples, spec)):
        batch_id = f"{dataset_name}_main_{batch_index:06d}"
        output_path = output_dir / f"{batch_id}.pt"
        payload = {
            "batch_id": batch_id,
            "output_path": str(output_path),
            "records": [
                {
                    "record_index": idx,
                    "text_id": example.text_id,
                    "dataset_name": dataset_name,
                    "kind": example.kind,
                    "prompt": example.prompt,
                    "prompt_key": _prompt_key(list(example.prompt_token_ids or [])),
                    "prompt_len": int(example.token_count or 0),
                    "prompt_token_ids": list(example.prompt_token_ids or []),
                    "content_token_mask": list(example.content_token_mask or []),
                    "tags": list(example.tags),
                }
                for idx, example in enumerate(batch_examples)
            ],
        }
        llm.collective_rpc("prepare_batch", args=(json.dumps(payload),))
        prompts = [TokensPrompt(prompt_token_ids=list(example.prompt_token_ids or [])) for example in batch_examples]
        llm.generate(prompts, sampling_params=sampling)
        llm.collective_rpc("transfer_and_save_async")
        output_paths.append(output_path)

    llm.collective_rpc("flush_saves")
    llm.collective_rpc("remove_hooks")
    return output_paths
